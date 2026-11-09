# Analysis plan (fixed before any physiological model is trained)

*Committed 8 Nov 2026, after M1 (windowing, shortcut baselines) and before M2 model results.
Any deviation from this plan is reported as such in the README.*

## Data and tasks
- WESAD, all 15 subjects, no exclusion. Known weak signals (chest temperature of S2 and S17,
  flat chest EDA of S14) are reported, not removed.
- Main task: stress vs non-stress (baseline + amusement). Secondary: baseline / stress / amusement.
  Meditation is never used for training.
- Windows lie entirely inside one condition run (labels from the RespiBAN label stream).
  Main window 60 s, step 5 s; 10 / 20 / 30 s for the latency question.

## Evaluation
1. Leave-one-subject-out (15 folds).
2. **Cross-protocol-order** (train on order A = amusement before stress, test on order B; and the reverse).
   Motivation: with LOSO, a model that only sees minutes-since-start reaches 95.7 % (binary) and
   93.6 % (3-class) pooled accuracy; cross-order removes this shortcut (66.9 % / 62.4 %).
- Metrics: per-subject macro-F1 and balanced accuracy (mean ± SD over subjects, primary);
  pooled accuracy and macro-F1 (only for comparison with Schmidt et al. 2018).
- Everything fitted from data (scaling, imputation, hyperparameters, graph-inference
  hyperparameters chosen by validation) is fitted on training subjects only. Hyperparameters are
  tuned by inner leave-one-subject-out on the training subjects, over small fixed grids.
- No per-subject normalisation that uses the test subject's whole session. Personalisation is a
  separate, labelled condition: features divided/centred by the test subject's own baseline run only.
- Random forests: 5 seeds, mean reported.

## Feature sets (fixed now)
- **Levels** (per window): ECG heart rate and HRV; BVP pulse rate, PRV and pulse amplitude; EDA tonic
  level/slope and phasic SCR statistics (chest, wrist); respiration rate, depth and variability;
  temperature mean/slope (chest, wrist); EMG envelope.
- **Motion**: accelerometer statistics (as in M1). Treated as a nuisance / shortcut group, reported separately.
- **G1 coupling graph** (per window, node series at 4 Hz, z-scored within the window):
  - chest nodes: heart rate (ECG), respiration, phasic EDA, EMG envelope, chest activity;
  - wrist nodes: pulse rate (BVP), pulse amplitude, phasic EDA, wrist activity.
  - Dependency matrix T = maximum absolute cross-correlation within ±5 s lag.
  - Three edge estimators: (a) T itself (correlation graph); (b) partial correlation (Ledoit-Wolf);
    (c) random-walk diffusion-kernel inference, min_w ||T − α expm(−L(w))||²_F + λ Σ w, w ≥ 0
    (Noda et al. 2014; Al-Ali et al. 2019; Karfoul & Almuhammad Alali 2026), which separates direct
    from indirect (multipath) coupling.
  - Features: edge weights, total strength, algebraic connectivity, α.
- **G2 test-time graph smoothing** (on predicted probabilities of the test subject, label-free):
  causal (past windows only) graph Tikhonov smoothing on a graph built from temporal adjacency and
  feature similarity; compared with a causal moving average over the same horizon.

### Implementation details fixed during M2 (before any model was trained)
- Before random-walk inference, T is symmetrically normalised (D^-1/2 T D^-1/2), because the
  diffusion kernel has unit row sums; without it the fit is poor (relative residual ~0.7 vs ~0.05).
- Random-walk lambda grid {0.005, 0.02, 0.05}, chosen by inner leave-one-subject-out.
- Wrist pulse peaks are detected on the inverted E4 BVP signal (more consistent timing vs ECG).

### Decisions fixed before M3 (before any classifier result)
- Primary model: logistic regression (median imputation, standardisation, class-balanced), with C
  and the random-walk lambda tuned by inner cross-validation grouped by subject (5 folds, for
  compute reasons, instead of inner LOSO). Grid: C in {0.01, 0.1, 1}; lambda in {0.005, 0.02, 0.05}.
- Robustness model: random forest (100 trees, min_samples_leaf 5, class-balanced, 5 seeds),
  run for "levels" and "levels + random-walk graph" only; it uses the lambda selected by the
  logistic-regression inner CV of the same outer fold.
- Holm correction over the primary family, separately for LOSO and for cross-order, logistic
  regression only: H1 (2 tasks), H2 (chest, wrist x 2 tasks), H3 (2 alternatives x chest, wrist
  x 2 tasks) = 14 paired Wilcoxon tests on per-subject macro-F1.

### Decisions fixed before M4 (after M3 results, before any M4 result)
**Personalisation (M4a).**
- Calibration = the test subject's first 2 minutes of the baseline run (secondary: 5 minutes); the
  same procedure is applied to every training subject. Every feature is centred on the mean of the
  calibration windows (windows ending inside the calibration period). No scaling by calibration SD.
- Windows starting inside the calibration period are removed from evaluation for every model in M4,
  personalised or not, so all M4 comparisons use identical test windows.
- Configurations (logistic regression, tuned as in M3): both/levels, both/levels+rw, both/levels+pcorr,
  chest/levels, chest/levels+rw, wrist/levels, wrist/levels+rw; each without and with personalisation.
- Sensitivity ("no-clock"): both/levels and both/levels+rw without absolute temperature and tonic
  skin-conductance level (temp_*_mean, eda_*_scl_mean), because early calibration plus slow drift can
  re-introduce the clock. These four features were identified as drifting in the M2 descriptive analysis.

**G2 test-time graph smoothing (M4b)**, applied to out-of-fold probabilities of the test subject,
causal (only windows whose end time is not later than the current window's end), fixed hyperparameters
(no tuning, to avoid any nested leakage):
- horizon H = 120 s (primary); sweep H in {30, 60, 120, 300} s reported in full, not used for selection;
- graph on the windows inside the horizon: W_ij = exp(-(dt_ij / 60 s)^2) * exp(-d_ij^2 / (2 s^2)),
  d = Euclidean distance between level-feature vectors standardised with training-subject statistics,
  s = median of the d_ij inside the horizon;
- Tikhonov: x = (I + L)^-1 p, output at the current window; 3-class: each class smoothed, renormalised;
- controls: causal moving average over the same horizon; Tikhonov with the temporal kernel only.
- Latency: time from the start of the stress run to the end of the first window predicted as stress.
- Bases: both/levels+rw and wrist/levels+rw, personalised (2 min); both/levels+rw not personalised.

**Hypotheses M4** (paired Wilcoxon on per-subject macro-F1, Holm within each scheme, 14 tests):
- H5: personalised > not personalised (both/levels, both/levels+rw; 2 tasks) = 4 tests.
- H6: with personalisation, levels+rw > levels (chest, wrist, both; 2 tasks) = 6 tests.
- H4: G2 (full graph) > moving average, H = 120 s (bases both and wrist, personalised; 2 tasks) = 4 tests.

### Decisions fixed before M5 (interpretation; after M4, before any M5 result)
- "Final model" for interpretation: logistic regression, chest + wrist levels + random-walk graph,
  without the four drifting features, no calibration (the most robust M4 configuration under
  cross-order). Fixed hyperparameters C = 0.1, lambda = 0.02 (middle of the grids), no tuning.
  It is selected after seeing M4, so its scores are not new evidence; M5 only interprets it.
- Decoupling robustness: per subject, OLS of the heart-rate~movement edge (random walk, lambda 0.02;
  chest and wrist) on condition (stress, amusement vs baseline) adjusted for movement level and
  variability, heart-rate level and breathing irregularity (z-scored within subject); the 15 adjusted
  stress coefficients are tested with a Wilcoxon signed-rank test. Plus a movement-matched comparison
  (stress vs non-stress windows inside the same within-subject movement quintile).
- Modality importance: leave-one-signal-group-out retraining under LOSO and cross-order
  (groups: ECG, BVP, chest EDA, wrist EDA, respiration, temperature, EMG, chest graph, wrist graph).
- Self-reports: the final model (trained without meditation) predicts every window of the test
  subject, including meditation. Per run: mean P(stress) vs STAI state, SAM arousal, PANAS 'stressed'
  (within-subject Spearman over 5 runs; between-subject Spearman of the TSST-minus-baseline changes).
- Failures: per-subject F1 vs self-reported stress change, heart-rate change and readme notes.
All M5 analyses are descriptive; p-values are reported without claims of confirmation.

### Decisions fixed before the TV-graph experiment (M6-B; after M7, before any WESAD computation)
Approved by Salman on 8 Nov 2026 (option B: nodes = signals inside one window).
- Nodes and node signals: chest graph (heart rate, respiration, phasic EDA, EMG envelope, chest movement),
  wrist graph (pulse rate, pulse amplitude, phasic EDA, wrist movement); each node's 60 s series on the
  4 Hz grid, smoothed by a 2 s moving average, z-scored inside the window and divided by sqrt(m), so that
  ||y_s|| = 1 and ||y_s - y_t|| = sqrt(2(1 - rho_st)).
- Joint objective, solved by alternating minimisation (X-step: ADMM network lasso; W-step: Kalofolias
  primal-dual log-degree graph learning):
  min_{X, W} 1/2 ||X - Y||_F^2 + lambda * sum_{s<t} W_st ||x_s - x_t||_2 - beta * sum_s log d_s + gamma * ||W||_F^2,
  W symmetric, W >= 0, d_s = sum_t W_st.
- beta = 1, gamma = 0.25 (graph scale); lambda in {0.1, 0.3, 0.6}, chosen by the inner cross-validation of the
  classifier exactly as in M3. These scale values may be revised once, after the synthetic stage only, before
  any WESAD window is processed; any revision is recorded here.
- Per-window features: learned edge weights, node degrees, algebraic connectivity, number of fused clusters
  (edges with ||x_s - x_t|| < 1e-3 merged), TV of Y on the learned graph.
- Comparison (logistic regression, M3 tuning, LOSO and cross-order, binary and 3-class), base = level features
  without the four drifting features: base; base + TV graph; base + random-walk graph; base + correlation graph;
  for chest, wrist and both.
- Known assumption: the TV term treats coupling as positive and uses no lag search (a 2 s smoothing tolerates
  delays of a few seconds).
- Hypotheses (paired Wilcoxon on per-subject macro-F1, Holm within each scheme, 8 tests):
  H8: base + TV graph > base (chest, wrist; 2 tasks). H9: base + TV graph > base + random-walk graph (chest, wrist; 2 tasks).
- Descriptive: stress effect on the heart-rate~movement edge (d_z); meditation windows called stress.

### Amendment 1 to M6-B (8 Nov 2026, approved by Salman; before any WESAD window is processed)
Reason (stage B1, synthetic data only, commit e1fb204): with beta = 1 the model depends on a single number,
theta = lambda / sqrt(2 gamma) (substituting w = v / sqrt(2 gamma); checked numerically to 1e-9). The
pre-specified values give theta = 0.14-0.85, where the learned weights are almost uniform. Values that
recover two planted clusters (theta ~ 3-10) also make independent signals collapse into one cluster
(93 % of trials at theta = 3.2, 13 % at 9.5, 10 % at 16), and a 2+2+1 structure is never recovered exactly.
The cause is a feedback loop in the joint model: fused nodes get zero distance, hence a larger weight,
hence stronger fusion. This is a design change, not only the scale revision allowed above.
- New objective (eta in [0, 1] mixes the distance seen by the graph between the data Y and the denoised X):
  min_{X, W} 1/2 ||X - Y||^2 + lambda * sum_{s<t} W_st [ (1 - eta) ||y_s - y_t|| + eta ||x_s - x_t|| ]
             - beta * sum_s log d_s + gamma * ||W||_F^2.
  eta = 1 is the original model; eta = 0 is convex graph learning from the total variation of Y (X = Y);
  0 < eta < 1 keeps the alternating updates while the distance keeps the floor (1 - eta) ||y_s - y_t|| > 0.
- Scale: beta = 1, gamma = 0.5, so that lambda = theta.
- Feature change: the number of fused clusters is replaced by the fusion degree
  F = 1 - TV_W(X) / TV_W(Y), TV_W(X) = sum_{s<t} W_st ||x_s - x_t||, in [0, 1]. At eta = 0, F = 0 by
  construction and is not used.
- Synthetic gate (run once, before WESAD). Grid eta in {0, 0.25, 0.5}, theta in {1, 2, 4, 8}; n = 5 nodes,
  m = 240 samples (60 s at 4 Hz); 100 trials per scenario; seed 2026. Each node series is white Gaussian
  noise (or a group source plus such noise) passed through the same preparation as WESAD
  (2 s moving average, z-score, divide by sqrt(m)). Source and noise have equal variance per sample,
  so the expected within-group correlation is 1 / (1 + s^2) for noise scale s.
  S1 null: 5 independent nodes. S2a: groups {0,1,2},{3,4}, within correlation 0.6. S2b: same groups,
  within correlation 0.3. S3: groups {0,1},{2,3},{4}, within correlation 0.6.
  A setting passes if all hold:
    (1) S1: share of trials with any fused edge (ADMM split variable ||v_e|| < 1e-6) <= 5 %;
    (2) S2a and S2b: mean within-group weight >= 3 x mean between-group weight in >= 90 % of trials;
    (3) S3: the same ratio >= 3 in >= 80 % of trials.
  Selection among passing settings: largest eta; then lowest S1 fusion rate; then highest minimum pass
  rate over S2a, S2b, S3; then smallest theta. If no setting passes, M6-B stops and is reported as a
  negative result; the random-walk graph (M3) remains the graph method.
  Descriptive only (not part of the decision): the same null with slow signals (AR(1), phi = 0.95 before
  the preparation), closer to heart-rate dynamics.
- The selected (eta, theta) is fixed for every WESAD run; neither is tuned on WESAD (this replaces the
  inner-CV choice of lambda above). Hypotheses H8-H9 and the comparison set are unchanged.
- Gate result (8 Nov 2026, experiments/m6_gate.py, results/m6/gate*.csv): passing settings were
  (eta, theta) = (0, 4), (0, 8), (0.25, 4), (0.25, 8); no eta = 0.5 setting passed (independent signals
  fused in 100 % of trials at theta >= 4). Selected by the rule above: eta = 0.25, theta = 4
  (S1 fusion 1 %, S2a 100 %, S2b 92 %, S3 100 %). Descriptive slow null: some edge fused in 15 % of trials.

### M6-C - online time-varying TV graph and a decoupling index (8 Nov 2026, approved by Salman)
Fixed before any M6-C computation and before the M6-B WESAD results (stage B2/B3) were seen.
Motivation: the robust M2/M5 finding (heart-rate~movement coupling weakens under stress, 15/15 subjects,
surviving movement matching) is a graph form of "additional heart rate" (heart rate beyond what movement
requires; Blix et al. 1974; van de Ven et al., Psychophysiology 2025). M6-C asks whether a causal,
time-varying coupling graph adds to the classic additional-heart-rate index.
- Model (causal, one convex problem per window t, given the previous graph):
  w_t = argmin_{w>=0} theta z_t^T w - beta sum_s log d_s + 2 gamma ||w||^2 + mu ||w - w_{t-1}||_1,
  z_t,e = ||y_t,s - y_t,r|| (prepared node signals as in M6-B). eta = 0 (no denoising step: the online
  problem must stay convex); theta = 4 (passed the M6-B gate at eta = 0); beta = 1, gamma = 0.5.
- mu chosen once by a synthetic gate (below), not tuned on WESAD.
- Timeline: per subject, 60 s windows ending every 5 s over the whole recording (labelled or not).
  A labelled window takes the estimate of the latest grid window ending at or before its end.
- Calibration: grid windows lying entirely inside the first 120 s of the subject's baseline run;
  excluded from every index evaluation below.
- Decoupling index: D_t = mean_p ( Wbar_pb - W_t,pb ), b = movement node of the device, p = its
  physiological nodes (chest: hr_ecg, resp, eda_ph_chest, emg_env; wrist: hr_bvp, pulse_amp,
  eda_ph_wrist), Wbar = mean over the calibration windows. Higher D = more decoupling.
- Comparator, additional heart rate: HR_t, ACT_t = window means of the device heart-rate and movement nodes;
  slope b fitted on the training subjects' baseline + amusement grid windows (within-subject centred least
  squares, LOSO); intercept a_i = mean over the subject's calibration windows of HR - b ACT;
  aHR_t = HR_t - a_i - b ACT_t.
- Combined score: logistic regression (standardised inputs, C = 1) on [aHR, D], trained on the training
  subjects' labelled grid windows (stress vs baseline + amusement, calibration excluded), LOSO.
- Hypotheses on per-subject AUC (paired Wilcoxon over 15 subjects, two-sided, Holm over these 4 tests):
  H10: combined > aHR alone for stress vs baseline + amusement (chest; wrist).
  H11: combined > aHR alone for stress vs meditation (chest; wrist); meditation is never used for training.
- H12: classification with the M3 pipeline (LR, inner-CV C), base = device levels without the four drifting
  features; base + online-graph features (edges, degrees, algebraic connectivity at the window) > base;
  chest and wrist, binary and 3-class; LOSO and cross-order; Holm over the 4 tests within each scheme.
  H8 and H9 (M6-B per-window TV graph) are run as registered, in the same pipeline.
- Descriptive: D alone and HR - calibration mean as scores; detection latency at stress onset and false-alarm
  rate (threshold = 95th percentile of the score on training subjects' baseline + amusement grid windows;
  detection = 3 consecutive grid windows above threshold after the stress run starts); time courses of D and
  aHR for all 15 subjects.
- Synthetic gate for mu (seed 2027, 100 trials per scenario): n = 5 nodes, 20 min at 4 Hz, the same
  preparation and grid; nodes 0 and 1 share a source (within correlation 0.6) before 600 s and are
  independent after ("decouple"), or the reverse ("couple"); other nodes independent white noise.
  Grid mu in {0, 0.25, 0.5, 1, 2, 4}. Measures on edge (0,1): AUC between windows ending <= 600 s and
  windows starting >= 600 s; delay = time from 600 s until the edge passes the midpoint of its pre and post
  medians and stays past it for 3 consecutive windows. Eligible: median delay <= median delay at mu = 0
  plus 30 s, in both scenarios. Selected: highest mean AUC over the two scenarios; ties within 0.005 go to
  the smaller mu. mu = 0 is always eligible, so the gate always selects a value; mu = 0 would mean that
  temporal regularisation does not help and M6-C reduces to the independent per-window graph.
- mu gate result (8 Nov 2026, experiments/m6c_gate.py, results/m6/gate_mu*.csv): mu = 0.25 selected
  (mean AUC 0.998 vs 0.992 at mu = 0; median delay 47.5 / 42.5 s vs 30 / 32.5 s). mu >= 0.5 exceeded the
  delay limit; mu = 4 froze the graph in the decouple scenario (AUC 0.07).

## Hypotheses (primary comparisons, paired Wilcoxon over the 15 subjects, Holm-corrected)
- H1: physiological levels beat motion-only under cross-order evaluation (binary and 3-class).
- H2: levels + G1 (random-walk edges) beat levels alone (macro-F1), for chest and for wrist.
- H3: random-walk edges beat correlation edges and partial-correlation edges as added features.
- H4: G2 graph smoothing beats the moving average with the same causal horizon.
Secondary, descriptive: drift of each feature within the baseline run ("clock-likeness");
comparison with self-reports; per-subject failures.

## Reporting
All configurations run are reported, including those that do not help. "Better than
Schmidt et al. 2018" is only stated for the same task, same LOSO protocol and same metric.
