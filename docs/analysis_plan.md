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
