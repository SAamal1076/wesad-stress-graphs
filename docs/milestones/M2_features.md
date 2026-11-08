# WESAD — M2 report: physiological features and coupling graphs (G1)

*8 Nov 2026. Repository `wesad-stress/` in the Claude session (commit 7761824). No classifier has been trained yet; nothing below was used to choose features (the feature sets were fixed in `docs/analysis_plan.md`, committed before M2).*

## What was built
- **Per-subject processing** (`src/wesad/physio.py`, NeuroKit2 0.3.0 + own code): ECG R-peaks, wrist pulse peaks and pulse amplitude, breaths and breath depth, EDA tonic/phasic split and SCR peaks (chest and wrist), temperature, EMG envelope, movement intensity. Unsupervised and per subject, so no leakage across subjects. Filters are zero-phase (offline; a limitation for real time).
- **38 level features per window** (60 s, step 5 s; 8143 windows): heart rate and HRV (ECG), pulse rate, PRV and pulse amplitude (BVP), EDA level/slope/SCRs (chest, wrist), respiration rate/variability/depth, temperature mean/slope, EMG.
- **G1 coupling graphs per window** (`src/wesad/graphs.py`): nodes on a 4 Hz grid, z-scored inside the window.
  - Chest graph: heart rate, respiration, phasic EDA, EMG envelope, chest movement (10 edges).
  - Wrist graph: pulse rate, pulse amplitude, phasic EDA, wrist movement (6 edges).
  - Dependency matrix T = max |cross-correlation| within ±5 s.
  - Edges by three estimators: correlation (T), partial correlation (Ledoit-Wolf), and **random-walk diffusion-kernel inference** (Salman's J4/C7 objective, min ||T − α·expm(−L(w))||² + λΣw, w ≥ 0), batched, with the exact gradient of the matrix exponential.
- **Tests: 12 pass**, including: analytic gradient = finite differences; on synthetic sparse graphs the random-walk estimate recovers edges better than reading T directly (AUC); on a chain a–b–c the indirect edge a–c is set to exactly 0 while T(a,c) = 0.16.
- One implementation detail fixed before any model and recorded in the plan: T is normalised D^-1/2 T D^-1/2 before random-walk inference (relative residual ~0.7 → ~0.05), because the diffusion kernel has unit row sums.

## Descriptive findings (paired over 15 subjects)
1. **Stress decouples heart rate from movement.** The heart-rate↔movement edge is weaker under stress than baseline in 15/15 subjects.
   - Chest, random walk: d_z = −2.0 (correlation −1.7, partial correlation −1.4). Wrist, random walk: −1.8.
   - Not under amusement (amusement vs baseline: −0.04 chest, −0.29 wrist), and stress vs amusement −1.0 / −1.2: the effect is specific to stress.
   - Reading: at rest, heart rate follows small movements; under stress it rises independently of movement (a non-metabolic heart-rate increase). To check: whether this is fully explained by speaking/standing in the stress task (M5), and the related literature before writing it in the README.
2. **Levels are stronger, couplings are less clock-like.** Median |d_z| stress vs baseline: levels 0.96, random-walk edges 0.48, correlation 0.33, partial correlation 0.25. Median drift inside the 20-min baseline (|Spearman ρ| with time): levels 0.22, coupling features 0.14–0.16. The most clock-like features are among the most stress-sensitive levels: chest temperature (ρ = 0.93), wrist and chest skin-conductance level (0.92, 0.90), wrist temperature (0.75). Coupling features are a complement, not a replacement.
3. **Wrist heart rate is unreliable during stress.** Median |HR(ECG) − HR(BVP)| per window: 1.1 bpm at baseline, 9.6 bpm under stress. More than 10 bpm off in 96–100% of stress windows for S11, S13, S16. So wrist PRV "effects" under stress (pulse-rate SD has the largest effect, d_z = 2.4) partly measure motion artefacts, not physiology. Important for the wrist-only (VR-like) claim.
4. Respiration depth rises under stress in 15/15 (d_z = 1.8) with almost no drift (0.08), but speaking during the stress test is a confound.

## Files
`results/m2/features_60s.csv` (8143 × 162), `feature_effects.csv`, `family_summary.csv`, `qc_bvp_hr_disagreement_gt10bpm.csv`; figures `figures/m2/coupling_edges_stress_effect.png`, `effect_vs_drift.png`.

## Proposed M3
Classifiers (logistic regression, random forest) under LOSO and cross-order, binary and 3-class: levels alone; levels + G1 (correlation / partial correlation / random walk, λ chosen by inner LOSO); chest, wrist, both; motion and time shortcuts alongside; paired Wilcoxon over subjects with Holm correction for H1–H3.
