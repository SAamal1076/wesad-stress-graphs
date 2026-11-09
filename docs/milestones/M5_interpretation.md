# WESAD — M5 report: interpretation

*8 Nov 2026. Analyses fixed in `docs/analysis_plan.md`, commit ac25d7c, before running). All M5 results are descriptive.*

**Final model interpreted:** logistic regression (C = 0.1), chest + wrist level features + random-walk coupling graph (λ = 0.02), without the four drifting features, no calibration. It was chosen after M4, so its scores are not new evidence: binary per-subject macro-F1 0.937 LOSO / 0.943 cross-order; 3-class 0.728 / 0.697.

## 1. Is the heart-rate↔movement decoupling just movement or speaking?
Per-subject change of the edge weight under stress (15 subjects):

| | Raw | Adjusted for movement level/variability, heart rate, breathing irregularity | Movement-matched (same within-subject movement quintile) |
|---|---|---|---|
| Chest (ECG) | 15/15 negative, p = 0.0001 | 11/15 negative, p = 0.048 | 15/15 negative, p = 0.0001 |
| Wrist (BVP) | 15/15 negative, p = 0.0001 | 15/15 negative, p = 0.0001 | 14/15 negative, p = 0.0002 |

- The decoupling survives matching on movement in both devices, and covariate adjustment on the wrist. On the chest, adjustment weakens it (part of it goes with heart-rate level and breathing), but it stays negative for most subjects.
- Speaking cannot be fully separated from stress in WESAD (the stress test is spoken), so the claim is: "beyond movement level, heart rate follows movement less under stress".

## 2. Which signals matter? (retraining without each group; drop in mean per-subject F1)
- Binary, cross-order: chest EDA −0.034, wrist pulse (BVP) −0.032, ECG −0.022, chest coupling graph −0.013, respiration −0.008, EMG −0.006, temperature slopes −0.005.
- Removing wrist EDA or the wrist graph slightly *improves* the model (+0.003 to +0.008): they add noise.
- 3-class, cross-order: chest EDA and the chest coupling graph matter most (both −0.031), then respiration (−0.024) and EMG (−0.019). The chest graph is one of the most useful groups for separating amusement from baseline.

## 3. Does the model follow the self-reports?
- **Within each person (5 runs):** mean predicted P(stress) rises with self-reported stress: median Spearman ρ = 0.36 with STAI (positive in 14/15 subjects), 0.45 with SAM arousal, 0.67 with PANAS "stressed".
- **Between persons:** the size of the model's stress response does not track how stressed people said they were (ρ = −0.10 with the STAI change, ρ = 0.33 with the arousal change, both n.s.). Mean P(stress) under stress is 0.91, near ceiling: the model detects the condition, not its intensity.
- **Out-of-task check — meditation (never seen in training):** 36% of meditation windows are classified as stress (baseline 3%, amusement 9%), in both meditation runs. Meditation has slow, deep breathing (median 8.8 breaths/min, depth 9.2 vs 6.6 at baseline), and stress also lowers breathing rate and deepens breaths (14.7/min, depth 10.4). Removing the chest coupling graph lowers meditation false alarms from 36% to 21% (EMG removed: 25%; respiration removed: 27%) with almost no loss of stress detection (92% → 91%).
- Reading: part of what the model calls "stress" is "breathing differs from ordinary rest" (speaking, or guided slow breathing), and the chest coupling graph carries much of that. This matters for VR, where tasks also change breathing and speech.

## 4. Per-subject failures (final model, binary)
- Worst cross-order: S13 (0.77) and S15 (0.78). S15 did not believe the stress-test cover story (readme) and had the smallest heart-rate rise (+1.6 bpm vs a median of ~20). S13 is not explained by the available notes.
- S2 is no longer a failure once the drifting features are removed (0.89; temperature sensor not fully attached).
- Across subjects, F1 follows the physiological response (Spearman ρ = 0.39 with the heart-rate change) more than the self-reported change (ρ = −0.07); n = 15, descriptive.

## Files
`results/m5/decoupling_robustness.csv`, `decoupling_summary.csv`, `modality_ablation.csv`, `selfreport_vs_prediction.csv`, `selfreport_within_subject.csv`, `meditation_false_alarms.csv`, `per_subject_failures.csv`; figures `figures/m5/decoupling_robustness.png`, `modality_ablation.png`, `selfreport_vs_prediction.png`.
