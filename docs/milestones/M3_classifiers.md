# WESAD — M3 results: classifiers under LOSO and cross-protocol-order

*8 Nov 2026. Repository `wesad-stress/`. Decisions (primary model, tuning, Holm family) were committed in `docs/analysis_plan.md` before any classifier was run (commit f79d868).*

## Setup
- 60 s windows, step 5 s; binary (stress vs baseline + amusement) and 3-class.
- Primary model: logistic regression, C and random-walk λ tuned by inner 5-fold CV grouped by subject on training subjects only. Robustness: random forest, 5 seeds (levels and levels + random walk only).
- Feature sets per sensor (chest / wrist / both): levels; levels + correlation graph; levels + partial-correlation graph; levels + random-walk graph; random-walk graph only. Shortcuts: motion, minutes-since-start.
- Metric: per-subject macro-F1 (mean over 15 subjects); pooled accuracy only for the paper comparison.

## Main numbers — binary, per-subject macro-F1 (logistic regression)
| Inputs | LOSO | Cross-order |
|---|---|---|
| Minutes since start (RF) | 0.951 | 0.546 |
| Motion only | 0.805 | 0.746 |
| Wrist levels | 0.889 | 0.841 |
| Chest levels | 0.879 | 0.874 |
| Chest + wrist levels | 0.918 | 0.875 |
| Chest + wrist levels + random-walk graph | 0.921 | 0.886 |
| Chest + wrist levels + partial-correlation graph | 0.932 | 0.892 |
| Random-walk graph only (chest + wrist) | 0.631 | 0.630 |

3-class, same order: time (RF) 0.903 / 0.424 (LR 0.335 / 0.399); motion 0.622 / 0.502; wrist levels 0.621 / 0.533; chest levels 0.681 / 0.678; both levels 0.693 / 0.631; both + random walk 0.738 / 0.687; both + partial correlation 0.752 / 0.679.

## Comparison with Schmidt et al. 2018 (LOSO, pooled accuracy / macro-F1)
- Binary: chest levels LR 90.7% (RF 93.1%) vs paper 93.1%; wrist levels LR 90.3% / 88.9% vs paper 88.3% / 86.1%; chest + wrist + partial-correlation graph LR **94.0% / 93.0%** vs paper's best 93.1% / 91.5% (chest, LDA).
- 3-class: chest + wrist + partial-correlation graph LR 79.6% / 75.8% vs paper's best 80.3% / 72.5% (chest, AdaBoost): higher F1, slightly lower accuracy.
- Caveats: the clock-only random forest reaches 95.7% binary with LOSO, so beating 93% under LOSO alone proves little; the paper used 0.25 s window shifts and different features; the highest numbers come from partial correlation, which was not the pre-specified main estimator.

## Hypothesis tests (paired Wilcoxon on per-subject macro-F1, Holm within each scheme, 14 tests)
- **H1 (physiology > motion): not significant after correction.** Binary LOSO: +0.11, 11/15 subjects better, p = 0.004, Holm 0.06. Cross-order: +0.13, 11/15, p = 0.04, Holm 0.58.
- **H2 (levels + random walk > levels): not supported.** Largest: chest 3-class LOSO +0.035, 11/15 better, p = 0.07, Holm 0.95. Binary differences are ≤ 0.005.
- **H3 (random walk > correlation / partial correlation): not supported.** Differences within ±0.02, no test below p = 0.17.
- So: adding coupling graphs gives small, consistent-looking gains for chest + wrist and in the 3-class task (≈ +3–4 points pooled), but with 15 subjects they are not statistically distinguishable from zero. The random-walk estimator is not better than simpler estimators as a classification feature.

## What generalises
- Physiology is robust to the protocol order: chest + wrist levels lose ~4 points of binary F1 from LOSO to cross-order, while the clock shortcut loses ~40.
- Chest levels hold almost unchanged (0.879 → 0.874); wrist drops more (0.889 → 0.841), consistent with the motion-contaminated wrist heart rate found in M2.
- Coupling graphs alone are weak across subjects (F1 ≈ 0.63) even though the heart-rate↔movement edge separates stress from baseline in 15/15 subjects within each person (M2). Coupling is informative relative to the person's own baseline, not in absolute values: this is what M4 (personalisation and test-time graph smoothing G2) should test.

## Per-subject failures (chest + wrist levels, LR, binary)
- S2 (cross-order 0.42, LOSO 0.80) and S13 (0.62 / 0.67) are the worst; S15 0.76 / 0.75. S2 is also among the low-performance subjects reported by PAMG-AT.
- 3-class: amusement vs baseline is the main confusion (as in the paper).

## Files
`results/m3/summary.csv`, `per_subject.csv`, `hypothesis_tests.csv`, `oof/*.npz` (out-of-fold probabilities, used by M4), `figures/m3/per_subject_f1.png`.
