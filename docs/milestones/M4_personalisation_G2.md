# WESAD — M4 results: personalisation and test-time graph smoothing (G2)

*8 Nov 2026. Repository `wesad-stress/` (commit 3f3aadf). All M4 settings were committed in `docs/analysis_plan.md` before running (commit 55eb58b). Evaluation excludes the windows overlapping the calibration period for every model, so all comparisons use identical test windows.*

## Personalisation: centre features on the test subject's first 2 minutes of baseline
Per-subject macro-F1, logistic regression, chest + wrist:

| Features | Binary LOSO none → calib | Binary cross-order none → calib | 3-class LOSO none → calib | 3-class cross-order none → calib |
|---|---|---|---|---|
| levels | 0.922 → 0.935 | 0.879 → 0.922 | 0.698 → 0.787 | 0.633 → 0.710 |
| levels + random walk | 0.924 → 0.931 | 0.889 → 0.927 | 0.743 → 0.775 | 0.690 → 0.706 |
| levels + partial corr. | 0.936 → 0.966 | 0.895 → 0.941 | 0.757 → 0.812 | 0.682 → 0.710 |

- Pooled LOSO accuracy, best configuration (levels + partial correlation + 2-min calibration): **97.3% binary, 86.5% 3-class** (paper: 93.1% / 80.3%). The paper used no calibration, so this is not a like-for-like comparison.
- **H5 not significant:** mean gains are positive, but only 7–10 of 15 subjects improve; the means are driven by a few large gains (for example S13 binary cross-order 0.64 → 0.99, S2 0.47 → 0.82) while others lose a little. Best p: 3-class LOSO levels, +0.089, p = 0.08, Holm 1.0.
- 5-minute calibration was not better than 2 minutes.

## "No-clock" sensitivity (without absolute temperature and tonic skin-conductance level)
- **Without calibration, removing the four drifting features helps generalisation across protocol order:** binary cross-order 0.879 → 0.928 (levels) and 0.889 → **0.943** (levels + random walk) — the best cross-order binary result of the project, with no calibration at all.
- S2 is the clearest case: 0.47 → 0.89. S2's chest temperature sensor was not fully attached (readme), so the temperature features were misleading.
- **With calibration, the 3-class gain partly comes back from the clock:** 3-class LOSO with calibration 0.787 with the drifting features vs 0.718 without. "Feature minus early baseline" grows with time, and baseline is always first, so calibration plus slow drift helps separate baseline from later conditions. This was anticipated in the plan and is reported as a limitation of baseline calibration in a fixed-order protocol.
- Without the drifting features, adding random-walk graph features gives small gains everywhere (binary cross-order +0.015, 3-class LOSO +0.026); exploratory, not in the tested family.

## H6: graph features with personalisation
Not supported: levels + random walk vs levels with calibration, differences −0.012 to +0.019, no p below 0.16.

## G2: causal graph smoothing of the test subject's probabilities
- Smoothing in general helps a little (H = 120 s: +0.5 to +2 points of F1; more at H = 300 s in 3-class).
- **H4 not supported: graph smoothing is not better than a moving average with the same horizon** (differences −0.008 to +0.001; in 3-class LOSO the moving average is better for 10/15 subjects, p = 0.03–0.05 uncorrected, Holm 0.48–0.65).
- Latency could not be measured meaningfully: only labelled windows are scored, the label-0 gap before the stress run is longer than the horizon, and the first stress window is already detected for the median subject (60 s, the window length). A continuous-stream evaluation including transition periods is needed.

## Overall reading after M3–M4
- The graph methods (G1 features, G2 smoothing) did not give statistically supported improvements with 15 subjects.
- The robust, defensible contributions are methodological:
  1. the clock shortcut and the cross-protocol-order evaluation that removes it;
  2. the feature-drift ("clock-likeness") analysis, and the gain from removing drifting features (0.943 cross-order binary without calibration);
  3. the stress-specific heart-rate↔movement decoupling (15/15 subjects within person, M2);
  4. a concrete, explained failure case (S2, detached temperature sensor).
- Relevance to VR-STRESS: VR training scenarios also run in fixed orders, so the same time-in-session shortcut and the same cross-order test apply.

## Files
`results/m4/personalisation_summary.csv`, `personalisation_per_subject.csv`, `g2_per_subject.csv`, `g2_latency.csv`, `hypothesis_tests.csv`; `figures/m4/personalisation_per_subject.png`.
