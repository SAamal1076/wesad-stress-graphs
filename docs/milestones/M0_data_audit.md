# WESAD — M0 data audit (before any processing)

*8 Nov 2026. 15 subjects (S2–S17, no S12). Numbers are computed from the original `pkl` files at 700 Hz. These are quick descriptive checks (simple peak detector), not the final processing pipeline.*

## 1. What the data contains
- 15 folders, each with `SX.pkl`, `SX_quest.csv` and `SX_readme.txt`. No NaN in any signal.
- Chest signals are in physical units (ECG mV, EDA µS, Temp °C, ACC g, Resp %); wrist ACC is in units of **1/64 g** (do not mix it with the chest).
- Recording length 87–118 min. The chest and all wrist signals have **identical lengths to the second** for everyone.
- Synchronisation: the cross-correlation peak between chest and wrist movement energy is at lag ≤ 1 s for everyone.

## 2. Labels and class balance
- **45 % of the recording is label 0** (transient) and 2.7 % is labels 5–7. The part actually used is **52 %** (42–58 % depending on the subject).
- One clean segment each of baseline / stress / amusement and two meditation segments, with no fragmentation. Shortest segment: 337 s.
- Total minutes: baseline 294, stress 166, amusement 93, meditation 197.
- Stress share: **30 %** if non-stress = baseline + amusement, **22 %** if meditation is added. Majority baseline: 53 % (3 classes), 39 % (4 classes). Hence macro-F1 and balanced accuracy.
- 60 s windows with a 5 s step: 8142 windows, but heavily overlapping. Only 714 **non-overlapping** windows (stress 160, amusement 90). The effective sample is small.
- **The protocol order differs:** Base > Fun > Medi1 > TSST > Medi2 (8 subjects), Base > TSST > Medi1 > Fun > Medi2 (6), and S14: Base > TSST > Medi2 > Fun > Medi1. **Baseline always comes first.**

## 3. The `quest.csv` trap
- Times are in **minutes.seconds with the trailing zero dropped**: `50.3` = 50:30 and `32` = 32:00.
- The `quest.csv` time axis leads the `pkl` axis by a constant offset per subject (33–155 s); the correct interpretation gives 0.00 s deviation within each subject. **Segment the signals by the labels, not by the quest times.**
- Every protocol condition is exactly 20 s longer than its label segment (75 of 75).

## 4. Signal quality
| Issue | Detail |
|---|---|
| S3 ADC glitch | 0.12 s (83 samples) inside stress at t = 2531 s: every chest channel at the converter limits. It produced temperatures of −122/+72 °C after downsampling. **Fixed in `prepare_wesad.py`** |
| Zero padding | The first sample (t = 0) of every downsampled signal was distorted (label 0). **Fixed with `padtype="line"`** |
| Chest temperature S2 and S17 | Sensor not fully attached (per the readme). S2: 28.9 °C in baseline, then a jump at ~23 min |
| Almost flat chest EDA | S14 (SD 0.08 µS) and S8 (0.19): no visible response. Between-subject differences in chest EDA are huge (median 1.0 µS for S10, 12.3 for S7) |
| Flat-looking wrist ACC/TEMP | Device quantisation (1/64 g and 0.02 °C), not dropouts |
| ECG | Positive R polarity for everyone (≥ 94 % of peaks). Clipping at ±1.5 mV is rare (< 0.1 %). 0.997–1.0 of RR intervals are plausible |
| Wrist BVP | Matches the ECG heart rate in baseline (median difference 1.0 bpm) but **8.4 bpm in stress**: it degrades with movement |

## 5. Does physiology change between conditions? (descriptive, n = 15)
Stress vs baseline (number of subjects, out of 15, whose value increased):
- ECG heart rate: 15/15, median +22 bpm; BVP heart rate +12.
- EDA: chest 13/15, wrist 14/15.
- RMSSD: decreases in 10/15 (p = 0.07, weak).
- Movement: wrist rises in 12/15, chest in 13/15.
- Breathing rate **decreases** in 12/15 (13.8 vs 15.7).
- Amusement does not differ from baseline in heart rate (difference 0.2).
- 44 tests without correction; what counts is consistency (15/15) and effect size.

## 6. Methodological warnings (leakage and interpretation)
1. **Movement is confounded with stress:** the TSST involves speaking and arithmetic, so ACC rises and breathing slows. A model may learn "speaking and moving" rather than "stress". Evaluate models without ACC.
2. **Time is confounded with baseline:** baseline always comes first; chest temperature rises in everything after it (15/15) and wrist temperature falls. Any feature that drifts with time separates "early" from "late". Be careful with temperature and tonic EDA and monitor their importance.
3. **EDA decreases over the first ~20 minutes** (see `subjects/S2.png`: from 6 to 1.2 µS during baseline). The EDA level in baseline is not a stable "state".
4. **Personalisation (research question 3):** the within-subject effect size is about twice the between-subject effect size for chest EDA (1.08 vs 0.56) and temperature (1.22 vs 0.53), but similar for heart rate (1.28 vs 1.66). Between-subject differences are large, so baseline normalisation matters for EDA and temperature. It must be computed **inside the training fold only**, and for the test subject from their own baseline minutes only, and stated explicitly.
5. **Label validity:** STAI-state rose in 15/15 after the TSST (median +8, p = 0.0006) and valence fell in 14/15. Even the subjects whose readme says they did not feel stressed (S6, S15) had STAI increases ≥ +6. According to the readme, S3 reported valence = 7 after stress.
6. The PANAS row after the TSST has 26 items (it adds Angry and Irritated) while the others have 24, so item positions shift (used: Stressed = 21, Nervous = 16).

## 7. What changed in `prepare_wesad.py` after the first real run
1. `repair_adc_glitches`: glitch samples are replaced by linear interpolation before downsampling.
2. `resample_poly(..., padtype="line")` instead of zero padding.

## 8. Files
`results/eda/`: `inventory.csv`, `label_segments.csv`, `schedule_check.csv`, `condition_features.csv`, `subject_qc.csv`, `questionnaires.csv`, `glitches.csv`. Figures in `figures/eda/` (and `subjects/` for each subject).
