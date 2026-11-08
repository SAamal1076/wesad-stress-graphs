# Stress detection on WESAD: shortcuts, generalisation and physiological coupling graphs

A leakage-safe, pre-registered analysis of acute-stress detection from wearable physiology on the open
[WESAD](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) dataset
(Schmidt et al., ICMI 2018), with a focus on **what generalises across people and across protocol order**.

The full end-to-end report (fundamentals, methods, results, discussion) is in [`docs/REPORT.md`](docs/REPORT.md).

## Key findings

1. **A clock beats the original paper.** Minutes since the start of the recording alone reach **95.7 %**
   pooled accuracy for stress vs non-stress under leave-one-subject-out (paper: 93.1 %), because the
   protocol schedule is almost identical for everyone.
2. **Cross-protocol-order evaluation removes the shortcut.** Training on one protocol order and testing on
   the other drops the clock to 66.9 %, while physiology keeps per-subject macro-F1 at 0.88–0.94.
3. **Drifting features hurt generalisation.** Removing four slowly drifting features (absolute temperature,
   tonic skin-conductance level) raises cross-order binary macro-F1 from 0.889 to **0.943** without any
   calibration, and fixes the worst subject (S2, detached temperature sensor).
4. **Stress decouples heart rate from movement.** The heart-rate↔movement coupling weakens under stress in
   15/15 subjects, survives matching on movement level, and does not occur under amusement.
5. **Graph methods did not give significant accuracy gains** (pre-registered tests, Holm-corrected, n = 15),
   and the model calls 36 % of (never-seen) meditation windows "stress": slow deep breathing resembles
   breathing during the spoken stress test.
6. **A learned total-variation graph, tracked causally in real time, confirms the decoupling but adds
   nothing to heart rate.** A synthetic test exposed false fusion in the planned joint model before any
   WESAD data was used; the amended convex online version works, but heart rate minus its rest level stays
   the best index (per-subject AUC 0.94 chest), and correcting heart rate for movement hurts because the
   stress test itself involves movement.

| Binary task, per-subject macro-F1 (mean of 15) | LOSO | Cross-protocol-order |
|---|---|---|
| Minutes since start only (random forest) | 0.951 | 0.546 |
| Movement only | 0.805 | 0.746 |
| Wrist physiology | 0.889 | 0.841 |
| Chest physiology | 0.879 | 0.874 |
| Chest + wrist physiology | 0.918 | 0.875 |
| Chest + wrist, no drifting features, + random-walk coupling graph | 0.933* | 0.943* |

\* scored without the first two baseline minutes (see report, section 9). Logistic regression unless stated.

<p align="center"><img src="figures/report/clock_vs_cross_order.png" width="820"></p>

## Method in one paragraph

60 s windows (step 5 s) inside one condition; tasks as in the paper (stress vs baseline + amusement;
baseline / stress / amusement). 38 physiological level features (ECG/HRV, wrist pulse, EDA, respiration,
temperature, EMG) and, per window, **coupling graphs** between heart rate, respiration, phasic EDA, EMG
and movement, estimated by correlation, partial correlation, and **random-walk diffusion-kernel graph
inference** (min ‖T̃ − α·expm(−L(w))‖² + λΣw, exact matrix-exponential gradient, unit-tested).
A second family of graphs is **learned from the total variation of the signals on the graph** (Kalofolias
log-degree graph learning, per window and as a causal time-varying graph with an l1 penalty on graph changes),
compared with the classic additional-heart-rate index.
Evaluation by leave-one-subject-out **and** cross-protocol-order, everything fitted inside the training
folds, per-subject metrics, paired Wilcoxon tests with Holm correction. Every stage's settings were
committed to [`docs/analysis_plan.md`](docs/analysis_plan.md) **before** it was run (see the commit history).

## Repository layout

```
scripts/        prepare_wesad.py (shrink WESAD to per-subject .npz), eda_audit.py, eda_figures.py (data audit)
src/wesad/      windows.py, evaluation.py (LOSO + leakage checks), physio.py (signal processing, features),
                graphs.py (dependency matrix, partial correlation, random-walk graph inference),
                tvgraph.py (joint graph-TV denoising + graph learning), tvgraph_online.py (causal time-varying graph)
experiments/    m1_* shortcuts and cross-order check · m2_* features and descriptive effects ·
                m3_* classifiers and tests · m4_* personalisation, no-clock sensitivity, G2 smoothing ·
                m5_interpret.py (decoupling robustness, ablation, self-reports, meditation, failures) ·
                m6_* / m6c_* learned TV graphs: synthetic gates, WESAD features, decoupling index, tests
tests/          leakage, windowing and graph-inference tests (pytest)
results/        summary tables (CSV) per milestone; per-window data is not redistributed
figures/        all figures
docs/           REPORT.md, analysis_plan.md (pre-registration), data audit, milestone reports
```

## Reproduce

1. Download WESAD (2.5 GB) from the [official link](https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx) and
   unzip it to `data/WESAD/` (folders `S2` … `S17`). Raw data is not included here (dataset terms).
2. `pip install -r requirements.txt` (Python 3.13 was used).
3. `bash run_all.sh` — runs the whole pipeline in order (about 1.5 hours on 2 CPU cores, 8 GB RAM).

## Data and licence

- Dataset: P. Schmidt, A. Reiss, R. Duerichen, C. Marberger, K. Van Laerhoven, *Introducing WESAD, a
  multimodal dataset for wearable stress and affect detection*, ICMI 2018. Scientific, non-commercial use.
- Code: MIT licence (see `LICENSE`).

## Authorship and AI assistance

Project direction, research questions, decisions and review: **Salman Almuhammad Alali**. The code, the
analyses and the reports were written and run with an AI assistant (Anthropic's Claude) under his
direction, milestone by milestone; every milestone was reviewed and approved before the next one started.
The random-walk graph-inference objective follows his own published work (Al-Ali et al. 2019;
Karfoul & Almuhammad Alali 2026).
