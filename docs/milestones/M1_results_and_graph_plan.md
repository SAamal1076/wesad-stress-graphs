# WESAD — M1 results and proposed graph method

*8 Nov 2026. Code in the session repository `wesad-stress/` (git commits aab84a5, 33ccea4). Proposal awaiting Salman's approval before M2.*

## M1 — what was built
- `src/wesad/windows.py`: windows that lie entirely inside one condition run (labels from the 350 Hz label stream). 60 s / step 5 s gives 8143 windows (baseline 3354, stress 1826, amusement 944, meditation 2019); 10/20/30 s indexes also saved.
- `src/wesad/evaluation.py`: leave-one-subject-out with a leakage assertion in every fold; per-subject macro-F1 and balanced accuracy (mean ± SD) plus pooled accuracy and macro-F1 (comparable with Schmidt et al.).
- `tests/test_m1.py`: 8 tests pass — every window checked sample by sample against the label stream; folds disjoint and complete; a spy model proves no test row is ever used for fitting.

## M1 — shortcut baselines (no physiology), 60 s windows, LOSO
| Task | Inputs | Model | Per-subject macro-F1 | Pooled accuracy |
|---|---|---|---|---|
| Binary | majority class | — | 0.41 | 70.2% |
| Binary | **minutes since start only** | RF | **0.95 ± 0.06** | **95.7%** |
| Binary | motion, chest + wrist | LR | 0.82 ± 0.08 | 85.1% |
| Binary | motion, wrist only | LR | 0.78 ± 0.12 | 80.7% |
| 3-class | majority class | — | 0.24 | 54.8% |
| 3-class | **minutes since start only** | RF | **0.90 ± 0.10** | **93.6%** |
| 3-class | motion, chest + wrist | LR | 0.64 ± 0.15 | 70.7% |

Paper reference (Schmidt et al. 2018, pooled accuracy): binary 93.1%, 3-class 80.3%.

**Key finding:** a model that only sees the clock beats the original paper. Within each protocol order the schedule is almost identical across subjects, so time identifies the condition. Any feature that drifts with time (temperature, tonic EDA, fatigue) or any model that sees the position of a window in the session can exploit this.

## Cross-protocol-order check (train on one order, test on the other)
- Order A (amusement before stress): S4, S5, S7, S8, S10, S13, S15, S17. Order B (stress first): S2, S3, S6, S9, S11, S14, S16.
- Time-only RF collapses: binary 95.7% → 66.9%, 3-class 93.6% → 62.4%. Motion holds: binary RF 84.0%, 3-class RF 68.0%.
- This test is added to the evaluation protocol: every reported model is evaluated with LOSO **and** cross-order.

## Prior graph work on WESAD (checked 8 Nov 2026)
- PAMG-AT (Yildiz & Subasi, bioRxiv, March 2026): physiological features as nodes, fixed knowledge-driven edges (cardiac–electrodermal, cardio-respiratory), graph attention plus a transformer over time windows; LOSO accuracy 94.6% chest, 91.8% wrist. Its low-performance subjects (S2, S3, S9) all belong to protocol order B — worth testing, not a claim.
- So "physiological graph + GNN on WESAD" is not new. Novelty has to come from elsewhere.

## Proposed graph method (two graph components, both from Salman's own research line)
1. **G1 — Learned physiological coupling graph per window (network physiology).**
   - Nodes: instantaneous heart rate (ECG or BVP), respiration, phasic and tonic EDA, temperature, and motion as an explicit nuisance node.
   - Edges inferred from the data of each window with graph learning (smoothness / constrained random-walk inference, as in Salman's J4/C7), with partial correlation as the simple baseline.
   - Features: edge weights and spectral descriptors; the hypothesis is that stress changes the coupling (for example cardio-respiratory decoupling), not only the levels.
   - Difference from PAMG-AT: edges are estimated per window and per person, so the coupling itself is the interpretable biomarker.
2. **G2 — Label-free test-time adaptation on the new person's window graph.**
   - At test time, the new person's past windows (causal, real-time) form a graph by temporal adjacency and feature similarity.
   - The classifier's stress probabilities are treated as a graph signal and denoised by graph Tikhonov or graph total-variation regularisation (Salman's graph-denoising work, and his test-time adaptation paper C9).
   - Mandatory control: a plain moving average of the same past probabilities, so any gain is attributable to the graph, not to smoothing.
   - Latency reported, since smoothing trades reaction time for accuracy.

## Rigour rules for M2–M5
- LOSO and cross-order for every model; shortcut baselines (time, motion) reported next to physiology.
- Ablations: features alone; + G1; + G2; + G1 + G2; moving average vs graph smoothing; chest vs wrist (wrist is the VR-relevant case).
- Hyperparameters tuned only with inner leave-one-subject-out on the training subjects; 5 seeds, mean reported.
- Paired per-subject comparison between methods (Wilcoxon, n = 15) with effect sizes; no claim of superiority from pooled accuracy alone.
- The analysis plan is committed to the repository before running the experiments.
- "Higher than the paper" is a hypothesis, not a promise; the stronger claim to aim for is good performance under cross-order evaluation with wrist signals only.
