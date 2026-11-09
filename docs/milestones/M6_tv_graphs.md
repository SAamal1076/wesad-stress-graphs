# WESAD — M6 report: learned total-variation graphs

*8 Nov 2026. Settings fixed in `docs/analysis_plan.md` before each computation: commit 6e887d0 (M6-B plan, H8–H9), 9a17db7 (amendment 1 after the synthetic test), 4fdffba (M6-C plan, H10–H12, μ gate). Salman approved the direction and delegated the decisions after each test.*

## 1. M6-B: per-window joint graph-TV denoising and graph learning

**Model.** min over X, W of ½‖X−Y‖² + λΣ W_st‖x_s−x_t‖ − βΣ log d_s + γ‖W‖². The nodes are the signals of one 60 s window. The method alternates ADMM (network lasso) for X with Kalofolias' primal-dual algorithm for W.

**Stage B1, solver and synthetic data.**
- Unit tests passed: two-node closed form, KKT conditions, monotone objective.
- With β = 1 the model depends on one number only, θ = λ/√(2γ) (checked numerically to 1e-9).
- The pre-specified θ of 0.14–0.85 learned near-uniform weights.
- θ of 3–10 recovers planted clusters, but at those values independent signals collapse into one cluster in 10–93% of trials.
- The cause is a feedback loop: fusion gives zero distance, a larger weight, then more fusion.

**Amendment 1** (before any WESAD window was processed):
- The distance becomes (1−η)‖y_s−y_t‖ + η‖x_s−x_t‖.
- The fusion degree F = 1 − TV_W(X)/TV_W(Y) replaces the number of clusters.
- (η, θ) is chosen once by a synthetic gate.

**Gate result:**

| Setting | Independent signals fused | 2 groups, r = 0.6 | 2 groups, r = 0.3 | 2+2+1 | Outcome |
|---|---|---|---|---|---|
| η = 0, θ = 4 | 0% | 100% | 93% | 100% | pass |
| **η = 0.25, θ = 4** | **1%** | **100%** | **92%** | **100%** | **selected** |
| η = 0.5, θ = 4 | 100% | 100% | 88% | 100% | fail |
| η = 1 (original model) | 100% | 100% | 79% | 96% | reference |

With slow independent signals (descriptive), some edge fused in 15% of trials.

**Stages B2/B3 on WESAD (descriptive):**
- **Fusion.** At least one fused edge in 48–63% of chest windows and 20–37% of wrist windows, in every condition. Real signals are slow, so the fusion degree is uninformative (stress effect d_z −0.37 chest).
- **Heart-rate↔movement edge under stress:**
  - TV graph: chest d_z −1.22 (15/15), wrist −0.49 (10/15);
  - random-walk graph: −2.01 / −1.78 (15/15);
  - lag-0 correlation: −1.27 / −1.08 (13/15).
- **Relation to correlation.** Median Spearman with the lag-0 correlation was 0.45 (chest) and 0.57 (wrist), and about half the edges were exactly zero. The TV graph is a sparse transform, not a copy of the correlation graph.

## 2. M6-C: causal time-varying TV graph and a decoupling index

**Model.** At each 60 s window ending every 5 s over the whole recording:

w_t = argmin over w ≥ 0 of θ z_tᵀw − βΣ log d + 2γ‖w‖² + μ‖w − w_{t−1}‖₁

- Settings: η = 0 (convex), θ = 4.
- Solver: primal-dual FBF with a second dual for the ℓ1 term. Tests check KKT conditions, optimality against perturbations, μ = 0 being equivalent to independent windows, and a large μ freezing the graph.

**μ gate (planted coupling change at 600 s).**
- Selected μ = 0.25: mean AUC 0.998 vs 0.992 at μ = 0; median delay 47.5 / 42.5 s vs 30 / 32.5 s.
- μ ≥ 0.5 exceeded the +30 s delay limit, and μ = 4 froze the graph.

**Indices** (calibration = the person's first 2 minutes of baseline, excluded from evaluation):
- **D:** mean decoupling of the physiology–movement edges from calibration.
- **aHR:** heart rate − a_i − b·movement, with b fitted on the training subjects (LOSO).
- **Combined:** a LOSO logistic regression on [aHR, D].

| Per-subject AUC (mean of 15) | aHR | D | aHR + D | HR − calibration mean |
|---|---|---|---|---|
| Chest, stress vs baseline + amusement | 0.847 | 0.590 | 0.840 | 0.937 |
| Wrist, stress vs baseline + amusement | 0.662 | 0.556 | 0.681 | 0.774 |
| Chest, stress vs meditation | 0.973 | 0.646 | 0.972 | 0.999 |
| Wrist, stress vs meditation | 0.741 | 0.434 | 0.702 | 0.880 |

**Hypothesis tests:**
- **H10** (combined > aHR, stress vs non-stress): chest Holm p = 0.069 (combined not better: 1/15 better, the rest tied or worse); wrist 10/15 better, Holm p = 0.42. **Not supported.**
- **H11** (combined > aHR, stress vs meditation): chest Holm p = 0.42; wrist combined *worse* (4/15 better, Holm p = 0.069). **Not supported.**

**Descriptive:**
- **D moves in the expected direction, but weakly.** Under stress D rises in 10/15 (chest, d_z 0.64) and 13/15 (wrist, 0.79) subjects. The online chest heart-rate↔movement edge drops in 14/15 (d_z −1.22).
- **Movement is confounded with stress.** Movement rises strongly during the stress test (chest d_z 2.26, 15/15). Correcting heart rate for movement therefore removes stress signal, and the raw change in heart rate beats aHR.
- **Latency** (threshold = 95th percentile on the training subjects' non-stress windows, 3 consecutive windows):

  | Score | Detected | Median latency | False alarms (baseline / amusement / meditation) |
  |---|---|---|---|
  | Chest aHR | 13/15 | 13 s | 1% / 19% / 14% |
  | Wrist aHR | 9/15 | 7.3 min | — |
  | D alone | 6/15 (chest), 2/15 (wrist) | — | — |

  Some chest latencies are about 1 s: heart rate rises before the labelled stress period (anticipation).

## 3. Classification (M3 pipeline; base = device levels without the four drifting features)

| Hypothesis | Result |
|---|---|
| H8: base + TV graph > base | +0.003 to +0.010, all Holm p = 1.00 |
| H9: TV graph > random-walk graph | −0.004 to +0.018, Holm p ≥ 0.67 |
| H12: base + online graph > base | −0.001 to +0.021, Holm p ≥ 0.92 |

Same pattern under LOSO and cross-order, for binary and 3-class, chest and wrist. **None supported.**

## 4. Conclusion

- The learned TV graph is a sound, tested method that is causal and fast enough for online use.
- The coupling it measures changes with stress consistently, but the change is small next to the change in heart-rate level. It adds nothing to heart-rate indices or classifiers with 15 subjects.
- **The synthetic gate was decisive:** it caught false fusion in the planned joint model before any WESAD window was processed.
- **For VR:** movement-aware stress indices can only be judged where movement varies independently of stress. WESAD's stress test, where people stand and speak, cannot provide that.

Files:
- `results/m6/`: gate.csv, gate_decision.csv, gate_mu*.csv, descriptive_*.csv, c3_*.csv.
- `figures/m6/`: index_aucs.png, timecourse_chest.png, timecourse_wrist.png.
