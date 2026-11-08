"""M6-B synthetic gate (amendment 1 in docs/analysis_plan.md, fixed before this script was run).

Grid eta in {0, 0.25, 0.5}, theta in {1, 2, 4, 8} (beta = 1, gamma = 0.5, so lambda = theta); n = 5 nodes,
m = 240 samples; 100 trials per scenario; seed 2026. The same simulated data are used for every setting.
eta = 1 (the original model) is run for reference only and cannot be selected.
Output: results/m6/gate.csv (one row per setting x scenario) and the selected setting on stdout.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wesad.tvgraph import edge_index, fit, fusion_degree, n_clusters, prepare_node_signals  # noqa: E402

N, M, TRIALS, SEED = 5, 240, 100, 2026
ETAS, THETAS, ETA_REF = [0.0, 0.25, 0.5], [1.0, 2.0, 4.0, 8.0], 1.0
BETA, GAMMA = 1.0, 0.5
FUSED_TOL = 1e-6


def grouped(rng, groups, corr):
    """Each node = its group's source + noise of scale s, with 1 / (1 + s^2) = corr."""
    s = np.sqrt(1.0 / corr - 1.0)
    groups = np.asarray(groups)
    src = rng.standard_normal((TRIALS, groups.max() + 1, M))
    return src[:, groups, :] + s * rng.standard_normal((TRIALS, N, M))


def slow_null(rng, phi=0.95, burn=500):
    e = rng.standard_normal((TRIALS, N, M + burn))
    x = np.zeros_like(e)
    for k in range(1, M + burn):
        x[:, :, k] = phi * x[:, :, k - 1] + e[:, :, k]
    return x[:, :, burn:]


def simulate():
    rng = np.random.default_rng(SEED)
    raw = {
        "S1_null": rng.standard_normal((TRIALS, N, M)),
        "S2a_two_groups_r0.6": grouped(rng, [0, 0, 0, 1, 1], 0.6),
        "S2b_two_groups_r0.3": grouped(rng, [0, 0, 0, 1, 1], 0.3),
        "S3_2+2+1_r0.6": grouped(rng, [0, 0, 1, 1, 2], 0.6),
        "D_slow_null": slow_null(rng),
    }
    groups = {"S2a_two_groups_r0.6": [0, 0, 0, 1, 1], "S2b_two_groups_r0.3": [0, 0, 0, 1, 1],
              "S3_2+2+1_r0.6": [0, 0, 1, 1, 2]}
    return {k: prepare_node_signals(v) for k, v in raw.items()}, groups


def mean_abs_corr(Y):
    s, t = edge_index(N)
    C = np.einsum("bim,bjm->bij", Y, Y)                  # rows have unit norm, so this is the correlation
    return np.abs(C[:, s, t]).mean()


def evaluate(Y, res, groups):
    s, t = edge_index(N)
    w = res["w"]
    row = {
        "fused_any": float(np.mean(n_clusters(res["V"], N, tol=FUSED_TOL) < N)),
        "fusion_F_mean": float(fusion_degree(Y, res["X"], w, N).mean()),
        "w_mean": float(w.mean()),
    }
    if groups is not None:
        g = np.asarray(groups)
        same = g[s] == g[t]
        within, between = w[:, same].mean(1), w[:, ~same].mean(1)
        ok = (within > 0) & (within >= 3 * between)
        ratio = np.where(between > 0, within / np.maximum(between, 1e-300), np.inf)
        row.update({"ratio_pass_rate": float(ok.mean()), "ratio_median": float(np.median(ratio))})
    return row


def main():
    data, groups = simulate()
    rows = []
    for eta in ETAS + [ETA_REF]:
        for theta in THETAS:
            for name, Y in data.items():
                res = fit(Y, lam=theta, beta=BETA, gamma=GAMMA, eta=eta)
                rows.append({"eta": eta, "theta": theta, "scenario": name, "mean_abs_corr": mean_abs_corr(Y),
                             **evaluate(Y, res, groups.get(name))})
                print(f"eta={eta:<4} theta={theta:<3} {name:<22} " + "  ".join(
                    f"{k}={v:.3g}" for k, v in rows[-1].items() if k not in ("eta", "theta", "scenario")), flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "m6"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "gate.csv", index=False)

    # pre-registered decision
    piv = df.pivot_table(index=["eta", "theta"], columns="scenario",
                         values=["fused_any", "ratio_pass_rate"], aggfunc="first")
    dec = pd.DataFrame({
        "S1_fused": piv[("fused_any", "S1_null")],
        "S2a_pass": piv[("ratio_pass_rate", "S2a_two_groups_r0.6")],
        "S2b_pass": piv[("ratio_pass_rate", "S2b_two_groups_r0.3")],
        "S3_pass": piv[("ratio_pass_rate", "S3_2+2+1_r0.6")],
    })
    dec["passes"] = ((dec.S1_fused <= 0.05) & (dec.S2a_pass >= 0.90) & (dec.S2b_pass >= 0.90)
                     & (dec.S3_pass >= 0.80))
    dec["eligible"] = dec.index.get_level_values("eta") != ETA_REF
    dec["min_recovery"] = dec[["S2a_pass", "S2b_pass", "S3_pass"]].min(1)
    dec.to_csv(out / "gate_decision.csv")
    print("\n", dec.to_string())
    cand = dec[dec.passes & dec.eligible].reset_index()
    if cand.empty:
        print("\nNO SETTING PASSES THE GATE -> M6-B stops (negative result).")
        return
    cand = cand.sort_values(["eta", "S1_fused", "min_recovery", "theta"], ascending=[False, True, False, True])
    best = cand.iloc[0]
    print(f"\nSELECTED: eta = {best.eta}, theta = {best.theta}")


if __name__ == "__main__":
    main()
