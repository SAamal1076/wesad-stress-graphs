"""M6-C synthetic gate for the temporal regularisation mu (fixed in docs/analysis_plan.md before running).

n = 5 nodes, 20 min at 4 Hz, 100 trials per scenario, seed 2027. Nodes 0 and 1 share a source (within
correlation 0.6) before 600 s and are independent after ("decouple"), or the reverse ("couple"); nodes 2-4
are independent. Windows: 60 s ending every 5 s, prepared exactly as on WESAD. Online graph with
theta = 4, eta = 0, beta = 1, gamma = 0.5 and mu in {0, 0.25, 0.5, 1, 2, 4}.
Measures on edge (0,1): AUC between windows ending <= 600 s and windows starting >= 600 s; delay until the
edge passes the midpoint of its pre/post medians and stays past it for 3 consecutive windows.
Rule: eligible if the median delay <= median delay at mu = 0 + 30 s in both scenarios; select the highest
mean AUC; ties within 0.005 go to the smaller mu.
"""
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wesad.tvgraph import prepare_node_signals  # noqa: E402
from wesad.tvgraph_online import online_fit  # noqa: E402

FS, DUR, CHANGE, WIN, STEP = 4, 1200, 600, 60, 5
N, TRIALS, SEED, CORR = 5, 100, 2027, 0.6
MUS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
THETA = 4.0
ENDS = np.arange(WIN, DUR + 1, STEP)


def simulate(rng, coupled_first):
    m = DUR * FS
    s = np.sqrt(1.0 / CORR - 1.0)
    X = rng.standard_normal((TRIALS, N, m)) * np.sqrt(1 + s ** 2)        # independent nodes, same variance
    shared = rng.standard_normal((TRIALS, m))
    k = CHANGE * FS
    sl = slice(0, k) if coupled_first else slice(k, m)
    for node in (0, 1):
        X[:, node, sl] = shared[:, sl] + s * rng.standard_normal((TRIALS, sl.stop - sl.start))
    W = np.stack([X[:, :, (e - WIN) * FS:e * FS] for e in ENDS], axis=1)   # (trials, T, n, m)
    nb, T, n, mm = W.shape
    return prepare_node_signals(W.reshape(nb * T, n, mm)).reshape(nb, T, n, mm)


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie), per trial."""
    gt = (pos[:, :, None] > neg[:, None, :]).mean((1, 2))
    eq = (pos[:, :, None] == neg[:, None, :]).mean((1, 2))
    return gt + 0.5 * eq


def delays(w, high_before):
    pre = ENDS <= CHANGE
    post = (ENDS - WIN) >= CHANGE
    mid = (np.median(w[:, pre], 1) + np.median(w[:, post], 1)) / 2
    side = (w < mid[:, None]) if high_before else (w > mid[:, None])
    out = np.full(len(w), np.inf)
    after = np.flatnonzero(ENDS >= CHANGE)
    for i in range(len(w)):
        for k in after[:-2]:
            if side[i, k] and side[i, k + 1] and side[i, k + 2]:
                out[i] = ENDS[k] - CHANGE
                break
    return out


def main():
    rng = np.random.default_rng(SEED)
    data = {"decouple": simulate(rng, True), "couple": simulate(rng, False)}
    pre, post = ENDS <= CHANGE, (ENDS - WIN) >= CHANGE
    rows = []
    for mu in MUS:
        for name, Y in data.items():
            t0 = time.time()
            W, _ = online_fit(Y, theta=THETA, mu=mu)
            w = W[:, :, 0]                                              # edge (0, 1) is the first edge
            hb = name == "decouple"
            a = auc(w[:, pre], w[:, post]) if hb else auc(w[:, post], w[:, pre])
            d = delays(w, hb)
            sd_pre = w[:, pre].std(1).mean()
            rows.append(dict(mu=mu, scenario=name, auc_mean=a.mean(), auc_min=a.min(),
                             delay_median=np.median(d), delay_p90=np.percentile(d, 90),
                             never_detected=float(np.isinf(d).mean()), sd_within_segment=sd_pre))
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()},
                  f"({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "m6"
    df.to_csv(out / "gate_mu.csv", index=False)

    piv = df.pivot(index="mu", columns="scenario", values=["auc_mean", "delay_median"])
    base = piv.loc[0.0, "delay_median"]
    elig = (piv["delay_median"] <= base + 30).all(axis=1)
    score = piv["auc_mean"].mean(axis=1)
    dec = pd.DataFrame({"mean_auc": score, "eligible": elig,
                        "delay_decouple": piv[("delay_median", "decouple")],
                        "delay_couple": piv[("delay_median", "couple")]})
    dec.to_csv(out / "gate_mu_decision.csv")
    print("\n", dec.round(3).to_string())
    cand = dec[dec.eligible]
    best = cand.mean_auc.max()
    chosen = min(m for m in cand.index if cand.loc[m, "mean_auc"] >= best - 0.005)
    print(f"\nSELECTED mu = {chosen}")


if __name__ == "__main__":
    main()
