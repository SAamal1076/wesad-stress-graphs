"""M6-C stage C3a - causal online TV graph over each subject's whole recording.

Grid: 60 s windows ending every 5 s from 60 s to the end of the recording (labelled or not). Node signals are
prepared as in M6-B; the online graph uses theta = 4, eta = 0, beta = 1, gamma = 0.5 and the mu selected by
the synthetic gate (results/m6/gate_mu_decision.csv). Subjects are batched; shorter recordings are padded by
repeating their last window, which cannot affect earlier (causal) estimates, and the padding is dropped.
Unsupervised: labels are only attached for later evaluation.
Output: results/m6/online_grid.csv (gitignored).
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wesad.windows import LABEL_KEY, SUBJECTS, condition_runs, load_subject  # noqa: E402
from wesad.physio import CHEST_NODES, GRID_FS, WRIST_NODES, node_series, process_subject  # noqa: E402
from wesad.graphs import edge_names  # noqa: E402
from wesad.tvgraph import prepare_node_signals  # noqa: E402
from wesad.tvgraph_online import online_fit  # noqa: E402

WIN, STEP, THETA, CALIB_S = 60.0, 5.0, 4.0, 120.0
DEV = {"chest": (CHEST_NODES, "hr_ecg", "act_chest"), "wrist": (WRIST_NODES, "hr_bvp", "act_wrist")}
OUT = ROOT / "results" / "m6"


def grid_table(s, P):
    T = len(P["t_grid"]) / GRID_FS
    ends = np.arange(WIN, np.floor(T / STEP) * STEP + 1e-9, STEP)
    d = load_subject(s)
    runs = condition_runs(d[LABEL_KEY], float(d["chest_ECG_fs"]), s)
    g = pd.DataFrame(dict(subject=s, start_s=ends - WIN, end_s=ends, label=0, run_id="", run_pos=0))
    for r in runs.itertuples(index=False):
        inside = (g.start_s >= r.start_s - 1e-9) & (g.end_s <= r.end_s + 1e-9)
        g.loc[inside, ["label", "run_id", "run_pos"]] = [r.label, r.run_id, r.run_pos]
    b0 = runs[runs.label == 1].start_s.iloc[0]
    g["calib"] = (g.start_s >= b0 - 1e-9) & (g.end_s <= b0 + CALIB_S + 1e-9)
    g["stress_start_s"] = runs[runs.label == 2].start_s.iloc[0]
    g["stress_end_s"] = runs[runs.label == 2].end_s.iloc[0]
    return g


def main(mu):
    t0 = time.time()
    tables, Ys = [], {dev: [] for dev in DEV}
    for s in SUBJECTS:
        P = process_subject(s)
        g = grid_table(s, P)
        for dev, (nodes, hr, act) in DEV.items():
            X = np.stack([node_series(P, a, b, nodes) for a, b in zip(g.start_s, g.end_s)])
            Ys[dev].append(prepare_node_signals(X))
            g[f"hr_{dev}"] = np.nanmean(X[:, nodes.index(hr)], axis=1)
            g[f"act_{dev}"] = np.nanmean(X[:, nodes.index(act)], axis=1)
        tables.append(g)
    print(f"grids built ({time.time() - t0:.0f} s): {[len(t) for t in tables]}", flush=True)

    for dev, (nodes, _, _) in DEV.items():
        Tmax = max(len(y) for y in Ys[dev])
        batch = np.stack([np.concatenate([y, np.repeat(y[-1:], Tmax - len(y), 0)]) for y in Ys[dev]])
        W, _ = online_fit(batch, theta=THETA, mu=mu)
        n = len(nodes)
        iu = np.triu_indices(n, 1)
        for i, g in enumerate(tables):
            w = W[i, :len(g)]
            A = np.zeros((len(g), n, n))
            A[:, iu[0], iu[1]] = w
            A = A + np.swapaxes(A, 1, 2)
            L = A.sum(2)[:, :, None] * np.eye(n)[None] - A
            for k, e in enumerate(edge_names(nodes)):
                g[f"g_on_{dev}_{e}"] = w[:, k]
            for k, nd in enumerate(nodes):
                g[f"g_on_{dev}_deg_{nd}"] = A[:, k].sum(1)
            g[f"g_on_{dev}_algconn"] = np.linalg.eigvalsh(L)[:, 1]
        print(f"{dev}: online graph done ({time.time() - t0:.0f} s)", flush=True)

    df = pd.concat(tables, ignore_index=True)
    df["mu"] = mu
    df.to_csv(OUT / "online_grid.csv", index=False)
    print("saved", df.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, required=True)
    main(ap.parse_args().mu)
