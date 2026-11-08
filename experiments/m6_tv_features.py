"""M6-B stage B2 - per-window learned TV coupling graph on every WESAD window (all conditions 1-4).

Setting fixed by the synthetic gate (amendment 1): eta = 0.25, theta = 4 (beta = 1, gamma = 0.5, lambda = theta).
Unsupervised and per window: no label and no other subject is used, so it is computed once before any
LOSO loop without leakage (same argument as M2).
Output: results/m6/tv_features_60s.csv (gitignored), one row per window of build_window_index(60, 5).
Columns g_tv_{device}_*: edge weights, node degrees, algebraic connectivity, TV of Y, fusion degree F.
Diagnostic columns diag_tv_{device}_*: number of fused clusters, number of flat (all-zero) nodes.
"""
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wesad.windows import SUBJECTS, build_window_index  # noqa: E402
from wesad.physio import CHEST_NODES, WRIST_NODES, node_series, process_subject  # noqa: E402
from wesad.graphs import edge_names  # noqa: E402
from wesad.tvgraph import descriptors, fit, n_clusters, prepare_node_signals  # noqa: E402

ETA, THETA, BETA, GAMMA = 0.25, 4.0, 1.0, 0.5
OUT = ROOT / "results" / "m6"
OUT.mkdir(parents=True, exist_ok=True)


def tv_graph_features(Xraw, nodes, dev):
    n = len(nodes)
    Y = prepare_node_signals(Xraw)
    res = fit(Y, lam=THETA, beta=BETA, gamma=GAMMA, eta=ETA)
    d = descriptors(Y, res, n)
    cols = {f"g_tv_{dev}_{e}": d["w"][:, k] for k, e in enumerate(edge_names(nodes))}
    cols.update({f"g_tv_{dev}_deg_{nd}": d["degree"][:, k] for k, nd in enumerate(nodes)})
    cols[f"g_tv_{dev}_algconn"] = d["algconn"]
    cols[f"g_tv_{dev}_tvy"] = d["tv_y"]
    cols[f"g_tv_{dev}_fusion"] = d["fusion"]
    cols[f"diag_tv_{dev}_nclusters"] = n_clusters(res["V"], n)
    cols[f"diag_tv_{dev}_flat_nodes"] = (np.abs(Y).sum(2) == 0).sum(1)
    return pd.DataFrame(cols)


def main():
    t0 = time.time()
    win = build_window_index(60.0, 5.0)
    parts = []
    for s in SUBJECTS:
        P = process_subject(s)
        w = win[win.subject == s].reset_index(drop=True)
        Xc = np.stack([node_series(P, a, b, CHEST_NODES) for a, b in zip(w.start_s, w.end_s)])
        Xw = np.stack([node_series(P, a, b, WRIST_NODES) for a, b in zip(w.start_s, w.end_s)])
        parts.append(pd.concat([w, tv_graph_features(Xc, CHEST_NODES, "chest"),
                                tv_graph_features(Xw, WRIST_NODES, "wrist")], axis=1))
        print(f"{s}: {len(w)} windows ({time.time() - t0:.0f} s)", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(OUT / "tv_features_60s.csv", index=False)
    print("saved", df.shape)


if __name__ == "__main__":
    main()
