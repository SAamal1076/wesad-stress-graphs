"""M2 - per-window physiological level features and G1 coupling-graph features.

Output: results/m2/features_{L}s.csv  (one row per window, all conditions 1-4).
Feature extraction is unsupervised and per window/per subject: no label and no other
subject is used, so it can be done once before the LOSO loop without leakage.
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wesad.windows import SUBJECTS, build_window_index  # noqa: E402
from wesad.physio import (CHEST_NODES, WRIST_NODES, node_series, process_subject,  # noqa: E402
                          window_level_features)
from wesad.graphs import (dependency_matrix, edge_names, graph_descriptors,  # noqa: E402
                          normalize_dependency, partial_correlation, rw_graph_inference,
                          zscore_rows)

OUT = ROOT / "results" / "m2"
OUT.mkdir(parents=True, exist_ok=True)
LAMBDAS = (0.005, 0.02, 0.05)
MAX_LAG = 20  # samples at 4 Hz = 5 s


def graph_features(Xw, nodes, cfg):
    """Xw: (W, n, m) raw node series -> DataFrame of corr / pcorr / rw features."""
    n = len(nodes)
    iu = np.triu_indices(n, 1)
    names = edge_names(nodes)
    Z = zscore_rows(np.nan_to_num(Xw))
    T = dependency_matrix(Z, MAX_LAG)
    cols = {}
    w_corr = T[:, iu[0], iu[1]]
    s, a = graph_descriptors(w_corr, n)
    cols.update({f"g_corr_{cfg}_{e}": w_corr[:, k] for k, e in enumerate(names)})
    cols[f"g_corr_{cfg}_strength"], cols[f"g_corr_{cfg}_algconn"] = s, a
    P = partial_correlation(Z)
    w_pc = P[:, iu[0], iu[1]]
    s, a = graph_descriptors(w_pc, n)
    cols.update({f"g_pcorr_{cfg}_{e}": w_pc[:, k] for k, e in enumerate(names)})
    cols[f"g_pcorr_{cfg}_strength"], cols[f"g_pcorr_{cfg}_algconn"] = s, a
    Tn = normalize_dependency(T)
    for lam in LAMBDAS:
        w, alpha, resid = rw_graph_inference(Tn, lam=lam, n_iter=600, step=0.2)
        s, a = graph_descriptors(w, n)
        tag = f"g_rw{lam:g}_{cfg}"
        cols.update({f"{tag}_{e}": w[:, k] for k, e in enumerate(names)})
        cols[f"{tag}_strength"], cols[f"{tag}_algconn"] = s, a
        cols[f"{tag}_alpha"], cols[f"{tag}_resid"] = alpha, resid
    return pd.DataFrame(cols)


def main(win_s):
    t0 = time.time()
    win = build_window_index(win_s, 5.0)
    parts = []
    for s in SUBJECTS:  # noqa: B007
        P = process_subject(s)
        w = win[win.subject == s].reset_index(drop=True)
        lev = pd.DataFrame([window_level_features(P, a, b) for a, b in zip(w.start_s, w.end_s)])
        Xc = np.stack([node_series(P, a, b, CHEST_NODES) for a, b in zip(w.start_s, w.end_s)])
        Xw = np.stack([node_series(P, a, b, WRIST_NODES) for a, b in zip(w.start_s, w.end_s)])
        g = pd.concat([graph_features(Xc, CHEST_NODES, "chest"),
                       graph_features(Xw, WRIST_NODES, "wrist")], axis=1)
        parts.append(pd.concat([w, lev, g], axis=1))
        print(f"{s}: {len(w)} windows ({time.time() - t0:.0f} s)", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(OUT / f"features_{int(win_s)}s.csv", index=False)
    print("saved", df.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=float, default=60.0)
    main(ap.parse_args().win)
