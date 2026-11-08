"""M6-B stage B3 - descriptive checks of the per-window TV graph on WESAD (no model is trained).

1. Real-data fusion: share of windows with at least one fused edge, per condition.
2. Heart-rate~movement edge: per-subject stress - baseline difference, d_z, sign count, Wilcoxon;
   same for the random-walk edge of M2 on the same windows.
3. What the TV graph adds: Spearman between TV edge weights and (a) lag-0 Pearson correlation of the
   prepared node signals, (b) the M2 correlation graph, (c) the M2 random-walk graph.
4. Stress effect on fusion degree and algebraic connectivity.
Outputs: results/m6/descriptive_*.csv
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wesad.windows import SUBJECTS  # noqa: E402
from wesad.physio import CHEST_NODES, WRIST_NODES, node_series, process_subject  # noqa: E402
from wesad.graphs import edge_names  # noqa: E402
from wesad.tvgraph import prepare_node_signals  # noqa: E402

R = ROOT / "results" / "m6"
NODES = {"chest": CHEST_NODES, "wrist": WRIST_NODES}
HR_MOVE = {"chest": "hr_ecg~act_chest", "wrist": "hr_bvp~act_wrist"}
LABELS = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}


def load():
    tv = pd.read_csv(R / "tv_features_60s.csv")
    m2 = pd.read_csv(ROOT / "results" / "m2" / "features_60s.csv")
    key = ["subject", "start_s", "end_s", "label"]
    assert (tv[key].values == m2[key].values).all(), "window order differs"
    keep = [c for c in m2.columns if c.startswith("g_corr_") or c.startswith("g_rw0.02_")]
    return pd.concat([tv, m2[keep]], axis=1)


def lag0_corr(df):
    """Lag-0 Pearson correlation of the prepared node signals, per window and edge."""
    out = {}
    for dev, nodes in NODES.items():
        iu = np.triu_indices(len(nodes), 1)
        vals = np.empty((len(df), len(iu[0])))
        for s in SUBJECTS:
            idx = np.flatnonzero(df.subject.values == s)
            P = process_subject(s)
            X = np.stack([node_series(P, a, b, nodes) for a, b in zip(df.start_s.values[idx], df.end_s.values[idx])])
            Y = prepare_node_signals(X)
            C = np.einsum("bim,bjm->bij", Y, Y)
            vals[idx] = C[:, iu[0], iu[1]]
        for k, e in enumerate(edge_names(nodes)):
            out[f"rho0_{dev}_{e}"] = vals[:, k]
    return pd.DataFrame(out, index=df.index)


def subject_effect(df, col, a=2, b=1):
    d = df.groupby(["subject", "label"])[col].mean().unstack()
    diff = (d[a] - d[b]).dropna()
    return dict(measure=col, n=len(diff), mean_diff=diff.mean(), d_z=diff.mean() / diff.std(ddof=1),
                n_negative=int((diff < 0).sum()), p_wilcoxon=wilcoxon(diff).pvalue)


def main():
    df = load()
    df = pd.concat([df, lag0_corr(df)], axis=1)

    rows = []
    for dev, nodes in NODES.items():
        n = len(nodes)
        fused = df[f"diag_tv_{dev}_nclusters"] < n
        for lab, name in LABELS.items():
            m = df.label == lab
            rows.append(dict(device=dev, condition=name, windows=int(m.sum()), any_fused=fused[m].mean(),
                             fusion_F_mean=df.loc[m, f"g_tv_{dev}_fusion"].mean(),
                             flat_node_windows=(df.loc[m, f"diag_tv_{dev}_flat_nodes"] > 0).mean()))
    fus = pd.DataFrame(rows)
    fus.to_csv(R / "descriptive_fusion.csv", index=False)
    print(fus.round(3).to_string(index=False))

    eff = []
    for dev in NODES:
        e = HR_MOVE[dev]
        for col in (f"g_tv_{dev}_{e}", f"g_rw0.02_{dev}_{e}", f"rho0_{dev}_{e}",
                    f"g_tv_{dev}_fusion", f"g_tv_{dev}_algconn"):
            eff.append(dict(device=dev, contrast="stress-baseline", **subject_effect(df, col)))
            eff.append(dict(device=dev, contrast="amusement-baseline", **subject_effect(df, col, a=3)))
    eff = pd.DataFrame(eff)
    eff.to_csv(R / "descriptive_effects.csv", index=False)
    print("\n", eff.round(3).to_string(index=False))

    sp = []
    for dev, nodes in NODES.items():
        for e in edge_names(nodes):
            tvw = df[f"g_tv_{dev}_{e}"]
            sp.append(dict(device=dev, edge=e,
                           rho_vs_lag0corr=spearmanr(tvw, df[f"rho0_{dev}_{e}"]).statistic,
                           rho_vs_m2corr=spearmanr(tvw, df[f"g_corr_{dev}_{e}"]).statistic,
                           rho_vs_rw=spearmanr(tvw, df[f"g_rw0.02_{dev}_{e}"]).statistic,
                           share_zero=(tvw < 1e-8).mean()))
    sp = pd.DataFrame(sp)
    sp.to_csv(R / "descriptive_spearman.csv", index=False)
    print("\n", sp.round(3).to_string(index=False))
    print("\nmedian over edges:\n", sp.groupby("device")[["rho_vs_lag0corr", "rho_vs_m2corr", "rho_vs_rw", "share_zero"]].median().round(3))


if __name__ == "__main__":
    main()
