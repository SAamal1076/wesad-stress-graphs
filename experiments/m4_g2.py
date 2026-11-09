"""M4b - G2: causal, label-free test-time graph smoothing of the test subject's probabilities.

For each window i of the test subject, the nodes are its windows that ended within the last H
seconds (including i). Edge weights combine temporal proximity and physiological similarity:
    W_jk = exp(-(dt_jk / 60)^2) * exp(-d_jk^2 / (2 s^2)),  s = median d inside the horizon,
d = distance between level-feature vectors standardised with TRAINING-subject statistics.
Graph Tikhonov: x = (I + L)^-1 p ; the output is x at window i.
Controls: causal moving average over the same horizon; Tikhonov with the temporal kernel only.
All hyperparameters are fixed (pre-specified); no tuning on any subject.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from wesad.windows import task_view  # noqa: E402
from wesad.physio import group_columns  # noqa: E402
from wesad.evaluation import per_subject_metrics  # noqa: E402
from m3_models import ORDER_A, ORDER_B  # noqa: E402

R4 = ROOT / "results" / "m4"
HORIZONS = (30, 60, 120, 300)
TAU = 60.0
BASES = [("both", "levels+rw", "calib2"), ("wrist", "levels+rw", "calib2"), ("both", "levels+rw", "none")]


def smooth_subject(end_s, P, F, H, kind):
    order = np.argsort(end_s)
    e, Pp, Fp = end_s[order], P[order], F[order]
    out = np.empty_like(Pp)
    for i in range(len(e)):
        j0 = np.searchsorted(e, e[i] - H, side="right")
        idx = np.arange(j0, i + 1)
        p = Pp[idx]
        if kind == "ma" or len(idx) == 1:
            out[i] = p.mean(axis=0)
            continue
        dt = e[idx][:, None] - e[idx][None, :]
        W = np.exp(-(dt / TAU) ** 2)
        if kind == "graph":
            f = Fp[idx]
            d = np.sqrt(((f[:, None, :] - f[None, :, :]) ** 2).sum(-1))
            s = np.median(d[np.triu_indices(len(idx), 1)])
            W = W * np.exp(-d ** 2 / (2 * max(s, 1e-6) ** 2))
        np.fill_diagonal(W, 0.0)
        L = np.diag(W.sum(1)) - W
        x = np.linalg.solve(np.eye(len(idx)) + L, p)
        out[i] = x[-1]
    out = np.clip(out, 1e-9, None)
    out /= out.sum(axis=1, keepdims=True)
    res = np.empty_like(out)
    res[order] = out
    return res


def standardised_features(t, cols, scheme):
    """Per test subject (or test group), z-score with statistics of the training subjects only."""
    X = t[cols].values.astype(float)
    Z = np.zeros_like(X)
    g = t.subject.values
    groups = [[s] for s in pd.unique(g)] if scheme == "loso" else [ORDER_B, ORDER_A]
    for test in groups:
        te = np.isin(g, test)
        tr = ~te
        med = np.nanmedian(X[tr], axis=0)
        Xi = np.where(np.isnan(X), med, X)
        mu, sd = Xi[tr].mean(0), Xi[tr].std(0) + 1e-9
        Z[te] = (Xi[te] - mu) / sd
    return Z


def latency(df):
    """Seconds from the start of the stress run to the end of the first window predicted stress."""
    out = {}
    for s, d in df.groupby("subject"):
        st = d[d.label == 2].sort_values("end_s")
        hit = st[st.pred == 1]
        out[s] = (hit.end_s.iloc[0] - st.start_s.min()) if len(hit) else np.nan
    return pd.Series(out)


def main():
    feats = pd.read_csv(ROOT / "results" / "m2" / "features_60s.csv")
    rows, lat_rows = [], []
    for task in ("binary", "three_class"):
        t = task_view(feats, task)
        for scheme in ("loso", "cross_order"):
            for sensor, name, mode in BASES:
                d = np.load(R4 / "oof" / f"{task}__{scheme}__{sensor}__{name}__{mode}.npz", allow_pickle=True)
                assert np.allclose(d["start_s"], t.start_s.values) and (d["subject"] == t.subject.values).all()
                cols = group_columns(t.columns, "chest_levels") * (sensor != "wrist") + group_columns(t.columns, "wrist_levels")
                Z = standardised_features(t, cols, scheme)
                ev = ~d["excl2"]
                y, g, P = d["y"], d["subject"], d["prob"]
                variants = {("raw", 0): P}
                for H in HORIZONS:
                    for kind in ("ma", "time", "graph"):
                        S = np.zeros_like(P)
                        for s in pd.unique(g):
                            m = g == s
                            S[m] = smooth_subject(d["end_s"][m], P[m], Z[m], H, kind)
                        variants[(kind, H)] = S
                for (kind, H), S in variants.items():
                    pred = S.argmax(1)
                    ps = per_subject_metrics(y[ev], pred[ev], g[ev])
                    base = dict(task=task, scheme=scheme, base=f"{sensor}:{name}:{mode}", smoother=kind, H=H)
                    for r in ps.itertuples():
                        rows.append(dict(**base, subject=r.subject, macro_f1=r.macro_f1, balanced_acc=r.balanced_acc))
                    lat = latency(pd.DataFrame(dict(subject=g, label=d["label"], start_s=d["start_s"],
                                                    end_s=d["end_s"], pred=pred))[ev])
                    lat_rows.append(dict(**base, latency_median_s=lat.median(), n_detected=int(lat.notna().sum())))
                print(task, scheme, sensor, name, mode, "done", flush=True)
    ps = pd.DataFrame(rows)
    ps.to_csv(R4 / "g2_per_subject.csv", index=False)
    lat = pd.DataFrame(lat_rows)
    lat.to_csv(R4 / "g2_latency.csv", index=False)
    summ = ps.groupby(["task", "scheme", "base", "smoother", "H"]).macro_f1.mean().unstack(["smoother", "H"]).round(3)
    pd.set_option("display.width", 250)
    print(summ.to_string())


if __name__ == "__main__":
    main()
