"""M3 - metrics, hypothesis tests (paired Wilcoxon + Holm) and figures from the OOF predictions."""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, f1_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wesad.evaluation import per_subject_metrics  # noqa: E402

R = ROOT / "results" / "m3"
FIG = ROOT / "figures" / "m3"
FIG.mkdir(parents=True, exist_ok=True)
PAPER = {("binary", "chest"): 93.12, ("binary", "wrist"): 88.33, ("binary", "both"): 92.28,
         ("three_class", "chest"): 80.34, ("three_class", "wrist"): 76.17, ("three_class", "both"): 79.57}


def load():
    rows, per_subj = [], []
    for f in sorted((R / "oof").glob("*.npz")):
        task, scheme, sensor, name = f.stem.split("__")
        d = np.load(f, allow_pickle=True)
        y, g = d["y"], d["subject"]
        models = {"LR": [d["lr_prob"]]}
        if "rf_prob" in d:
            models["RF"] = list(d["rf_prob"])
        for m, probs in models.items():
            ps = [per_subject_metrics(y, p.argmax(1), g) for p in probs]
            ps_mean = pd.concat(ps).groupby("subject", sort=False)[["macro_f1", "balanced_acc", "acc"]].mean().reset_index()
            ps_mean = ps_mean.assign(task=task, scheme=scheme, sensor=sensor, set=name, model=m)
            per_subj.append(ps_mean)
            rows.append(dict(task=task, scheme=scheme, sensor=sensor, set=name, model=m,
                             f1_mean=ps_mean.macro_f1.mean(), f1_sd=ps_mean.macro_f1.std(ddof=1),
                             balacc_mean=ps_mean.balanced_acc.mean(), worst_f1=ps_mean.macro_f1.min(),
                             pooled_acc=np.mean([accuracy_score(y, p.argmax(1)) for p in probs]),
                             pooled_f1=np.mean([f1_score(y, p.argmax(1), average="macro") for p in probs])))
    return pd.DataFrame(rows), pd.concat(per_subj, ignore_index=True)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def tests(ps):
    rows = []
    for scheme in ("loso", "cross_order"):
        fam = []
        for task in ("binary", "three_class"):
            fam.append(("H1", task, "both", "levels", "motion", "motion"))
            for sensor in ("chest", "wrist"):
                fam.append(("H2", task, sensor, "levels+rw", sensor, "levels"))
                fam.append(("H3", task, sensor, "levels+rw", sensor, "levels+corr"))
                fam.append(("H3", task, sensor, "levels+rw", sensor, "levels+pcorr"))
        for h, task, s1, a, s2, b in fam:
            q = ps[(ps.model == "LR") & (ps.task == task) & (ps.scheme == scheme)]
            xa = q[(q.sensor == s1) & (q.set == a)].set_index("subject").macro_f1
            xb = q[(q.sensor == s2) & (q.set == b)].set_index("subject").macro_f1
            d = (xa - xb.reindex(xa.index)).dropna()
            p = wilcoxon(d).pvalue if (d != 0).sum() > 5 else 1.0
            rows.append(dict(scheme=scheme, hyp=h, task=task, compare=f"{s1}:{a} vs {s2}:{b}",
                             mean_diff=d.mean(), median_diff=d.median(), n_better=int((d > 0).sum()),
                             n_worse=int((d < 0).sum()), p=p))
        idx = [i for i, r in enumerate(rows) if r["scheme"] == scheme]
        adj = holm([rows[i]["p"] for i in idx])
        for i, a_ in zip(idx, adj):
            rows[i]["p_holm"] = a_
    return pd.DataFrame(rows)


def main():
    summ, ps = load()
    summ.to_csv(R / "summary.csv", index=False)
    ps.to_csv(R / "per_subject.csv", index=False)
    t = tests(ps)
    t.to_csv(R / "hypothesis_tests.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 300)
    for task in ("binary", "three_class"):
        print(f"\n===== {task} =====")
        s = summ[summ.task == task].copy()
        s["cfg"] = s.sensor + ":" + s.set
        piv = s.pivot_table(index=["cfg"], columns=["scheme", "model"], values="f1_mean").round(3)
        print("per-subject macro-F1 (mean over 15 subjects)"); print(piv.to_string())
        lo = s[(s.scheme == "loso") & (s.model == "LR")].set_index("cfg")[["pooled_acc", "pooled_f1", "worst_f1"]].round(3)
        print("\nLOSO pooled (paper comparison):"); print(lo.to_string())
    print("\n===== hypothesis tests (LR, per-subject macro-F1, Holm within scheme) =====")
    print(t.round(4).to_string(index=False))

    # chosen lambda / C
    for f in sorted(R.glob("meta_*.json")):
        meta = json.load(open(f))
        for m in meta:
            if m["set"] in ("levels+rw", "rw_only"):
                ks = pd.Series([c["k"] for c in m["chosen"]]).value_counts().to_dict()
                print(f"{f.stem} {m['sensor']}:{m['set']} lambda index counts {ks}")

    # figure: per-subject macro-F1 under both schemes, LR, binary and 3-class
    cfgs = [("motion", "motion"), ("time", "time"), ("wrist", "levels"), ("wrist", "levels+rw"),
            ("chest", "levels"), ("chest", "levels+rw"), ("both", "levels"), ("both", "levels+rw")]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, task in zip(axs, ("binary", "three_class")):
        for j, (sensor, name) in enumerate(cfgs):
            for k, (scheme, col) in enumerate((("loso", "#2a78d6"), ("cross_order", "#eb6834"))):
                q = ps[(ps.model == "LR") & (ps.task == task) & (ps.scheme == scheme) & (ps.sensor == sensor) & (ps.set == name)]
                x = j + (k - 0.5) * 0.3
                ax.scatter(np.full(len(q), x) + np.random.default_rng(j).uniform(-0.05, 0.05, len(q)), q.macro_f1,
                           s=12, color=col, alpha=0.6, edgecolor="white", linewidth=0.4)
                ax.plot([x - 0.12, x + 0.12], [q.macro_f1.mean()] * 2, color="#0b0b0b", lw=2)
        ax.set_xticks(range(len(cfgs)))
        ax.set_xticklabels([f"{s}\n{n}" for s, n in cfgs], fontsize=8)
        ax.set_title(f"{task}: per-subject macro-F1 (LR); blue = LOSO, orange = cross-order", loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("macro-F1 per subject")
    fig.tight_layout(); fig.savefig(FIG / "per_subject_f1.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
