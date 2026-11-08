"""M4 - per-subject metrics, H4-H6 tests (Holm within scheme), summary tables and figure."""
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
sys.path.insert(0, str(ROOT / "experiments"))
from wesad.evaluation import per_subject_metrics  # noqa: E402
from m3_stats import holm  # noqa: E402

R4 = ROOT / "results" / "m4"
FIG = ROOT / "figures" / "m4"
FIG.mkdir(parents=True, exist_ok=True)


def load_m4a():
    rows, summ = [], []
    for f in sorted((R4 / "oof").glob("*.npz")):
        task, scheme, sensor, name, mode = f.stem.split("__")
        d = np.load(f, allow_pickle=True)
        ev = ~d["excl5" if mode == "calib5" else "excl2"]
        y, p, g = d["y"][ev], d["prob"][ev].argmax(1), d["subject"][ev]
        ps = per_subject_metrics(y, p, g).assign(task=task, scheme=scheme, sensor=sensor, set=name, mode=mode)
        rows.append(ps)
        summ.append(dict(task=task, scheme=scheme, sensor=sensor, set=name, mode=mode,
                         f1_mean=ps.macro_f1.mean(), f1_sd=ps.macro_f1.std(ddof=1), worst=ps.macro_f1.min(),
                         pooled_acc=accuracy_score(y, p), pooled_f1=f1_score(y, p, average="macro")))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summ)


def paired(xa, xb):
    d = (xa - xb.reindex(xa.index)).dropna()
    p = wilcoxon(d).pvalue if (d != 0).sum() > 5 else 1.0
    return dict(mean_diff=d.mean(), n_better=int((d > 0).sum()), n_worse=int((d < 0).sum()), p=p)


def main():
    ps, summ = load_m4a()
    ps.to_csv(R4 / "personalisation_per_subject.csv", index=False)
    summ.to_csv(R4 / "personalisation_summary.csv", index=False)
    g2 = pd.read_csv(R4 / "g2_per_subject.csv")
    sel = lambda q, **kw: q.loc[np.logical_and.reduce([q[k] == v for k, v in kw.items()])].set_index("subject").macro_f1  # noqa: E731

    tests = []
    for scheme in ("loso", "cross_order"):
        fam = []
        for task in ("binary", "three_class"):
            for s in ("levels", "levels+rw"):
                fam.append(("H5", task, f"both:{s} calib2 vs none",
                            sel(ps, task=task, scheme=scheme, sensor="both", set=s, mode="calib2"),
                            sel(ps, task=task, scheme=scheme, sensor="both", set=s, mode="none")))
            for sensor in ("chest", "wrist", "both"):
                fam.append(("H6", task, f"{sensor}: levels+rw vs levels (calib2)",
                            sel(ps, task=task, scheme=scheme, sensor=sensor, set="levels+rw", mode="calib2"),
                            sel(ps, task=task, scheme=scheme, sensor=sensor, set="levels", mode="calib2")))
            for base in ("both:levels+rw:calib2", "wrist:levels+rw:calib2"):
                fam.append(("H4", task, f"{base} G2 vs moving average (H=120 s)",
                            sel(g2, task=task, scheme=scheme, base=base, smoother="graph", H=120),
                            sel(g2, task=task, scheme=scheme, base=base, smoother="ma", H=120)))
        res = [dict(scheme=scheme, hyp=h, task=t, compare=c, **paired(a, b)) for h, t, c, a, b in fam]
        for r, padj in zip(res, holm([r["p"] for r in res])):
            r["p_holm"] = padj
        tests += res
    tests = pd.DataFrame(tests)
    tests.to_csv(R4 / "hypothesis_tests.csv", index=False)

    pd.set_option("display.width", 250, "display.max_rows", 300)
    for task in ("binary", "three_class"):
        s = summ[summ.task == task].copy()
        s["cfg"] = s.sensor + ":" + s.set
        print(f"\n===== {task}: per-subject macro-F1 (eval windows), LR =====")
        print(s.pivot_table(index="cfg", columns=["scheme", "mode"], values="f1_mean").round(3).to_string())
        print("pooled accuracy LOSO:")
        print(s[s.scheme == "loso"].pivot_table(index="cfg", columns="mode", values="pooled_acc").round(3).to_string())
    print("\n===== hypothesis tests =====")
    print(tests.round(4).to_string(index=False))
    lat = pd.read_csv(R4 / "g2_latency.csv")
    print("\n===== stress-onset latency (median over subjects, s) =====")
    print(lat[lat.base == "both:levels+rw:calib2"].pivot_table(index=["task", "scheme"], columns=["smoother", "H"], values="latency_median_s").round(0).to_string())

    # figure: personalisation effect per subject (binary and 3-class, cross-order and LOSO)
    fig, axs = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    for ax, (task, scheme) in zip(axs, [(t, s) for t in ("binary", "three_class") for s in ("loso", "cross_order")]):
        a = sel(ps, task=task, scheme=scheme, sensor="both", set="levels+rw", mode="none")
        b = sel(ps, task=task, scheme=scheme, sensor="both", set="levels+rw", mode="calib2").reindex(a.index)
        for s_ in a.index:
            ax.plot([0, 1], [a[s_], b[s_]], color="#b9b8b2", lw=0.9)
        ax.scatter(np.zeros(len(a)), a, color="#2a78d6", s=16, zorder=3)
        ax.scatter(np.ones(len(b)), b, color="#eb6834", s=16, zorder=3)
        ax.plot([0, 1], [a.mean(), b.mean()], color="#0b0b0b", lw=2.2)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["no calibration", "2-min calibration"], fontsize=8)
        ax.set_title(f"{task}, {scheme}", loc="left", fontsize=10); ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("macro-F1 per subject (chest + wrist, levels + RW graph)")
    fig.tight_layout(); fig.savefig(FIG / "personalisation_per_subject.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
