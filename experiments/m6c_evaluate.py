"""M6-C stage C3b - evaluation (pre-registered in docs/analysis_plan.md).

Part 1  Indices on the causal grid: decoupling index D, additional heart rate (aHR), their LOSO logistic
        combination, and HR minus its calibration mean; per-subject AUCs; H10-H11 (paired Wilcoxon, Holm).
Part 2  Descriptive latency at stress onset and false-alarm rates.
Part 3  Classification with the M3 pipeline: H8, H9 (M6-B per-window TV graph) and H12 (online graph).
Part 4  Figures.
Outputs: results/m6/c3_*.csv, figures/m6/*.png
"""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from wesad.windows import SUBJECTS, task_view  # noqa: E402
from wesad.physio import CHEST_NODES, WRIST_NODES, group_columns  # noqa: E402
from wesad.evaluation import assert_no_subject_leak, per_subject_metrics  # noqa: E402
from m3_models import LAMBDAS, inner_select, make_lr, outer_splits  # noqa: E402
from m3_stats import holm  # noqa: E402
from m4_personalise import CLOCK  # noqa: E402

R = ROOT / "results" / "m6"
FIG = ROOT / "figures" / "m6"
FIG.mkdir(parents=True, exist_ok=True)
DEV = {"chest": (CHEST_NODES, "act_chest"), "wrist": (WRIST_NODES, "act_wrist")}
COND = {1: ("baseline", "#2a78d6"), 2: ("stress", "#eb6834"), 3: ("amusement", "#1baf7a"),
        4: ("meditation", "#eda100")}
SCORE_COLORS = {"aHR": "#2a78d6", "D": "#eb6834", "combined": "#1baf7a"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"


def wtest(d):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    return wilcoxon(d).pvalue if (d != 0).sum() > 5 else 1.0


# ------------------------------------------------------------------ part 1: indices
def add_indices(g):
    """D (unsupervised, person-relative) for both devices."""
    for dev, (nodes, act) in DEV.items():
        phys = [nd for nd in nodes if nd != act]
        cols = [f"g_on_{dev}_{min(p, act, key=nodes.index)}~{max(p, act, key=nodes.index)}" for p in phys]
        calib = g[g.calib].groupby("subject")[cols].mean()
        wbar = calib.loc[g.subject].values
        g[f"D_{dev}"] = (wbar - g[cols].values).mean(1)
        hr_cal = g[g.calib].groupby("subject")[f"hr_{dev}"].mean()
        g[f"dHR_{dev}"] = g[f"hr_{dev}"].values - hr_cal.loc[g.subject].values
    return g


def loso_scores(g):
    """aHR (LOSO population slope + personal calibration intercept) and the LOSO combined score."""
    lab = g.label.isin([1, 2, 3]) & ~g.calib
    for dev in DEV:
        hr, act, D = f"hr_{dev}", f"act_{dev}", f"D_{dev}"
        g[f"aHR_{dev}"] = np.nan
        g[f"comb_{dev}"] = np.nan
        for s in SUBJECTS:
            tr_subj = g.subject != s
            nb = tr_subj & g.label.isin([1, 3]) & g[hr].notna()
            c = g[nb].groupby("subject")[[hr, act]].transform(lambda x: x - x.mean())
            b = float((c[hr] * c[act]).sum() / (c[act] ** 2).sum())
            cal = g.calib & (g.subject != s) | g.calib & (g.subject == s)
            a_i = (g.loc[cal, hr] - b * g.loc[cal, act]).groupby(g.loc[cal, "subject"]).mean()
            ahr = g[hr] - a_i.loc[g.subject].values - b * g[act]
            tr = tr_subj & lab & ahr.notna()
            X_tr = np.c_[ahr[tr], g.loc[tr, D]]
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X_tr, g.loc[tr, "label"] == 2)
            te = g.subject == s
            g.loc[te, f"aHR_{dev}"] = ahr[te]
            ok = te & ahr.notna()
            g.loc[ok, f"comb_{dev}"] = m.decision_function(np.c_[ahr[ok], g.loc[ok, D]])
            g.loc[te, f"b_{dev}"] = b
    return g


def subject_aucs(g):
    rows = []
    for dev in DEV:
        for s, d in g[~g.calib].groupby("subject"):
            for contrast, neg in (("stress_vs_nonstress", [1, 3]), ("stress_vs_meditation", [4])):
                m = d.label.isin([2] + neg)
                for score in ("aHR", "D", "comb", "dHR"):
                    v = d.loc[m, f"{score}_{dev}"]
                    ok = v.notna()
                    y = (d.loc[m, "label"] == 2)[ok]
                    rows.append(dict(device=dev, subject=s, contrast=contrast, score=score,
                                     auc=roc_auc_score(y, v[ok]) if y.nunique() == 2 else np.nan))
    return pd.DataFrame(rows)


def index_tests(auc):
    rows = []
    for contrast, h in (("stress_vs_nonstress", "H10"), ("stress_vs_meditation", "H11")):
        for dev in DEV:
            p = auc[(auc.device == dev) & (auc.contrast == contrast)].pivot(index="subject", columns="score", values="auc")
            diff = p["comb"] - p["aHR"]
            rows.append(dict(hypothesis=h, device=dev, contrast=contrast,
                             auc_aHR=p["aHR"].mean(), auc_D=p["D"].mean(), auc_comb=p["comb"].mean(),
                             auc_dHR=p["dHR"].mean(), median_diff=diff.median(),
                             n_better=int((diff > 0).sum()), n=int(diff.notna().sum()), p=wtest(diff)))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p.values)
    return t


# ------------------------------------------------------------------ part 2: latency, false alarms
def latency(g):
    rows = []
    for dev in DEV:
        for score in ("aHR", "D", "comb"):
            col = f"{score}_{dev}"
            for s in SUBJECTS:
                tr = (g.subject != s) & g.label.isin([1, 3]) & ~g.calib
                thr = np.nanpercentile(g.loc[tr, col], 95)
                d = g[g.subject == s].sort_values("end_s")
                above = (d[col] > thr).values
                run3 = above & np.r_[above[1:], False] & np.r_[above[2:], False, False]
                cand = (d.end_s.values >= d.stress_start_s.values) & (d.end_s.values <= d.stress_end_s.values) & run3
                lat = d.end_s.values[cand][0] - d.stress_start_s.values[0] if cand.any() else np.inf
                fa = {COND[k][0]: float(np.mean(above[(d.label.values == k) & ~d.calib.values]))
                      for k in (1, 3, 4)}
                rows.append(dict(device=dev, score=score, subject=s, threshold=thr, latency_s=lat,
                                 hit_rate_in_stress=float(np.mean(above[d.label.values == 2])),
                                 **{f"false_alarm_{k}": v for k, v in fa.items()}))
    lat = pd.DataFrame(rows)
    summ = lat.groupby(["device", "score"]).agg(
        latency_median_s=("latency_s", "median"), detected=("latency_s", lambda x: int(np.isfinite(x).sum())),
        hit_rate=("hit_rate_in_stress", "mean"), fa_baseline=("false_alarm_baseline", "mean"),
        fa_amusement=("false_alarm_amusement", "mean"), fa_meditation=("false_alarm_meditation", "mean")).reset_index()
    return lat, summ


# ------------------------------------------------------------------ part 3: classification
def online_to_windows(feats, g):
    """Online-graph columns at the latest grid window ending at or before each labelled window."""
    cols = [c for c in g.columns if c.startswith("g_on_")]
    key = g.set_index(["subject", "end_s"])[cols]
    end_grid = np.floor(feats.end_s.values / 5.0 + 1e-9) * 5.0
    idx = pd.MultiIndex.from_arrays([feats.subject.values, end_grid])
    out = key.reindex(idx)
    assert out.notna().all().all(), "unmatched windows"
    return out.reset_index(drop=True)


def feature_sets(cols, dev):
    base = [c for c in group_columns(cols, f"{dev}_levels") if c not in CLOCK]
    tv = [c for c in cols if c.startswith(f"g_tv_{dev}_")]
    on = [c for c in cols if c.startswith(f"g_on_{dev}_")]
    rw = [[c for c in cols if c.startswith(f"g_rw{lam}_{dev}_")] for lam in LAMBDAS]
    return {"base": [base], "base+tv": [base + tv], "base+rw": [base + r for r in rw], "base+online": [base + on]}


def fold(t, y, g, tr, te, cands):
    assert_no_subject_leak(g, tr, te)
    k, C = inner_select(t.iloc[tr], y[tr], g[tr], cands)
    m = make_lr(C).fit(t.iloc[tr][cands[k]].values, y[tr])
    return te, m.predict(t.iloc[te][cands[k]].values)


def classification(feats, jobs=2):
    rows = []
    for task in ("binary", "three_class"):
        t = task_view(feats, task)
        y, g = t.y.values, t.subject.values
        for scheme in ("loso", "cross_order"):
            for dev in DEV:
                for name, cands in feature_sets(t.columns, dev).items():
                    res = Parallel(n_jobs=jobs)(delayed(fold)(t, y, g, tr, te, cands) for tr, te in outer_splits(g, scheme))
                    pred = np.empty(len(y), int)
                    for te, p in res:
                        pred[te] = p
                    ps = per_subject_metrics(y, pred, g)
                    rows.append(ps.assign(task=task, scheme=scheme, device=dev, set=name))
                    print(f"{task:11s} {scheme:11s} {dev:5s} {name:12s} F1 {ps.macro_f1.mean():.3f}", flush=True)
    return pd.concat(rows, ignore_index=True)


def class_tests(ps):
    rows = []
    for scheme in ("loso", "cross_order"):
        fam = []
        for h, a, b in (("H8", "base+tv", "base"), ("H9", "base+tv", "base+rw"), ("H12", "base+online", "base")):
            for dev in DEV:
                for task in ("binary", "three_class"):
                    q = ps[(ps.scheme == scheme) & (ps.device == dev) & (ps.task == task)]
                    p = q.pivot(index="subject", columns="set", values="macro_f1")
                    diff = p[a] - p[b]
                    fam.append(dict(scheme=scheme, hypothesis=h, device=dev, task=task, f1_a=p[a].mean(),
                                    f1_b=p[b].mean(), mean_diff=diff.mean(), n_better=int((diff > 0).sum()),
                                    p=wtest(diff)))
        fam = pd.DataFrame(fam)
        m89 = fam.hypothesis.isin(["H8", "H9"])
        fam.loc[m89, "p_holm"] = holm(fam.loc[m89, "p"].values)
        fam.loc[~m89, "p_holm"] = holm(fam.loc[~m89, "p"].values)
        rows.append(fam)
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------ part 4: figures
def fig_timecourses(g, dev):
    fig, axes = plt.subplots(5, 3, figsize=(13, 13), sharey=True)
    for ax, s in zip(axes.ravel(), SUBJECTS):
        d = g[g.subject == s].sort_values("end_s")
        t = d.end_s.values / 60
        for k, (name, col) in COND.items():
            m = (d.label.values == k).astype(float)
            ax.fill_between(t, -6, 10, where=m > 0, color=col, alpha=0.18, lw=0, step="mid")
        for score, ls, col in (("aHR", "-", MUTED), ("D", "-", INK)):
            v = d[f"{score}_{dev}"].values
            cal = d.calib.values
            z = (v - np.nanmean(v[cal])) / (np.nanstd(v[~np.isnan(v) & (d.label.values == 1)]) + 1e-9)
            ax.plot(t, z, ls, color=col, lw=1.2 if score == "D" else 1.6, label=score)
        ax.set_title(s, fontsize=10, color=INK, loc="left")
        ax.set_ylim(-6, 10)
        ax.grid(axis="y", color="#e8e7e3", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
    handles = [plt.Line2D([], [], color=INK, lw=1.2, label="decoupling index D"),
               plt.Line2D([], [], color=MUTED, lw=1.6, label="additional heart rate (aHR)")]
    handles += [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.3, label=n) for n, c in COND.values()]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, fontsize=9)
    fig.supxlabel("time since recording start (min)", color=INK2)
    fig.supylabel("score, in units of the baseline SD (0 = calibration mean)", color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / f"timecourse_{dev}.png", dpi=130)
    plt.close(fig)


def fig_aucs(auc):
    fig, axes = plt.subplots(1, 4, figsize=(13, 4), sharey=True)
    panels = [(dev, c) for c in ("stress_vs_nonstress", "stress_vs_meditation") for dev in DEV]
    for ax, (dev, contrast) in zip(axes, panels):
        p = auc[(auc.device == dev) & (auc.contrast == contrast)].pivot(index="subject", columns="score", values="auc")
        order = ["aHR", "D", "comb"]
        for _, r in p[order].iterrows():
            ax.plot(range(3), r.values, color="#d5d4cf", lw=0.8, zorder=1)
        for k, sc in enumerate(order):
            key = "combined" if sc == "comb" else sc
            ax.scatter(np.full(len(p), k), p[sc], s=36, color=SCORE_COLORS[key], edgecolor="#fcfcfb", lw=1.5, zorder=2)
            ax.text(k, 1.02, f"{p[sc].mean():.2f}", ha="center", fontsize=9, color=INK)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
        ax.set_xticks(range(3), ["aHR", "D", "aHR + D"])
        ax.set_title(f"{dev}: {contrast.replace('_', ' ')}", fontsize=10, color=INK, loc="left")
        ax.set_ylim(0, 1.08)
        ax.set_xlim(-0.45, 2.45)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=9)
    axes[0].set_ylabel("per-subject AUC (numbers = mean)", color=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "index_aucs.png", dpi=130)
    plt.close(fig)


def main():
    g = pd.read_csv(R / "online_grid.csv")
    g = loso_scores(add_indices(g))
    auc = subject_aucs(g)
    auc.to_csv(R / "c3_auc_per_subject.csv", index=False)
    it = index_tests(auc)
    it.to_csv(R / "c3_index_tests.csv", index=False)
    print(it.round(3).to_string(index=False))

    lat, lsum = latency(g)
    lat.to_csv(R / "c3_latency_per_subject.csv", index=False)
    lsum.to_csv(R / "c3_latency_summary.csv", index=False)
    print("\n", lsum.round(3).to_string(index=False))
    print("\naHR slope b (bpm per movement unit), range over folds:",
          {dev: (round(g[f'b_{dev}'].min(), 2), round(g[f'b_{dev}'].max(), 2)) for dev in DEV})

    fig_timecourses(g, "chest")
    fig_timecourses(g, "wrist")
    fig_aucs(auc)

    feats = pd.read_csv(ROOT / "results" / "m2" / "features_60s.csv")
    tv = pd.read_csv(R / "tv_features_60s.csv")
    feats = pd.concat([feats, tv[[c for c in tv.columns if c.startswith("g_tv_")]],
                       online_to_windows(feats, g)], axis=1)
    ps = classification(feats)
    ps.to_csv(R / "c3_class_per_subject.csv", index=False)
    ct = class_tests(ps)
    ct.to_csv(R / "c3_class_tests.csv", index=False)
    print("\n", ct.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
