"""M5 - interpretation (descriptive): decoupling robustness, modality importance,
self-report validation (incl. meditation as an out-of-task check), per-subject failures."""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from wesad.windows import SUBJECTS, task_view  # noqa: E402
from wesad.physio import GRID_FS, group_columns, process_subject  # noqa: E402
from wesad.evaluation import assert_no_subject_leak, per_subject_metrics  # noqa: E402
from m3_models import outer_splits  # noqa: E402
from m4_personalise import CLOCK  # noqa: E402

R = ROOT / "results" / "m5"
FIG = ROOT / "figures" / "m5"
R.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
EDGE = {"chest": "g_rw0.02_chest_hr_ecg~act_chest", "wrist": "g_rw0.02_wrist_hr_bvp~act_wrist"}
COND_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}


def final_columns(cols):
    lev = group_columns(cols, "chest_levels") + group_columns(cols, "wrist_levels")
    lev = [c for c in lev if c not in CLOCK]
    graph = [c for c in cols if c.startswith("g_rw0.02_chest_") or c.startswith("g_rw0.02_wrist_")]
    return lev + graph


def lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))


def add_activity(feats):
    """Movement level/variability per window from the 4 Hz activity nodes."""
    for s in SUBJECTS:
        P = process_subject(s)
        w = feats[feats.subject == s]
        for dev in ("chest", "wrist"):
            a = P[f"act_{dev}"]
            seg = [a[int(round(x * GRID_FS)):int(round(y * GRID_FS))] for x, y in zip(w.start_s, w.end_s)]
            feats.loc[w.index, f"act_{dev}_mean"] = [np.log(v.mean() + 1e-6) for v in seg]
            feats.loc[w.index, f"act_{dev}_std"] = [np.log(v.std() + 1e-6) for v in seg]
    return feats


# ------------------------------------------------------------------ 1. decoupling robustness
def decoupling(feats):
    rows = []
    d = feats[feats.label.isin([1, 2, 3])]
    for dev, edge in EDGE.items():
        hr = "ecg_hr_mean" if dev == "chest" else "bvp_hr_mean"
        covs = [f"act_{dev}_mean", f"act_{dev}_std", hr, "resp_int_cv"]
        for s, g in d.groupby("subject"):
            g = g.dropna(subset=[edge] + covs)
            z = (g[covs] - g[covs].mean()) / g[covs].std()
            X_unadj = np.c_[np.ones(len(g)), g.label == 2, g.label == 3].astype(float)
            X_adj = np.c_[X_unadj, z.values]
            y = g[edge].values
            b_un = np.linalg.lstsq(X_unadj, y, rcond=None)[0][1]
            b_adj = np.linalg.lstsq(X_adj, y, rcond=None)[0][1]
            # movement-matched: stress vs non-stress inside within-subject quintiles of movement variability
            q = pd.qcut(g[f"act_{dev}_std"], 5, labels=False, duplicates="drop")
            diffs = [g[(q == k) & (g.label == 2)][edge].mean() - g[(q == k) & (g.label != 2)][edge].mean()
                     for k in np.unique(q) if ((q == k) & (g.label == 2)).sum() >= 3 and ((q == k) & (g.label != 2)).sum() >= 3]
            rows.append(dict(device=dev, subject=s, stress_coef_unadjusted=b_un, stress_coef_adjusted=b_adj,
                             matched_diff=np.mean(diffs) if diffs else np.nan, n_matched_bins=len(diffs)))
    res = pd.DataFrame(rows)
    res.to_csv(R / "decoupling_robustness.csv", index=False)
    summ = []
    for dev, g in res.groupby("device"):
        for col in ("stress_coef_unadjusted", "stress_coef_adjusted", "matched_diff"):
            v = g[col].dropna()
            summ.append(dict(device=dev, measure=col, median=v.median(), n_negative=int((v < 0).sum()),
                             n=len(v), p=wilcoxon(v).pvalue))
    summ = pd.DataFrame(summ)
    summ.to_csv(R / "decoupling_summary.csv", index=False)
    return res, summ


# ------------------------------------------------------------------ 2. modality importance
GROUPS = {"ECG": ("ecg_",), "BVP": ("bvp_",), "EDA chest": ("eda_chest_",), "EDA wrist": ("eda_wrist_",),
          "respiration": ("resp_",), "temperature": ("temp_",), "EMG": ("emg_",),
          "graph chest": ("g_rw0.02_chest_",), "graph wrist": ("g_rw0.02_wrist_",)}


def loso_f1(t, cols, scheme):
    y, g = t.y.values, t.subject.values
    pred = np.empty(len(y), int)
    for tr, te in outer_splits(g, scheme):
        assert_no_subject_leak(g, tr, te)
        pred[te] = lr().fit(t.iloc[tr][cols].values, y[tr]).predict(t.iloc[te][cols].values)
    return per_subject_metrics(y, pred, g).set_index("subject").macro_f1


def ablation(feats):
    rows = []
    for task in ("binary", "three_class"):
        t = task_view(feats, task)
        full = final_columns(t.columns)
        for scheme in ("loso", "cross_order"):
            base = loso_f1(t, full, scheme)
            for name, pref in GROUPS.items():
                cols = [c for c in full if not c.startswith(pref)]
                f = loso_f1(t, cols, scheme)
                d = base - f.reindex(base.index)
                rows.append(dict(task=task, scheme=scheme, group=name, n_features=len(full) - len(cols),
                                 full_f1=base.mean(), drop_f1=f.mean(), loss=d.mean(),
                                 n_subjects_hurt=int((d > 0).sum())))
            print(task, scheme, "full F1 %.3f" % base.mean(), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(R / "modality_ablation.csv", index=False)
    return res


# ------------------------------------------------------------------ 3. self-reports
def predictions_all_windows(feats, drop=()):
    """LOSO final model (binary, trained on labels 1-3) predicting every window incl. meditation."""
    cols = [c for c in final_columns(feats.columns) if not c.startswith(tuple(drop))] if drop else final_columns(feats.columns)
    tr_rows = feats.label.isin([1, 2, 3]).values
    y = (feats.label == 2).astype(int).values
    p = np.empty(len(feats))
    g = feats.subject.values
    for s in SUBJECTS:
        tr = tr_rows & (g != s)
        te = g == s
        p[te] = lr().fit(feats.loc[tr, cols].values, y[tr]).predict_proba(feats.loc[te, cols].values)[:, 1]
    return p


def self_reports(feats, p):
    q = pd.read_csv(ROOT / "results" / "eda" / "questionnaires.csv")  # one row per subject x position (1-5)
    runs = feats.assign(p=p).groupby(["subject", "run_pos", "label"]).p.mean().reset_index()
    m = runs.merge(q, left_on=["subject", "run_pos"], right_on=["subject", "position"])
    m.to_csv(R / "selfreport_vs_prediction.csv", index=False)
    within = []
    for s, g in m.groupby("subject"):
        within.append(dict(subject=s, rho_stai=spearmanr(g.p, g.stai_state).statistic,
                           rho_arousal=spearmanr(g.p, g.sam_arousal).statistic,
                           rho_panas_stressed=spearmanr(g.p, g.panas_stressed).statistic))
    within = pd.DataFrame(within)
    piv = m.pivot_table(index="subject", columns="label", values=["p", "stai_state", "sam_arousal"], aggfunc="mean")
    dp = piv[("p", 2)] - piv[("p", 1)]
    between = dict(rho_dP_dSTAI=spearmanr(dp, piv[("stai_state", 2)] - piv[("stai_state", 1)]),
                   rho_dP_dArousal=spearmanr(dp, piv[("sam_arousal", 2)] - piv[("sam_arousal", 1)]))
    med = dict(meditation_p_mean=float(m[m.label == 4].p.mean()), baseline_p_mean=float(m[m.label == 1].p.mean()),
               stress_p_mean=float(m[m.label == 2].p.mean()), amusement_p_mean=float(m[m.label == 3].p.mean()),
               meditation_windows_predicted_stress=float((p[feats.label.values == 4] > 0.5).mean()))
    return m, within, between, med


# ------------------------------------------------------------------ 4. failures
def failures(m, feats):
    ps = pd.read_csv(ROOT / "results" / "m4" / "personalisation_per_subject.csv")
    f1 = ps[(ps.task == "binary") & (ps.sensor == "both") & (ps["set"] == "levels+rw-noclock") & (ps["mode"] == "none")]
    f1 = f1.pivot_table(index="subject", columns="scheme", values="macro_f1")
    piv = m.pivot_table(index="subject", columns="label", values="stai_state")
    hr = feats[feats.label.isin([1, 2])].groupby(["subject", "label"]).ecg_hr_mean.mean().unstack()
    out = f1.assign(stai_change=piv[2] - piv[1], hr_change=hr[2] - hr[1]).round(3)
    notes = {"S2": "chest temperature sensor not fully attached", "S3": "sunny baseline; valence 7 after stress",
             "S5": "may have slept in meditation 1", "S6": "not really stressed by the interview",
             "S8": "stressful day; cold room during stress", "S15": "did not believe the TSST cover story",
             "S16": "cold room during stress", "S17": "chest temperature sensor not fully attached"}
    out["readme_note"] = [notes.get(s, "") for s in out.index]
    out = out.reindex(sorted(out.index, key=lambda s: int(s[1:])))
    out.to_csv(R / "per_subject_failures.csv")
    corr = {c: spearmanr(out[c], out.cross_order).statistic for c in ("stai_change", "hr_change")}
    return out, corr


def figures(dec, abl, m):
    # decoupling: unadjusted vs adjusted coefficient per subject
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, dev in zip(axs, ("chest", "wrist")):
        g = dec[dec.device == dev]
        for _, r in g.iterrows():
            ax.plot([0, 1, 2], [r.stress_coef_unadjusted, r.stress_coef_adjusted, r.matched_diff], color="#b9b8b2", lw=0.9)
        for k, c in enumerate(("stress_coef_unadjusted", "stress_coef_adjusted", "matched_diff")):
            ax.scatter(np.full(len(g), k), g[c], color="#6250d6", s=16, zorder=3)
            ax.plot([k - 0.15, k + 0.15], [g[c].median()] * 2, color="#0b0b0b", lw=2.2)
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["raw", "adjusted for\nmovement, HR,\nbreathing", "movement-\nmatched"], fontsize=8)
        ax.set_title(f"{dev}: stress effect on heart-rate~movement edge", loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("change in edge weight under stress (per subject)")
    fig.tight_layout(); fig.savefig(FIG / "decoupling_robustness.png", dpi=150); plt.close(fig)

    # ablation
    a = abl[(abl.task == "binary")].pivot_table(index="group", columns="scheme", values="loss")
    a = a.sort_values("cross_order")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = np.arange(len(a))
    ax.barh(y + 0.2, a["loso"], height=0.38, color="#2a78d6", label="LOSO")
    ax.barh(y - 0.2, a["cross_order"], height=0.38, color="#eb6834", label="cross-order")
    ax.set_yticks(y); ax.set_yticklabels(a.index); ax.axvline(0, color="#52514e", lw=0.8)
    ax.set_xlabel("drop in mean per-subject macro-F1 when the group is removed (binary)")
    ax.legend(frameon=False, loc="lower right"); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "modality_ablation.png", dpi=150); plt.close(fig)

    # self-report vs predicted stress probability, one dot per subject x run
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (col, lab) in zip(axs, (("stai_state", "STAI state (6-24)"), ("sam_arousal", "SAM arousal (1-9)"))):
        for lbl, c in COND_COLORS.items():
            g = m[m.label == lbl]
            ax.scatter(g[col] + np.random.default_rng(lbl).uniform(-0.15, 0.15, len(g)), g.p, color=c, s=22,
                       edgecolor="white", linewidth=0.5, label={1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}[lbl])
        ax.set_xlabel(lab); ax.set_ylabel("mean predicted P(stress) in the run")
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "selfreport_vs_prediction.png", dpi=150); plt.close(fig)


def main():
    feats = pd.read_csv(ROOT / "results" / "m2" / "features_60s.csv")
    feats = add_activity(feats.copy())
    dec, dsum = decoupling(feats)
    print(dsum.round(4).to_string(index=False))
    abl = ablation(feats)
    pd.set_option("display.width", 200)
    print(abl.pivot_table(index="group", columns=["task", "scheme"], values="loss").round(3).to_string())
    p = predictions_all_windows(feats)
    m, within, between, med = self_reports(feats, p)
    print("\nwithin-subject Spearman (5 runs): median", within[["rho_stai", "rho_arousal", "rho_panas_stressed"]].median().round(2).to_dict(),
          "| n positive STAI:", int((within.rho_stai > 0).sum()))
    print("between-subject:", {k: (round(v.statistic, 2), round(v.pvalue, 3)) for k, v in between.items()})
    print("mean P(stress) by condition:", {k: round(v, 3) for k, v in med.items()})
    within.to_csv(R / "selfreport_within_subject.csv", index=False)
    # out-of-task check: which signal group makes meditation look like stress?
    med_rows = [dict(dropped="none", meditation_false_alarm=float((p[feats.label.values == 4] > 0.5).mean()))]
    for name, pref in GROUPS.items():
        pp = predictions_all_windows(feats, drop=pref)
        med_rows.append(dict(dropped=name, meditation_false_alarm=float((pp[feats.label.values == 4] > 0.5).mean()),
                             stress_hit_rate=float((pp[feats.label.values == 2] > 0.5).mean())))
    med_rows[0]["stress_hit_rate"] = float((p[feats.label.values == 2] > 0.5).mean())
    medt = pd.DataFrame(med_rows)
    medt.to_csv(R / "meditation_false_alarms.csv", index=False)
    print(medt.round(3).to_string(index=False))
    mm = m[m.label == 4].assign(which=lambda d: d.groupby("subject").run_pos.rank().astype(int))
    print("meditation P(stress) by meditation number:", mm.groupby("which").p.mean().round(3).to_dict())
    print(feats.groupby("label")[["resp_rate", "resp_depth_mean", "resp_int_cv", "ecg_hr_mean"]].median().round(2).to_string())
    fail, corr = failures(m, feats)
    print(fail.to_string()); print("Spearman with cross-order F1:", {k: round(v, 2) for k, v in corr.items()})
    figures(dec, abl, m)


if __name__ == "__main__":
    main()
