"""M2 - descriptive checks of the features (no model, no feature selection).

* within-subject effect of the conditions on every feature (paired over 15 subjects)
* 'clock-likeness': drift of each feature with time inside the 20-min baseline run
* signal-quality table (BVP vs ECG heart rate)
These numbers describe the data; they are NOT used to choose features (fixed in the plan).
"""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wesad.physio import CHEST_NODES, WRIST_NODES  # noqa: E402
from wesad.graphs import edge_names  # noqa: E402

R = ROOT / "results" / "m2"
FIG = ROOT / "figures" / "m2"
FIG.mkdir(parents=True, exist_ok=True)
META = ["subject", "label", "run_id", "run_pos", "start_s", "end_s", "t_mid_min", "t_in_run_s", "win_s", "step_s"]


def family(c):
    for k in ("g_corr_", "g_pcorr_", "g_rw0.005_", "g_rw0.02_", "g_rw0.05_"):
        if c.startswith(k):
            return k.strip("_").replace("g_", "")
    return "levels"


def main():
    df = pd.read_csv(R / "features_60s.csv")
    feats = [c for c in df.columns if c not in META and not c.endswith("_valid")]
    means = df[df.label.isin([1, 2, 3])].groupby(["subject", "label"])[feats].mean()

    rows = []
    for c in feats:
        m = means[c].unstack()
        rec = dict(feature=c, family=family(c))
        for (a, b, tag) in ((2, 1, "stress_vs_base"), (2, 3, "stress_vs_amus"), (3, 1, "amus_vs_base")):
            d = (m[a] - m[b]).dropna()
            sd = d.std(ddof=1)
            rec[f"dz_{tag}"] = d.mean() / sd if sd > 0 else 0.0
            rec[f"up_{tag}"] = int((d > 0).sum())
            rec[f"p_{tag}"] = wilcoxon(d).pvalue if (d != 0).sum() > 5 else np.nan
        # drift inside the baseline run (same condition, ~20 min)
        rhos = []
        for s, g in df[df.label == 1].groupby("subject"):
            x = g[c].values
            if np.nanstd(x) > 0:
                rhos.append(abs(spearmanr(g.t_mid_min, x, nan_policy="omit").statistic))
        rec["drift_abs_rho_baseline"] = np.nanmedian(rhos) if rhos else np.nan
        rows.append(rec)
    eff = pd.DataFrame(rows)
    eff.to_csv(R / "feature_effects.csv", index=False)

    # family summary (rw shown for lambda = 0.02, the middle of the grid)
    fam = eff[eff.family.isin(["levels", "corr", "pcorr", "rw0.02"])].copy()
    fam["abs_dz_sb"] = fam.dz_stress_vs_base.abs()
    fam["abs_dz_sa"] = fam.dz_stress_vs_amus.abs()
    summ = fam.groupby("family").agg(n=("feature", "size"),
                                     median_abs_dz_stress_vs_base=("abs_dz_sb", "median"),
                                     n_abs_dz_gt_0_8=("abs_dz_sb", lambda s: int((s > 0.8).sum())),
                                     median_abs_dz_stress_vs_amus=("abs_dz_sa", "median"),
                                     median_drift=("drift_abs_rho_baseline", "median"))
    summ.to_csv(R / "family_summary.csv")
    print(summ.round(3).to_string())

    # ---- figure 1: effect of stress on every coupling edge, by estimator
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [10, 6], "wspace": 0.08})
    for ax, (cfg, nodes) in zip(axs, (("chest", CHEST_NODES), ("wrist", WRIST_NODES))):
        names = edge_names(nodes)
        M = np.array([[eff.set_index("feature").loc[f"g_{m}_{cfg}_{e}", "dz_stress_vs_base"] for e in names]
                      for m in ("corr", "pcorr", "rw0.02")])
        im = ax.imshow(M, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        ax.set_yticks(range(3)); ax.set_yticklabels(["correlation", "partial corr.", "random walk"] if cfg == "chest" else [])
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(M[i, j]) > 1.2 else "black")
        ax.set_title(f"{cfg}: stress vs baseline (d_z, n=15)", loc="left", fontsize=10)
        ax.grid(False)
    fig.colorbar(im, ax=axs, shrink=0.7, pad=0.02, label="d_z (paired, n=15); blue = weaker coupling under stress")
    fig.savefig(FIG / "coupling_edges_stress_effect.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---- figure 2: effect size vs clock-likeness
    colors = {"levels": "#2a78d6", "corr": "#eb6834", "pcorr": "#1baf7a", "rw0.02": "#6250d6"}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for f_, g in fam.groupby("family"):
        ax.scatter(g.drift_abs_rho_baseline, g.abs_dz_sb, s=22, color=colors[f_], label=f_, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("drift inside the baseline run (median |Spearman ρ| with time)")
    ax.set_ylabel("|d_z| stress vs baseline")
    for _, r in fam[fam.drift_abs_rho_baseline > 0.5].iterrows():
        ax.annotate(r.feature, (r.drift_abs_rho_baseline, r.abs_dz_sb), fontsize=7, xytext=(-4, 5),
                    textcoords="offset points", ha="right", color="#52514e")
    ax.legend(frameon=False)
    ax.set_title("Each dot is a feature: right = clock-like, up = stress-sensitive", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "effect_vs_drift.png", dpi=150); plt.close(fig)

    # ---- quality table: wrist vs chest heart rate
    df["hr_absdiff"] = (df.ecg_hr_mean - df.bvp_hr_mean).abs()
    q = df.assign(bad=df.hr_absdiff > 10).pivot_table(index="subject", columns="label", values="bad", aggfunc="mean")
    q.to_csv(R / "qc_bvp_hr_disagreement_gt10bpm.csv")

    top = eff.assign(a=eff.dz_stress_vs_base.abs()).sort_values("a", ascending=False)
    print("\nTop 15 features by |d_z| stress vs baseline:")
    print(top[["feature", "family", "dz_stress_vs_base", "up_stress_vs_base", "dz_stress_vs_amus", "drift_abs_rho_baseline"]].head(15).round(2).to_string(index=False))
    print("\nTop graph features:")
    print(top[top.family.isin(["corr", "pcorr", "rw0.02"])][["feature", "dz_stress_vs_base", "up_stress_vs_base", "dz_stress_vs_amus", "dz_amus_vs_base", "drift_abs_rho_baseline"]].head(12).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
