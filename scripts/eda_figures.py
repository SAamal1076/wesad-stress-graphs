"""Figures for the WESAD exploratory audit. Reads results/eda/*.csv and data/wesad_small/*.npz."""
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

R = pathlib.Path("/home/claude/results/eda")
S = pathlib.Path("/home/claude/data/wesad_small")
F = pathlib.Path("/home/claude/figures/eda")
(F / "subjects").mkdir(parents=True, exist_ok=True)

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
NAME = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
sid_key = lambda s: int(s[1:])
seg = pd.read_csv(R / "label_segments.csv")
cond = pd.read_csv(R / "condition_features.csv")
subs = sorted(seg.subject.unique(), key=sid_key)


def legend_conditions(ax, **kw):
    hs = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in COL]
    ax.legend(hs, [NAME[k] for k in COL], frameon=False, ncol=4, **kw)


# ---- 1. protocol timeline (Gantt) --------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
for i, s in enumerate(subs):
    d = seg[(seg.subject == s)]
    tot = d.dur_s.sum() / 60
    ax.barh(i, tot, color="#d9d8d3", height=0.62)  # whole recording (label 0 / 5-7 = grey)
    for _, r in d[d.label.between(1, 4) & (d.dur_s > 30)].iterrows():
        ax.barh(i, r.dur_s / 60, left=r.start_s / 60, color=COL[int(r.label)], height=0.62, edgecolor=SURF, linewidth=1)
ax.set_yticks(range(len(subs))); ax.set_yticklabels(subs); ax.invert_yaxis()
ax.set_xlabel("minutes since start of the synchronised recording")
ax.set_title("Protocol timeline per subject (grey = label 0 / 5-7, not used)", loc="left", color=INK)
legend_conditions(ax, loc="lower right", bbox_to_anchor=(1, 1.0))
fig.tight_layout(); fig.savefig(F / "01_protocol_timeline.png", dpi=150); plt.close(fig)

# ---- 2. class balance --------------------------------------------------------------------
m = seg[seg.label.between(1, 4) & (seg.dur_s > 30)].groupby(["subject", "label"]).dur_s.sum().unstack() / 60
m = m.reindex(subs)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.4, 1]})
bottom = np.zeros(len(m))
for k in (1, 2, 3, 4):
    a1.bar(m.index, m[k], bottom=bottom, color=COL[k], edgecolor=SURF, linewidth=1.5, width=0.75)
    bottom += m[k].values
a1.set_ylabel("labelled minutes"); a1.set_title("Labelled minutes per subject", loc="left")
a1.tick_params(axis="x", rotation=60); a1.grid(axis="x", visible=False)
tot = m.sum()
a2.bar([NAME[k] for k in tot.index], tot.values, color=[COL[k] for k in tot.index], width=0.6)
for x, v in enumerate(tot.values):
    a2.text(x, v + 4, f"{v:.0f}", ha="center", color=INK)
a2.set_title("Total minutes (15 subjects)", loc="left"); a2.grid(axis="x", visible=False)
a2.tick_params(axis="x", rotation=25)
fig.tight_layout(); fig.savefig(F / "02_class_balance.png", dpi=150); plt.close(fig)

# ---- 3. condition effects: paired lines --------------------------------------------------
feats = [("hr_ecg", "Heart rate, ECG (bpm)"), ("rmssd_ms", "RMSSD (ms)"), ("eda_chest", "EDA chest (µS)"),
         ("eda_wrist", "EDA wrist (µS)"), ("temp_chest", "Skin/chest temp (°C)"), ("temp_wrist", "Wrist temp (°C)"),
         ("resp_rate_bpm", "Respiration (breaths/min)"), ("acc_wrist_dyn_g", "Wrist motion (SD of |acc|, g)")]
fig, axs = plt.subplots(2, 4, figsize=(13, 6.4))
xs = [1, 2, 3, 4]
for ax, (f, title) in zip(axs.ravel(), feats):
    P = cond.pivot(index="subject", columns="label", values=f)
    for s in P.index:
        ax.plot(xs, P.loc[s, xs].values, color="#b9b8b2", lw=0.8, zorder=1)
    for k in xs:
        ax.scatter([k] * len(P), P[k], s=14, color=COL[k], edgecolor=SURF, linewidth=0.6, zorder=3)
    ax.plot(xs, P[xs].median().values, color=INK, lw=2.2, zorder=4)
    ax.set_xticks(xs); ax.set_xticklabels(["base", "stress", "amus.", "medit."], fontsize=8)
    ax.set_title(title, loc="left", fontsize=10); ax.grid(axis="x", visible=False)
fig.suptitle("One line per subject (15), black = median across subjects", x=0.01, ha="left", color=INK2, fontsize=10)
fig.tight_layout(); fig.savefig(F / "03_condition_effects.png", dpi=150); plt.close(fig)

# ---- 4. between-subject spread vs condition effect (EDA chest) ---------------------------
P = cond.pivot(index="subject", columns="label", values="eda_chest").reindex(subs)
P = P.sort_values(1)
fig, ax = plt.subplots(figsize=(9, 4.6))
for i, s in enumerate(P.index):
    ax.plot([1, 2], [i, i], alpha=0)  # keep axis sane
    ax.plot([P.loc[s, 1], P.loc[s, 2]], [i, i], color="#b9b8b2", lw=2, zorder=1)
    ax.scatter(P.loc[s, 1], i, color=COL[1], s=36, zorder=3, edgecolor=SURF)
    ax.scatter(P.loc[s, 2], i, color=COL[2], s=36, zorder=3, edgecolor=SURF)
ax.set_yticks(range(len(P))); ax.set_yticklabels(P.index)
ax.set_xlabel("mean chest EDA (µS)")
ax.set_title("Chest EDA: baseline (blue) → stress (orange), subjects sorted by baseline", loc="left")
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(F / "04_eda_between_vs_within.png", dpi=150); plt.close(fig)

# ---- 5. ECG snippets with simple R-peak markers (baseline, 8 s) --------------------------
fig, axs = plt.subplots(5, 3, figsize=(13, 9.5))
for ax, s in zip(axs.ravel(), subs):
    d = np.load(S / f"{s}.npz")
    ecg, fs, lab = d["chest_ECG"].ravel(), float(d["chest_ECG_fs"]), d["chest_ECG_label"]
    i0 = np.flatnonzero(lab == 1)[0] + int(300 * fs)
    x = ecg[i0:i0 + int(8 * fs)]
    t = np.arange(len(x)) / fs
    ax.plot(t, x, color=COL[1], lw=0.8)
    p, _ = find_peaks(x, height=np.percentile(x, 99.0) * 0.5, distance=int(0.3 * fs))
    ax.scatter(t[p], x[p], color=INK, s=10, zorder=3)
    ax.set_title(s, loc="left", fontsize=9); ax.set_ylim(-1.6, 1.6); ax.tick_params(labelsize=7)
fig.suptitle("Chest ECG, 8 s inside baseline (mV); dots = peaks above a simple threshold", x=0.01, ha="left", color=INK2, fontsize=10)
fig.tight_layout(); fig.savefig(F / "05_ecg_snippets.png", dpi=130); plt.close(fig)

# ---- 6. self-reports ---------------------------------------------------------------------
q = pd.read_csv(R / "questionnaires.csv")
q["c"] = q.condition.replace({"Medi 1": "Medi", "Medi 2": "Medi"})
qm = q.groupby(["subject", "c"])[["stai_state", "sam_arousal", "sam_valence"]].mean().reset_index()
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
order = ["Base", "TSST", "Fun", "Medi"]; colmap = dict(zip(order, [COL[1], COL[2], COL[3], COL[4]]))
for ax, (v, t) in zip(axs, [("stai_state", "STAI state (6-24)"), ("sam_arousal", "SAM arousal (1-9)"), ("sam_valence", "SAM valence (1-9)")]):
    P = qm.pivot(index="subject", columns="c", values=v)[order]
    for s in P.index:
        ax.plot(range(4), P.loc[s].values, color="#b9b8b2", lw=0.8)
    for j, c in enumerate(order):
        ax.scatter([j] * len(P), P[c], s=14, color=colmap[c], edgecolor=SURF, linewidth=0.6, zorder=3)
    ax.plot(range(4), P.median().values, color=INK, lw=2.2)
    ax.set_xticks(range(4)); ax.set_xticklabels(["base", "TSST", "fun", "medit."]); ax.set_title(t, loc="left")
    ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(F / "06_self_reports.png", dpi=150); plt.close(fig)

# ---- 7. per-subject overview -------------------------------------------------------------
def overview(s):
    d = np.load(S / f"{s}.npz")
    rows = [("chest_EDA", "EDA chest µS"), ("wrist_EDA", "EDA wrist µS"), ("chest_Temp", "Temp chest °C"),
            ("wrist_TEMP", "Temp wrist °C"), ("chest_Resp", "Resp"), ("accn", "|ACC| chest g (1 s SD)"), ("wrist_ACCn", "|ACC| wrist g (1 s SD)")]
    fig, axs = plt.subplots(len(rows), 1, figsize=(11, 10), sharex=True)
    for ax, (k, ttl) in zip(axs, rows):
        if k == "accn":
            a = np.linalg.norm(d["chest_ACC"], axis=1); fs = float(d["chest_ACC_fs"]); lab = d["chest_ACC_label"]
            n = int(len(a) // fs); x = a[:int(n * fs)].reshape(n, -1).std(axis=1) if float(fs).is_integer() else a
        elif k == "wrist_ACCn":
            a = np.linalg.norm(d["wrist_ACC"] / 64.0, axis=1); fs = 32.0; lab = d["wrist_ACC_label"]
            n = len(a) // 32; x = a[:n * 32].reshape(n, 32).std(axis=1)
        else:
            x = d[k].ravel(); fs = float(d[k + "_fs"]); lab = d[k + "_label"]
        if k in ("accn", "wrist_ACCn"):
            fs_plot = 1.0; lab = lab[::int(round(len(lab) / len(x)))][:len(x)]
        else:
            fs_plot = fs
        step = max(1, int(fs_plot // 4)) if fs_plot > 4 else 1
        t = np.arange(len(x)) / fs_plot / 60
        for c in (1, 2, 3, 4):
            mk = np.zeros(len(x), bool); L = min(len(lab), len(x)); mk[:L] = lab[:L] == c
            if mk.any():
                r0 = np.flatnonzero(np.diff(np.r_[0, mk.astype(int), 0]) == 1); r1 = np.flatnonzero(np.diff(np.r_[0, mk.astype(int), 0]) == -1)
                for a0, a1 in zip(r0, r1):
                    ax.axvspan(t[a0], t[min(a1, len(t) - 1)], color=COL[c], alpha=0.16, lw=0)
        ax.plot(t[::step], x[::step], color=INK, lw=0.6)
        ax.set_ylabel(ttl, fontsize=8); ax.tick_params(labelsize=8)
    axs[-1].set_xlabel("minutes")
    legend_conditions(axs[0], loc="lower right", bbox_to_anchor=(1, 1.02), fontsize=8)
    fig.suptitle(s, x=0.01, ha="left"); fig.tight_layout(); fig.savefig(F / "subjects" / f"{s}.png", dpi=110); plt.close(fig)

for s in subs:
    overview(s)
print("figures done")
