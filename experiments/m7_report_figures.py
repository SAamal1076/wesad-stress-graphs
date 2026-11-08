"""M7 - the two summary figures of the report, drawn from the result tables."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "report"
FIG.mkdir(parents=True, exist_ok=True)
INK, QUIET, GRID, MUTE, ACC = "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2", "#2a78d6"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GRID, "xtick.color": QUIET, "ytick.color": INK})

# ---- 1. LOSO vs cross-order (binary, per-subject macro-F1)
m3 = pd.read_csv(ROOT / "results" / "m3" / "summary.csv")
m4 = pd.read_csv(ROOT / "results" / "m4" / "personalisation_summary.csv")
b3 = m3[m3.task == "binary"]
get3 = lambda sensor, name, model, scheme: float(b3[(b3.sensor == sensor) & (b3["set"] == name) & (b3.model == model) & (b3.scheme == scheme)].f1_mean.iloc[0])  # noqa: E731
b4 = m4[(m4.task == "binary") & (m4.sensor == "both") & (m4["set"] == "levels+rw-noclock") & (m4["mode"] == "none")]
rows = [("Minutes since start only (RF)", get3("time", "time", "RF", "loso"), get3("time", "time", "RF", "cross_order")),
        ("Movement only", get3("motion", "motion", "LR", "loso"), get3("motion", "motion", "LR", "cross_order")),
        ("Wrist physiology", get3("wrist", "levels", "LR", "loso"), get3("wrist", "levels", "LR", "cross_order")),
        ("Chest physiology", get3("chest", "levels", "LR", "loso"), get3("chest", "levels", "LR", "cross_order")),
        ("Chest + wrist physiology", get3("both", "levels", "LR", "loso"), get3("both", "levels", "LR", "cross_order")),
        ("No drifting features + RW graph*", float(b4[b4.scheme == "loso"].f1_mean.iloc[0]), float(b4[b4.scheme == "cross_order"].f1_mean.iloc[0]))]
fig, ax = plt.subplots(figsize=(9, 4.2))
for i, (name, lo, cr) in enumerate(rows):
    y = len(rows) - 1 - i
    c = ACC if i == 0 else MUTE
    ax.plot([min(lo, cr), max(lo, cr)], [y, y], color=c, lw=2.5, alpha=0.6)
    ax.scatter([lo], [y], s=60, facecolor="white", edgecolor=c, linewidth=2, zorder=3)
    ax.scatter([cr], [y], s=60, color=c, zorder=3)
    ax.text(max(lo, cr) + 0.01, y, f"{cr - lo:+.2f}", va="center", color=INK if i == 0 else QUIET,
            fontweight="bold" if i == 0 else "normal")
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows][::-1])
ax.set_xlim(0.5, 1.02)
ax.grid(axis="x", color=GRID)
ax.set_xlabel("mean per-subject macro-F1 (binary, 15 subjects); hollow = LOSO, filled = cross-protocol-order", color=QUIET)
ax.set_title("The clock wins under LOSO and collapses across protocol orders", loc="left", fontsize=12, fontweight="bold")
fig.text(0.01, 0.01, "* scored without the first 2 baseline minutes. Logistic regression unless marked RF.", fontsize=8, color=QUIET)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(FIG / "clock_vs_cross_order.png", dpi=150); plt.close(fig)

# ---- 2. meditation false alarms by removed signal group
med = pd.read_csv(ROOT / "results" / "m5" / "meditation_false_alarms.csv")
med["label"] = med.dropped.replace({"none": "Nothing removed (final model)", "graph chest": "Chest coupling graph", "graph wrist": "Wrist coupling graph", "respiration": "Respiration", "temperature": "Temperature slopes", "BVP": "BVP (wrist pulse)"})
med = pd.concat([med[med.dropped == "none"], med[med.dropped != "none"].sort_values("meditation_false_alarm")])
fig, ax = plt.subplots(figsize=(9, 4.6))
y = list(range(len(med)))[::-1]
cols = [ACC if d == "graph chest" else MUTE for d in med.dropped]
ax.barh(y, med.meditation_false_alarm * 100, color=cols, height=0.62)
ref = float(med[med.dropped == "none"].meditation_false_alarm.iloc[0]) * 100
ax.axvline(ref, color=QUIET, ls="--", lw=1)
for yy, (_, r) in zip(y, med.iterrows()):
    ax.text(r.meditation_false_alarm * 100 + 0.6, yy, f"{r.meditation_false_alarm * 100:.0f}%", va="center", color=QUIET)
    ax.text(52, yy, f"{r.stress_hit_rate * 100:.0f}%", va="center", color=QUIET)
ax.text(52, max(y) + 0.75, "stress detected", color=QUIET, fontsize=9)
ax.set_ylim(-0.6, max(y) + 1.3)
ax.set_yticks(y); ax.set_yticklabels(med.label)
ax.set_xlim(0, 58); ax.set_xlabel("meditation windows called stress (%), one signal group removed and the model retrained", color=QUIET)
ax.set_title("Removing the chest coupling graph cuts meditation false alarms from 36% to 21%", loc="left", fontsize=12, fontweight="bold", pad=14)
ax.grid(axis="x", color=GRID)
fig.tight_layout(); fig.savefig(FIG / "meditation_false_alarms.png", dpi=150); plt.close(fig)
print("saved", list(FIG.glob("*.png")))
