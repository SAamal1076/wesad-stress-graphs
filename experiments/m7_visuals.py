"""Visual summary assets: SVG panels drawn from the repository's real data and results.

Used by experiments/m7_portfolio.py (README image) and by the presentation. Every shape and number comes from:
- the processed signals of one typical participant (S9: the median person for the heart-rate~movement change)
- the condition runs of all 15 subjects (session timetables)
- per-subject macro-F1 summaries (README table) and the M2 coupling-graph features
"""
import html
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wesad.windows import LABEL_KEY, SUBJECTS, condition_runs, load_subject  # noqa: E402

ORDER_A = ["S4", "S5", "S7", "S8", "S10", "S13", "S15", "S17"]   # amusement before stress
ORDER_B = ["S2", "S3", "S6", "S9", "S11", "S14", "S16"]          # stress before amusement
COND = {1: ("Baseline (rest)", "#2a78d6"), 3: ("Amusement", "#1baf7a"), 2: ("Stress test", "#eb6834"),
        4: ("Meditation", "#eda100")}
STRESS, REST = "#eb6834", "#2a78d6"
INK, INK2, MUTED, RULE, PAPER = "#16202e", "#4a5566", "#8b95a5", "#dce1e8", "#f7f8fa"
TYPICAL = "S9"
SLOPES = [("Minutes since start only", 0.951, 0.546, MUTED, "7 6", 3),
          ("Physiology, all features", 0.918, 0.875, "#7d8aa3", None, 3),
          ("Physiology without drifting features*", 0.933, 0.943, INK, None, 5)]
EDGE = "g_rw0.02_chest_hr_ecg~act_chest"
CSS = f"""
svg text {{ font-family: 'Inter', sans-serif; fill: {INK2}; }}
svg .s {{ font-size: 15px; }} svg .m {{ font-size: 17px; }}
svg .b {{ font-size: 17px; font-weight: 650; fill: {INK}; }}
svg .v {{ font-family: 'Inter Display','Inter',sans-serif; font-size: 24px; font-weight: 650; }}
svg .big {{ font-family: 'Inter Display','Inter',sans-serif; font-size: 44px; font-weight: 700; }}
"""


def _svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">'
            f'<style>{CSS}</style>{body}</svg>')


# ------------------------------------------------------------------ data
def runs_table():
    rows = []
    for s in SUBJECTS:
        d = load_subject(s)
        for r in condition_runs(d[LABEL_KEY], float(d["chest_ECG_fs"]), s).itertuples(index=False):
            rows.append(dict(subject=s, label=r.label, start=r.start_s / 60, end=r.end_s / 60))
    return pd.DataFrame(rows)


def session_signals(subject=TYPICAL, step_s=2.0):
    from wesad.physio import process_subject
    P = process_subject(subject)
    t = P["t_grid"]
    fs = float(P["eda_fs_chest"])
    eda = np.interp(t, np.arange(len(P["eda_tonic_chest"])) / fs, P["eda_tonic_chest"] + P["eda_phasic_chest"])

    def smooth(x, sec, how="median"):
        s = pd.Series(x).interpolate(limit_direction="both")
        r = s.rolling(int(sec * 4), center=True, min_periods=1)
        return (r.median() if how == "median" else r.mean()).values

    k = int(step_s * 4)
    return dict(t=t[::k] / 60, hr=smooth(P["hr_ecg"], 30)[::k], eda=smooth(eda, 20)[::k],
                act=smooth(P["act_chest"], 6, "mean")[::k])


def coupling_data():
    f = pd.read_csv(ROOT / "results" / "m2" / "features_60s.csv")
    cols = [c for c in f.columns if c.startswith("g_rw0.02_chest_") and "~" in c]
    per = f[f.label.isin([1, 2])].groupby(["subject", "label"])[cols].mean()
    means = per.groupby("label").mean()
    m = per[EDGE].unstack()
    return means, m


# ------------------------------------------------------------------ panels
def band_rects(runs, x, y, h, rx=3):
    out = []
    for t in runs.itertuples(index=False):
        out.append(f'<rect x="{x(t.start):.1f}" y="{y}" width="{max(x(t.end) - x(t.start), 2):.1f}" height="{h}" '
                   f'rx="{rx}" fill="{COND[t.label][1]}" opacity="0.13"/>')
    return out


def svg_session(sig, runs, width=1408):
    """Three stacked tracks of one participant's whole session, condition bands behind."""
    left, right, top = 190, 30, 58
    tmax = float(np.ceil(sig["t"][-1] / 5) * 5)
    pw = width - left - right
    x = lambda m: left + pw * m / tmax  # noqa: E731
    tracks = [("Heart rate", "beats per minute", sig["hr"], INK),
              ("Skin conductance", "sweat, microsiemens", sig["eda"], INK),
              ("Movement", "chest accelerometer", sig["act"], INK)]
    th, gap = 120, 26
    p = []
    r = runs[runs.subject == TYPICAL]
    total_h = top + 3 * th + 2 * gap
    p += band_rects(r, x, top - 6, total_h - top + 12)
    # condition names above the bands
    for t in r.itertuples(index=False):
        name = COND[t.label][0].split(" (")[0]
        p.append(f'<text x="{(x(t.start) + x(t.end)) / 2:.1f}" y="{top - 16}" class="s" text-anchor="middle" '
                 f'fill="{COND[t.label][1]}" style="fill:{COND[t.label][1]};font-weight:650">{name}</text>')
    for i, (name, unit, y, col) in enumerate(tracks):
        y0 = top + i * (th + gap)
        lo, hi = np.nanpercentile(y, 0.5), np.nanpercentile(y, 99.5)
        yy = lambda v: y0 + th - (th - 8) * (np.clip(v, lo, hi) - lo) / (hi - lo + 1e-12) - 4  # noqa: E731
        pts = " ".join(f"{x(tt):.1f},{yy(v):.1f}" for tt, v in zip(sig["t"], y))
        p.append(f'<line x1="{left}" x2="{left + pw}" y1="{y0 + th}" y2="{y0 + th}" stroke="{RULE}"/>')
        p.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.6" stroke-linejoin="round"/>')
        p.append(f'<text x="{left - 20}" y="{y0 + th / 2 - 4}" class="b" text-anchor="end">{name}</text>'
                 f'<text x="{left - 20}" y="{y0 + th / 2 + 18}" class="s" text-anchor="end">{unit}</text>')
    axis = top + 3 * th + 2 * gap + 26
    for m in range(0, int(tmax) + 1, 10):
        p.append(f'<text x="{x(m)}" y="{axis}" class="s" text-anchor="middle">{m}</text>')
    p.append(f'<text x="{left + pw}" y="{axis + 24}" class="s" text-anchor="end">minutes</text>')
    return _svg(width, axis + 34, "".join(p))


def svg_timetables(runs, width=1408):
    left, right, row_h, gap, group_gap, tmax = 160, 30, 20, 7, 40, 115.0
    pw = width - left - right
    x = lambda m: left + pw * m / tmax  # noqa: E731
    parts, lx = [], left
    for lab in (1, 3, 2, 4):
        name, col = COND[lab]
        parts.append(f'<rect x="{lx}" y="6" width="16" height="16" rx="4" fill="{col}"/>'
                     f'<text x="{lx + 24}" y="20" class="m">{name}</text>')
        lx += 46 + 9.6 * len(name)
    y0 = y = 74
    for gname, members in (("Order A: amusement, then stress test", ORDER_A),
                           ("Order B: stress test, then amusement", ORDER_B)):
        parts.append(f'<text x="{left}" y="{y - 12}" class="b">{gname}</text>')
        for s in members:
            r = runs[runs.subject == s]
            parts.append(f'<text x="{left - 14}" y="{y + 15}" class="s" text-anchor="end">{s}</text>'
                         f'<line x1="{x(0)}" x2="{x(r.end.max())}" y1="{y + row_h / 2}" y2="{y + row_h / 2}" '
                         f'stroke="{RULE}" stroke-width="2"/>')
            for t in r.itertuples(index=False):
                parts.append(f'<rect x="{x(t.start) + 1:.1f}" y="{y}" width="{max(x(t.end) - x(t.start) - 2, 2):.1f}" '
                             f'height="{row_h}" rx="4" fill="{COND[t.label][1]}"/>')
            y += row_h + gap
        y += group_gap
    axis_y = y - group_gap + 6
    for m in range(0, 111, 10):
        parts.append(f'<line x1="{x(m)}" x2="{x(m)}" y1="{y0 - 4}" y2="{axis_y}" stroke="{RULE}" opacity="0.6"/>'
                     f'<text x="{x(m)}" y="{axis_y + 20}" class="s" text-anchor="middle">{m}</text>')
    parts.append(f'<text x="{left + pw}" y="{axis_y + 44}" class="s" text-anchor="end">'
                 f'minutes since the start of the recording</text>')
    return _svg(width, axis_y + 52, "".join(parts))


def svg_benchmark_bars(width=640):
    rows = [("Original WESAD paper (physiology)", 93.1, "#7d8aa3"), ("Minutes since start, nothing else", 95.7, STRESS)]
    p, y = [], 18
    lo, hi, left, bw = 85, 100, 0, width - 90
    x = lambda v: left + bw * (v - lo) / (hi - lo)  # noqa: E731
    for name, v, col in rows:
        p.append(f'<text x="{left}" y="{y + 16}" class="m">{name}</text>'
                 f'<rect x="{left}" y="{y + 28}" width="{x(v) - left:.1f}" height="30" rx="4" fill="{col}"/>'
                 f'<text x="{x(v) + 12:.1f}" y="{y + 51}" class="v" style="fill:{INK}">{v:.1f}%</text>')
        y += 82
    p.append(f'<text x="{left}" y="{y + 8}" class="s">Accuracy, stress vs non-stress, leave-one-person-out. '
             f'Axis starts at 85%.</text>')
    return _svg(width, y + 18, "".join(p))


def svg_slopes(width=760, height=440):
    top, bottom, left, right = 30, 56, 44, 300
    X0, X1 = left + 110, width - right
    yv = lambda v: top + (height - top - bottom) * (1.0 - v) / 0.5  # noqa: E731
    p = []
    for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        p.append(f'<line x1="{left}" x2="{X1 + 30}" y1="{yv(v)}" y2="{yv(v)}" stroke="{RULE}"/>'
                 f'<text x="{left - 10}" y="{yv(v) + 5}" class="s" text-anchor="end">{v:.1f}</text>')
    for xx, lab in ((X0, "Tested on new people"), (X1, "New people + new schedule")):
        p.append(f'<text x="{xx}" y="{height - 18}" class="b" text-anchor="middle">{lab}</text>')
    for name, a, b, col, dash, sw in SLOPES:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p.append(f'<line x1="{X0}" y1="{yv(a)}" x2="{X1}" y2="{yv(b)}" stroke="{col}" stroke-width="{sw}" '
                 f'stroke-linecap="round"{d}/>')
        for xx, v in ((X0, a), (X1, b)):
            p.append(f'<circle cx="{xx}" cy="{yv(v)}" r="8" fill="{col}" stroke="{PAPER}" stroke-width="2.5"/>')
    placed = []
    for name, a, b, col, _, _ in sorted(SLOPES, key=lambda r: -r[2]):
        yy = yv(b)
        if placed and yy - placed[-1] < 44:
            yy = placed[-1] + 44
        placed.append(yy)
        wgt = 700 if col == INK else 500
        p.append(f'<text x="{X1 + 22}" y="{yy}" class="v" style="fill:{col};font-weight:{wgt}">{b:.3f}</text>'
                 f'<text x="{X1 + 22}" y="{yy + 20}" class="s">{html.escape(name)}</text>')
    prev = None
    for name, a, b, col, _, _ in sorted(SLOPES, key=lambda r: -r[1]):
        yy = yv(a) + 5
        if prev is not None and yy - prev < 19:
            yy = prev + 19
        prev = yy
        p.append(f'<text x="{X0 - 18}" y="{yy:.1f}" class="s" text-anchor="end" style="fill:{col}">{a:.3f}</text>')
    # arrow on the collapsing line
    p.append(f'<text x="{(X0 + X1) / 2 + 26}" y="{yv(0.76)}" class="s" style="fill:{MUTED};font-style:italic">'
             f'the clock collapses</text>')
    return _svg(width, height, "".join(p))


def svg_split(width=520, height=300):
    """Train on one protocol order, test on the other: people icons."""
    def person(cx, cy, col, s=1.0):
        return (f'<circle cx="{cx}" cy="{cy - 15 * s}" r="{9 * s}" fill="{col}"/>'
                f'<path d="M{cx - 15 * s},{cy + 18 * s} Q{cx - 15 * s},{cy - 2 * s} {cx},{cy - 2 * s} '
                f'Q{cx + 15 * s},{cy - 2 * s} {cx + 15 * s},{cy + 18 * s} Z" fill="{col}"/>')
    p = []
    p.append(f'<text x="10" y="26" class="b">Train: 8 people, order A</text>')
    for i in range(8):
        p.append(person(30 + i * 40, 76, INK))
    p.append(f'<text x="10" y="150" class="b">Test: 7 other people, order B</text>')
    for i in range(7):
        p.append(person(30 + i * 40, 200, STRESS))
    p.append(f'<text x="10" y="262" class="s">then the reverse. Nothing from the test people or their</text>'
             f'<text x="10" y="284" class="s">schedule is used to fit the model.</text>')
    p.append(f'<path d="M{width - 70},70 C{width - 20},100 {width - 20},160 {width - 70},190" fill="none" '
             f'stroke="{INK2}" stroke-width="2.5" marker-end="url(#ar)"/>'
             f'<defs><marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" '
             f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>')
    return _svg(width, height, "".join(p))


def icon(kind, x, y, s=1.0, col=INK):
    """Small line icons drawn by hand (48x48 box)."""
    g = f'<g transform="translate({x},{y}) scale({s})" fill="none" stroke="{col}" stroke-width="2.6" ' \
        f'stroke-linecap="round" stroke-linejoin="round">'
    d = {
        "wave": '<path d="M2,26 L10,26 L14,14 L19,38 L24,8 L29,32 L33,26 L46,26"/>'
                '<path d="M8,4 L8,46 M40,4 L40,46" stroke-dasharray="3 4"/>',
        "bars": '<path d="M6,44 L6,30 M16,44 L16,18 M26,44 L26,24 M36,44 L36,10 M2,44 L46,44"/>',
        "graph": '<circle cx="10" cy="12" r="5"/><circle cx="38" cy="10" r="5"/><circle cx="24" cy="38" r="5"/>'
                 '<circle cx="42" cy="34" r="4"/><path d="M15,12 L33,10 M12,17 L21,33 M36,15 L27,33 M29,38 L38,35"/>',
        "clockx": '<circle cx="24" cy="24" r="18"/><path d="M24,13 L24,24 L32,29"/><path d="M6,42 L42,6" '
                  f'stroke="{STRESS}" stroke-width="3.4"/>',
        "sigmoid": '<path d="M2,44 L46,44 M2,4 L2,44"/><path d="M4,40 C18,40 18,8 44,8" stroke-width="3"/>',
        "split": '<rect x="2" y="8" width="20" height="32" rx="4"/><rect x="26" y="8" width="20" height="32" rx="4" '
                 f'stroke="{STRESS}"/><path d="M12,18 L12,30 M36,18 L36,30" />',
        "person": '<circle cx="24" cy="14" r="8"/><path d="M8,44 C8,28 40,28 40,44"/><path d="M36,6 L44,6 M40,2 L40,10"/>',
        "lens": '<circle cx="20" cy="20" r="13"/><path d="M30,30 L44,44" stroke-width="3.4"/><path d="M14,20 L26,20 '
                'M20,14 L20,26"/>',
        "watch": '<circle cx="24" cy="27" r="17"/><path d="M24,27 L24,16 M18,4 L30,4 M24,4 L24,10 M24,27 L31,32"/>',
        "heartmove": '<path d="M16,40 C2,30 4,16 12,16 C16,16 18,20 18,20 C18,20 20,16 24,16 C32,16 34,30 18,40 Z"/>'
                     '<path d="M30,10 C34,6 40,6 44,10 M30,18 C34,14 40,14 44,18"/>',
        "band": '<path d="M16,2 L32,2 L32,12 M16,2 L16,12 M16,36 L16,46 L32,46 L32,36"/><rect x="10" y="12" width="28" '
                'height="24" rx="6"/><path d="M18,24 L22,24 L25,18 L28,30 L30,24"/>',
        "repo": '<circle cx="12" cy="10" r="5"/><circle cx="12" cy="38" r="5"/><circle cx="36" cy="18" r="5"/>'
                '<path d="M12,15 L12,33 M36,23 C36,30 20,28 14,34"/>',
    }[kind]
    return g + d + "</g>"


def svg_coupling(width=1408, height=380):
    means, m = coupling_data()
    n_down = int((m[2] < m[1]).sum())
    names = ["Heart rate", "Breathing", "Skin conductance", "Muscle", "Movement"]
    keys = ["hr_ecg", "resp", "eda_ph_chest", "emg_env", "act_chest"]
    ang = [-90, -18, 54, 126, 198]
    p = []

    def network(cx, cy, lab, title, col):
        R = 88
        pos = [(cx + R * np.cos(np.radians(a)), cy + R * np.sin(np.radians(a))) for a in ang]
        out = [f'<text x="{cx}" y="{cy - R - 52}" class="b" text-anchor="middle" style="fill:{col}">{title}</text>']
        for i in range(5):
            for j in range(i + 1, 5):
                w = means.loc[lab, f"g_rw0.02_chest_{keys[i]}~{keys[j]}"]
                hl = (i, j) == (0, 4)
                out.append(f'<line x1="{pos[i][0]:.1f}" y1="{pos[i][1]:.1f}" x2="{pos[j][0]:.1f}" y2="{pos[j][1]:.1f}" '
                           f'stroke="{col if hl else "#c3cad5"}" stroke-width="{2 + 26 * w:.1f}" '
                           f'stroke-linecap="round" opacity="{1 if hl else 0.8}"/>')
        for (px, py), nm, a in zip(pos, names, ang):
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="13" fill="{INK}" stroke="{PAPER}" stroke-width="3"/>')
            dx, anchor, dy = (20, "start", 5) if -90 < a < 90 else (-20, "end", 5)
            if a == -90:
                dx, anchor, dy = 0, "middle", -24
            if a == 198:
                dx, anchor, dy = 0, "middle", 38
            out.append(f'<text x="{px + dx:.1f}" y="{py + dy:.1f}" class="s" text-anchor="{anchor}">{nm}</text>')
        w = means.loc[lab, EDGE]
        mx, my = (pos[0][0] + pos[4][0]) / 2, (pos[0][1] + pos[4][1]) / 2
        out.append(f'<text x="{mx - 16:.1f}" y="{my - 10:.1f}" class="v" text-anchor="end" style="fill:{col}">'
                   f'{w:.2f}</text>')
        return out

    p += network(200, 210, 1, "At rest", REST)
    p += network(600, 210, 2, "During the stress test", STRESS)
    # one line per person
    X0, X1, top, bot = 960, 1200, 60, height - 60
    lo, hi = 0.15, 0.5
    yv = lambda v: top + (bot - top) * (hi - v) / (hi - lo)  # noqa: E731
    for v in (0.2, 0.3, 0.4, 0.5):
        p.append(f'<line x1="{X0 - 24}" x2="{X1 + 24}" y1="{yv(v)}" y2="{yv(v)}" stroke="{RULE}"/>'
                 f'<text x="{X0 - 34}" y="{yv(v) + 5}" class="s" text-anchor="end">{v:.1f}</text>')
    for s, r in m.iterrows():
        p.append(f'<line x1="{X0}" y1="{yv(r[1]):.1f}" x2="{X1}" y2="{yv(r[2]):.1f}" stroke="{STRESS}" '
                 f'stroke-width="1.8" opacity="0.7"/>'
                 f'<circle cx="{X0}" cy="{yv(r[1]):.1f}" r="4" fill="{REST}"/>'
                 f'<circle cx="{X1}" cy="{yv(r[2]):.1f}" r="4" fill="{STRESS}"/>')
    p.append(f'<text x="{X0}" y="{height - 26}" class="b" text-anchor="middle">Rest</text>'
             f'<text x="{X1}" y="{height - 26}" class="b" text-anchor="middle">Stress</text>'
             f'<text x="{(X0 + X1) / 2}" y="30" class="b" text-anchor="middle">Every person: {n_down} of 15 drop</text>'
             f'<text x="{X1 + 40}" y="{yv(0.33)}" class="s">heart-rate↔movement</text>'
             f'<text x="{X1 + 40}" y="{yv(0.33) + 20}" class="s">coupling weight</text>')
    return _svg(width, height, "".join(p)), n_down


def svg_steps(width=1408):
    steps = [("wave", "Cut", "One-minute windows,", "each inside one condition"),
             ("bars", "Describe", "38 body measures + a graph", "of how signals move together"),
             ("clockx", "Remove the clock", "Drop 4 measures that drift", "with time (temperature, sweat level)"),
             ("sigmoid", "Classify", "Simple, readable model:", "p = σ(w·z + b)"),
             ("split", "Test fairly", "New people and a new", "schedule, never seen in training")]
    cw = width / 5
    p = []
    for i, (ic, t, l1, l2) in enumerate(steps):
        cx = i * cw + cw / 2
        col = STRESS if ic == "clockx" else INK
        p.append(f'<circle cx="{cx}" cy="58" r="50" fill="#fff" stroke="{RULE}" stroke-width="2"/>')
        p.append(icon(ic, cx - 30, 28, 1.25, INK))
        if i < 4:
            p.append(f'<path d="M{cx + 64},58 L{cx + cw - 64},58" stroke="{MUTED}" stroke-width="2.5" '
                     f'marker-end="url(#ar2)"/>')
        p.append(f'<text x="{cx}" y="140" class="b" text-anchor="middle" style="font-size:20px;fill:{col}">'
                 f'{i + 1}. {t}</text>'
                 f'<text x="{cx}" y="168" class="m" text-anchor="middle">{html.escape(l1)}</text>'
                 f'<text x="{cx}" y="192" class="m" text-anchor="middle">{html.escape(l2)}</text>')
    p.append(f'<defs><marker id="ar2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
             f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>')
    return _svg(width, 206, "".join(p))


def svg_vr_tiles(width=1408):
    tiles = [("person", "Works on a new trainee", ["Tested on people and on", "a schedule it never saw"]),
             ("lens", "Interpretable", ["Named body measures and", "a readable coupling marker"]),
             ("watch", "Near real time", ["Updates every 5 s; stress", "onset flagged in a median", "of 13 s"]),
             ("heartmove", "Body + behaviour", ["Links body signals to", "movement; ready for head,", "hand and gaze in VR"]),
             ("repo", "Reproducible", ["Plan fixed in git before", "each analysis; one script", "reruns everything"])]
    cw, p = width / 5, []
    for i, (ic, t, lines) in enumerate(tiles):
        x0 = i * cw + 10
        p.append(f'<rect x="{x0}" y="0" width="{cw - 20}" height="226" rx="14" fill="#fff" stroke="{RULE}" '
                 f'stroke-width="1.5"/>')
        p.append(icon(ic, x0 + 24, 24, 1.1, INK))
        p.append(f'<text x="{x0 + 24}" y="114" class="b" style="font-size:19px">{t}</text>')
        for k, ln in enumerate(lines):
            p.append(f'<text x="{x0 + 22}" y="{146 + 24 * k}" class="m" style="font-size:15px">{html.escape(ln)}</text>')
    return _svg(width, 230, "".join(p))


def build_all():
    runs = runs_table()
    sig = session_signals()
    coupling, n_down = svg_coupling()
    check_sequences(runs)
    return dict(session=svg_session(sig, runs), timetables=svg_timetables(runs), bench=svg_benchmark_bars(),
                sequences=svg_sequences(),
                slopes=svg_slopes(), split=svg_split(), steps=svg_steps(), coupling=coupling,
                tiles=svg_vr_tiles(), n_down=n_down)




def svg_realtime(subject="S5", width=1408, height=420):
    """Causal heart-rate index over one whole session: threshold and the first detection after stress begins.
    S5 is the median participant for detection delay (results/m6/c3_latency_per_subject.csv)."""
    sys.path.insert(0, str(ROOT / "experiments"))
    import m6c_evaluate as E
    g = E.loso_scores(E.add_indices(pd.read_csv(ROOT / "results" / "m6" / "online_grid.csv")))
    lat = pd.read_csv(ROOT / "results" / "m6" / "c3_latency_per_subject.csv")
    row = lat[(lat.device == "chest") & (lat.score == "aHR") & (lat.subject == subject)].iloc[0]
    d = g[g.subject == subject].sort_values("end_s")
    t, v = d.end_s.values / 60, d["aHR_chest"].values
    runs = runs_table()
    r = runs[runs.subject == subject]
    left, right, top, bot = 200, 40, 40, height - 70
    tmax = float(np.ceil(t.max() / 10) * 10)
    lo, hi = np.nanpercentile(v, 1), np.nanpercentile(v, 99.5)
    lo, hi = min(lo, row.threshold - 5), max(hi, row.threshold + 5)
    x = lambda m: left + (width - left - right) * m / tmax  # noqa: E731
    y = lambda u: bot - (bot - top) * (np.clip(u, lo, hi) - lo) / (hi - lo)  # noqa: E731
    p = band_rects(r, x, top, bot - top)
    for tt in r.itertuples(index=False):
        name = COND[tt.label][0].split(" (")[0]
        p.append(f'<text x="{(x(tt.start) + x(tt.end)) / 2:.1f}" y="{top - 12}" class="s" text-anchor="middle" '
                 f'style="fill:{COND[tt.label][1]};font-weight:650">{name}</text>')
    pts = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in zip(t, v) if np.isfinite(b))
    p.append(f'<line x1="{left}" x2="{width - right}" y1="{y(row.threshold):.1f}" y2="{y(row.threshold):.1f}" '
             f'stroke="{INK2}" stroke-width="2" stroke-dasharray="8 6"/>'
             f'<text x="{width - right}" y="{y(row.threshold) - 10:.1f}" class="s" text-anchor="end">alarm threshold '
             f'(set on other people)</text>')
    p.append(f'<polyline points="{pts}" fill="none" stroke="{INK}" stroke-width="2"/>')
    s0 = d.stress_start_s.iloc[0] / 60
    det = s0 + row.latency_s / 60
    p.append(f'<line x1="{x(s0):.1f}" x2="{x(s0):.1f}" y1="{top}" y2="{bot}" stroke="{STRESS}" stroke-width="2"/>'
             f'<circle cx="{x(det):.1f}" cy="{y(row.threshold):.1f}" r="9" fill="{STRESS}" stroke="{PAPER}" '
             f'stroke-width="3"/>'
             f'<text x="{x(s0) - 16:.1f}" y="{top + 34:.1f}" class="b" text-anchor="end" style="fill:{STRESS}">'
             f'flagged {row.latency_s:.0f} s after the stress test began</text>')
    for m in range(0, int(tmax) + 1, 10):
        p.append(f'<text x="{x(m)}" y="{bot + 26}" class="s" text-anchor="middle">{m}</text>')
    p.append(f'<text x="{width - right}" y="{bot + 52}" class="s" text-anchor="end">minutes</text>'
             f'<text x="{left - 16}" y="{(top + bot) / 2 - 6}" class="b" text-anchor="end">Heart rate</text>'
             f'<text x="{left - 16}" y="{(top + bot) / 2 + 16}" class="s" text-anchor="end">above rest, bpm</text>'
             f'<text x="{left - 16}" y="{(top + bot) / 2 + 36}" class="s" text-anchor="end">(movement-adjusted)</text>')
    return _svg(width, height, "".join(p))


def render_pngs(svgs, out_dir):
    """Render SVG strings to 2x PNGs with the local fonts (labels inside SVGs need the page's fonts)."""
    from playwright.sync_api import sync_playwright
    out_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path="/usr/bin/google-chrome")
        pg = b.new_page(device_scale_factor=2)
        for name, s in svgs.items():
            pg.set_content(f'<html><body style="margin:0;background:{PAPER}">{s}</body></html>')
            pg.wait_for_timeout(150)
            pg.locator("svg").first.screenshot(path=str(out_dir / f"{name}.png"))
        b.close()



# ------------------------------------------------------------------ protocol orders and graphical abstract
SEQ = {"A": [1, 3, 4, 2, 4], "B": [1, 2, 4, 3, 4]}       # read from the condition runs of every subject


def check_sequences(runs):
    """Confirm SEQ against the data: every subject of an order has that exact sequence of conditions."""
    for name, members in (("A", ORDER_A), ("B", ORDER_B)):
        for s in members:
            seq = runs[runs.subject == s].sort_values("start").label.tolist()
            assert seq == SEQ[name], (s, seq)


def _seq_row(x0, y0, w, h, seq, gap=10, fs=17, short=False):
    names = {1: "Rest", 3: "Amusement", 4: "Meditation", 2: "Stress test"}
    if short:
        names = {1: "Rest", 3: "Fun", 4: "Calm", 2: "Stress"}
    bw = (w - gap * (len(seq) - 1)) / len(seq)
    out = []
    for i, lab in enumerate(seq):
        x = x0 + i * (bw + gap)
        col = COND[lab][1]
        out.append(f'<rect x="{x:.1f}" y="{y0}" width="{bw:.1f}" height="{h}" rx="8" fill="{col}" opacity="0.16"/>'
                   f'<rect x="{x:.1f}" y="{y0}" width="6" height="{h}" rx="3" fill="{col}"/>'
                   f'<text x="{x + bw / 2 + 3:.1f}" y="{y0 + h / 2 + fs * 0.35:.1f}" text-anchor="middle" '
                   f'style="font-size:{fs}px;fill:{INK};font-weight:{700 if lab == 2 else 500}">{names[lab]}</text>')
        if i < len(seq) - 1:
            out.append(f'<text x="{x + bw + gap / 2:.1f}" y="{y0 + h / 2 + 5:.1f}" text-anchor="middle" '
                       f'style="font-size:14px;fill:{MUTED}">›</text>')
    return out


def svg_sequences(width=1408):
    p = []
    for k, (name, n, y) in enumerate((("Order A", len(ORDER_A), 0), ("Order B", len(ORDER_B), 70))):
        p.append(f'<text x="0" y="{y + 26}" class="b">{name}</text>'
                 f'<text x="0" y="{y + 48}" class="s">{n} people</text>')
        p += _seq_row(150, y, width - 150, 56, SEQ[name[-1]])
    return _svg(width, 128, "".join(p))


def svg_graphical_abstract(width=1600, height=680):
    means, m = coupling_data()
    n_down = int((m[2] < m[1]).sum())
    p = [f'<rect width="{width}" height="{height}" fill="{PAPER}"/>']
    p.append(f'<text x="48" y="62" style="font-family:Lora,serif;font-size:36px;font-weight:600;fill:{INK}">'
             f'Stress detection that still works when the schedule changes</text>'
             f'<text x="48" y="94" class="m">Open WESAD dataset: 15 people, chest and wrist wearables, '
             f'a lab stress test. Every number below comes from that data.</text>')
    top, ph, pw, gap = 124, 430, 352, 40
    xs = [48 + i * (pw + gap) for i in range(4)]
    heads = [("1", "The data"), ("2", "A hidden shortcut"), ("3", "A fair test"), ("4", "The result")]
    for i, (x, (n, t)) in enumerate(zip(xs, heads)):
        p.append(f'<rect x="{x}" y="{top}" width="{pw}" height="{ph}" rx="18" fill="#fff" stroke="{RULE}" '
                 f'stroke-width="1.5"/>'
                 f'<circle cx="{x + 34}" cy="{top + 38}" r="17" fill="{STRESS if i == 1 else INK}"/>'
                 f'<text x="{x + 34}" y="{top + 44}" text-anchor="middle" style="font-size:17px;font-weight:700;'
                 f'fill:#fff">{n}</text>'
                 f'<text x="{x + 62}" y="{top + 45}" class="b" style="font-size:21px">{t}</text>')
        if i < 3:
            ax = x + pw + 6
            p.append(f'<path d="M{ax},{top + ph / 2} L{ax + gap - 14},{top + ph / 2}" stroke="{MUTED}" '
                     f'stroke-width="3" marker-end="url(#gaar)"/>')
    p.append(f'<defs><marker id="gaar" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" '
             f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>')
    # 1: data
    x = xs[0] + 26
    p.append(icon("heartmove", x, top + 82, 1.0) + icon("band", x + 64, top + 82, 1.0))
    p.append(f'<text x="{x + 132}" y="{top + 100}" class="m">Chest + wrist sensors</text>'
             f'<text x="{x + 132}" y="{top + 124}" class="m">15 people</text>')
    p.append(f'<text x="{x}" y="{top + 182}" class="s">Two orders of events:</text>')
    for k, (nm, cnt) in enumerate((("A", len(ORDER_A)), ("B", len(ORDER_B)))):
        y = top + 202 + k * 92
        p.append(f'<text x="{x}" y="{y + 20}" class="b" style="font-size:16px">Order {nm}, {cnt} people</text>')
        p += _seq_row(x, y + 32, pw - 52, 42, SEQ[nm], gap=6, fs=13, short=True)
    p.append(f'<text x="{x}" y="{top + ph - 26}" class="s">Stress test early or late in the session</text>')
    # 2: shortcut
    x = xs[1] + 26
    p.append(icon("clockx", x + pw / 2 - 26 - 46, top + 78, 1.9, INK).replace(f'stroke="{STRESS}"', 'stroke="none"'))
    p.append(f'<text x="{xs[1] + pw / 2}" y="{top + 214}" text-anchor="middle" class="m">Minutes since the start, '
             f'nothing else:</text>'
             f'<text x="{xs[1] + pw / 2}" y="{top + 270}" text-anchor="middle" '
             f'style="font-family:Inter Display,Inter,sans-serif;font-size:56px;font-weight:700;fill:{STRESS}">95.7%</text>'
             f'<text x="{xs[1] + pw / 2}" y="{top + 300}" text-anchor="middle" class="m">accuracy on new people</text>')
    bx, bw = x, pw - 52
    for k, (lab, v, col) in enumerate((("Original paper", 93.1, "#7d8aa3"), ("Clock only", 95.7, STRESS))):
        yy = top + 334 + k * 40
        w = bw * 0.62 * (v - 85) / 15
        p.append(f'<text x="{bx}" y="{yy + 17}" class="s">{lab}</text>'
                 f'<rect x="{bx + 118}" y="{yy + 2}" width="{w:.1f}" height="20" rx="4" fill="{col}"/>'
                 f'<text x="{bx + 126 + w:.1f}" y="{yy + 18}" class="s" style="font-weight:650;fill:{INK}">{v}%</text>')
    # 3: fair test
    x = xs[2] + 26

    def dots(n, col, y0):
        return "".join(f'<circle cx="{x + 14 + i * 34}" cy="{y0}" r="12" fill="{col}"/>' for i in range(n))
    p.append(f'<text x="{x}" y="{top + 96}" class="b" style="font-size:16px">Train on order A</text>' + dots(8, INK, top + 122))
    p.append(f'<text x="{x}" y="{top + 172}" class="b" style="font-size:16px">Test on order B, then reverse</text>'
             + dots(7, STRESS, top + 198))
    for k, (ic, l1, l2) in enumerate((("clockx", "Drop 4 measures that", "drift with time"),
                                     ("graph", "A simple model plus a", "body-signal coupling graph"))):
        yy = top + 250 + k * 84
        p.append(icon(ic, x, yy, 0.95))
        p.append(f'<text x="{x + 62}" y="{yy + 20}" class="m">{l1}</text>'
                 f'<text x="{x + 62}" y="{yy + 42}" class="m">{l2}</text>')
    # 4: result: mini slope + mini network
    x = xs[3] + 26
    X0, X1, y0, y1 = x + 48, x + 196, top + 84, top + 244
    yv = lambda v: y1 - (y1 - y0) * (v - 0.5) / 0.5  # noqa: E731
    for name, a, b, col, dash, sw in SLOPES[::2]:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p.append(f'<line x1="{X0}" y1="{yv(a):.1f}" x2="{X1}" y2="{yv(b):.1f}" stroke="{col}" stroke-width="{sw}"{d}/>'
                 f'<circle cx="{X0}" cy="{yv(a):.1f}" r="6" fill="{col}"/><circle cx="{X1}" cy="{yv(b):.1f}" r="6" fill="{col}"/>'
                 f'<text x="{X1 + 12}" y="{yv(b) + 7:.1f}" style="font-size:20px;font-weight:700;fill:{col}">{b:.3f}</text>')
    p.append(f'<text x="{X0 - 12}" y="{yv(0.94) - 14:.1f}" class="s">body</text>'
             f'<text x="{(X0 + X1) / 2 - 24}" y="{yv(0.64):.1f}" class="s" text-anchor="end" style="fill:{MUTED}">clock</text>'
             f'<text x="{X0}" y="{y1 + 30}" class="s" text-anchor="middle">LOSO</text>'
             f'<text x="{X1}" y="{y1 + 30}" class="s" text-anchor="middle">new order</text>')
    cx, cy, R = x + 40, top + 356, 34
    ang = [-90, -18, 54, 126, 198]
    pos = [(cx + R * np.cos(np.radians(a)), cy + R * np.sin(np.radians(a))) for a in ang]
    for i in range(5):
        for j in range(i + 1, 5):
            hl = (i, j) == (0, 4)
            p.append(f'<line x1="{pos[i][0]:.1f}" y1="{pos[i][1]:.1f}" x2="{pos[j][0]:.1f}" y2="{pos[j][1]:.1f}" '
                     f'stroke="{STRESS if hl else "#c3cad5"}" stroke-width="{5 if hl else 2}"/>')
    for px, py in pos:
        p.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" fill="{INK}"/>')
    p.append(f'<text x="{cx + 56}" y="{top + 340}" class="m" style="font-weight:650;fill:{INK}">Heart stops following</text>'
             f'<text x="{cx + 56}" y="{top + 362}" class="m" style="font-weight:650;fill:{INK}">movement under stress</text>'
             f'<text x="{cx + 56}" y="{top + 386}" class="m">{n_down} of 15 people</text>')
    # bottom strip
    sy = top + ph + 34
    p.append(f'<rect x="48" y="{sy}" width="{width - 96}" height="72" rx="14" fill="{INK}"/>'
             f'<text x="80" y="{sy + 44}" style="font-size:20px;font-weight:700;fill:#fff">For VR training:</text>')
    for k, (ic, t) in enumerate((("person", "works on a new trainee"), ("lens", "interpretable"),
                                 ("watch", "near real time"), ("heartmove", "body + behaviour"))):
        xx = 270 + k * 320
        p.append(icon(ic, xx, sy + 16, 0.82, "#ffffff") + f'<text x="{xx + 52}" y="{sy + 44}" '
                 f'style="font-size:19px;fill:#e6eaf0">{t}</text>')
    return _svg(width, height, "".join(p))


if __name__ == "__main__":
    out = ROOT / "figures" / "report" / "visuals"
    out.mkdir(parents=True, exist_ok=True)
    v = build_all()
    v["realtime"] = svg_realtime()
    v["graphical_abstract"] = svg_graphical_abstract()
    svgs = {k: s for k, s in v.items() if isinstance(s, str)}
    for k, s in svgs.items():
        (out / f"{k}.svg").write_text(s, encoding="utf-8")
    render_pngs(svgs, out)
    print("saved", sorted(p.name for p in out.glob("*.png")))
