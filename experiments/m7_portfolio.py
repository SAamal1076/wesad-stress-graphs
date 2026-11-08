"""One-page visual summary of the project for the README (figures/report/portfolio.png).

Every number and shape comes from the repository's own results:
- session timetables: condition runs of each subject (src/wesad/windows.py)
- slope chart: per-subject macro-F1, results/m3 and results/m4 summaries (also in the README table)
- coupling edge: mean random-walk heart-rate~movement weight per subject, results/m2/features_60s.csv
Run: python experiments/m7_portfolio.py   (needs playwright + a Chrome/Chromium binary to render the PNG)
"""
import html
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wesad.windows import LABEL_KEY, SUBJECTS, condition_runs, load_subject  # noqa: E402

OUT = ROOT / "figures" / "report"
ORDER_A = ["S4", "S5", "S7", "S8", "S10", "S13", "S15", "S17"]   # amusement before stress
ORDER_B = ["S2", "S3", "S6", "S9", "S11", "S14", "S16"]          # stress before amusement
COND = {1: ("Baseline", "#2a78d6"), 3: ("Amusement", "#1baf7a"), 2: ("Stress", "#eb6834"),
        4: ("Meditation", "#eda100")}
INK, INK2, MUTED, RULE, PAPER = "#16202e", "#4a5566", "#8b95a5", "#dce1e8", "#f7f8fa"

# per-subject macro-F1, binary, mean of 15 (README table; * scored without the first 2 baseline minutes)
SLOPES = [("Minutes since start only", 0.951, 0.546, MUTED, "6 5", 2.5),
          ("Physiology, all features", 0.918, 0.875, "#7d8aa3", None, 2.5),
          ("Physiology without drifting features*", 0.933, 0.943, INK, None, 4.5)]


def runs_table():
    rows = []
    for s in SUBJECTS:
        d = load_subject(s)
        for r in condition_runs(d[LABEL_KEY], float(d["chest_ECG_fs"]), s).itertuples(index=False):
            rows.append(dict(subject=s, label=r.label, start=r.start_s / 60, end=r.end_s / 60))
    return pd.DataFrame(rows)


def svg_timetables(runs, width=1408):
    left, right, row_h, gap, group_gap, tmax = 150, 24, 20, 7, 34, 115.0
    plot_w = width - left - right
    x = lambda m: left + plot_w * m / tmax  # noqa: E731
    y0, parts = 64, []
    # legend
    lx = left
    for lab in (1, 3, 2, 4):
        name, col = COND[lab]
        parts.append(f'<rect x="{lx}" y="8" width="14" height="14" rx="3" fill="{col}"/>'
                     f'<text x="{lx + 22}" y="20" class="small">{name}</text>')
        lx += 36 + 9 * len(name)
    y = y0
    for gname, members in (("Order A: amusement, then stress", ORDER_A), ("Order B: stress, then amusement", ORDER_B)):
        parts.append(f'<text x="{left}" y="{y - 10}" class="label">{gname}</text>')
        for s in members:
            r = runs[runs.subject == s]
            end = r.end.max()
            parts.append(f'<text x="{left - 14}" y="{y + 15}" class="small" text-anchor="end">{s}</text>')
            parts.append(f'<line x1="{x(0)}" x2="{x(end)}" y1="{y + row_h / 2}" y2="{y + row_h / 2}" '
                         f'stroke="{RULE}" stroke-width="2"/>')
            for t in r.itertuples(index=False):
                w = max(x(t.end) - x(t.start) - 2, 2)
                parts.append(f'<rect x="{x(t.start) + 1:.1f}" y="{y}" width="{w:.1f}" height="{row_h}" rx="4" '
                             f'fill="{COND[t.label][1]}"><title>{s} {COND[t.label][0]}: '
                             f'{t.start:.0f}-{t.end:.0f} min</title></rect>')
            y += row_h + gap
        y += group_gap
    axis_y = y - group_gap + 6
    for m in range(0, 111, 10):
        parts.append(f'<line x1="{x(m)}" x2="{x(m)}" y1="{y0 - 4}" y2="{axis_y}" stroke="{RULE}" '
                     f'stroke-width="1" opacity="0.6"/>')
        parts.append(f'<text x="{x(m)}" y="{axis_y + 20}" class="small" text-anchor="middle">{m}</text>')
    parts.append(f'<text x="{left + plot_w}" y="{axis_y + 44}" class="small" text-anchor="end">'
                 f'minutes since the start of the recording</text>')
    h = axis_y + 52
    return f'<svg viewBox="0 0 {width} {h}" width="{width}" height="{h}" role="img">{"".join(parts)}</svg>'


def svg_slopes(width=760, height=430):
    top, bottom, left, right = 34, 54, 44, 300
    ymin, ymax = 0.5, 1.0
    X0, X1 = left + 110, width - right
    yv = lambda v: top + (height - top - bottom) * (ymax - v) / (ymax - ymin)  # noqa: E731
    p = []
    for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        p.append(f'<line x1="{left}" x2="{X1 + 30}" y1="{yv(v)}" y2="{yv(v)}" stroke="{RULE}" stroke-width="1"/>'
                 f'<text x="{left - 12}" y="{yv(v) + 5}" class="small" text-anchor="end">{v:.1f}</text>')
    for xx, lab in ((X0, "Leave-one-person-out"), (X1, "Other protocol order")):
        p.append(f'<text x="{xx}" y="{height - 18}" class="label" text-anchor="middle">{lab}</text>')
    label_y = {}
    for name, a, b, col, dash, sw in SLOPES:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p.append(f'<line x1="{X0}" y1="{yv(a)}" x2="{X1}" y2="{yv(b)}" stroke="{col}" stroke-width="{sw}"'
                 f' stroke-linecap="round"{d}/>')
        for xx, v in ((X0, a), (X1, b)):
            p.append(f'<circle cx="{xx}" cy="{yv(v)}" r="7" fill="{col}" stroke="{PAPER}" stroke-width="2"/>')
        label_y[name] = yv(b)
    # right-end labels, nudged apart so the two physiology lines do not collide
    ys = sorted(label_y.items(), key=lambda kv: kv[1])
    placed = []
    for name, yy in ys:
        if placed and yy - placed[-1] < 40:
            yy = placed[-1] + 40
        placed.append(yy)
        _, a, b, col, _, _ = next(s for s in SLOPES if s[0] == name)
        weight = "600" if col == INK else "400"
        p.append(f'<text x="{X1 + 20}" y="{yy - 2}" class="value" fill="{col}" font-weight="{weight}">{b:.3f}</text>'
                 f'<text x="{X1 + 20}" y="{yy + 18}" class="small">{html.escape(name)}</text>')
    prev = None
    for name, a, b, col, _, _ in sorted(SLOPES, key=lambda r: -r[1]):
        yy = yv(a) + 5
        if prev is not None and yy - prev < 19:
            yy = prev + 19
        prev = yy
        p.append(f'<text x="{X0 - 16}" y="{yy:.1f}" class="small" text-anchor="end" fill="{col}">{a:.3f}</text>')
    return f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">{"".join(p)}</svg>'


def svg_coupling(width=640, height=340):
    f = pd.read_csv(ROOT / "results" / "m2" / "features_60s.csv")
    c = "g_rw0.02_chest_hr_ecg~act_chest"
    m = f[f.label.isin([1, 2])].groupby(["subject", "label"])[c].mean().unstack()
    n_down = int((m[2] < m[1]).sum())
    p = []
    # node diagram: five chest signals, the heart-rate~movement edge highlighted
    names = ["Heart rate", "Breathing", "Skin conductance", "Muscle", "Movement"]
    cx, cy, R = 150, 175, 74
    ang = [-90, -18, 54, 126, 198]
    pos = [(cx + R * np.cos(np.radians(a)), cy + R * np.sin(np.radians(a))) for a in ang]
    for i in range(5):
        for j in range(i + 1, 5):
            hl = (i, j) == (0, 4)
            p.append(f'<line x1="{pos[i][0]:.1f}" y1="{pos[i][1]:.1f}" x2="{pos[j][0]:.1f}" y2="{pos[j][1]:.1f}" '
                     f'stroke="{"#eb6834" if hl else RULE}" stroke-width="{5 if hl else 2}"/>')
    for (px, py), nm, a in zip(pos, names, ang):
        p.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="11" fill="{INK}" stroke="{PAPER}" stroke-width="3"/>')
        dx = 18 if -90 < a < 90 else -18
        anchor = "start" if -90 < a < 90 else "end"
        dy = 5
        if a == -90:
            dx, anchor, dy = 0, "middle", -20
        if a == 198:                                   # Movement: label under the node so it stays inside
            dx, anchor, dy = 0, "middle", 34
        p.append(f'<text x="{px + dx:.1f}" y="{py + dy:.1f}" class="small" text-anchor="{anchor}">{nm}</text>')
    # per-person slope: rest -> stress for that edge
    X0, X1, top, bot = 400, 590, 44, height - 50
    lo, hi = 0.15, 0.5
    yv = lambda v: top + (bot - top) * (hi - v) / (hi - lo)  # noqa: E731
    for v in (0.2, 0.3, 0.4, 0.5):
        p.append(f'<line x1="{X0 - 20}" x2="{X1 + 20}" y1="{yv(v)}" y2="{yv(v)}" stroke="{RULE}"/>'
                 f'<text x="{X0 - 30}" y="{yv(v) + 5}" class="small" text-anchor="end">{v:.1f}</text>')
    for s, r in m.iterrows():
        p.append(f'<line x1="{X0}" y1="{yv(r[1]):.1f}" x2="{X1}" y2="{yv(r[2]):.1f}" stroke="#eb6834" '
                 f'stroke-width="1.6" opacity="0.75"/>')
        p.append(f'<circle cx="{X0}" cy="{yv(r[1]):.1f}" r="3.5" fill="#2a78d6"/>'
                 f'<circle cx="{X1}" cy="{yv(r[2]):.1f}" r="3.5" fill="#eb6834"/>')
    p.append(f'<text x="{X0}" y="{height - 18}" class="label" text-anchor="middle">Rest</text>'
             f'<text x="{X1}" y="{height - 18}" class="label" text-anchor="middle">Stress</text>'
             f'<text x="{(X0 + X1) / 2}" y="22" class="small" text-anchor="middle">'
             f'one line per person</text>')
    svg = f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">{"".join(p)}</svg>'
    return svg, n_down


def page(runs):
    timetables = svg_timetables(runs)
    slopes = svg_slopes()
    coupling, n_down = svg_coupling()
    steps = [
        ("Cut", "60 s windows inside one condition, every 5 s."),
        ("Describe", "38 physiological features and a coupling graph learned between heart rate, "
                     "breathing, skin conductance, muscle and movement."),
        ("Remove the clock", "Drop four features that drift with time in session: skin temperature and "
                             "tonic skin conductance."),
        ("Classify", "Logistic regression, p\u00a0=\u00a0σ(w·z\u00a0+\u00a0b), tuned inside the training people only."),
        ("Test on new people and a new schedule", "Leave-one-person-out, and train on one protocol order, "
                                                  "test on the other."),
    ]
    step_html = "".join(f'<li><span class="num">{i}</span><h3>{html.escape(t)}</h3><p>{html.escape(d)}</p></li>'
                        for i, (t, d) in enumerate(steps, 1))
    fit = [
        ("Works for a new trainee", "Evaluated on people never seen in training, and on a protocol order never "
                                    "seen in training, with drift checks that expose time-in-session shortcuts."),
        ("Interpretable", "A linear model on named physiological features, and body↔movement coupling as a "
                          "readable, person-relative marker of stress."),
        ("Near-real-time", "Causal graph learning updated every 5 s; a heart-rate index flags the start of "
                           "the stressor in a median 13 s."),
        ("Physiology with behaviour", "Coupling between body signals and movement, built to take more "
                                      "behavioural channels such as head, hand and gaze in VR."),
        ("Reproducible", "Every analysis committed to git before it was run, leakage tests, 25 unit tests, "
                         "one script reruns everything from the raw data."),
    ]
    fit_html = "".join(f'<tr><th>{html.escape(a)}</th><td>{html.escape(b)}</td></tr>' for a, b in fit)
    css = f"""
    @font-face {{ font-family: 'Lora'; src: local('Lora'); }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: {PAPER}; color: {INK}; font-family: 'Inter', sans-serif; width: 1600px; }}
    main {{ padding: 88px 96px 64px; }}
    h1 {{ font-family: 'Lora', serif; font-weight: 600; font-size: 64px; line-height: 1.08;
          letter-spacing: -0.01em; max-width: 1180px; }}
    .stand {{ font-size: 23px; line-height: 1.5; color: {INK2}; max-width: 1060px; margin-top: 24px; }}
    h2 {{ font-family: 'Lora', serif; font-weight: 600; font-size: 34px; line-height: 1.2; margin-bottom: 8px; }}
    .lead {{ font-size: 19px; line-height: 1.5; color: {INK2}; max-width: 900px; margin-bottom: 26px; }}
    section {{ margin-top: 76px; }}
    .rule {{ border-top: 2px solid {INK}; padding-top: 30px; }}
    svg text {{ font-family: 'Inter', sans-serif; fill: {INK2}; }}
    svg .small {{ font-size: 15px; }}
    svg .label {{ font-size: 16px; font-weight: 600; fill: {INK}; }}
    svg .value {{ font-size: 22px; font-family: 'Inter Display', 'Inter', sans-serif; }}
    .callout {{ display: flex; gap: 40px; align-items: baseline; margin-top: 10px; padding: 22px 28px;
                background: #fff; border-left: 6px solid #eb6834; border-radius: 0 10px 10px 0; }}
    .callout b {{ font-family: 'Inter Display', 'Inter', sans-serif; font-size: 48px; font-weight: 700;
                  font-variant-numeric: tabular-nums; }}
    .callout span {{ font-size: 20px; line-height: 1.45; color: {INK2}; max-width: 900px; }}
    .two {{ display: grid; grid-template-columns: 760px 1fr; gap: 72px; align-items: start; }}
    .stats {{ list-style: none; display: grid; gap: 34px; padding-top: 30px; }}
    .stats b {{ display: block; font-family: 'Inter Display', 'Inter', sans-serif; font-size: 58px;
                font-weight: 700; line-height: 1; font-variant-numeric: tabular-nums; }}
    .stats span {{ display: block; font-size: 18px; line-height: 1.45; color: {INK2}; margin-top: 8px;
                   max-width: 470px; }}
    .note {{ font-size: 15px; color: {MUTED}; margin-top: 6px; }}
    ol.steps {{ list-style: none; display: grid; grid-template-columns: repeat(5, 1fr); gap: 0;
                counter-reset: s; position: relative; }}
    ol.steps li {{ padding: 0 26px 0 0; position: relative; }}
    ol.steps li + li {{ padding-left: 26px; border-left: 1px solid {RULE}; }}
    ol.steps .num {{ display: inline-flex; width: 40px; height: 40px; border-radius: 50%; background: {INK};
                     color: #fff; font-weight: 700; font-size: 19px; align-items: center; justify-content: center; }}
    ol.steps li:nth-child(3) .num {{ background: #eb6834; }}
    ol.steps h3 {{ font-size: 21px; line-height: 1.25; margin: 16px 0 8px; font-weight: 650; }}
    ol.steps p {{ font-size: 17px; line-height: 1.5; color: {INK2}; }}
    .fit {{ display: grid; grid-template-columns: 640px 1fr; gap: 72px; align-items: start; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ text-align: left; vertical-align: top; padding: 15px 0; border-top: 1px solid {RULE};
              font-size: 17.5px; line-height: 1.45; }}
    th {{ width: 230px; padding-right: 24px; font-weight: 650; color: {INK}; }}
    td {{ color: {INK2}; }}
    tr:first-child th, tr:first-child td {{ border-top: 2px solid {INK}; }}
    footer {{ margin-top: 72px; padding-top: 22px; border-top: 1px solid {RULE}; display: flex;
              justify-content: space-between; gap: 40px; font-size: 16px; color: {INK2}; line-height: 1.5; }}
    footer b {{ color: {INK}; }}
    """
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Stress detection that still works when the schedule changes</title><style>{css}</style></head>
<body><main>
<h1>Stress detection that still works when the schedule changes</h1>
<p class="stand">A leakage-checked, pre-registered study of acute-stress detection from chest and wrist wearables
on the open WESAD dataset (15 people). It shows how the benchmark can be won by reading the clock, and builds a
model that reads the body instead.</p>

<section class="rule">
<h2>Fifteen sessions, two timetables</h2>
<p class="lead">Every recording follows almost the same schedule, so the time since the start already tells you
which condition is running. This is the shortcut a stress model must not learn.</p>
{timetables}
<div class="callout"><b>95.7%</b><span>accuracy for stress vs non-stress from the minutes since the start alone,
under the standard leave-one-person-out test: higher than the original benchmark (93.1%).</span></div>
</section>

<section class="rule two">
<div>
<h2>Change the protocol order and the shortcut collapses</h2>
<p class="lead">Training on one protocol order and testing on the other separates reading the clock from reading
the body.</p>
{slopes}
<p class="note">Per-person macro-F1 for stress vs non-stress, mean of 15 people.
* Scored without the first two baseline minutes.</p>
</div>
<ul class="stats">
<li><b>0.943*</b><span>macro-F1 on a protocol order the model never saw, without any personal calibration</span></li>
<li><b>94.0%</b><span>leave-one-person-out accuracy, against 93.1% in the original WESAD paper, also
leave-one-person-out</span></li>
<li><b>4</b><span>features that drift with time in session; removing them is what made the model robust to a new
schedule</span></li>
</ul>
</section>

<section class="rule">
<h2>How the model reads the body</h2>
<p class="lead">Simple, interpretable steps, each fitted only on the people used for training.</p>
<ol class="steps">{step_html}</ol>
</section>

<section class="rule fit">
<div>
<h2>Under stress, the heart stops following movement</h2>
<p class="lead">A learned coupling graph between body signals shows the heart-rate↔movement link weakening under
stress in {n_down} of 15 people, and not under amusement.</p>
{coupling}
</div>
<div>
<h2>What carries over to stress estimation in VR training</h2>
<p class="lead">The study was designed around what a VR training system needs from a stress model.</p>
<table>{fit_html}</table>
</div>
</section>

<footer><div><b>Salman Almuhammad Alali</b>. Code written with an AI assistant (Claude) under the author's
direction.</div><div>Data: WESAD, Schmidt et al., ICMI 2018, scientific non-commercial use.</div></footer>
</main></body></html>"""


def render(html_path, png_path):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path="/usr/bin/google-chrome")
        pg = b.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=2)
        pg.goto(html_path.resolve().as_uri())
        pg.wait_for_timeout(300)
        pg.screenshot(path=str(png_path), full_page=True)
        b.close()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    h = OUT / "portfolio.html"
    h.write_text(page(runs_table()), encoding="utf-8")
    render(h, OUT / "portfolio.png")
    print("saved", OUT / "portfolio.png")
