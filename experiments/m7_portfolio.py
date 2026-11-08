"""One-page visual summary of the project for the README (figures/report/portfolio.png).

Panels are drawn by experiments/m7_visuals.py from the repository's real data and results.
Run: python experiments/m7_portfolio.py   (needs playwright: `playwright install chromium`, or set CHROME_PATH)
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from m7_visuals import INK, INK2, MUTED, PAPER, RULE, build_all, launch_chromium  # noqa: E402

OUT = ROOT / "figures" / "report"


def page(v):
    css = f"""
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: {PAPER}; color: {INK}; font-family: 'Inter', sans-serif; width: 1600px; }}
    main {{ padding: 84px 96px 60px; }}
    h1 {{ font-family: 'Lora', serif; font-weight: 600; font-size: 66px; line-height: 1.07; max-width: 1200px; }}
    .stand {{ font-size: 24px; line-height: 1.5; color: {INK2}; max-width: 1120px; margin-top: 26px; }}
    section {{ margin-top: 72px; border-top: 2px solid {INK}; padding-top: 30px; }}
    h2 {{ font-family: 'Lora', serif; font-weight: 600; font-size: 38px; line-height: 1.2; }}
    .lead {{ font-size: 20px; line-height: 1.55; color: {INK2}; max-width: 1060px; margin: 10px 0 28px; }}
    .two {{ display: grid; grid-template-columns: 520px 760px; gap: 128px; align-items: start; }}
    .bench {{ display: grid; grid-template-columns: 640px 1fr; gap: 72px; align-items: center; margin-top: 30px; }}
    .bench p {{ font-size: 20px; line-height: 1.55; color: {INK2}; }}
    .bench p b {{ color: {INK}; }}
    .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0; margin-top: 40px;
              border-top: 1px solid {RULE}; }}
    .stats div {{ padding: 26px 30px 0 0; }}
    .stats div + div {{ padding-left: 30px; border-left: 1px solid {RULE}; }}
    .stats b {{ display: block; font-family: 'Inter Display','Inter',sans-serif; font-size: 60px; font-weight: 700;
                line-height: 1; font-variant-numeric: tabular-nums; }}
    .stats span {{ display: block; font-size: 18px; line-height: 1.45; color: {INK2}; margin-top: 10px; }}
    .note {{ font-size: 15px; color: {MUTED}; margin-top: 10px; }}
    footer {{ margin-top: 64px; padding-top: 22px; border-top: 1px solid {RULE}; display: flex;
              justify-content: space-between; gap: 40px; font-size: 16px; color: {INK2}; line-height: 1.5; }}
    footer b {{ color: {INK}; }}
    """
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Stress detection that still works when the schedule changes</title><style>{css}</style></head>
<body><main>
<h1>Stress detection that still works when the schedule changes</h1>
<p class="stand">Can a wearable recognise acute stress in a person it has never seen, without being fooled by the
timetable of the experiment? On the open WESAD dataset (15 people, chest and wrist sensors), this study finds a
hidden shortcut in the standard benchmark, removes it, and keeps a simple model that reads the body.</p>

<section>
<h2>What stress looks like in the body</h2>
<p class="lead">One typical participant's whole session. During the stress test (a public speech and mental
arithmetic), heart rate and sweating rise. Skin conductance also creeps up all session long: a slow drift that
quietly tracks the time.</p>
{v['session']}
</section>

<section>
<h2>The benchmark has a hidden shortcut</h2>
<p class="lead">The study ran in two orders of events: 8 people did amusement before the stress test, 7 did it
after. Within each order, all sessions follow almost the same timetable, so the time since the start already
reveals which condition is running.</p>
{v['sequences']}
<div style="height:34px"></div>
{v['timetables']}
<div class="bench">{v['bench']}
<p>A model that reads <b>only the clock</b> beats the published benchmark. High scores on this dataset can
therefore mean "knows the schedule" rather than "recognises stress".</p></div>
</section>

<section>
<h2>A fairer test removes it</h2>
<p class="lead">Train on people who did the conditions in one order, test on people who did them in the other
order. Reading the clock no longer works; reading the body still does.</p>
<div class="two">{v['split']}<div>{v['slopes']}
<p class="note">Per-person macro-F1 for stress vs non-stress, mean of 15 people.
* Scored without the first two baseline minutes.</p></div></div>
<div class="stats">
<div><b>0.943*</b><span>macro-F1 on new people and a schedule never seen in training, with no personal
calibration</span></div>
<div><b>94.0%</b><span>accuracy on new people, against 93.1% in the original WESAD paper, same
leave-one-person-out test</span></div>
<div><b>{v['n_down']} of 15</b><span>people in whom heart rate decouples from movement under stress</span></div>
</div>
</section>

<section>
<h2>The model in five simple steps</h2>
<p class="lead">Every step is fitted only on the people used for training, so nothing about the test person
leaks in.</p>
{v['steps']}
</section>

<section>
<h2>Under stress, the heart stops following movement</h2>
<p class="lead">Each line between two signals shows how strongly they move together, learned with a random-walk
graph method. At rest, heart rate tracks movement; during the stress test that link weakens, in every one of the
15 people, and not during amusement.</p>
{v['coupling']}
</section>

<section>
<h2>Why it matters for stress estimation in VR training</h2>
<p class="lead">A VR training system needs a stress model that works on a new trainee, explains itself, reacts in
time, and combines body and behaviour. The study was built around those needs.</p>
{v['tiles']}
</section>

<footer><div><b>Salman Almuhammad Alali</b>. Code written with an AI assistant (Claude) under the author's
direction.</div><div>Data: WESAD, Schmidt et al., ICMI 2018, scientific non-commercial use.</div></footer>
</main></body></html>"""


def render(html_path, png_path, width=1600):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = launch_chromium(pw)
        pg = b.new_page(viewport={"width": width, "height": 1000}, device_scale_factor=2)
        pg.goto(html_path.resolve().as_uri())
        pg.wait_for_timeout(300)
        pg.screenshot(path=str(png_path), full_page=True)
        b.close()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    h = OUT / "portfolio.html"
    h.write_text(page(build_all()), encoding="utf-8")
    render(h, OUT / "portfolio.png")
    from PIL import Image
    im = Image.open(OUT / "portfolio.png").convert("RGB")
    im.save(OUT / "portfolio.png", optimize=True)
    print("saved", OUT / "portfolio.png", im.size)
