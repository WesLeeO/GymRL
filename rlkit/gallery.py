from __future__ import annotations

import html
import re
from pathlib import Path

from rlkit.utils import ROOT, RUNS

VIDEOS = ROOT / "videos"

GAMES = [
    ("cartpole", "CartPole", "Push the cart left or right to keep the pole upright. +1 per step, 500 max."),
    ("acrobot", "Acrobot", "Swing the two-link arm above the line. −1 per step until it gets there."),
    ("pendulum", "Pendulum", "Apply torque to swing the pendulum up and hold it upright. Closer to 0 is better."),
    ("lunarlander", "LunarLander", "Land the spacecraft between the flags. Solved at 200."),
    ("pong", "Pong", "Atari Pong from raw pixels, first to 21 points. Return is your points minus the opponent's."),
]

LESSONS = {
    "01_q_learning": "Q-learning",
    "02_dqn": "DQN",
    "02_double_dqn": "Double DQN",
    "03_ppo": "PPO",
    "04_ddpg": "DDPG",
}


def _runs() -> dict[str, list[tuple[str, Path]]]:
    found: dict[str, list[tuple[str, Path]]] = {}
    if not VIDEOS.exists():
        return found
    for folder in sorted(p for p in VIDEOS.iterdir() if p.is_dir()):
        m = re.match(r"(\d\d_[a-z_]+?)_([a-z0-9-]+)$", folder.name)
        if not m or m.group(1) not in LESSONS:
            continue
        stages = []
        for sub in sorted(p for p in folder.iterdir() if p.is_dir() and p.name.isdigit()):
            mp4 = next(iter(sorted(sub.glob("*.mp4"))), None)
            if mp4:
                stages.append((sub.name, mp4))
        loose = sorted(folder.glob("*.mp4"))
        if loose:
            stages.append(("final", loose[-1]))
        if stages:
            found.setdefault(m.group(2), []).append((folder.name, stages))
    return found


def _stage_label(name: str, last: int, lesson: str) -> str:
    if name == "final":
        return "after training (--video)"
    n = int(name)
    unit = "episode" if lesson == "01_q_learning" else "step"
    pct = f" · {round(100 * n / last)}%" if last else ""
    return f"{unit} {n:,}{pct}"


def build_gallery() -> Path:
    runs = _runs()
    order = [g for g, _, _ in GAMES]
    titles = {g: (t, d) for g, t, d in GAMES}
    sections = []
    for game in order:
        if game not in runs:
            continue
        title, desc = titles.get(game, (game, ""))
        rows = []
        for folder, stages in runs[game]:
            lesson = re.match(r"(\d\d_[a-z_]+?)_[a-z0-9-]+$", folder).group(1)
            numeric = [int(s) for s, _ in stages if s.isdigit()]
            last = max(numeric) if numeric else 0
            cards = "".join(
                f'<figure><video src="{html.escape(mp4.relative_to(VIDEOS).as_posix())}" muted loop autoplay playsinline controls></video>'
                f"<figcaption>{html.escape(_stage_label(s, last, lesson))}</figcaption></figure>"
                for s, mp4 in stages
            )
            curve = RUNS / f"{folder}.png"
            curve_html = (
                f'<details><summary>Learning curves</summary><img src="../runs/{curve.name}" alt="Learning curves for {html.escape(folder)}"></details>'
                if curve.exists() else ""
            )
            rows.append(
                f'<div class="run"><h3>{html.escape(LESSONS[lesson])} <span>{html.escape(folder)}</span></h3>'
                f'<div class="strip">{cards}</div>{curve_html}</div>'
            )
        sections.append(
            f'<section><h2>{html.escape(title)}</h2><p class="desc">{html.escape(desc)}</p>{"".join(rows)}</section>'
        )

    body = "".join(sections) or '<p class="desc">No videos yet. Train with <code>--snapshots 5</code>.</p>'
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Progress</title>
<style>
:root {{ --bg:#f7f7f5; --fg:#1d1d1b; --muted:#6b6b66; --card:#ffffff; --line:#e3e3de; --accent:#2f6fde; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161615; --fg:#ecece8; --muted:#9a9a93; --card:#21211f; --line:#33332f; --accent:#7aa7ff; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
main {{ max-width:1200px; margin:0 auto; padding:32px 16px 64px; }}
h1 {{ font-size:26px; margin:0 0 4px; }}
.lead {{ color:var(--muted); margin:0 0 32px; }}
section {{ margin-bottom:48px; }}
h2 {{ font-size:20px; margin:0; padding-bottom:4px; border-bottom:1px solid var(--line); }}
.desc {{ color:var(--muted); margin:6px 0 16px; }}
.run {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; margin-bottom:16px; }}
h3 {{ font-size:16px; margin:0 0 10px; }}
h3 span {{ color:var(--muted); font-weight:400; font-size:13px; margin-left:6px; }}
.strip {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(220px, 1fr)); gap:12px; }}
figure {{ margin:0; }}
video {{ width:100%; border-radius:6px; background:#000; display:block; }}
figcaption {{ font-size:13px; color:var(--muted); margin-top:4px; font-variant-numeric:tabular-nums; }}
details {{ margin-top:12px; }}
summary {{ cursor:pointer; color:var(--accent); font-size:14px; }}
details img {{ width:100%; max-width:900px; margin-top:8px; border-radius:6px; background:#fff; }}
code {{ font-family:ui-monospace, Menlo, monospace; font-size:13px; }}
</style>
</head>
<body>
<main>
<h1>Agent progress</h1>
<p class="lead">Each row is one training run, filmed at evenly spaced points from left (untrained) to right (end of training). Regenerate with <code>uv run python -m rlkit.gallery</code>.</p>
{body}
</main>
</body>
</html>
"""
    VIDEOS.mkdir(exist_ok=True)
    out = VIDEOS / "index.html"
    out.write_text(page)
    return out


if __name__ == "__main__":
    path = build_gallery()
    print(f"Gallery -> {path.relative_to(ROOT)}  (open: open {path.relative_to(ROOT)})")
