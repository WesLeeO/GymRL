from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont

from rlkit.utils import ROOT, RUNS, moving_average

VIDEOS = ROOT / "videos"
ASSETS = ROOT / "assets"

GAMES = {
    "cartpole": ("CartPole", [("01_q_learning", "Q-learning"), ("02_dqn", "DQN"), ("03_ppo", "PPO")]),
    "acrobot": ("Acrobot", [("02_dqn", "DQN"), ("03_ppo", "PPO")]),
    "pendulum": ("Pendulum", [("04_ddpg", "DDPG")]),
    "lunarlander": ("LunarLander", [("02_dqn", "DQN"), ("03_ppo", "PPO")]),
    "pong": ("Pong", [("02_dqn", "DQN"), ("03_ppo", "PPO")]),
}


def _font(size: int):
    for name in ("/System/Library/Fonts/Helvetica.ttc", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def episode_returns(run: str) -> np.ndarray | None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    logs = sorted((RUNS / "tb").glob(f"{run}_*"))
    if not logs:
        return None
    acc = EventAccumulator(str(logs[-1]), size_guidance={"scalars": 0})
    acc.Reload()
    if "episode/return" not in acc.Tags()["scalars"]:
        return None
    return np.array([e.value for e in acc.Scalars("episode/return")])


def reward_plot(game: str) -> Path | None:
    title, algos = GAMES[game]
    data = [(label, episode_returns(f"{lesson}_{game}")) for lesson, label in algos]
    data = [(label, r) for label, r in data if r is not None]
    if not data:
        return None
    fig = Figure(figsize=(4.2 * len(data), 3.2))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, len(data), squeeze=False)[0]
    for ax, (label, r) in zip(axes, data):
        window = 500 if len(r) >= 5000 else 50 if len(r) >= 250 else 10
        ax.plot(np.arange(1, len(r) + 1), r, color="#9ab8e6", lw=0.6, label="each episode")
        ax.plot(np.arange(1, len(r) + 1), moving_average(r, window), color="#1f4e9c", lw=2,
                label=f"mean of last {window} episodes")
        ax.set_title(f"{label} on {title}")
        ax.set_xlabel("episode")
        ax.set_ylabel("return")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    ASSETS.mkdir(exist_ok=True)
    out = ASSETS / f"rewards_{game}.png"
    fig.savefig(out, dpi=110)
    return out


GIF_OPTS = {
    "pong": dict(width=200, step=8, speed=2.0, colors=32, max_frames=1500, label_bottom=True),
}


def trained_gif(run: str, label: str, width: int = 240, step: int | None = None, max_frames: int = 400,
                speed: float = 1.0, colors: int = 64, label_bottom: bool = False) -> Path | None:
    folder = VIDEOS / run
    if not folder.exists():
        return None
    stages = sorted((p for p in folder.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    mp4s = [m for s in stages for m in sorted(s.glob("*.mp4"))[:1]]
    if not mp4s:
        return None
    from itertools import islice

    fps = float(iio.immeta(mp4s[-1]).get("fps") or 30)
    if step is None:
        step = 4 if fps > 40 else 2
    def render(f):
        img = Image.fromarray(f)
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
        draw = ImageDraw.Draw(img)
        pos = (8, img.height - 24) if label_bottom else (8, 6)
        box = draw.textbbox(pos, label, font=_font(16))
        draw.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(255, 255, 255))
        draw.text(pos, label, fill=(30, 30, 30), font=_font(16))
        return img.quantize(colors=colors, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)

    frames, last, last_kept = [], None, False
    for i, f in enumerate(islice(iio.imiter(mp4s[-1]), 0, max_frames * step)):
        last, last_kept = f, i % step == 0
        if last_kept:
            frames.append(render(f))
    if last is not None and not last_kept:
        frames.append(render(last))
    frame_ms = max(30, int(1000 * step / fps / speed))
    durations = [frame_ms] * (len(frames) - 1) + [1500]
    ASSETS.mkdir(exist_ok=True)
    out = ASSETS / f"{run}.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False)
    return out


def build_media() -> list[Path]:
    made = []
    for game, (_, algos) in GAMES.items():
        made.append(reward_plot(game))
        for lesson, label in algos:
            made.append(trained_gif(f"{lesson}_{game}", label, **GIF_OPTS.get(game, {})))
    return [p for p in made if p]


if __name__ == "__main__":
    for p in build_media():
        print(f"{p.relative_to(ROOT)}  {p.stat().st_size / 1e6:.2f} MB")
