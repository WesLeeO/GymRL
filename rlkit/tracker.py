from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from rlkit.utils import ROOT, RUNS, moving_average, record_video


class Tracker:
    def __init__(
        self,
        name: str,
        live: bool = False,
        window: int = 100,
        print_every: float = 5.0,
        live_every: float = 1.0,
        tensorboard: bool = True,
    ):
        self.name = name
        self.window = window
        self.print_every = print_every
        self.live_every = live_every
        self.returns: list[float] = []
        self.return_steps: list[int] = []
        self.metrics: dict[str, list[tuple[int, float]]] = {}
        self._start = self._last_print = self._last_draw = time.time()
        self._last_metrics: dict[str, float] = {}

        self.writer = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = RUNS / "tb" / f"{name}_{datetime.now():%m%d-%H%M%S}"
            self.writer = SummaryWriter(str(log_dir))
            print(f"TensorBoard logs -> {log_dir.relative_to(ROOT)}  (view: uv run tensorboard --logdir runs/tb)")

        self.live = live
        if live:
            import matplotlib.pyplot as plt

            plt.ion()
            self.plt = plt
            self.fig, (self.ax_ret, self.ax_met) = plt.subplots(2, 1, figsize=(9, 7))
            self.fig.canvas.manager.set_window_title(f"Training: {name}")
            self.fig.tight_layout(pad=3)
            plt.show(block=False)

    def log(self, step: int, **metrics: float) -> None:
        for k, v in metrics.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            v = float(v)
            self._last_metrics[k] = v
            self.metrics.setdefault(k, []).append((step, v))
            if self.writer:
                self.writer.add_scalar(f"train/{k}", v, step)

    def episode(self, ep_return: float, step: int, **metrics: float) -> None:
        self.returns.append(float(ep_return))
        self.return_steps.append(step)
        if self.writer:
            self.writer.add_scalar("episode/return", ep_return, step)
            self.writer.add_scalar(f"episode/return_avg{self.window}", self.recent_mean(), step)
        if metrics:
            self.log(step, **metrics)

        now = time.time()
        if now - self._last_print >= self.print_every:
            self._last_print = now
            parts = [f"ep {len(self.returns):5d}", f"step {step:8d}", f"return {ep_return:8.1f}", f"avg{self.window} {self.recent_mean():8.1f}"]
            parts += [f"{k} {v:.3g}" for k, v in self._last_metrics.items()]
            parts.append(f"{now - self._start:5.0f}s")
            print(" | ".join(parts))
        if self.live and now - self._last_draw >= self.live_every:
            self._last_draw = now
            self.draw()

    def recent_mean(self, n: int | None = None) -> float:
        n = n or self.window
        return float(np.mean(self.returns[-n:])) if self.returns else float("-inf")

    def _plot(self, ax_ret, ax_met) -> None:
        ax_ret.clear()
        ax_ret.plot(self.return_steps, self.returns, alpha=0.3, label="episode return")
        ax_ret.plot(self.return_steps, moving_average(self.returns, self.window), label=f"moving avg ({self.window})")
        ax_ret.set_xlabel("environment step")
        ax_ret.set_ylabel("return")
        ax_ret.set_title(self.name)
        ax_ret.grid(alpha=0.3)
        ax_ret.legend(loc="upper left")

        ax_met.clear()
        for k, series in self.metrics.items():
            steps, vals = zip(*series)
            vals = np.asarray(vals)
            span = np.ptp(vals) or 1.0
            ax_met.plot(steps, (vals - vals.min()) / span, label=f"{k} [{vals.min():.3g} … {vals.max():.3g}]")
        ax_met.set_xlabel("environment step")
        ax_met.set_ylabel("scaled to 0–1")
        ax_met.set_title("training signals")
        ax_met.grid(alpha=0.3)
        if self.metrics:
            ax_met.legend(loc="upper left", fontsize=8)

    def draw(self) -> None:
        self._plot(self.ax_ret, self.ax_met)
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def close(self) -> Path:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        RUNS.mkdir(exist_ok=True)
        path = RUNS / f"{self.name}.png"
        fig = Figure(figsize=(9, 7))
        FigureCanvasAgg(fig)
        ax1, ax2 = fig.subplots(2, 1)
        self._plot(ax1, ax2)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        print(f"Saved learning curves -> {path.relative_to(ROOT)}")
        if self.writer:
            self.writer.close()
        if self.live:
            self.draw()
            print("Close the plot window to finish.")
            self.plt.ioff()
            self.plt.show()
        return path


class Snapshots:
    def __init__(self, env_id: str, name: str, total: int, count: int, max_steps: int | None = None, **make_kwargs):
        self.env_id, self.name, self.make_kwargs, self.max_steps = env_id, name, make_kwargs, max_steps
        self.points = sorted({int(total * i / max(count - 1, 1)) for i in range(count)}) if count > 0 else []
        self.points = [min(p, total - 1) for p in self.points]

    def maybe_record(self, progress: int, policy: Callable, writer=None) -> None:
        while self.points and progress >= self.points[0]:
            at = self.points.pop(0)
            folder = record_video(self.env_id, policy, f"{self.name}/{at:08d}", max_steps=self.max_steps, **self.make_kwargs)
            if writer is not None:
                self._to_tensorboard(folder, writer, at)

    def finish(self, progress: int, policy: Callable, writer=None) -> None:
        if self.points:
            self.points = [progress]
            self.maybe_record(progress, policy, writer)
            from rlkit.gallery import build_gallery

            print(f"Updated video gallery -> {build_gallery().relative_to(ROOT)}")

    @staticmethod
    def _to_tensorboard(folder: Path, writer, step: int) -> None:
        try:
            import imageio.v3 as iio
            import torch

            from itertools import islice

            import numpy as np

            mp4 = next(folder.glob("*.mp4"))
            frames = np.stack(list(islice(iio.imiter(mp4), 0, 1200, 2)))
            video = torch.as_tensor(frames).permute(0, 3, 1, 2).unsqueeze(0)
            writer.add_video("agent/playing", video, step, fps=15)
        except Exception as e:
            print(f"(could not add video to TensorBoard: {e})")
