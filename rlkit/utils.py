from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def moving_average(x, window: int = 100) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = x[max(0, i - window + 1) : i + 1].mean()
    return out


def watch(env_id: str, policy: Callable[[np.ndarray], object], episodes: int = 3, **make_kwargs) -> list[float]:
    from rlkit.envs import make

    env = make(env_id, render_mode="human", **make_kwargs)
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            obs, r, terminated, truncated, _ = env.step(policy(obs))
            total += r
            done = terminated or truncated
        print(f"episode {ep}: return {total:.1f}")
        returns.append(total)
    env.close()
    return returns


def record_video(env_id: str, policy: Callable[[np.ndarray], object], name: str, episodes: int = 1,
                 max_steps: int | None = None, **make_kwargs) -> Path:
    from rlkit.envs import make

    folder = ROOT / "videos" / name
    env = make(env_id, render_mode="rgb_array", record_to=folder, **make_kwargs)
    for _ in range(episodes):
        obs, _ = env.reset()
        done, t = False, 0
        while not done:
            obs, _, terminated, truncated, _ = env.step(policy(obs))
            t += 1
            done = terminated or truncated or (max_steps is not None and t >= max_steps)
    env.close()
    print(f"Saved video -> {folder.relative_to(ROOT)}")
    return folder
