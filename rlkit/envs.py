from __future__ import annotations

from pathlib import Path

import gymnasium as gym


def is_atari(env_id: str) -> bool:
    return "NoFrameskip" in env_id or env_id.startswith("ALE/")


def make(env_id: str, render_mode: str | None = None, record_to: Path | None = None, **kwargs) -> gym.Env:
    if is_atari(env_id):
        import ale_py

        gym.register_envs(ale_py)
    env = gym.make(env_id, render_mode=render_mode, **kwargs)
    if record_to is not None:
        fps = 60 if is_atari(env_id) else None
        env = gym.wrappers.RecordVideo(env, str(record_to), episode_trigger=lambda _: True, disable_logger=True, fps=fps)
    if is_atari(env_id):
        env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True, noop_max=30)
        env = gym.wrappers.FrameStackObservation(env, 4)
    return env


def test_return(env_id: str, policy, episodes: int = 10, seed: int = 20_000, **kwargs) -> float:
    env = make(env_id, **kwargs)
    total = 0.0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            obs, r, terminated, truncated, _ = env.step(policy(obs))
            total += r
            done = terminated or truncated
    env.close()
    return total / episodes
