from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

from rlkit import Snapshots, Tracker, record_video, set_seed, watch


class Discretizer:
    def __init__(self, low: np.ndarray, high: np.ndarray, bins: int):
        self.edges = [np.linspace(l, h, bins) for l, h in zip(low, high)]
        self.base = bins + 1
        self.n_states = self.base ** len(self.edges)

    def __call__(self, obs: np.ndarray) -> int:
        idx = 0
        for value, edges in zip(obs, self.edges):
            idx = idx * self.base + int(np.digitize(value, edges))
        return idx


ENVS = {
    "cartpole": dict(id="CartPole-v1", episodes=30_000, alpha=0.1, gamma=0.99, solved=400,
                     low=[-2.4, -3.0, -0.2095, -3.5], high=[2.4, 3.0, 0.2095, 3.5]),
}


def epsilon_greedy(q_row: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(len(q_row)))
    best = np.flatnonzero(q_row == q_row.max())
    return int(rng.choice(best))


def train(args):
    cfg = ENVS[args.env]
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = gym.make(cfg["id"])
    to_state = Discretizer(np.array(cfg["low"]), np.array(cfg["high"]), args.bins)
    n_states = to_state.n_states
    n_actions = env.action_space.n

    q = np.zeros((n_states, n_actions))
    alpha, gamma = cfg["alpha"], cfg["gamma"]
    episodes = cfg["episodes"]
    eps_start, eps_end = 1.0, 0.01
    eps_decay_episodes = int(0.8 * episodes)

    name = f"01_q_learning_{args.env}"
    tracker = Tracker(name, live=args.live, window=100, print_every=2.0)
    snaps = Snapshots(cfg["id"], name, episodes, args.snapshots)
    greedy = lambda obs: int(np.argmax(q[to_state(obs)]))

    step = 0
    for ep in range(episodes):
        snaps.maybe_record(ep, greedy, tracker.writer)
        epsilon = max(eps_end, eps_start - (eps_start - eps_end) * ep / eps_decay_episodes)
        obs, _ = env.reset(seed=args.seed + ep)
        s = to_state(obs)
        done, ep_return, td_errors = False, 0.0, []

        while not done:
            a = epsilon_greedy(q[s], epsilon, rng)
            obs, r, terminated, truncated, _ = env.step(a)
            s2 = to_state(obs)
            target = r if terminated else r + gamma * q[s2].max()
            td_errors.append(abs(target - q[s, a]))
            q[s, a] += alpha * (target - q[s, a])
            s = s2
            ep_return += r
            step += 1
            done = terminated or truncated

        tracker.episode(ep_return, step, epsilon=epsilon, td_error=float(np.mean(td_errors)),
                        visited=float(np.count_nonzero(q.any(axis=1))))
        if ep > 1000 and tracker.recent_mean() >= cfg["solved"] and epsilon <= eps_end + 1e-9:
            print(f"Solved at episode {ep}: avg return {tracker.recent_mean():.2f}")
            break

    snaps.finish(ep, greedy, tracker.writer)
    env.close()
    print(f"Visited {np.count_nonzero(q.any(axis=1))}/{n_states} table rows")
    tracker.close()
    return q, to_state, cfg


def evaluate(q, to_state, cfg, episodes: int = 100) -> float:
    env = gym.make(cfg["id"])
    total = 0.0
    for ep in range(episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        done = False
        while not done:
            obs, r, terminated, truncated, _ = env.step(int(np.argmax(q[to_state(obs)])))
            total += r
            done = terminated or truncated
    env.close()
    return total / episodes


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="cartpole", choices=list(ENVS))
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--live", action="store_true", help="live plot window while training")
    p.add_argument("--snapshots", type=int, default=0, help="record N videos spread over training")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--video", action="store_true")
    args = p.parse_args()

    q, to_state, cfg = train(args)
    print(f"Greedy policy, 100 eval episodes: avg return {evaluate(q, to_state, cfg):.2f}")

    policy = lambda obs: int(np.argmax(q[to_state(obs)]))
    if args.watch:
        watch(cfg["id"], policy)
    if args.video:
        record_video(cfg["id"], policy, f"01_q_learning_{args.env}")
