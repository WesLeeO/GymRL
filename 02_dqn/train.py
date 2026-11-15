from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlkit import Snapshots, Tracker, is_atari, make, record_video, set_seed, test_return, watch

BASE = dict(kwargs={}, lr=2.5e-4, buffer=50_000, eps_decay=20_000, eps_end=0.02, target_every=1_000, start=1_000,
            batch=128, hidden=128, train_every=1, window=20, eval_episodes=20, video_steps=None,
            test_every=None, test_episodes=10)

ENVS = {
    "cartpole": dict(BASE, id="CartPole-v1", steps=150_000, solved=475),
    "acrobot": dict(BASE, id="Acrobot-v1", steps=150_000, solved=-90),
    "lunarlander": dict(BASE, id="LunarLander-v3", steps=500_000, lr=5e-4, buffer=100_000, eps_decay=60_000,
                        eps_end=0.05, start=5_000, hidden=256, window=50, solved=None, test_every=20_000),
    "pong": dict(BASE, id="PongNoFrameskip-v4", steps=3_000_000, lr=1e-4, buffer=100_000, eps_decay=1_000_000,
                 eps_end=0.01, start=50_000, batch=32, train_every=4, window=10, eval_episodes=5, video_steps=2_500,
                 solved=18),
}


class MLPQNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class CNNQNet(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.net(x.float() / 255.0)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape, obs_dtype):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.pos
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.terminated[i] = next_obs, terminated
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        t = lambda x: torch.as_tensor(x[idx], device=device)
        return t(self.obs), t(self.actions), t(self.rewards), t(self.next_obs), t(self.terminated)


def linear_schedule(start: float, end: float, duration: int, t: int) -> float:
    return max(end, start + (end - start) * t / duration)


def build(cfg, env, device):
    n_actions = env.action_space.n
    if is_atari(cfg["id"]):
        return CNNQNet(n_actions).to(device)
    return MLPQNet(env.observation_space.shape[0], n_actions, cfg["hidden"]).to(device)


def to_input(obs, cfg, device):
    x = torch.as_tensor(np.asarray(obs), device=device)
    return x if is_atari(cfg["id"]) else x.float()


def train(args):
    cfg = dict(ENVS[args.env])
    for k in ("lr", "target_every"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    set_seed(args.seed)
    atari = is_atari(cfg["id"])
    device = torch.device("mps" if atari and torch.backends.mps.is_available() else "cpu")
    env = make(cfg["id"], **cfg["kwargs"])
    obs_shape = env.observation_space.shape

    q_net = build(cfg, env, device)
    target_net = copy.deepcopy(q_net)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg["lr"])
    buffer = ReplayBuffer(cfg["buffer"], obs_shape, np.uint8 if atari else np.float32)
    gamma, batch_size = 0.99, args.batch_size or cfg["batch"]
    total_steps = args.steps or cfg["steps"]
    window = cfg["window"]

    name = f"02_{'double_' if args.double else ''}dqn_{args.env}"
    tracker = Tracker(name, live=args.live, window=window)
    snaps = Snapshots(cfg["id"], name, total_steps, args.snapshots, max_steps=cfg["video_steps"], **cfg["kwargs"])

    @torch.no_grad()
    def greedy(o):
        return int(q_net(to_input(o, cfg, device).unsqueeze(0)).argmax().item())

    best_avg, best_state = float("-inf"), copy.deepcopy(q_net.state_dict())
    obs, _ = env.reset(seed=args.seed)
    ep_return = 0.0
    for step in range(total_steps):
        snaps.maybe_record(step, greedy, tracker.writer)
        epsilon = linear_schedule(1.0, cfg["eps_end"], cfg["eps_decay"], step)

        action = env.action_space.sample() if np.random.rand() < epsilon else greedy(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        ep_return += reward
        if terminated or truncated:
            tracker.episode(ep_return, step, epsilon=epsilon)
            if not cfg["test_every"] and len(tracker.returns) >= window and tracker.recent_mean(window) > best_avg:
                best_avg, best_state = tracker.recent_mean(window), copy.deepcopy(q_net.state_dict())
            obs, _ = env.reset()
            ep_return = 0.0

        if step < cfg["start"] or step % cfg["train_every"]:
            continue

        o, a, r, o2, term = buffer.sample(batch_size, device)

        with torch.no_grad():
            if args.double:
                best_a = q_net(o2).argmax(dim=1, keepdim=True)
                next_q = target_net(o2).gather(1, best_a).squeeze(1)
            else:
                next_q = target_net(o2).max(dim=1).values
            td_target = r + gamma * (1.0 - term) * next_q

        q_sa = q_net(o).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q_sa, td_target)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
        optimizer.step()

        if step % 1000 < cfg["train_every"]:
            tracker.log(step, loss=loss.item(), q_mean=q_sa.mean().item(), grad_norm=grad_norm.item())
        if step % cfg["target_every"] < cfg["train_every"]:
            target_net.load_state_dict(q_net.state_dict())

        if cfg["test_every"] and (step + 1) % cfg["test_every"] < cfg["train_every"]:
            score = test_return(cfg["id"], greedy, cfg["test_episodes"], **cfg["kwargs"])
            tracker.log(step, test_return=score)
            print(f"step {step + 1}: test return {score:.1f} over {cfg['test_episodes']} episodes")
            if score > best_avg:
                best_avg, best_state = score, copy.deepcopy(q_net.state_dict())

        if cfg["solved"] is not None and len(tracker.returns) >= window and tracker.recent_mean(window) >= cfg["solved"]:
            print(f"Solved at step {step}: avg{window} return {tracker.recent_mean(window):.1f}")
            break

    env.close()
    print(f"Keeping the best weights ({'test' if cfg['test_every'] else f'avg{window}'} return {best_avg:.1f})")
    q_net.load_state_dict(best_state)
    snaps.finish(step, greedy, tracker.writer)
    tracker.close()
    return q_net, cfg, device


@torch.no_grad()
def evaluate(q_net, cfg, device) -> float:
    env = make(cfg["id"], **cfg["kwargs"])
    total = 0.0
    for ep in range(cfg["eval_episodes"]):
        obs, _ = env.reset(seed=10_000 + ep)
        done = False
        while not done:
            a = int(q_net(to_input(obs, cfg, device).unsqueeze(0)).argmax().item())
            obs, r, te, tr, _ = env.step(a)
            total += r
            done = te or tr
    env.close()
    return total / cfg["eval_episodes"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="cartpole", choices=list(ENVS))
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--target-every", type=int, default=None)
    p.add_argument("--double", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--live", action="store_true", help="live plot window while training")
    p.add_argument("--snapshots", type=int, default=0, help="record N videos spread over training")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--video", action="store_true")
    args = p.parse_args()

    q_net, cfg, device = train(args)
    print(f"Greedy policy, {cfg['eval_episodes']} eval episodes: avg return {evaluate(q_net, cfg, device):.1f}")
    torch.save(q_net.state_dict(), f"02_dqn/{args.env}.pt")

    policy = lambda obs: int(q_net(to_input(obs, cfg, device).unsqueeze(0)).argmax().item())
    with torch.no_grad():
        if args.watch:
            watch(cfg["id"], policy, **cfg["kwargs"])
        if args.video:
            record_video(cfg["id"], policy, f"02_dqn_{args.env}", max_steps=cfg["video_steps"], **cfg["kwargs"])
