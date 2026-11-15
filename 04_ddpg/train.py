from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlkit import Snapshots, Tracker, make, record_video, set_seed, watch

BASE = dict(kwargs={}, gamma=0.99, start=2_000, window=10, eval_episodes=10)

ENVS = {
    "pendulum": dict(BASE, id="Pendulum-v1", steps=30_000),
}


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, max_action, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, act_dim), nn.Tanh())
        self.register_buffer("max_action", torch.as_tensor(max_action, dtype=torch.float32))

    def forward(self, obs):
        return self.net(obs) * self.max_action


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity, self.pos, self.size = capacity, 0, 0
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros((capacity, act_dim), np.float32)
        self.rew = np.zeros(capacity, np.float32)
        self.term = np.zeros(capacity, np.float32)

    def add(self, o, a, r, o2, term):
        i = self.pos
        self.obs[i], self.act[i], self.rew[i], self.next_obs[i], self.term[i] = o, a, r, o2, term
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return tuple(torch.as_tensor(x[idx]) for x in (self.obs, self.act, self.rew, self.next_obs, self.term))


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.mul_(1 - tau).add_(tau * sp)


def train(args):
    cfg = ENVS[args.env]
    set_seed(args.seed)
    env = make(cfg["id"], **cfg["kwargs"])
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    low, high = env.action_space.low, env.action_space.high
    max_action = high

    actor = Actor(obs_dim, act_dim, max_action)
    critic = Critic(obs_dim, act_dim)
    actor_target, critic_target = copy.deepcopy(actor), copy.deepcopy(critic)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    buffer = ReplayBuffer(200_000, obs_dim, act_dim)

    gamma, tau, batch_size = cfg["gamma"], 0.005, 256
    noise_std = args.noise * max_action
    start_steps = cfg["start"]
    total_steps = args.steps or cfg["steps"]

    name = f"04_ddpg_{args.env}"
    tracker = Tracker(name, live=args.live, window=cfg["window"])
    snaps = Snapshots(cfg["id"], name, total_steps, args.snapshots, **cfg["kwargs"])
    deterministic = lambda o: actor(torch.as_tensor(o, dtype=torch.float32)).numpy()

    obs, _ = env.reset(seed=args.seed)
    ep_return = 0.0
    for step in range(total_steps):
        with torch.no_grad():
            snaps.maybe_record(step, deterministic, tracker.writer)

        if step < start_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = deterministic(obs)
            action = np.clip(action + np.random.normal(0, noise_std, size=act_dim), low, high).astype(np.float32)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        ep_return += reward
        if terminated or truncated:
            tracker.episode(ep_return, step)
            obs, _ = env.reset()
            ep_return = 0.0

        if step < start_steps:
            continue

        o, a, r, o2, term = buffer.sample(batch_size)

        with torch.no_grad():
            y = r + gamma * (1 - term) * critic_target(o2, actor_target(o2))
        q = critic(o, a)
        critic_loss = F.mse_loss(q, y)
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        actor_loss = -critic(o, actor(o)).mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        soft_update(critic_target, critic, tau)
        soft_update(actor_target, actor, tau)
        if step % 200 == 0:
            tracker.log(step, critic_loss=critic_loss.item(), actor_loss=actor_loss.item(), q_mean=q.mean().item())

    env.close()
    with torch.no_grad():
        snaps.finish(step, deterministic, tracker.writer)
    tracker.close()
    return actor, cfg


@torch.no_grad()
def evaluate(actor, cfg) -> float:
    episodes = cfg["eval_episodes"]
    env = make(cfg["id"], **cfg["kwargs"])
    total = 0.0
    for ep in range(episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        done = False
        while not done:
            obs, r, te, tr, _ = env.step(actor(torch.as_tensor(obs, dtype=torch.float32)).numpy())
            total += r
            done = te or tr
    env.close()
    return total / episodes


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="pendulum", choices=list(ENVS))
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--live", action="store_true", help="live plot window while training")
    p.add_argument("--snapshots", type=int, default=0, help="record N videos spread over training")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--video", action="store_true")
    args = p.parse_args()

    actor, cfg = train(args)
    print(f"Deterministic policy, {cfg['eval_episodes']} eval episodes: avg return {evaluate(actor, cfg):.1f}")
    torch.save(actor.state_dict(), f"04_ddpg/{args.env}.pt")

    policy = lambda obs: actor(torch.as_tensor(obs, dtype=torch.float32)).detach().numpy()
    if args.watch:
        watch(cfg["id"], policy, **cfg["kwargs"])
    if args.video:
        record_video(cfg["id"], policy, f"04_ddpg_{args.env}", **cfg["kwargs"])
