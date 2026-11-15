from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from rlkit import Snapshots, Tracker, is_atari, make, record_video, set_seed, watch

BASE = dict(kwargs={}, n_envs=8, n_steps=256, gamma=0.99, lam=0.95, clip=0.2, epochs=4, minibatches=4,
            vf_coef=0.5, ent_coef=0.01, lr=2.5e-4, hidden=64, window=50, eval_episodes=20, video_steps=None)

ENVS = {
    "cartpole": dict(BASE, id="CartPole-v1", steps=500_000, solved=490),
    "acrobot": dict(BASE, id="Acrobot-v1", steps=300_000, solved=-85),
    "lunarlander": dict(BASE, id="LunarLander-v3", steps=2_000_000, n_envs=16, n_steps=512, gamma=0.999, lam=0.98,
                        minibatches=32, lr=3e-4, solved=220),
    "pong": dict(BASE, id="PongNoFrameskip-v4", steps=6_000_000, n_steps=128, clip=0.1, window=10,
                 eval_episodes=5, video_steps=2_500, solved=18),
}


def layer(i, o, std=np.sqrt(2)):
    l = nn.Linear(i, o)
    nn.init.orthogonal_(l.weight, std)
    nn.init.zeros_(l.bias)
    return l


def conv(i, o, k, s):
    c = nn.Conv2d(i, o, k, stride=s)
    nn.init.orthogonal_(c.weight, np.sqrt(2))
    nn.init.zeros_(c.bias)
    return c


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.critic = nn.Sequential(layer(obs_dim, hidden), nn.Tanh(), layer(hidden, hidden), nn.Tanh(), layer(hidden, 1, std=1.0))
        self.actor = nn.Sequential(layer(obs_dim, hidden), nn.Tanh(), layer(hidden, hidden), nn.Tanh(), layer(hidden, n_actions, std=0.01))

    def value(self, x):
        return self.critic(x).squeeze(-1)

    def dist(self, x) -> Categorical:
        return Categorical(logits=self.actor(x))

    def dist_value(self, x):
        return self.dist(x), self.value(x)


class CNNActorCritic(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.body = nn.Sequential(
            conv(4, 32, 8, 4), nn.ReLU(),
            conv(32, 64, 4, 2), nn.ReLU(),
            conv(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
            layer(64 * 7 * 7, 512), nn.ReLU(),
        )
        self.actor = layer(512, n_actions, std=0.01)
        self.critic = layer(512, 1, std=1.0)

    def _features(self, x):
        return self.body(x.float() / 255.0)

    def value(self, x):
        return self.critic(self._features(x)).squeeze(-1)

    def dist(self, x) -> Categorical:
        return Categorical(logits=self.actor(self._features(x)))

    def dist_value(self, x):
        h = self._features(x)
        return Categorical(logits=self.actor(h)), self.critic(h).squeeze(-1)


def compute_gae(rewards, values, dones, last_value, last_done, gamma, lam):
    n_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(last_value)
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_nonterminal, next_value = 1.0 - last_done, last_value
        else:
            next_nonterminal, next_value = 1.0 - dones[t + 1], values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def ppo_loss(model, obs, actions, old_logprobs, advantages, returns, clip_eps, vf_coef, ent_coef):
    dist, values = model.dist_value(obs)
    new_logprobs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    ratio = (new_logprobs - old_logprobs).exp()
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = ((values - returns) ** 2).mean()
    total = policy_loss + vf_coef * value_loss - ent_coef * entropy

    with torch.no_grad():
        clipfrac = ((ratio - 1).abs() > clip_eps).float().mean().item()
        approx_kl = ((ratio - 1) - (new_logprobs - old_logprobs)).mean().item()
    return total, dict(pl=policy_loss.item(), vl=value_loss.item(), ent=entropy.item(), kl=approx_kl, clip=clipfrac)


def to_input(obs, atari, device):
    x = torch.as_tensor(np.asarray(obs), device=device)
    return x if atari else x.float()


def train(args):
    cfg = ENVS[args.env]
    set_seed(args.seed)
    atari = is_atari(cfg["id"])
    device = torch.device("mps" if atari and torch.backends.mps.is_available() else "cpu")
    n_envs, n_steps = cfg["n_envs"], cfg["n_steps"]
    gamma, lam, clip_eps, epochs, n_minibatches = cfg["gamma"], cfg["lam"], cfg["clip"], cfg["epochs"], cfg["minibatches"]
    vf_coef, ent_coef, lr = cfg["vf_coef"], cfg["ent_coef"], cfg["lr"]
    total_steps = args.steps or cfg["steps"]
    batch_size = n_envs * n_steps
    n_updates = total_steps // batch_size

    fns = [lambda: make(cfg["id"], **cfg["kwargs"]) for _ in range(n_envs)]
    vec = gym.vector.AsyncVectorEnv if atari else gym.vector.SyncVectorEnv
    envs = vec(fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    obs_shape = envs.single_observation_space.shape
    n_actions = envs.single_action_space.n
    model = (CNNActorCritic(n_actions) if atari else ActorCritic(obs_shape[0], n_actions, cfg["hidden"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    name = f"03_ppo_{args.env}"
    window = cfg["window"]
    tracker = Tracker(name, live=args.live, window=window, print_every=5.0)
    snaps = Snapshots(cfg["id"], name, n_updates * batch_size, args.snapshots, max_steps=cfg["video_steps"], **cfg["kwargs"])
    greedy = lambda o: int(model.dist(to_input(o, atari, device).unsqueeze(0)).probs.argmax())

    buf_obs = torch.zeros((n_steps, n_envs, *obs_shape), dtype=torch.uint8 if atari else torch.float32, device=device)
    buf_act = torch.zeros(n_steps, n_envs, dtype=torch.long, device=device)
    buf_logp = torch.zeros(n_steps, n_envs, device=device)
    buf_rew = torch.zeros(n_steps, n_envs, device=device)
    buf_done = torch.zeros(n_steps, n_envs, device=device)
    buf_val = torch.zeros(n_steps, n_envs, device=device)

    obs = to_input(envs.reset(seed=args.seed)[0], atari, device)
    done = torch.zeros(n_envs, device=device)
    global_step = 0
    for update in range(n_updates):
        with torch.no_grad():
            snaps.maybe_record(global_step, greedy, tracker.writer)
        optimizer.param_groups[0]["lr"] = lr * (1 - update / n_updates)

        for t in range(n_steps):
            buf_obs[t], buf_done[t] = obs, done
            with torch.no_grad():
                d, v = model.dist_value(obs)
                action = d.sample()
                buf_logp[t], buf_val[t] = d.log_prob(action), v
            buf_act[t] = action
            next_obs, reward, term, trunc, infos = envs.step(action.cpu().numpy())
            buf_rew[t] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            obs = to_input(next_obs, atari, device)
            done = torch.as_tensor(np.logical_or(term, trunc), dtype=torch.float32, device=device)
            global_step += n_envs
            if "episode" in infos:
                for i in np.flatnonzero(infos["_episode"]):
                    tracker.episode(infos["episode"]["r"][i], global_step)

        with torch.no_grad():
            adv, ret = compute_gae(buf_rew, buf_val, buf_done, model.value(obs), done, gamma, lam)

        b_obs = buf_obs.reshape(batch_size, *obs_shape)
        b_act, b_logp = buf_act.reshape(-1), buf_logp.reshape(-1)
        b_adv, b_ret = adv.reshape(-1), ret.reshape(-1)
        mb_size = batch_size // n_minibatches
        infos_epoch = []
        for _ in range(epochs):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mb_size):
                mb = perm[start : start + mb_size]
                loss, info = ppo_loss(model, b_obs[mb], b_act[mb], b_logp[mb], b_adv[mb], b_ret[mb], clip_eps, vf_coef, ent_coef)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                infos_epoch.append(info)

        mean_info = {k: float(np.mean([i[k] for i in infos_epoch])) for k in infos_epoch[0]}
        tracker.log(global_step, policy_loss=mean_info["pl"], value_loss=mean_info["vl"], entropy=mean_info["ent"],
                    approx_kl=mean_info["kl"], clip_frac=mean_info["clip"], lr=optimizer.param_groups[0]["lr"])

        if len(tracker.returns) >= window and tracker.recent_mean(window) >= cfg["solved"]:
            print(f"Solved at step {global_step}: avg{window} return {tracker.recent_mean(window):.1f}")
            break

    envs.close()
    with torch.no_grad():
        snaps.finish(global_step, greedy, tracker.writer)
    tracker.close()
    return model, cfg, device


@torch.no_grad()
def evaluate(model, cfg, device, greedy: bool = True) -> float:
    env = make(cfg["id"], **cfg["kwargs"])
    atari = is_atari(cfg["id"])
    total = 0.0
    for ep in range(cfg["eval_episodes"]):
        obs, _ = env.reset(seed=10_000 + ep)
        done = False
        while not done:
            d = model.dist(to_input(obs, atari, device).unsqueeze(0))
            a = int(d.probs.argmax()) if greedy else int(d.sample())
            obs, r, te, tr, _ = env.step(a)
            total += r
            done = te or tr
    env.close()
    return total / cfg["eval_episodes"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="cartpole", choices=list(ENVS))
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--live", action="store_true", help="live plot window while training")
    p.add_argument("--snapshots", type=int, default=0, help="record N videos spread over training")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--video", action="store_true")
    args = p.parse_args()

    model, cfg, device = train(args)
    print(f"Greedy policy, {cfg['eval_episodes']} eval episodes: avg return {evaluate(model, cfg, device):.1f}")
    torch.save(model.state_dict(), f"03_ppo/{args.env}.pt")

    atari = is_atari(cfg["id"])
    policy = lambda obs: int(model.dist(to_input(obs, atari, device).unsqueeze(0)).probs.argmax())
    with torch.no_grad():
        if args.watch:
            watch(cfg["id"], policy, **cfg["kwargs"])
        if args.video:
            record_video(cfg["id"], policy, f"03_ppo_{args.env}", max_steps=cfg["video_steps"], **cfg["kwargs"])
