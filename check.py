from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
LESSONS = {1: "01_q_learning", 2: "02_dqn", 3: "03_ppo", 4: "04_ddpg"}


def load(lesson: int):
    path = ROOT / LESSONS[lesson] / "train.py"
    spec = importlib.util.spec_from_file_location(f"lesson{lesson}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def l1_discretizer(m):
    d = m.Discretizer(np.array([0.0, 0.0]), np.array([1.0, 1.0]), bins=3)
    assert d.n_states == 16
    assert d(np.array([-1.0, -1.0])) == 0, "both values below every edge should give state 0"
    assert d(np.array([2.0, 2.0])) == 15, "both values above every edge should give the last state"
    seen = {d(np.array([x, y])) for x in (-1, 0.2, 0.7, 2) for y in (-1, 0.2, 0.7, 2)}
    assert seen == set(range(16)), f"every bucket combination should map to a unique state, got {sorted(seen)}"


def l1_epsilon_greedy(m):
    rng = np.random.default_rng(0)
    q = np.array([0.0, 5.0, 1.0])
    assert all(m.epsilon_greedy(q, 0.0, rng) == 1 for _ in range(50)), "epsilon=0 must always pick the argmax"
    picks = [m.epsilon_greedy(q, 1.0, rng) for _ in range(600)]
    assert set(picks) == {0, 1, 2}, "epsilon=1 must pick uniformly random actions"
    ties = {m.epsilon_greedy(np.zeros(4), 0.0, rng) for _ in range(200)}
    assert len(ties) > 1, "break ties randomly, otherwise an all-zero table always picks action 0"


def l2_replay_buffer(m):
    b = m.ReplayBuffer(3, (2,), np.float32)
    for i in range(5):
        b.add(np.full(2, i), i % 2, float(i), np.full(2, i + 1), i == 4)
    assert b.size == 3, f"size should stop at capacity (3), got {b.size}"
    assert b.pos == 5 % 3, f"pos should wrap around with %, got {b.pos}"
    assert sorted(b.rewards.tolist()) == [2.0, 3.0, 4.0], "the oldest transitions should be overwritten first"
    assert b.terminated[(5 - 1) % 3] == 1.0


def l2_network_output(m):
    net = m.MLPQNet(4, 2)
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(-0.5)
    out = net(torch.ones(1, 4))
    assert (out < 0).any(), "Q-values must be able to be negative — no ReLU on the last layer!"


def l3_gae(m):
    T, N = 3, 1
    rewards = torch.ones(T, N)
    values = torch.zeros(T, N)
    dones = torch.zeros(T, N)
    adv, ret = m.compute_gae(rewards, values, dones, torch.zeros(N), torch.zeros(N), gamma=0.5, lam=1.0)

    expected = torch.tensor([[1 + 0.5 + 0.25], [1 + 0.5], [1.0]])
    assert torch.allclose(adv, expected), f"expected {expected.flatten().tolist()}, got {adv.flatten().tolist()}"
    assert torch.allclose(ret, adv + values)

    dones[2] = 1
    adv, _ = m.compute_gae(rewards, values, dones, torch.zeros(N), torch.zeros(N), gamma=0.5, lam=1.0)
    assert torch.allclose(adv, torch.tensor([[1.5], [1.0], [1.0]])), f"GAE leaked across an episode end: {adv.flatten().tolist()}"


def l3_ppo_loss(m):
    torch.manual_seed(0)
    model = m.ActorCritic(4, 2)
    obs = torch.randn(64, 4)
    with torch.no_grad():
        d = model.dist(obs)
        act = d.sample()
        logp = d.log_prob(act)
    adv, ret = torch.randn(64), torch.randn(64)
    loss, info = m.ppo_loss(model, obs, act, logp, adv, ret, clip_eps=0.2, vf_coef=0.5, ent_coef=0.0)
    assert loss.requires_grad, "the loss must keep its gradient"
    assert abs(info["clip"]) < 1e-6, "with an unchanged policy the ratio is 1, so nothing should be clipped"
    assert abs(info["pl"]) < 1e-5, "with ratio 1 and normalised advantages, policy loss should be ~0"

    loss, info = m.ppo_loss(model, obs, act, logp - 1.0, torch.ones(64) + torch.randn(64) * 0.01, ret, 0.2, 0.0, 0.0)
    assert info["clip"] > 0.9, "a ratio of e (~2.7) should be clipped almost everywhere"


def l4_soft_update(m):
    src, tgt = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    with torch.no_grad():
        for p in src.parameters():
            p.fill_(1.0)
        for p in tgt.parameters():
            p.fill_(0.0)
    m.soft_update(tgt, src, 0.1)
    assert all(torch.allclose(p, torch.full_like(p, 0.1)) for p in tgt.parameters()), "target should move 10% toward source"
    assert all(torch.allclose(p, torch.ones_like(p)) for p in src.parameters()), "don't change the source network"


def l4_actor_range(m):
    actor = m.Actor(3, 1, np.array([2.0], dtype=np.float32))
    out = actor(torch.randn(256, 3) * 100)
    assert out.abs().max() <= 2.0 + 1e-5, "actions must stay inside [-max_action, max_action]"


TESTS = {1: [l1_discretizer, l1_epsilon_greedy], 2: [l2_replay_buffer, l2_network_output],
         3: [l3_gae, l3_ppo_loss], 4: [l4_soft_update, l4_actor_range]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("lesson", type=int, choices=list(TESTS))
    args = p.parse_args()
    m = load(args.lesson)
    failed = 0
    for test in TESTS[args.lesson]:
        name = test.__name__[3:]
        try:
            test(m)
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception:
            failed += 1
            print(f"  ERROR {name}:")
            traceback.print_exc()
    print("All checks passed." if not failed else f"{failed} check(s) failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
