"""Reference DQN for Tier-A (attitude tracking). Ours, not the students'.

Validates ``AttitudeEnv``, the Tier-A scoring harness and the visualiser. The
DQN Bellman-target/loss pair is the hole students fill in
``tier_a/scaffold/starter_attitude.ipynb`` -- mirroring how ``agent/qlearning.py``'s
tabular Bellman update is the hole in ``scaffold/starter.ipynb``.

Small state (10D) and a small action set (27) keep this "a conceptual leap,
not full deep RL" (the same framing already used for the existing Tier 3
LIDAR net): a 2-hidden-layer MLP, plain experience replay, no target-network
requirement, no continuous-control algorithm. ``torch`` is confined to this
file and never imported anywhere under ``env_attitude/`` (see
``tests/test_invariants_attitude.py``'s torch-isolation check).
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from tier_a.env_attitude.env import AttitudeEnv

# Measured, not assumed: for this net's size (two 64-unit layers, batch 64),
# MPS's per-call dispatch overhead dominates and is ~7x SLOWER than CPU on
# this machine (11.7s vs 1.7s for the 150-episode smoke config). CPU wins for
# small tensors; MPS only pays off at larger batch/model sizes than this
# tier intentionally uses (see the plan's time-boxing section).
DEVICE = torch.device("cpu")

# -- potential-based shaping (INVARIANT 4's contract, adapted to continuous obs) --
# obs layout: [e_lat, e_alt, e_vz, e_phi, e_theta, e_psi, p, q, r, kappa]
W_POS = 1.0
W_ATT = 0.3


def potential(obs: np.ndarray) -> float:
    """Phi(s) = -(weighted tracking-error norm). Reads only the agent's own
    observation -- never a tree position (INVARIANT 2's spirit)."""
    e_lat, e_alt = obs[0], obs[1]
    e_phi, e_theta, e_psi = obs[3], obs[4], obs[5]
    return -(W_POS * math.hypot(e_lat, e_alt) + W_ATT * math.hypot(e_phi, e_theta, e_psi))


def shaping_F(obs: np.ndarray, obs2: np.ndarray, gamma: float, terminal: bool = False) -> float:
    """gamma*Phi(s') - Phi(s), with Phi(s')=0 at a true terminal -- same
    contract as ``agent.shaping.shaping_F``, adapted to continuous obs."""
    phi_next = 0.0 if terminal else potential(obs2)
    return gamma * phi_next - potential(obs)


class QNet(nn.Module):
    """Tiny 2-hidden-layer MLP -- the function approximator, not the algorithm."""

    def __init__(self, obs_dim: int = 10, n_actions: int = 27, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Plain uniform experience replay -- no prioritization, no target network."""

    def __init__(self, capacity: int = 20000, seed: int = 0):
        self.buf: deque = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, s, a, r, s2, terminal) -> None:
        self.buf.append((s, a, r, s2, terminal))

    def sample(self, batch_size: int):
        batch = self._rng.sample(self.buf, min(batch_size, len(self.buf)))
        s, a, r, s2, terminal = zip(*batch)
        return (
            torch.as_tensor(np.array(s), dtype=torch.float32, device=DEVICE),
            torch.as_tensor(a, dtype=torch.long, device=DEVICE),
            torch.as_tensor(r, dtype=torch.float32, device=DEVICE),
            torch.as_tensor(np.array(s2), dtype=torch.float32, device=DEVICE),
            torch.as_tensor(terminal, dtype=torch.bool, device=DEVICE),
        )

    def __len__(self) -> int:
        return len(self.buf)


def dqn_update(qnet: QNet, target_net: QNet, optimizer: torch.optim.Optimizer, batch, gamma: float) -> float:
    """One gradient step over a minibatch. Returns the scalar loss.

    ``target_net`` (a periodically-synced copy of ``qnet``) supplies the
    bootstrap value: computing ``max_a Q(s',a)`` from the *same* network being
    updated is the classic source of overoptimistic value estimates in vanilla
    online DQN (confirmed empirically here -- without it, the greedy policy
    measurably underperformed a random baseline). This is the standard,
    minimal fix, not a "full deep RL" addition.

    Bootstrap through truncation, not through a real terminal -- exactly the
    trap ``CLAUDE.md`` calls out for the tabular kit, carried over here.
    """
    s, a, r, s2, terminal = batch
    q_sa = qnet(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        target = r + gamma * target_net(s2).max(dim=1).values * (~terminal).float()
    loss = nn.functional.mse_loss(q_sa, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


@dataclass
class HistoryAttitude:
    """Per-episode training log; the visualiser and harness both read this."""

    ret: list = field(default_factory=list)
    success: list = field(default_factory=list)
    collision: list = field(default_factory=list)
    departed: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    epsilon: list = field(default_factory=list)

    # Periodic *greedy* evaluation -- sample efficiency is read off this
    # curve, not the exploration episodes (same reasoning as agent.qlearning).
    eval_ep: list = field(default_factory=list)
    eval_success: list = field(default_factory=list)
    eval_tracking_rmse: list = field(default_factory=list)


def greedy_rollout_attitude(env: AttitudeEnv, qnet: QNet, max_steps: int = 2000) -> dict:
    """Run argmax_a Q(s,a) with no exploration. Returns outcome + tracking RMSE."""
    obs = env.reset()
    sq_err: list[float] = []
    for _ in range(max_steps):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            a = int(qnet(obs_t).argmax(dim=1).item())
        obs2, r, done, info = env.step(a)
        sq_err.append(float(obs2[0]) ** 2 + float(obs2[1]) ** 2)  # e_lat^2 + e_alt^2
        obs = obs2
        if done or info["truncated"]:
            break
    rmse = float(np.sqrt(np.mean(sq_err))) if sq_err else 0.0
    return {
        "success": bool(info["goal"]), "collision": bool(info["collision"]),
        "departed": bool(info["departed"] or info["loss_of_control"]),
        "truncated": bool(info["truncated"]), "tracking_rmse": rmse, "steps": len(sq_err),
    }


def train_attitude(
    env: AttitudeEnv,
    episodes: int = 1200,
    gamma: float = 0.99,
    lr: float = 1e-3,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_frac: float = 0.6,
    shaping: bool = True,
    seed: int = 0,
    batch_size: int = 64,
    buffer_capacity: int = 20000,
    train_every: int = 4,
    target_sync_every: int = 200,
    eval_every: int = 50,
) -> tuple[QNet, HistoryAttitude]:
    """Reference DQN training loop. Mirrors ``agent.qlearning.train``'s shape:
    epsilon-greedy exploration, optional potential-based shaping, periodic
    greedy evaluation. ``train_every`` batches gradient updates every N env
    steps rather than every step -- keeps the *smoke* config (150 episodes)
    finishing in seconds, matching the kit's existing pytest time budget.
    ``target_sync_every`` hard-syncs a target network every N gradient steps
    (see ``dqn_update``'s docstring for why this is not optional here).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    qnet = QNet(env.obs_dim, env.n_actions).to(DEVICE)
    target_net = QNet(env.obs_dim, env.n_actions).to(DEVICE)
    target_net.load_state_dict(qnet.state_dict())
    target_net.eval()
    optimizer = torch.optim.Adam(qnet.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity, seed=seed)
    hist = HistoryAttitude()
    decay_eps = max(1, int(episodes * eps_decay_frac))
    global_step = 0
    update_count = 0

    for ep in range(episodes):
        eps = max(eps_end, eps_start + (eps_end - eps_start) * ep / decay_eps)
        obs = env.reset()
        total = 0.0
        steps = 0
        info = {"goal": False, "collision": False, "departed": False, "loss_of_control": False}

        while True:
            if rng.random() < eps:
                a = int(rng.integers(env.n_actions))
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    a = int(qnet(obs_t).argmax(dim=1).item())
            obs2, r, done, info = env.step(a)
            steps += 1
            global_step += 1
            total += r

            terminal = info["collision"] or info["goal"] or info["departed"] or info["loss_of_control"]
            r_train = r + (shaping_F(obs, obs2, gamma, terminal) if shaping else 0.0)
            buffer.push(obs, a, r_train, obs2, terminal)

            if len(buffer) >= batch_size and global_step % train_every == 0:
                dqn_update(qnet, target_net, optimizer, buffer.sample(batch_size), gamma)
                update_count += 1
                if update_count % target_sync_every == 0:
                    target_net.load_state_dict(qnet.state_dict())

            obs = obs2
            if done or info["truncated"]:
                break

        hist.ret.append(total)
        hist.success.append(bool(info["goal"]))
        hist.collision.append(bool(info["collision"]))
        hist.departed.append(bool(info["departed"] or info["loss_of_control"]))
        hist.steps.append(steps)
        hist.epsilon.append(eps)

        if eval_every and (ep % eval_every == 0 or ep == episodes - 1):
            roll = greedy_rollout_attitude(env, qnet)
            hist.eval_ep.append(ep)
            hist.eval_success.append(roll["success"])
            hist.eval_tracking_rmse.append(roll["tracking_rmse"])

    return qnet, hist
