"""Pirouette guidance: a classical PD handles moment-to-moment attitude
stabilization (unchanged from ``agent/hybrid_attitude.py``); a tiny tabular
Q-learner sits on top and decides only ONE thing -- what roll angle to
target, from level (0 deg) through a full knife-edge pass (85 deg ==
``PHI_MAX_PIROUETTE``) -- based on what the dense 64-ray LIDAR
(``AttitudeEnv.lidar_scan_dense``) senses nearby and the drone's OWN current
roll.

Why *own roll*, not ``e_lat`` (hybrid's guidance signal): hybrid's decision
was "which side to correct toward" -- inherently needs to know which side of
the centreline the drone has drifted to. This decision is "how hard to
bank," and ``env_attitude.occupancy.presented_half_width`` is symmetric in
the sign of phi, so bank direction doesn't change whether a *centred*
two-tree gate is passable. Tracking the agent's own roll instead gives the
table something hybrid's design didn't have: a partially-rolled state is
entered *during approach*, before ``lidar_min`` bottoms out -- directly
targeting hybrid's own diagnosed root cause (README: "the danger state is
visited too rarely per episode before collision ends it").

Never imports torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from tier_a.admin.agent.qtable_attitude import _bin, shaping_F
from tier_d.env.constants import TREE_R
from tier_a.env_attitude.constants import ACTIONS_ATTITUDE, D_TAU, PHI_MAX_PIROUETTE
from tier_a.env_attitude.env import AttitudeEnv
from tier_a.env_attitude.occupancy import presented_half_width

KP, KD = 1.0, 0.3  # same gains as agent/hybrid_attitude.py's proven PD

# --- action: 7 discrete roll targets, the steepest exactly at the physical
# envelope boundary AttitudeEnv(body_aware_collision=True) already enforces --
ROLL_TARGETS = tuple(math.radians(d) for d in (0, 30, -30, 60, -60, 85, -85))
assert math.isclose(max(ROLL_TARGETS), PHI_MAX_PIROUETTE)
N_ROLL_ACTIONS = len(ROLL_TARGETS)  # 7

# --- state: bin(lidar_min) x bin(own |phi|) = 9 -----------------------------
_LMIN_EDGES = (0.55, 0.75)   # danger / caution / clear, metres -- tuned to
# THIS task's gate range (TREE_R+H_BODY=0.48 .. TREE_R+R_BODY=0.65), not
# reused unmodified from hybrid's (1.0, 1.6), which was tuned for a wider gate.
_PHI_OWN_EDGES = (math.radians(20.0), math.radians(50.0))  # level / partial-roll / knife-edge
N_PIROUETTE_STATES = 3 * 3  # lidar_min bin (3) x own |phi| bin (3)

W_MARGIN = 2.0      # same order as hybrid's W_PROX=2.0
SAFE_MARGIN = 0.15  # m, buffer beyond the bare collision threshold


def pirouette_state(obs: np.ndarray, lidar: np.ndarray, phi_now: float) -> int:
    """(lidar_min bin, own |phi| bin) -> single index in [0, 9)."""
    b_lmin = _bin(float(lidar.min()), _LMIN_EDGES)
    b_phi = _bin(abs(phi_now), _PHI_OWN_EDGES)
    return b_lmin * 3 + b_phi


def proactive_margin_penalty(phi: float, lidar_min: float, tree_r: float = TREE_R,
                              w: float = W_MARGIN, safe: float = SAFE_MARGIN) -> float:
    """Direct (non-shaping) reward term: rewards banking BEFORE lidar_min
    forces it. Unlike hybrid's proximity_penalty (blind to the agent's own
    roll, fixed PROX_SAFE tuned for a point-mass body), this compares the
    sensed clearance against ``tree_r + presented_half_width(phi)`` -- the
    exact expression ``collides_oriented`` uses for the real collision test,
    so "margin" here means real physical clearance at the CURRENT roll, not
    an arbitrary fixed radius."""
    margin = lidar_min - tree_r - presented_half_width(phi)
    return -w * max(0.0, safe - margin) / safe


def pd_action(obs: np.ndarray, phi_bias: float = 0.0, theta_bias: float = 0.0,
              psi_bias: float = 0.0) -> int:
    """Identical in form to agent.hybrid_attitude.pd_action -- kept as a
    local copy (not imported) since this module's docstring/tuning notes are
    specific to the pirouette task; the gains (KP, KD) are the same proven
    values."""
    e_phi = obs[3] - phi_bias
    e_theta = obs[4] - theta_bias
    e_psi = obs[5] - psi_bias
    p, q, r = obs[6], obs[7], obs[8]
    cmd = (-(KP * e_phi + KD * p), -(KP * e_theta + KD * q), -(KP * e_psi + KD * r))
    tau = tuple(D_TAU if c > 0 else (-D_TAU if c < 0 else 0.0) for c in cmd)
    return ACTIONS_ATTITUDE.index(tau)


@dataclass
class HistoryPirouette:
    ret: list = field(default_factory=list)
    success: list = field(default_factory=list)
    collision: list = field(default_factory=list)
    departed: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    epsilon: list = field(default_factory=list)
    max_roll: list = field(default_factory=list)
    min_lidar: list = field(default_factory=list)
    q_snapshots: list = field(default_factory=list)
    eval_ep: list = field(default_factory=list)
    eval_success: list = field(default_factory=list)
    eval_steps: list = field(default_factory=list)


def greedy_rollout_pirouette(env: AttitudeEnv, Q: np.ndarray, max_steps: int = 2000,
                             record: bool = False) -> dict:
    """PD stabilizes every step; the roll target is argmax(Q[pirouette_state])."""
    obs = env.reset()
    lidar = env.lidar_scan_dense()
    phi_now = float(obs[3])  # e_phi == phi - phi_ref; phi_ref==0 on these local straight gates
    s = pirouette_state(obs, lidar, phi_now)
    sq_err: list[float] = []
    rolls: list[float] = []
    lmins: list[float] = []
    traj = {"xyz": [env.xyz.copy()], "attitude": [env.attitude]} if record else None

    for _ in range(max_steps):
        target = ROLL_TARGETS[int(np.argmax(Q[s]))]
        a = pd_action(obs, phi_bias=target)
        obs, r, done, info = env.step(a)
        lidar = env.lidar_scan_dense()
        phi_now = float(obs[3])
        s = pirouette_state(obs, lidar, phi_now)
        sq_err.append(float(obs[0]) ** 2 + float(obs[1]) ** 2)
        rolls.append(abs(phi_now))
        lmins.append(float(lidar.min()))
        if record:
            traj["xyz"].append(env.xyz.copy())
            traj["attitude"].append(env.attitude)
        if done or info["truncated"]:
            break

    rmse = float(np.sqrt(np.mean(sq_err))) if sq_err else 0.0
    out = {
        "success": bool(info["goal"]), "collision": bool(info["collision"]),
        "departed": bool(info["departed"] or info["loss_of_control"]),
        "tracking_rmse": rmse, "steps": len(sq_err),
        "max_roll": float(np.max(rolls)) if rolls else 0.0,
        "min_lidar": float(np.min(lmins)) if lmins else 2.5,
    }
    if record:
        out["trajectory"] = {k: np.array(v) for k, v in traj.items()}
    return out


def train_pirouette_guidance(
    env: AttitudeEnv,
    episodes: int = 3000,
    alpha: float = 0.3,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_frac: float = 0.6,
    shaping: bool = True,
    seed: int = 0,
    Q_init: np.ndarray | None = None,
    snapshot_every: int = 0,
    eval_every: int = 0,
) -> tuple[np.ndarray, HistoryPirouette]:
    """Only the 9x7 pirouette table is learned; PD handles every torque
    command. ``Q_init`` (optional) warm-starts from a previous curriculum
    stage's table -- shapes match exactly (9x7) across all stages, since the
    state/action design doesn't change, only the gate's clearance does."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_PIROUETTE_STATES, N_ROLL_ACTIONS), dtype=np.float64) if Q_init is None else Q_init.copy()
    hist = HistoryPirouette()
    decay_eps = max(1, int(episodes * eps_decay_frac))

    for ep in range(episodes):
        eps = max(eps_end, eps_start + (eps_end - eps_start) * ep / decay_eps)
        obs = env.reset()
        lidar = env.lidar_scan_dense()
        phi_now = float(obs[3])
        s = pirouette_state(obs, lidar, phi_now)
        total = 0.0
        rolls: list[float] = []
        lmins: list[float] = []
        info = {"goal": False, "collision": False, "departed": False, "loss_of_control": False}

        while True:
            a_idx = int(rng.integers(N_ROLL_ACTIONS)) if rng.random() < eps else int(np.argmax(Q[s]))
            a = pd_action(obs, phi_bias=ROLL_TARGETS[a_idx])
            obs2, r, done, info = env.step(a)
            lidar2 = env.lidar_scan_dense()
            phi_now2 = float(obs2[3])
            s2 = pirouette_state(obs2, lidar2, phi_now2)

            terminal = info["collision"] or info["goal"] or info["departed"] or info["loss_of_control"]
            r_train = r + (shaping_F(obs, obs2, gamma, terminal) if shaping else 0.0)
            r_train += proactive_margin_penalty(phi_now2, float(lidar2.min()))

            target_q = r_train if terminal else r_train + gamma * float(np.max(Q[s2]))
            Q[s, a_idx] += alpha * (target_q - Q[s, a_idx])

            total += r
            rolls.append(abs(phi_now2))
            lmins.append(float(lidar2.min()))

            obs, s, lidar, phi_now = obs2, s2, lidar2, phi_now2
            if done or info["truncated"]:
                break

        hist.ret.append(total)
        hist.success.append(bool(info["goal"]))
        hist.collision.append(bool(info["collision"]))
        hist.departed.append(bool(info["departed"] or info["loss_of_control"]))
        hist.steps.append(len(rolls))
        hist.epsilon.append(eps)
        hist.max_roll.append(float(np.max(rolls)) if rolls else 0.0)
        hist.min_lidar.append(float(np.min(lmins)) if lmins else 2.5)
        if snapshot_every and (ep % snapshot_every == 0 or ep == episodes - 1):
            hist.q_snapshots.append((ep, Q.copy()))
        if eval_every and (ep % eval_every == 0 or ep == episodes - 1):
            roll = greedy_rollout_pirouette(env, Q)
            hist.eval_ep.append(ep)
            hist.eval_success.append(roll["success"])
            hist.eval_steps.append(roll["steps"])

    return Q, hist
