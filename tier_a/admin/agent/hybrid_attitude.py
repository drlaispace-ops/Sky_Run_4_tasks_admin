"""Hybrid controller: a classical PD handles moment-to-moment attitude
stabilization (proven, in ``env_attitude`` and this repo's own diagnostics,
to track the reference and hold level flight precisely); a tiny tabular
Q-learner sits on top and decides only ONE thing -- whether to bias the
roll target left, right, or not at all, based on what the 3D LIDAR senses
nearby.

Why: ``agent/qtable_attitude.py``'s end-to-end tabular Q-learning discretizes
attitude error into 3 bins, which is too coarse to reproduce continuous PD's
precision (confirmed: it rarely beats a hand-tuned PD controller on the same
gauntlet). Rather than throwing that away, this module keeps the part PD is
already good at (stabilization) and scopes learning down to the part that
is genuinely a *decision*, not a control-loop: react to a sensed obstacle.
That decision has a 9-state, 3-action table -- small enough to converge in
a few hundred episodes, not tens of thousands.

Never imports torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from tier_a.admin.agent.qtable_attitude import PROX_SAFE, _bin, shaping_F
from tier_a.env_attitude.constants import ACTIONS_ATTITUDE, D_TAU
from tier_a.env_attitude.env import AttitudeEnv

KP, KD = 1.0, 0.3  # same gains as the hand-tuned PD used throughout this session's diagnostics

BIAS_ANGLE = math.radians(12.0)
# Index 0 = swerve toward -roll, 1 = no bias, 2 = swerve toward +roll. Which
# physical direction each corresponds to is for the Q-table to learn from
# reward, not to be hard-coded here -- see the module docstring.
BIAS_ACTIONS = (-BIAS_ANGLE, 0.0, BIAS_ANGLE)
N_BIAS_ACTIONS = len(BIAS_ACTIONS)

_LMIN_EDGES = (1.0, 1.6)  # danger / caution / clear, metres -- wider than qtable_attitude's
# so the "danger" state is entered well before the gate, giving Q-learning
# enough steps per episode to actually sample it (a 0.6m gap only stays
# inside a tight 0.6m LIDAR threshold for a handful of steps -- too sparse
# to learn from in a few thousand episodes).
_ELAT_EDGES = (-0.15, 0.15)  # drifting left / centred / drifting right
N_GUIDANCE_STATES = 3 * 3  # e_lat bin (3) x lidar_min bin (3)


W_PROX = 2.0


def guidance_state(obs: np.ndarray, lidar: np.ndarray) -> int:
    """(e_lat bin, lidar_min bin) -> single index in [0, 9).

    A LIDAR *bearing* alone is ambiguous in a two-tree gate: centred between
    both trunks, "nearest obstacle" flip-flops between left and right with no
    signal for which way is actually safe. ``e_lat`` (which side of the
    reference centreline the drone has already drifted to) is the
    unambiguous "which way to correct" signal; ``lidar_min`` (from the 3D
    scan) supplies "how urgently," so the sensor still does real work here --
    it is what distinguishes "drifting but far from anything" from "drifting
    and about to clip a trunk."
    """
    e_lat = float(obs[0])
    lmin = float(lidar.min())
    b_lat = _bin(e_lat, _ELAT_EDGES)
    b_lmin = _bin(lmin, _LMIN_EDGES)
    return b_lat * 3 + b_lmin


def proximity_penalty(lidar_min: float) -> float:
    """Direct (non-shaping) reward term: discourage the guidance layer from
    picking a bias that leaves the drone too close to an obstacle."""
    return -W_PROX * max(0.0, PROX_SAFE - lidar_min) / PROX_SAFE


def pd_action(obs: np.ndarray, phi_bias: float = 0.0, theta_bias: float = 0.0,
              psi_bias: float = 0.0) -> int:
    """The proven hand-tuned PD, parameterized by a target bias added to the
    reference attitude. ``obs[3:6]`` are already errors relative to the
    reference (``actual - ref``), so subtracting a bias from the error is
    equivalent to tracking ``ref + bias`` instead of ``ref``."""
    e_phi = obs[3] - phi_bias
    e_theta = obs[4] - theta_bias
    e_psi = obs[5] - psi_bias
    p, q, r = obs[6], obs[7], obs[8]
    cmd = (-(KP * e_phi + KD * p), -(KP * e_theta + KD * q), -(KP * e_psi + KD * r))
    tau = tuple(D_TAU if c > 0 else (-D_TAU if c < 0 else 0.0) for c in cmd)
    return ACTIONS_ATTITUDE.index(tau)


@dataclass
class HistoryHybrid:
    ret: list = field(default_factory=list)
    success: list = field(default_factory=list)
    collision: list = field(default_factory=list)
    departed: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    epsilon: list = field(default_factory=list)
    mean_rate: list = field(default_factory=list)
    min_lidar: list = field(default_factory=list)
    q_snapshots: list = field(default_factory=list)
    eval_ep: list = field(default_factory=list)
    eval_success: list = field(default_factory=list)
    eval_steps: list = field(default_factory=list)


def greedy_rollout_hybrid(env: AttitudeEnv, Q: np.ndarray, max_steps: int = 2000,
                          record: bool = False) -> dict:
    """PD stabilizes every step; the guidance bias is argmax(Q[guidance_state])."""
    obs = env.reset()
    lidar = env.lidar_scan()
    s = guidance_state(obs, lidar)
    sq_err: list[float] = []
    rates: list[float] = []
    lmins: list[float] = []
    traj = {"xyz": [env.xyz.copy()], "attitude": [env.attitude]} if record else None

    for _ in range(max_steps):
        bias = BIAS_ACTIONS[int(np.argmax(Q[s]))]
        a = pd_action(obs, phi_bias=bias)
        obs, r, done, info = env.step(a)
        lidar = env.lidar_scan()
        s = guidance_state(obs, lidar)
        sq_err.append(float(obs[0]) ** 2 + float(obs[1]) ** 2)
        rates.append(math.sqrt(float(obs[6]) ** 2 + float(obs[7]) ** 2 + float(obs[8]) ** 2))
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
        "mean_rate": float(np.mean(rates)) if rates else 0.0,
        "min_lidar": float(np.min(lmins)) if lmins else 2.5,
    }
    if record:
        out["trajectory"] = {k: np.array(v) for k, v in traj.items()}
    return out


def train_hybrid_guidance(
    env: AttitudeEnv,
    episodes: int = 3000,
    alpha: float = 0.3,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_frac: float = 0.6,
    shaping: bool = True,
    seed: int = 0,
    snapshot_every: int = 0,
    eval_every: int = 0,
) -> tuple[np.ndarray, HistoryHybrid]:
    """Only the 9x3 guidance table is learned; PD handles every torque command."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_GUIDANCE_STATES, N_BIAS_ACTIONS), dtype=np.float64)
    hist = HistoryHybrid()
    decay_eps = max(1, int(episodes * eps_decay_frac))

    for ep in range(episodes):
        eps = max(eps_end, eps_start + (eps_end - eps_start) * ep / decay_eps)
        obs = env.reset()
        lidar = env.lidar_scan()
        s = guidance_state(obs, lidar)
        total = 0.0
        rates: list[float] = []
        lmins: list[float] = []
        info = {"goal": False, "collision": False, "departed": False, "loss_of_control": False}

        while True:
            bias_idx = int(rng.integers(N_BIAS_ACTIONS)) if rng.random() < eps else int(np.argmax(Q[s]))
            a = pd_action(obs, phi_bias=BIAS_ACTIONS[bias_idx])
            obs2, r, done, info = env.step(a)
            lidar2 = env.lidar_scan()
            s2 = guidance_state(obs2, lidar2)

            terminal = info["collision"] or info["goal"] or info["departed"] or info["loss_of_control"]
            r_train = r + (shaping_F(obs, obs2, gamma, terminal) if shaping else 0.0)
            r_train += proximity_penalty(float(lidar2.min()))

            target = r_train if terminal else r_train + gamma * float(np.max(Q[s2]))
            Q[s, bias_idx] += alpha * (target - Q[s, bias_idx])

            total += r
            rates.append(math.sqrt(float(obs2[6]) ** 2 + float(obs2[7]) ** 2 + float(obs2[8]) ** 2))
            lmins.append(float(lidar2.min()))

            obs, s, lidar = obs2, s2, lidar2
            if done or info["truncated"]:
                break

        hist.ret.append(total)
        hist.success.append(bool(info["goal"]))
        hist.collision.append(bool(info["collision"]))
        hist.departed.append(bool(info["departed"] or info["loss_of_control"]))
        hist.steps.append(len(rates))
        hist.epsilon.append(eps)
        hist.mean_rate.append(float(np.mean(rates)) if rates else 0.0)
        hist.min_lidar.append(float(np.min(lmins)) if lmins else 2.5)
        if snapshot_every and (ep % snapshot_every == 0 or ep == episodes - 1):
            hist.q_snapshots.append((ep, Q.copy()))
        if eval_every and (ep % eval_every == 0 or ep == episodes - 1):
            roll = greedy_rollout_hybrid(env, Q)
            hist.eval_ep.append(ep)
            hist.eval_success.append(roll["success"])
            hist.eval_steps.append(roll["steps"])

    return Q, hist
