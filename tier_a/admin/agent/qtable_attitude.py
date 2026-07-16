"""Tabular Q-learning over a discretized (trajectory-error + attitude-rate +
3D-LIDAR) state -- the project's native RL style (no neural net), applied to
Tier-A. Where ``agent/qnet_attitude.py`` trades a continuous state for
function approximation, this module trades continuous precision for a
genuine Q-table: same discipline as the base kit's ``agent/qlearning.py``.

Extending the state with 3D-LIDAR gives the agent something the trajectory-
only DQN variant never had: reactive awareness of a nearby tree it can sense
but whose coordinates it never reads directly (INVARIANT 2's spirit, same
trust boundary as Tier 3's 2D LIDAR). Two behavioural objectives are added as
DIRECT reward terms (not potential-based shaping, since they are genuine
preferences over behaviour, not a policy-invariant speedup trick):

    stability   -- penalise angular-rate magnitude, discourages tumbling
    reactivity  -- penalise LIDAR proximity intrusion, discourages close calls

This module never imports torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from tier_a.env_attitude.env import AttitudeEnv
from tier_a.env_attitude.lidar3d import LIDAR_RANGE_3D, N_RAYS_3D, body_bearings

# --- potential-based shaping (own copy: this module stays torch-free, and
# independent of agent/qnet_attitude.py's identical-in-spirit definition,
# mirroring how scaffold/starter.ipynb keeps its own copy of agent/qlearning.py's
# logic rather than importing it) -------------------------------------------
W_POS, W_ATT = 1.0, 0.3


def potential(obs: np.ndarray) -> float:
    """Phi(s) = -(weighted tracking-error norm). Never reads tree positions."""
    e_lat, e_alt = obs[0], obs[1]
    e_phi, e_theta, e_psi = obs[3], obs[4], obs[5]
    return -(W_POS * math.hypot(e_lat, e_alt) + W_ATT * math.hypot(e_phi, e_theta, e_psi))


def shaping_F(obs: np.ndarray, obs2: np.ndarray, gamma: float, terminal: bool = False) -> float:
    """gamma*Phi(s') - Phi(s), with Phi(s')=0 at a true terminal (INVARIANT 4)."""
    phi_next = 0.0 if terminal else potential(obs2)
    return gamma * phi_next - potential(obs)


# --- safety/reactivity reward augmentation ----------------------------------
W_STAB = 0.5     # N*m*s-scale penalty weight on angular-rate magnitude
W_PROX = 2.5     # penalty weight on LIDAR proximity intrusion
PROX_SAFE = 1.0  # metres: at/beyond this nearest-obstacle range, no penalty applies


def safety_bonus(p: float, q: float, r: float, lidar_min: float) -> float:
    """Direct (non-shaping) reward term: discourage spinning and close calls."""
    rate_mag = math.sqrt(p * p + q * q + r * r)
    prox_penalty = max(0.0, PROX_SAFE - lidar_min) / PROX_SAFE
    return -(W_STAB * rate_mag + W_PROX * prox_penalty)


# --- discretization ----------------------------------------------------------
# obs layout: [e_lat, e_alt, e_vz, e_phi, e_theta, e_psi, p, q, r, kappa]
_EDGES_LAT = (-0.5, -0.15, 0.15, 0.5)   # 5 bins
_EDGES_ALT = (-0.15, 0.15)              # 3 bins
_EDGES_PSI = (-0.3, 0.3)                # 3 bins
_EDGES_PHI = (-0.2, 0.2)                # 3 bins
_EDGES_THETA = (-0.2, 0.2)              # 3 bins
_EDGES_RATE = (0.8, 2.0)                # 3 bins: calm / moderate / spinning
_EDGES_LMIN = (0.6, 1.2)                # 3 bins: danger / caution / clear (metres)
_N_BEARING_SECTORS = 3                  # ahead / left / right

N_BINS = (5, 3, 3, 3, 3, 3, 3, _N_BEARING_SECTORS)
N_STATES = 1
for _n in N_BINS:
    N_STATES *= _n

_BEARINGS = body_bearings(N_RAYS_3D)


def _bin(value: float, edges: tuple[float, ...]) -> int:
    idx = 0
    for e in edges:
        if value > e:
            idx += 1
    return idx


def _bearing_sector(bearing: float) -> int:
    """0=ahead (within +-60deg of body-forward), 1=left, 2=right."""
    b = ((bearing + math.pi) % (2.0 * math.pi)) - math.pi
    if abs(b) < math.radians(60):
        return 0
    return 1 if b > 0 else 2


def discretize(obs: np.ndarray, lidar: np.ndarray) -> int:
    """Continuous (obs, lidar) -> a single integer in [0, N_STATES)."""
    e_lat, e_alt, _e_vz, e_phi, e_theta, e_psi, p, q, r, _kappa = obs
    rate_mag = math.sqrt(p * p + q * q + r * r)
    i_min = int(np.argmin(lidar))
    lmin = float(lidar[i_min])
    near_enough_to_care = lmin < PROX_SAFE * 1.5
    bearing_sector = _bearing_sector(float(_BEARINGS[i_min])) if near_enough_to_care else 0

    bins = (
        _bin(e_lat, _EDGES_LAT), _bin(e_alt, _EDGES_ALT), _bin(e_psi, _EDGES_PSI),
        _bin(e_phi, _EDGES_PHI), _bin(e_theta, _EDGES_THETA), _bin(rate_mag, _EDGES_RATE),
        _bin(lmin, _EDGES_LMIN), bearing_sector,
    )
    idx = 0
    for b, n in zip(bins, N_BINS):
        idx = idx * n + b
    return idx


@dataclass
class HistoryTabular:
    """Per-episode training log, extended with the two behavioural metrics
    this module specifically targets: mean angular-rate (anti-tumbling) and
    closest LIDAR approach (obstacle reactivity)."""

    ret: list = field(default_factory=list)
    success: list = field(default_factory=list)
    collision: list = field(default_factory=list)
    departed: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    epsilon: list = field(default_factory=list)
    mean_rate: list = field(default_factory=list)
    min_lidar: list = field(default_factory=list)
    # (episode, Q.copy()) -- powers the progressive-training filmstrip.
    q_snapshots: list = field(default_factory=list)

    # Periodic *greedy* evaluation. Sample efficiency and behavioural trend
    # must be read off this curve, not the exploration episodes above: with
    # eps floored at eps_end the exploring agent keeps taking random detours
    # long after the greedy policy has actually improved.
    eval_ep: list = field(default_factory=list)
    eval_success: list = field(default_factory=list)
    eval_steps: list = field(default_factory=list)
    eval_mean_rate: list = field(default_factory=list)
    eval_min_lidar: list = field(default_factory=list)


def greedy_rollout_tabular(env: AttitudeEnv, Q: np.ndarray, max_steps: int = 2000,
                           record: bool = False) -> dict:
    """argmax_a Q(s,a), no exploration. Optionally records the flown pose
    trajectory for visualization."""
    obs = env.reset()
    lidar = env.lidar_scan()
    s = discretize(obs, lidar)
    sq_err: list[float] = []
    rates: list[float] = []
    lmins: list[float] = []
    traj = {"xyz": [env.xyz.copy()], "attitude": [env.attitude]} if record else None

    for _ in range(max_steps):
        a = int(np.argmax(Q[s]))
        obs, r, done, info = env.step(a)
        lidar = env.lidar_scan()
        s = discretize(obs, lidar)
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
        "min_lidar": float(np.min(lmins)) if lmins else LIDAR_RANGE_3D,
    }
    if record:
        out["trajectory"] = {k: np.array(v) for k, v in traj.items()}
    return out


def train_tabular_attitude(
    env: AttitudeEnv,
    episodes: int = 3000,
    alpha: float = 0.15,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_frac: float = 0.6,
    shaping: bool = True,
    seed: int = 0,
    snapshot_every: int = 0,
    eval_every: int = 0,
) -> tuple[np.ndarray, HistoryTabular]:
    """Tabular Q-learning with potential-based shaping plus a direct
    stability/reactivity reward augmentation. Mirrors ``agent.qlearning.train``'s
    shape and epsilon schedule.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, env.n_actions), dtype=np.float64)
    hist = HistoryTabular()
    decay_eps = max(1, int(episodes * eps_decay_frac))

    for ep in range(episodes):
        eps = max(eps_end, eps_start + (eps_end - eps_start) * ep / decay_eps)
        obs = env.reset()
        lidar = env.lidar_scan()
        s = discretize(obs, lidar)
        total = 0.0
        rates: list[float] = []
        lmins: list[float] = []
        info = {"goal": False, "collision": False, "departed": False, "loss_of_control": False}

        while True:
            a = int(rng.integers(env.n_actions)) if rng.random() < eps else int(np.argmax(Q[s]))
            obs2, r, done, info = env.step(a)
            lidar2 = env.lidar_scan()
            s2 = discretize(obs2, lidar2)

            terminal = info["collision"] or info["goal"] or info["departed"] or info["loss_of_control"]
            r_train = r + (shaping_F(obs, obs2, gamma, terminal) if shaping else 0.0)
            r_train += safety_bonus(float(obs2[6]), float(obs2[7]), float(obs2[8]), float(lidar2.min()))

            target = r_train if terminal else r_train + gamma * float(np.max(Q[s2]))
            Q[s, a] += alpha * (target - Q[s, a])

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
        hist.min_lidar.append(float(np.min(lmins)) if lmins else LIDAR_RANGE_3D)
        if snapshot_every and (ep % snapshot_every == 0 or ep == episodes - 1):
            hist.q_snapshots.append((ep, Q.copy()))
        if eval_every and (ep % eval_every == 0 or ep == episodes - 1):
            roll = greedy_rollout_tabular(env, Q)
            hist.eval_ep.append(ep)
            hist.eval_success.append(roll["success"])
            hist.eval_steps.append(roll["steps"])
            hist.eval_mean_rate.append(roll["mean_rate"])
            hist.eval_min_lidar.append(roll["min_lidar"])

    return Q, hist
