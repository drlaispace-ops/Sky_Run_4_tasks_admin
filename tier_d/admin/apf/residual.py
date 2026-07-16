"""APF + learned residual: the track the concept doc explicitly discarded.

Rebuilt here as a controlled baseline, not as the main line. The controller is

    a*(s) = argmax_a [ Q_res(s, a) + beta * prior(s, a) ]

where ``prior`` is the cosine alignment between action ``a`` and the APF
descent direction ``-grad U`` at cell ``s``. With ``Q_res == 0`` the policy is
exactly APF, so training starts from the classical controller and learns only
the *correction* -- which is what "residual" means.

READ THIS BEFORE COMPARING TRACKS
--------------------------------
``prior`` is computed from tree coordinates. This track therefore has full map
access and **violates INVARIANT 2 on purpose**. It is cartography, not RL. It
exists to make one point precisely: even *with* the map, hand-built APF stalls
in local minima, while the map-blind Q-learner does not. Never score this
against student submissions.
"""

from __future__ import annotations

import numpy as np

from tier_d.admin.agent.qlearning import History
from tier_d.admin.agent.shaping import shaping_F
from tier_d.admin.apf.field import grad_U
from tier_d.env.constants import ACTIONS, GRID_N, N_ACTIONS
from tier_d.env.gridworld import GridWorld
from tier_d.env.occupancy import cell_center


def apf_prior(trees: np.ndarray) -> np.ndarray:
    """(n_states, N_ACTIONS) cosine alignment of each action with -grad U."""
    prior = np.zeros((GRID_N * GRID_N, N_ACTIONS))
    dirs = np.array([(dc, dr) for dr, dc, _ in ACTIONS], dtype=float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    for r in range(GRID_N):
        for c in range(GRID_N):
            xy = np.asarray(cell_center(r, c))
            g = grad_U(xy, trees)
            n = float(np.linalg.norm(g))
            if n < 1e-12:
                continue  # stationary point: prior is silent, residual decides
            prior[r * GRID_N + c] = dirs @ (-g / n)
    return prior


def apf_greedy_rollout(env: GridWorld, prior: np.ndarray, max_steps: int = 400) -> dict:
    """Pure APF on the grid (Q_res == 0). Included to show the trap on-grid."""
    return residual_rollout(env, np.zeros_like(prior), prior, beta=1.0, max_steps=max_steps)


def residual_rollout(
    env: GridWorld,
    Q: np.ndarray,
    prior: np.ndarray,
    beta: float = 1.0,
    max_steps: int = 400,
) -> dict:
    s = env.reset()
    path = [env.xy]
    length = 0.0
    visits: dict[int, int] = {}

    for _ in range(max_steps):
        a = int(np.argmax(Q[s] + beta * prior[s]))
        s, _, done, info = env.step(a)
        path.append(env.xy)
        length += info["moved"]
        if info["collision"]:
            return {"success": False, "collision": True, "trapped": False,
                    "length": length, "path": np.array(path)}
        if info["goal"]:
            return {"success": True, "collision": False, "trapped": False,
                    "length": length, "path": np.array(path)}
        visits[s] = visits.get(s, 0) + 1
        if visits[s] > 3:  # deterministic policy revisiting a cell == stalled
            return {"success": False, "collision": False, "trapped": True,
                    "length": length, "path": np.array(path)}
        if done:
            break
    return {"success": False, "collision": False, "trapped": True,
            "length": length, "path": np.array(path)}


def train_residual(
    env: GridWorld,
    prior: np.ndarray,
    episodes: int = 3000,
    alpha: float = 0.15,
    gamma: float = 0.99,
    beta: float = 1.0,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_frac: float = 0.6,
    shaping: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, History]:
    """Q-learning over the residual, acting under ``Q + beta*prior``."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, N_ACTIONS))
    hist = History()
    decay = max(1, int(episodes * eps_decay_frac))

    for ep in range(episodes):
        eps = max(eps_end, eps_start + (eps_end - eps_start) * ep / decay)
        s = env.reset()
        total, length = 0.0, 0.0

        while True:
            if rng.random() < eps:
                a = int(rng.integers(N_ACTIONS))
            else:
                a = int(np.argmax(Q[s] + beta * prior[s]))
            s2, r, done, info = env.step(a)
            length += info["moved"]
            total += r

            terminal = info["collision"] or info["goal"]
            r_train = r + (shaping_F(s, s2, gamma, terminal) if shaping else 0.0)
            # Bootstrap under the *behaviour-consistent* greedy score, so the
            # prior shapes the target as well as the action.
            boot = 0.0 if terminal else float(np.max(Q[s2] + beta * prior[s2]))
            # Wiewiora look-ahead advice: the *scored* quantity Q+beta*prior is
            # what bootstraps, so the TD error is taken against it, not Q alone.
            td = r_train + gamma * boot - (Q[s, a] + beta * prior[s, a])
            Q[s, a] += alpha * td

            s = s2
            if done:
                break

        hist.length.append(length)
        hist.steps.append(env._steps)
        hist.success.append(bool(info["goal"]))
        hist.collision.append(bool(info["collision"]))
        hist.ret.append(total)
        hist.epsilon.append(eps)

    return Q, hist
