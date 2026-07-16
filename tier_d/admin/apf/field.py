"""Classical Artificial Potential Field (Khatib 1986) and -grad U flight.

This is the Tier 0 "Field Scanner" asset and, deliberately, the *foil* for the
RL track. APF hand-writes the potential surface; Q-learning grows it from
experience. APF's known defect (Koren & Borenstein 1991) is that the sum of an
attractive and several repulsive fields can have spurious local minima where
the drone stalls short of the goal.

Concept thesis #2 says the optimal RL value function is a navigation function
and has no such trap. ``tier_d/apf/compare.py`` demonstrates exactly that, on a seed
found by search rather than by assertion.
"""

from __future__ import annotations

import numpy as np

from tier_d.env.constants import DOMAIN, GOAL_XY, START_XY, TREE_R

# Gains chosen by sweep (see tier_d/admin/reports/apf_gain_sweep.md): this is a *fair*
# classical APF -- it reaches the goal on ~2/3 of feasible forests and never
# collides. A stronger K_REP would trap it everywhere and make the RL
# comparison dishonest.
K_ATT: float = 1.0
K_REP: float = 0.5
RHO_0: float = 0.6  # repulsion cuts off beyond this distance from a tree surface
D_STAR: float = 4.0  # attractive field goes conic beyond this (bounds the pull)


def attractive_potential(xy: np.ndarray, goal: np.ndarray = np.asarray(GOAL_XY)) -> np.ndarray:
    """Quadratic near the goal, conic far from it. Shape (...,)."""
    d = np.linalg.norm(xy - goal, axis=-1)
    quad = 0.5 * K_ATT * d**2
    cone = D_STAR * K_ATT * d - 0.5 * K_ATT * D_STAR**2
    return np.where(d <= D_STAR, quad, cone)


def repulsive_potential(xy: np.ndarray, trees: np.ndarray) -> np.ndarray:
    """Sum of per-tree barriers, active only within RHO_0 of the tree surface."""
    out = np.zeros(xy.shape[:-1])
    if len(trees) == 0:
        return out
    for t in trees:
        rho = np.linalg.norm(xy - t, axis=-1) - TREE_R
        rho = np.maximum(rho, 1e-3)
        active = rho < RHO_0
        out = out + np.where(active, 0.5 * K_REP * (1.0 / rho - 1.0 / RHO_0) ** 2, 0.0)
    return out


def total_potential(xy: np.ndarray, trees: np.ndarray) -> np.ndarray:
    """U = U_att + U_rep. This is the surface the Field Scanner draws."""
    return attractive_potential(xy) + repulsive_potential(xy, trees)


def grad_U(xy: np.ndarray, trees: np.ndarray) -> np.ndarray:
    """Analytic gradient of ``total_potential`` at a single point. Shape (2,)."""
    goal = np.asarray(GOAL_XY)
    d_vec = xy - goal
    d = float(np.linalg.norm(d_vec))
    if d < 1e-9:
        g = np.zeros(2)
    elif d <= D_STAR:
        g = K_ATT * d_vec
    else:
        g = D_STAR * K_ATT * d_vec / d

    for t in trees:
        diff = xy - t
        dist = float(np.linalg.norm(diff))
        rho = max(dist - TREE_R, 1e-3)
        if rho < RHO_0:
            mag = K_REP * (1.0 / rho - 1.0 / RHO_0) * (1.0 / rho**2)
            g = g - mag * diff / max(dist, 1e-9)
    return g


def apf_flight(
    trees: np.ndarray,
    step: float = 0.05,
    max_steps: int = 4000,
    goal_tol: float = 0.3,
    stall_window: int = 200,
    stall_eps: float = 1e-3,
) -> dict:
    """Gradient descent on U from A. Returns path, outcome and stall diagnosis.

    ``trapped`` means the walker stopped moving without reaching the goal --
    a local minimum of U. ``collided`` means it was pushed inside a tree, which
    a correct repulsive field should never allow.
    """
    x = np.asarray(START_XY, dtype=float)
    goal = np.asarray(GOAL_XY)
    path = [x.copy()]
    length = 0.0

    for i in range(max_steps):
        g = grad_U(x, trees)
        gn = float(np.linalg.norm(g))
        if gn < 1e-12:
            break
        nx = x - step * g / gn  # unit-speed descent; magnitude only sets direction
        nx = np.clip(nx, 0.0, DOMAIN)
        length += float(np.linalg.norm(nx - x))
        x = nx
        path.append(x.copy())

        if float(np.linalg.norm(x - goal)) < goal_tol:
            return {
                "success": True,
                "trapped": False,
                "collided": False,
                "length": length,
                "path": np.array(path),
                "stall_xy": None,
            }

        if len(path) > stall_window:
            recent = np.array(path[-stall_window:])
            if float(np.linalg.norm(recent[-1] - recent[0])) < stall_eps * stall_window:
                collided = _inside_tree(x, trees)
                return {
                    "success": False,
                    "trapped": not collided,
                    "collided": collided,
                    "length": length,
                    "path": np.array(path),
                    "stall_xy": x.copy(),
                }

    return {
        "success": False,
        "trapped": True,
        "collided": _inside_tree(x, trees),
        "length": length,
        "path": np.array(path),
        "stall_xy": x.copy(),
    }


def _inside_tree(xy: np.ndarray, trees: np.ndarray) -> bool:
    if len(trees) == 0:
        return False
    return bool(np.any(np.linalg.norm(trees - xy, axis=1) < TREE_R))


def potential_grid(trees: np.ndarray, n: int = 200, clip: float = 60.0) -> tuple[np.ndarray, ...]:
    """(X, Y, U) for surface plots. U is clipped -- the barriers go to infinity."""
    ax = np.linspace(0.0, DOMAIN, n)
    X, Y = np.meshgrid(ax, ax)
    P = np.stack([X, Y], axis=-1)
    U = total_potential(P, trees)
    return X, Y, np.minimum(U, clip)
