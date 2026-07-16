"""Potential-based reward shaping (Ng, Harada & Russell 1999).

INVARIANT 4. The shaping term is exactly

    F(s, s') = gamma * Phi(s') - Phi(s),      Phi(s) = -dist(cell(s), goal)

which the policy-invariance theorem guarantees leaves the optimal policy
unchanged. It only speeds learning; it cannot bend the task.

The theorem's episodic form requires ``Phi(terminal) = 0``. That is not
cosmetic: with a terminal collision state the agent never leaves, a non-zero
potential there would leak a bias into the return and the guarantee is void.
Both terminals are handled by ``shaping_F(..., terminal=True)``.

Note the goal already satisfies this naturally (``dist == 0``), so only the
collision terminal actually needs the special case.
"""

from __future__ import annotations

import math

from tier_d.env.constants import CELL, GRID_N
from tier_d.env.occupancy import GOAL_CELL


def potential(state: int) -> float:
    """Phi(s) = -Euclidean distance (world units) from cell(s) to the goal."""
    r, c = divmod(state, GRID_N)
    gr, gc = GOAL_CELL
    return -math.hypot(r - gr, c - gc) * CELL


def shaping_F(state: int, next_state: int, gamma: float, terminal: bool = False) -> float:
    """gamma*Phi(s') - Phi(s), with Phi(s')=0 when s' is a true terminal.

    ``terminal`` means goal or collision -- NOT a time-limit truncation, which
    is an artefact of the training loop rather than a property of the MDP.
    """
    phi_next = 0.0 if terminal else potential(next_state)
    return gamma * phi_next - potential(state)
