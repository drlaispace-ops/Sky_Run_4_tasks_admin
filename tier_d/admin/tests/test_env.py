"""Phase 1 checks: the grid MDP."""

from __future__ import annotations

import numpy as np
import pytest

from tier_d.env.constants import ACTIONS, GRID_N, N_ACTIONS, R_COLLISION, R_GOAL, R_STEP
from tier_d.env.gridworld import GridWorld
from tier_d.env.occupancy import GOAL_CELL, START_CELL, occupancy_grid, xy_to_cell


def test_start_and_goal_land_on_cell_centres():
    assert START_CELL == (2, 2)
    assert GOAL_CELL == (22, 22)
    assert xy_to_cell(1.0, 1.0) == START_CELL
    assert xy_to_cell(9.0, 9.0) == GOAL_CELL


def test_reset_returns_start_and_clears_steps():
    env = GridWorld(16, 0)
    env.step(7)
    s = env.reset()
    assert s == env.encode(START_CELL)
    assert env._steps == 0


def test_wall_of_domain_keeps_agent_in_bounds():
    env = GridWorld(0, 0)
    env.reset()
    env._rc = (0, 0)
    for a, (dr, dc, _) in enumerate(ACTIONS):
        env._rc = (0, 0)
        s, r, done, info = env.step(a)
        rr, cc = env.decode(s)
        assert 0 <= rr < GRID_N and 0 <= cc < GRID_N
        if dr < 0 or dc < 0:  # move would leave the domain
            assert (rr, cc) == (0, 0)
            assert r == R_STEP
            assert not info["collision"]


def test_goal_is_terminal_and_pays_r_goal():
    env = GridWorld(0, 0)
    env.reset()
    env._rc = (21, 21)
    _, r, done, info = env.step(ACTIONS.index((1, 1, ACTIONS[7][2])))
    assert info["goal"] and done and r == R_GOAL


def test_entering_an_occupied_cell_is_a_terminal_collision():
    env = GridWorld(16, 0)
    env.reset()
    blocked = np.argwhere(env.blocked)
    r, c = blocked[0]
    # Stand orthogonally adjacent and walk into it.
    env._rc = (int(r), int(c) - 1)
    _, rew, done, info = env.step(ACTIONS.index((0, 1, ACTIONS[3][2])))
    assert info["collision"] and done and rew == R_COLLISION


@pytest.mark.parametrize("seed", range(6))
def test_random_agent_never_occupies_a_blocked_cell(seed):
    """A step either collides (terminal) or lands on a free cell. Never both, never neither."""
    env = GridWorld(16, seed)
    rng = np.random.default_rng(seed)
    for _ in range(40):
        s = env.reset()
        for _ in range(200):
            s, _, done, info = env.step(int(rng.integers(N_ACTIONS)))
            rc = env.decode(s)
            assert not env.blocked[rc], "agent stood inside an obstacle"
            if done:
                break


def test_collision_and_truncation_are_distinct():
    env = GridWorld(0, 0, max_steps=3)
    env.reset()
    for _ in range(3):
        _, _, done, info = env.step(0)  # walk north into the wall repeatedly
    assert done and info["truncated"] and not info["collision"] and not info["goal"]


def test_state_space_is_exactly_the_grid():
    """INVARIANT 2, structural: no obstacle information is encoded in the state."""
    for n in (0, 8, 16, 24):
        env = GridWorld(n, 1)
        assert env.n_states == GRID_N * GRID_N
        assert env.n_actions == N_ACTIONS


def test_occupancy_is_empty_without_trees():
    assert not occupancy_grid(np.zeros((0, 2))).any()
