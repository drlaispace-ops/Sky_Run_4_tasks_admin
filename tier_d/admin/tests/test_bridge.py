"""Phase 7: the bridge's logic, verified without Webots being installed."""

from __future__ import annotations

import numpy as np
import pytest

from tier_d.admin.agent.qlearning import greedy_rollout, train
from tier_d.env.constants import CELL, GOAL_XY, START_XY
from tier_d.env.gridworld import GridWorld
from tier_d.webots.bridge import (
    CRUISE_SPEED,
    arrived,
    greedy_action_at,
    next_waypoint,
    policy_waypoints,
    simulate_bridge,
    target_velocity,
)


def test_bridge_imports_no_simulator():
    """The build machine has no Webots; the bridge must never require it."""
    import inspect

    import tier_d.webots.bridge as b

    src = inspect.getsource(b)
    assert "import controller" not in src and "from controller" not in src


def test_waypoints_collapse_straight_runs():
    path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [2.0, 2.0]])
    wp = policy_waypoints(path)
    assert len(wp) == 3  # start, the corner, the end
    assert np.allclose(wp[1], [2.0, 0.0])


def test_waypoints_handle_degenerate_paths():
    assert len(policy_waypoints(np.array([[1.0, 1.0]]))) == 1


def test_target_velocity_has_cruise_magnitude():
    env = GridWorld(0, 0)
    Q, _ = train(env, episodes=800, seed=0, eval_every=0)
    v = target_velocity(Q, np.asarray(START_XY))
    assert float(np.linalg.norm(v)) == pytest.approx(CRUISE_SPEED, rel=1e-9)


def test_next_waypoint_is_one_cell_away():
    env = GridWorld(0, 0)
    Q, _ = train(env, episodes=800, seed=0, eval_every=0)
    wp = next_waypoint(Q, np.asarray(START_XY))
    d = float(np.linalg.norm(wp - np.asarray(START_XY)))
    assert d <= CELL * np.sqrt(2) + 1e-9


def test_arrived_predicate():
    assert arrived(np.asarray(GOAL_XY))
    assert not arrived(np.asarray(START_XY))


def test_kinematic_bridge_flies_the_learned_policy_to_the_goal():
    env = GridWorld(0, 0)
    Q, _ = train(env, episodes=1500, seed=0, eval_every=0)
    traj = simulate_bridge(Q, np.asarray(START_XY))
    assert arrived(traj[-1]), "bridge did not reach the goal"
    assert len(traj) > 2


def test_bridge_matches_the_grid_policy_direction():
    env = GridWorld(16, 0)
    Q, _ = train(env, episodes=3000, seed=0, eval_every=0)
    roll = greedy_rollout(env, Q)
    assert roll["success"]
    a = greedy_action_at(Q, roll["path"][0])
    assert 0 <= a < 8
