"""env_attitude/online_planner.py: periodic-replan A* over a LIDAR-built belief."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tier_d.env.occupancy import GOAL_CELL, START_CELL, cell_center, xy_to_cell
from tier_d.env.constants import GOAL_XY, START_XY
from tier_a.env_attitude.mapping import BeliefMap
from tier_a.env_attitude.online_planner import OnlinePlanner, PlannerConfig


def test_empty_belief_yields_a_plan_to_the_goal():
    """Fully-UNKNOWN belief -> optimistic-unknown planning grid -> A* finds
    *some* path straight through, since nothing is known BLOCKED yet."""
    belief = BeliefMap()
    planner = OnlinePlanner(goal_xy=GOAL_XY, config=PlannerConfig(replan_every_steps=1))
    waypoints = planner.maybe_replan(belief, START_XY, psi=0.0)
    assert waypoints is not None
    assert len(waypoints) >= 2
    # last waypoint should be at (or very near) the goal cell centre
    assert np.allclose(waypoints[-1], cell_center(*GOAL_CELL), atol=1e-9)


def test_first_two_waypoints_are_anchored_to_current_pose():
    belief = BeliefMap()
    planner = OnlinePlanner(goal_xy=GOAL_XY, config=PlannerConfig(replan_every_steps=1, lead_in=0.3))
    xy = (2.0, 2.0)
    psi = math.radians(45.0)
    waypoints = planner.maybe_replan(belief, xy, psi)
    assert waypoints is not None
    assert np.allclose(waypoints[0], xy)
    expected_lead_in = (2.0 + 0.3 * math.cos(psi), 2.0 + 0.3 * math.sin(psi))
    assert np.allclose(waypoints[1], expected_lead_in)


def test_does_not_replan_before_the_periodic_interval_but_does_after():
    belief = BeliefMap()
    n = 5
    planner = OnlinePlanner(goal_xy=GOAL_XY, config=PlannerConfig(replan_every_steps=n))
    assert planner.maybe_replan(belief, START_XY, 0.0) is not None  # always plans once, immediately

    fired = [planner.maybe_replan(belief, START_XY, 0.0) is not None for _ in range(n + 1)]
    assert fired[:-1] == [False] * n  # nothing fires again until the interval elapses
    assert fired[-1] is True  # ...then the periodic timer fires


def test_replans_early_when_a_known_obstacle_falls_on_the_current_path():
    belief = BeliefMap()
    planner = OnlinePlanner(goal_xy=GOAL_XY, config=PlannerConfig(replan_every_steps=1000, lookahead_cells=6))
    waypoints = planner.maybe_replan(belief, START_XY, 0.0)
    assert waypoints is not None
    r, c = planner._last_path_cells[min(2, len(planner._last_path_cells) - 1)]
    belief.state[r, c] = 2  # BLOCKED -- simulate a newly-discovered tree on-path

    assert planner.maybe_replan(belief, START_XY, 0.0) is not None  # fires early, not at step 1000


def test_believed_blocked_start_or_goal_returns_none_instead_of_crashing():
    belief = BeliefMap()
    start_cell = xy_to_cell(*START_XY)
    belief.state[start_cell] = 2  # BLOCKED
    planner = OnlinePlanner(goal_xy=GOAL_XY, config=PlannerConfig(replan_every_steps=1))
    assert planner.maybe_replan(belief, START_XY, 0.0) is None


def test_a_discovered_obstacle_forces_a_longer_detouring_path():
    """Blocking the straight-line corridor between A and B must make the
    planned path longer than the empty-belief plan -- proof the planner is
    actually routing around what it believes, not ignoring it."""
    belief_empty = BeliefMap()
    planner_a = OnlinePlanner(goal_xy=GOAL_XY, config=PlannerConfig(replan_every_steps=1))
    wp_empty = planner_a.maybe_replan(belief_empty, START_XY, 0.0)
    len_empty = float(np.sum(np.hypot(*np.diff(wp_empty, axis=0).T)))

    belief_blocked = BeliefMap()
    # Block a band of cells straddling the direct diagonal from A to B.
    mid_r, mid_c = (START_CELL[0] + GOAL_CELL[0]) // 2, (START_CELL[1] + GOAL_CELL[1]) // 2
    for dr in range(-1, 2):
        for dc in range(-4, 5):
            r, c = mid_r + dr, mid_c + dc
            if 0 <= r < belief_blocked.state.shape[0] and 0 <= c < belief_blocked.state.shape[1]:
                if (r, c) not in (START_CELL, GOAL_CELL):
                    belief_blocked.state[r, c] = 2  # BLOCKED
    planner_b = OnlinePlanner(goal_xy=GOAL_XY, config=PlannerConfig(replan_every_steps=1))
    wp_blocked = planner_b.maybe_replan(belief_blocked, START_XY, 0.0)
    assert wp_blocked is not None
    len_blocked = float(np.sum(np.hypot(*np.diff(wp_blocked, axis=0).T)))

    assert len_blocked > len_empty
