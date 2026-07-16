"""Tier-A's incremental occupancy belief (env_attitude/mapping.py), built
purely from 2D LIDAR returns -- the input to env_attitude/online_planner.py.
"""

from __future__ import annotations

import numpy as np

from tier_d.env.occupancy import xy_to_cell
from tier_a.env_attitude.lidar2d import scan2d
from tier_a.env_attitude.mapping import BLOCKED, FREE, UNKNOWN, BeliefMap


def test_starts_fully_unknown_and_optimistically_passable():
    belief = BeliefMap()
    assert np.all(belief.state == UNKNOWN)
    assert not np.any(belief.planning_grid())  # nothing blocked yet
    assert belief.explored_fraction() == 0.0


def test_scan_toward_a_tree_marks_a_cell_close_to_its_true_location_blocked():
    """A straight-on ray hitting a tree marks a cell near (not necessarily
    pixel-exact, given the surface-return quantization -- see mapping.py's
    HIT_MARGIN comment) the tree's own true occupancy cell as BLOCKED."""
    tree_xy = (5.0, 5.0)
    origin = np.array([1.0, 5.0])  # due west of the tree, ray 0 (heading 0) points at it
    trees = np.array([tree_xy])
    d = scan2d(origin, 0.0, trees, n_rays=4, max_range=10.0)

    belief = BeliefMap()
    belief.update_from_scan(origin, d, 0.0, n_rays=4, max_range=10.0)

    true_cell = xy_to_cell(*tree_xy)
    blocked_cells = np.argwhere(belief.planning_grid())
    assert len(blocked_cells) > 0
    dists = np.max(np.abs(blocked_cells - np.array(true_cell)), axis=1)
    assert dists.min() <= 1  # Chebyshev distance <= 1 cell


def test_cells_along_a_clear_ray_become_free():
    origin = np.array([1.0, 5.0])
    trees = np.zeros((0, 2))
    d = scan2d(origin, 0.0, trees, n_rays=4, max_range=3.0)  # saturates at max_range

    belief = BeliefMap()
    belief.update_from_scan(origin, d, 0.0, n_rays=4, max_range=3.0)

    # a point 2m along the clear ray (heading 0) should have been marked FREE
    r, c = xy_to_cell(3.0, 5.0)
    assert belief.state[r, c] == FREE


def test_blocked_is_sticky_never_downgraded_by_a_later_free_reading():
    belief = BeliefMap()
    r, c = 12, 12
    belief.state[r, c] = BLOCKED

    tree_xy = (5.0, 5.0)  # cell (12, 12) by construction of CELL/GRID_N
    origin = np.array([1.0, 5.0])
    trees = np.zeros((0, 2))  # this scan sees nothing -- no real tree
    d = scan2d(origin, 0.0, trees, n_rays=4, max_range=10.0)
    belief.update_from_scan(origin, d, 0.0, n_rays=4, max_range=10.0)

    assert belief.state[r, c] == BLOCKED


def test_explored_fraction_increases_after_a_scan():
    belief = BeliefMap()
    trees = np.zeros((0, 2))
    d = scan2d(np.array([5.0, 5.0]), 0.0, trees, n_rays=8, max_range=3.0)
    belief.update_from_scan(np.array([5.0, 5.0]), d, 0.0, n_rays=8, max_range=3.0)
    assert belief.explored_fraction() > 0.0
