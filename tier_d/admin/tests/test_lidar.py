"""Analytic LIDAR: closed-form ray/circle, no physics engine."""

from __future__ import annotations

import numpy as np
import pytest

from tier_d.env.constants import DOMAIN, LIDAR_RANGE, TREE_R
from tier_d.env.forest import make_forest
from tier_d.env.lidar import endpoints, ray_dirs, scan, scan_path


def test_ray_directions_are_unit_and_evenly_spaced():
    d = ray_dirs(16)
    assert d.shape == (16, 2)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    assert d[0] == pytest.approx([1.0, 0.0])


def test_tree_directly_east_is_seen_at_its_surface():
    """A tree centred 2.0 east of the drone is reported at 2.0 - TREE_R."""
    trees = np.array([[7.0, 5.0]])
    d = scan(np.array([5.0, 5.0]), trees, n_rays=4, max_range=5.0)
    assert d[0] == pytest.approx(2.0 - TREE_R, abs=1e-6)  # ray 0 points +x


def test_empty_forest_reports_only_walls():
    d = scan(np.array([5.0, 5.0]), np.zeros((0, 2)), n_rays=4, max_range=10.0)
    assert np.allclose(d, 5.0)  # 5.0 to each wall from the centre


def test_no_hit_saturates_at_max_range():
    d = scan(np.array([5.0, 5.0]), np.zeros((0, 2)), n_rays=8, max_range=1.0)
    assert np.allclose(d, 1.0)


def test_reading_never_exceeds_range_and_is_positive():
    trees = make_forest(16, 0)
    d = scan(np.array([1.0, 1.0]), trees)
    assert np.all(d > 0.0) and np.all(d <= LIDAR_RANGE + 1e-9)


def test_lidar_never_sees_through_a_tree_it_stands_beside():
    """Reading toward a tree is never longer than the distance to its surface."""
    trees = np.array([[3.0, 1.0]])
    origin = np.array([1.0, 1.0])
    d = scan(origin, trees, n_rays=4, max_range=5.0)
    surface = np.linalg.norm(trees[0] - origin) - TREE_R
    assert d[0] <= surface + 1e-9


def test_walls_bound_every_ray_from_a_corner():
    d = scan(np.array([0.1, 0.1]), np.zeros((0, 2)), n_rays=8, max_range=DOMAIN)
    assert np.all(d <= DOMAIN * np.sqrt(2))
    assert d[4] == pytest.approx(0.1, abs=1e-9)  # ray 4 points -x, wall at x=0


def test_endpoints_lie_at_the_reported_distance():
    trees = make_forest(8, 1)
    o = np.array([5.0, 5.0])
    d = scan(o, trees)
    e = endpoints(o, d)
    assert np.allclose(np.linalg.norm(e - o, axis=1), d)


def test_scan_path_shape():
    trees = make_forest(8, 1)
    path = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    s = scan_path(path, trees)
    assert s.shape == (3, 16)
