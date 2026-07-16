"""Tier-A's 2D, yaw-rotating LIDAR (env_attitude/lidar2d.py) -- the sensor
feeding the online occupancy belief (env_attitude/mapping.py). Mirrors
tests/test_lidar.py's structure for the base kit's world-frame-fixed sensor,
with the added yaw-rotation checks this module needs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tier_d.env.constants import DOMAIN, LIDAR_RANGE, TREE_R
from tier_d.env.forest import make_forest
from tier_a.env_attitude.lidar2d import endpoints2d, ray_dirs2d, scan2d


def test_ray_directions_are_unit_and_evenly_spaced_at_zero_yaw():
    d = ray_dirs2d(0.0, 16)
    assert d.shape == (16, 2)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    assert d[0] == pytest.approx([1.0, 0.0])


def test_ray_directions_rotate_rigidly_with_yaw():
    """Every ray direction rotates by exactly psi -- yaw-only, not attitude-coupled."""
    psi = math.radians(37.0)
    base = ray_dirs2d(0.0, 8)
    rotated = ray_dirs2d(psi, 8)
    c, s = math.cos(psi), math.sin(psi)
    R = np.array([[c, -s], [s, c]])
    expected = base @ R.T
    assert np.allclose(rotated, expected, atol=1e-9)


def test_tree_directly_ahead_is_seen_at_its_surface():
    """A tree 2.0m along the current heading is reported at 2.0 - TREE_R,
    regardless of which way that heading points."""
    psi = math.radians(90.0)  # heading +y
    origin = np.array([5.0, 5.0])
    trees = np.array([[5.0, 7.0]])  # 2.0 north
    d = scan2d(origin, psi, trees, n_rays=4, max_range=5.0)
    assert d[0] == pytest.approx(2.0 - TREE_R, abs=1e-6)  # ray 0 points along heading


def test_empty_forest_reports_only_walls():
    d = scan2d(np.array([5.0, 5.0]), 0.0, np.zeros((0, 2)), n_rays=4, max_range=10.0)
    assert np.allclose(d, 5.0)


def test_no_hit_saturates_at_max_range():
    d = scan2d(np.array([5.0, 5.0]), 0.3, np.zeros((0, 2)), n_rays=8, max_range=1.0)
    assert np.allclose(d, 1.0)


def test_reading_never_exceeds_range_and_is_positive():
    trees = make_forest(8, 0)
    d = scan2d(np.array([1.0, 1.0]), 0.5, trees)
    assert np.all(d > 0.0) and np.all(d <= LIDAR_RANGE + 1e-9)


def test_lidar_never_sees_through_a_tree_it_stands_beside():
    trees = np.array([[3.0, 1.0]])
    origin = np.array([1.0, 1.0])
    d = scan2d(origin, 0.0, trees, n_rays=4, max_range=5.0)
    surface = np.linalg.norm(trees[0] - origin) - TREE_R
    assert d[0] <= surface + 1e-9


def test_walls_bound_every_ray_from_a_corner():
    d = scan2d(np.array([0.1, 0.1]), 0.0, np.zeros((0, 2)), n_rays=8, max_range=DOMAIN)
    assert np.all(d <= DOMAIN * math.sqrt(2))


def test_endpoints_lie_at_the_reported_distance():
    trees = make_forest(8, 1)
    o = np.array([5.0, 5.0])
    psi = 1.1
    d = scan2d(o, psi, trees)
    e = endpoints2d(o, psi, d)
    assert np.allclose(np.linalg.norm(e - o, axis=1), d)
