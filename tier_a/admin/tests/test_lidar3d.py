"""Tier-A 3D LIDAR checks: body-frame rotation and ray/cylinder geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tier_d.env.constants import TREE_R
from tier_a.env_attitude.lidar3d import (
    LIDAR_RANGE_3D,
    N_CIRCLES_DENSE,
    N_RAYS_3D,
    N_RAYS_3D_DENSE,
    N_RAYS_3D_SPHERICAL,
    RAYS_PER_CIRCLE_DENSE,
    body_bearings,
    body_ray_dirs_dense,
    body_ray_dirs_spherical,
    endpoints3d,
    endpoints3d_dense,
    endpoints3d_spherical,
    scan3d,
    scan3d_dense,
    scan3d_spherical,
)


def test_no_trees_returns_max_range_on_every_ray():
    d = scan3d((5.0, 5.0, 1.6), 0.0, 0.0, 0.0, np.zeros((0, 2)))
    assert np.allclose(d, LIDAR_RANGE_3D)


def test_forward_ray_hits_a_tree_directly_ahead():
    trees = np.array([[5.0, 5.0]])
    d = scan3d((3.0, 5.0, 1.6), 0.0, 0.0, 0.0, trees)
    assert d[0] == pytest.approx(2.0 - TREE_R, abs=1e-6)


def test_yaw_rotates_the_whole_ray_ring_with_the_body():
    """A 90deg yaw should turn the forward ray away from a tree that used to
    be directly ahead -- the sensor is body-fixed, not world-fixed."""
    trees = np.array([[5.0, 5.0]])
    d = scan3d((3.0, 5.0, 1.6), 0.0, 0.0, math.pi / 2, trees)
    assert d[0] == pytest.approx(LIDAR_RANGE_3D)


def test_tree_above_trunk_height_is_not_hit():
    trees = np.array([[5.0, 5.0]])
    d = scan3d((3.0, 5.0, 10.0), 0.0, 0.0, 0.0, trees)
    assert d[0] == pytest.approx(LIDAR_RANGE_3D)


def test_body_bearings_cover_a_full_ring_starting_at_zero():
    b = body_bearings(N_RAYS_3D)
    assert len(b) == N_RAYS_3D
    assert b[0] == pytest.approx(0.0)
    assert b[-1] == pytest.approx(2 * math.pi * (N_RAYS_3D - 1) / N_RAYS_3D)


def test_endpoints_are_consistent_with_scan_distances():
    trees = np.array([[5.0, 5.0]])
    origin = np.array([3.0, 5.0, 1.6])
    d = scan3d(origin, 0.0, 0.0, 0.0, trees)
    ends = endpoints3d(origin, 0.0, 0.0, 0.0, d)
    assert ends.shape == (N_RAYS_3D, 3)
    dist_back = np.linalg.norm(ends - origin, axis=1)
    assert np.allclose(dist_back, d)


def test_scan_never_exceeds_max_range():
    rng = np.random.default_rng(0)
    trees = rng.uniform(0, 10, size=(6, 2))
    for _ in range(5):
        origin = rng.uniform(1, 9, size=2).tolist() + [1.6]
        phi, theta, psi = rng.uniform(-0.3, 0.3, size=3)
        d = scan3d(origin, phi, theta, psi, trees)
        assert np.all(d <= LIDAR_RANGE_3D + 1e-9)
        assert np.all(d > 0.0)


# --- spherical variant: 3 great circles (XY, XZ, YZ), not one flat ring ----
def test_spherical_ray_dirs_are_unit_and_split_across_three_circles():
    dirs = body_ray_dirs_spherical(16)
    assert dirs.shape == (16, 3)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)


def test_spherical_ray_dirs_include_out_of_plane_rays():
    """The whole point versus body_ray_dirs: some rays must have a nonzero
    z-component, i.e. actually point above/below the horizontal plane."""
    dirs = body_ray_dirs_spherical(16)
    assert np.any(np.abs(dirs[:, 2]) > 0.5)


def test_spherical_scan_matches_plain_scan_on_the_horizontal_ring_portion():
    """Both variants use the same ray-vs-cylinder solve; a tree directly
    ahead on the horizontal ring must read identically either way."""
    trees = np.array([[5.0, 5.0]])
    d_flat = scan3d((3.0, 5.0, 1.6), 0.0, 0.0, 0.0, trees)
    d_sph = scan3d_spherical((3.0, 5.0, 1.6), 0.0, 0.0, 0.0, trees)
    assert d_flat[0] == pytest.approx(d_sph[0])  # both rays 0 point body-forward


def test_spherical_scan_never_exceeds_max_range_and_default_count_is_16():
    assert N_RAYS_3D_SPHERICAL == 16
    trees = np.array([[5.0, 5.0], [4.0, 6.0]])
    d = scan3d_spherical((3.0, 5.0, 1.6), 0.1, -0.1, 0.4, trees)
    assert len(d) == 16
    assert np.all(d <= LIDAR_RANGE_3D + 1e-9)
    assert np.all(d > 0.0)


def test_spherical_endpoints_are_consistent_with_scan_distances():
    trees = np.array([[5.0, 5.0]])
    origin = np.array([3.0, 5.0, 1.6])
    d = scan3d_spherical(origin, 0.0, 0.0, 0.0, trees)
    ends = endpoints3d_spherical(origin, 0.0, 0.0, 0.0, d)
    assert ends.shape == (N_RAYS_3D_SPHERICAL, 3)
    assert np.allclose(np.linalg.norm(ends - origin, axis=1), d)


# --- dense variant: 8 great circles x 8 rays = 64 --------------------------
def test_dense_defaults_are_8_circles_of_8_rays_64_total():
    assert N_CIRCLES_DENSE == 8
    assert RAYS_PER_CIRCLE_DENSE == 8
    assert N_RAYS_3D_DENSE == 64
    dirs = body_ray_dirs_dense()
    assert dirs.shape == (64, 3)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)


def test_dense_circle_zero_matches_the_xz_great_circle_exactly():
    """k=0 (beta=0) must recover exactly the same XZ circle
    body_ray_dirs_spherical uses -- no drift between the two designs."""
    dirs = body_ray_dirs_dense(n_circles=8, rays_per_circle=8)
    k0 = dirs[0:8]
    a = 2.0 * np.pi * np.arange(8) / 8
    expected_xz = np.stack([np.cos(a), np.zeros(8), np.sin(a)], axis=1)
    assert np.allclose(k0, expected_xz)


def test_dense_circle_at_beta_pi_over_2_matches_the_horizontal_ring():
    """k=4 of 8 circles (beta=pi/2) must recover the same XY (horizontal)
    ring as body_ray_dirs_spherical's, up to the y-sign convention baked
    into this construction's rotation direction (same set of directions)."""
    dirs = body_ray_dirs_dense(n_circles=8, rays_per_circle=8)
    k4 = dirs[4 * 8:5 * 8]
    a = 2.0 * np.pi * np.arange(8) / 8
    expected_xy_signed = np.stack([np.cos(a), -np.sin(a), np.zeros(8)], axis=1)
    assert np.allclose(k4, expected_xy_signed)


def test_dense_scan_matches_plain_scan_on_the_forward_ray():
    trees = np.array([[5.0, 5.0]])
    d_flat = scan3d((3.0, 5.0, 1.6), 0.0, 0.0, 0.0, trees)
    d_dense = scan3d_dense((3.0, 5.0, 1.6), 0.0, 0.0, 0.0, trees)
    assert d_dense[0] == pytest.approx(d_flat[0])  # ray 0 of every circle is body-forward


def test_dense_scan_never_exceeds_max_range():
    rng = np.random.default_rng(1)
    trees = rng.uniform(0, 10, size=(6, 2))
    for _ in range(4):
        origin = rng.uniform(1, 9, size=2).tolist() + [1.6]
        phi, theta, psi = rng.uniform(-0.4, 0.4, size=3)
        d = scan3d_dense(origin, phi, theta, psi, trees)
        assert len(d) == N_RAYS_3D_DENSE
        assert np.all(d <= LIDAR_RANGE_3D + 1e-9)
        assert np.all(d > 0.0)


def test_dense_endpoints_are_consistent_with_scan_distances():
    trees = np.array([[5.0, 5.0]])
    origin = np.array([3.0, 5.0, 1.6])
    d = scan3d_dense(origin, 0.0, 0.0, 0.0, trees)
    ends = endpoints3d_dense(origin, 0.0, 0.0, 0.0, d)
    assert ends.shape == (N_RAYS_3D_DENSE, 3)
    assert np.allclose(np.linalg.norm(ends - origin, axis=1), d)


def test_dense_ray_count_is_configurable_independent_of_the_default():
    dirs = body_ray_dirs_dense(n_circles=4, rays_per_circle=6)
    assert dirs.shape == (24, 3)
