"""Еліпсоїд vs циліндр: якорі, крос-чек з env_attitude, консервативність, точність глибини."""

from __future__ import annotations

import numpy as np
import pytest

from tier_b.envs.collision_ellipsoid import SEMI_AXES, check_forest, shadow_matrix

A, B, C = SEMI_AXES


def rot_x(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _support_h(rot, n):
    m = shadow_matrix(rot)
    n = np.asarray(n, dtype=np.float64)
    return float(np.sqrt(n @ m @ n))


def test_level_flight_shadow_is_circle_of_radius_a():
    m = shadow_matrix(np.eye(3))
    assert np.allclose(m, np.diag([A**2, B**2]))
    for ang in np.linspace(0, 2 * np.pi, 17):
        assert _support_h(np.eye(3), [np.cos(ang), np.sin(ang)]) == pytest.approx(A)


def test_pure_roll_matches_env_attitude_presented_half_width():
    """Той самий вираз, що в env_attitude/occupancy.py — пряма порівнюваність
    з Tier-A+ (там R_BODY/H_BODY інші, тому передаємо наші півосі явно)."""
    from tier_a.env_attitude.occupancy import presented_half_width

    for phi in np.linspace(0, np.pi / 2, 19):
        h = _support_h(rot_x(phi), [0.0, 1.0])
        assert h == pytest.approx(presented_half_width(phi, A, C), abs=1e-12)


def test_knife_edge_presents_c_level_presents_a():
    assert _support_h(rot_x(0.0), [0, 1]) == pytest.approx(A)
    assert _support_h(rot_x(np.pi / 2), [0, 1]) == pytest.approx(C)


def test_far_tree_no_collision_close_tree_collides():
    trees = np.array([[1.0, 0.0, 0.2]])
    hit, depth = check_forest([0, 0, 1.0], np.eye(3), trees)
    assert not hit and depth == 0.0
    hit, depth = check_forest([0.75, 0.0, 1.0], np.eye(3), trees)
    assert hit and depth == pytest.approx(0.2 + A - 0.25)


def test_above_tree_top_is_free():
    trees = np.array([[0.0, 0.0, 0.2]])
    hit, _ = check_forest([0.0, 0.0, 4.2], np.eye(3), trees, tree_height=4.0)
    assert not hit


def test_roll_opens_the_gap():
    """Фізичний сенс усього треку: щілина, непрохідна рівно, прохідна у крені."""
    gap = 0.16  # L4
    r = 0.2
    trees = np.array([[3.0, -(gap / 2 + r), r], [3.0, gap / 2 + r, r]])
    hit_level, _ = check_forest([3.0, 0.0, 1.0], np.eye(3), trees)
    hit_knife, _ = check_forest([3.0, 0.0, 1.0], rot_x(np.radians(75)), trees)
    assert hit_level and not hit_knife


def _sample_surface(pos, rot, n=4000):
    """Fibonacci-сфера → поверхня еліпсоїда у світових координатах."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    u = np.stack([np.sin(phi) * np.cos(theta),
                  np.sin(phi) * np.sin(theta),
                  np.cos(phi)], axis=1)
    return np.asarray(pos) + (rot @ (np.diag(SEMI_AXES) @ u.T)).T


def test_conservative_never_reports_free_on_sampled_contact():
    rng = np.random.default_rng(0)
    for _ in range(60):
        phi, theta = rng.uniform(0, np.pi / 2), rng.uniform(-0.4, 0.4)
        c1, s1 = np.cos(phi), np.sin(phi)
        c2, s2 = np.cos(theta), np.sin(theta)
        rot = rot_x(phi) @ np.array([[c2, 0, s2], [0, 1, 0], [-s2, 0, c2]])
        pos = np.array([0.0, rng.uniform(0.15, 0.45), 1.0])
        trees = np.array([[0.0, 0.0, 0.2]])
        pts = _sample_surface(pos, rot)
        sampled_hit = bool(np.any(np.hypot(pts[:, 0], pts[:, 1]) <= 0.2))
        analytic_hit, _ = check_forest(pos, rot, trees)
        if sampled_hit:
            assert analytic_hit  # консервативність: жодних хибних "вільно"


def test_depth_matches_dense_boundary_sampling_within_5mm():
    rng = np.random.default_rng(1)
    theta_grid = np.linspace(0, 2 * np.pi, 20000, endpoint=False)
    circle = np.stack([np.cos(theta_grid), np.sin(theta_grid)], axis=1)
    for _ in range(40):
        phi = rng.uniform(0.2, 1.4)
        rot = rot_x(phi)
        h_true_dir = _support_h(rot, [0, 1])
        # ставимо дерево так, щоб перекриття було неглибоке (0..2 см)
        overlap = rng.uniform(0.0, 0.02)
        pos = np.array([0.0, 0.0, 1.0])
        tree_y = 0.2 + h_true_dir - overlap
        trees = np.array([[0.0, tree_y, 0.2]])
        _, depth = check_forest(pos, rot, trees)
        # «істинна» глибина через щільну дискретизацію межі тіні
        m = shadow_matrix(rot)
        w, v = np.linalg.eigh((m + m.T) / 2)
        sqrt_m = v @ np.diag(np.sqrt(np.maximum(w, 0.0))) @ v.T
        bnd = pos[:2] + circle @ sqrt_m.T
        d_min = float(np.min(np.linalg.norm(bnd - np.array([0.0, tree_y]), axis=1)))
        depth_true = max(0.0, 0.2 - d_min)
        assert depth == pytest.approx(depth_true, abs=5e-3)
        assert depth >= depth_true - 1e-9  # консервативність і по глибині
