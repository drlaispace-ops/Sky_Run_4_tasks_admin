"""Golden-тест 34-променевого набору: конструкція за ТЗ, порядок детермінований."""

from __future__ import annotations

import numpy as np

from tier_b.envs.lidar import N_RAYS, RAY_DIRS, ray_dirs_34

S = np.sqrt(2.0) / 2.0


def test_exactly_34_unit_rays():
    d = ray_dirs_34()
    assert d.shape == (34, 3) and N_RAYS == 34
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-12)


def test_no_duplicates():
    d = np.round(ray_dirs_34(), 9)
    assert len({tuple(r) for r in d}) == 34


def test_closed_under_negation():
    """Кожному променю відповідає протилежний — сканер симетричний."""
    d = np.round(ray_dirs_34(), 9)
    dirs_set = {tuple(r) for r in d}
    assert all(tuple(-np.array(r)) in dirs_set for r in dirs_set)


def test_plane_membership_counts():
    d = ray_dirs_34()
    xy = np.sum(np.abs(d[:, 2]) < 1e-9)
    yz = np.sum(np.abs(d[:, 0]) < 1e-9)
    xz = np.sum(np.abs(d[:, 1]) < 1e-9)
    assert xy == 8            # повне коло XOY
    assert yz == 8            # 6 нових + ±y з XOY
    assert xz == 8            # 4 нових + ±x, ±z
    # бісектрисні кола: усі 8 (±½,±½,±s) та всі 8 (±s,±½,±½)
    r = np.round(d, 9)
    dirs_set = {tuple(v) for v in r}
    h, s9 = round(0.5, 9), round(S, 9)
    for sx in (h, -h):
        for sy in (h, -h):
            for sz in (s9, -s9):
                assert (sx, sy, sz) in dirs_set
                assert (sz, sx, sy) in dirs_set


def test_first_rays_are_the_xoy_ring_in_order():
    d = ray_dirs_34()
    assert np.allclose(d[0], [1, 0, 0], atol=1e-12)
    assert np.allclose(d[2], [0, 1, 0], atol=1e-12)
    assert np.allclose(d[4], [-1, 0, 0], atol=1e-12)
    assert np.allclose(d[1], [S, S, 0], atol=1e-12)


def test_module_level_cache_matches_function():
    assert np.array_equal(RAY_DIRS, ray_dirs_34())
