"""INVARIANT 1: reproducibility."""

from __future__ import annotations

import numpy as np
import pytest

from tier_d.env.constants import AB_CLEARANCE, BORDER_MARGIN, DOMAIN, GOAL_XY, MIN_TREE_SEP, START_XY
from tier_d.env.forest import Mulberry32, make_forest

# Golden sequence: pins the RNG so a browser port can be checked against it.
# Regenerating these numbers means the forest changed and every cached L* is stale.
GOLDEN_SEED_0 = [
    0.26642920868471265,
    0.0003297457005828619,
    0.2232720274478197,
    0.1462021479383111,
    0.46732782293111086,
]


def test_mulberry32_golden_sequence():
    rng = Mulberry32(0)
    got = [rng() for _ in range(5)]
    assert got == pytest.approx(GOLDEN_SEED_0, abs=1e-15)


def test_mulberry32_outputs_are_unit_interval():
    rng = Mulberry32(12345)
    vals = [rng() for _ in range(2000)]
    assert all(0.0 <= v < 1.0 for v in vals)
    assert 0.4 < float(np.mean(vals)) < 0.6  # not obviously biased


def test_same_seed_same_forest_bit_for_bit():
    a, b = make_forest(16, 7), make_forest(16, 7)
    assert np.array_equal(a, b)
    assert a.tobytes() == b.tobytes()


def test_different_seeds_differ():
    assert not np.array_equal(make_forest(16, 0), make_forest(16, 1))


@pytest.mark.parametrize("seed", range(8))
def test_forest_respects_all_three_constants(seed):
    trees = make_forest(20, seed)
    A, B = np.asarray(START_XY), np.asarray(GOAL_XY)

    for t in trees:
        assert BORDER_MARGIN - 1e-9 <= t[0] <= DOMAIN - BORDER_MARGIN + 1e-9
        assert BORDER_MARGIN - 1e-9 <= t[1] <= DOMAIN - BORDER_MARGIN + 1e-9
        assert np.hypot(*(t - A)) >= AB_CLEARANCE - 1e-9
        assert np.hypot(*(t - B)) >= AB_CLEARANCE - 1e-9

    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            assert np.hypot(*(trees[i] - trees[j])) >= MIN_TREE_SEP - 1e-9


def test_growing_n_extends_the_same_prefix():
    """make_forest(n) is a prefix of make_forest(n+k): required for L* monotonicity."""
    small, big = make_forest(6, 3), make_forest(14, 3)
    assert np.array_equal(small, big[: len(small)])


def test_dense_request_degrades_deterministically_not_forever():
    a, b = make_forest(400, 3), make_forest(400, 3)
    assert np.array_equal(a, b)
    assert len(a) < 400  # rejection saturates; it must not hang
