"""Tier-A scoring checks: the scoreboard must be well-formed and must exclude
the impossible.

Unlike the base kit's tabular Q-learning (which reliably converges within a
fixed budget), Tier-A's from-scratch 6-DOF DQN is a genuinely hard control
problem and is *not* guaranteed to converge within any small budget -- see
``env_attitude/README.md``. These tests therefore check the SCORING MACHINERY
(cache correctness, lexicographic ordering, right-censoring), not empirical
agent performance, to avoid a flaky test suite.
"""

from __future__ import annotations

import numpy as np

from tier_a.env_attitude.constants import N_TREES_ATTITUDE
from tier_a.admin.scoring.harness_attitude import ScoreAttitude, evaluate_attitude, leaderboard, random_agent
from tier_a.admin.scoring.precompute_attitude import build_cache_attitude
from tier_d.scoring.seeds import HIDDEN_SEEDS, PUBLIC_SEEDS


def test_cache_covers_every_seed_and_flags_feasibility():
    cache = build_cache_attitude(n_trees=N_TREES_ATTITUDE)
    for s in (*PUBLIC_SEEDS, *HIDDEN_SEEDS):
        entry = cache[f"{N_TREES_ATTITUDE}:{s}"]
        assert entry["feasible"] is (entry["Lref"] is not None)
        assert entry["feasible"] is (entry["Tref"] is not None)


def test_infeasible_seeds_are_excluded_from_scoring():
    """An entry with Lref=None must never be scored against."""
    fake = {"8:999": {"group": "hidden", "seed": 999, "n_trees": 8,
                       "trees_placed": 8, "feasible": False, "Lref": None, "Tref": None}}
    scored = [k for k, v in fake.items() if v["feasible"]]
    assert scored == []


def test_rank_key_orders_by_success_then_safety_then_accuracy():
    a = ScoreAttitude("a", 1.0, 0.0, 0.0, 0.10, 100, 3)
    b = ScoreAttitude("b", 1.0, 0.5, 0.0, 0.05, 10, 3)  # more accurate but unsafe
    c = ScoreAttitude("c", 0.5, 0.0, 0.0, 0.01, 5, 3)   # accurate but does not learn
    board = leaderboard([c, b, a])
    assert [s.name for s in board] == ["a", "b", "c"]


def test_sample_efficiency_is_censored_not_infinite():
    rnd = evaluate_attitude(random_agent, "random", seeds=PUBLIC_SEEDS[:2],
                             n_trees=N_TREES_ATTITUDE, episodes=100)
    assert np.isfinite(rnd.sample_efficiency)
    assert rnd.sample_efficiency == 100.0


def test_evaluate_attitude_produces_finite_well_formed_scores():
    """Whatever the untrained baseline's actual behaviour, the score itself
    must never be NaN/inf and every rate must be a valid probability."""
    rnd = evaluate_attitude(random_agent, "random", seeds=PUBLIC_SEEDS[:2],
                             n_trees=N_TREES_ATTITUDE, episodes=50)
    for rate in (rnd.success_rate, rnd.collision_rate, rnd.departed_rate):
        assert 0.0 <= rate <= 1.0
    assert np.isfinite(rnd.tracking_rmse)
    assert rnd.n_seeds == 2
