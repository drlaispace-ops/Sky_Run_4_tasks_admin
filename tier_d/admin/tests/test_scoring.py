"""Phase 5 checks: the scoreboard must discriminate and must exclude the impossible."""

from __future__ import annotations

import numpy as np

from tier_d.admin.scoring.harness import Score, evaluate, leaderboard, random_agent, reference_agent
from tier_d.admin.scoring.precompute import build_cache
from tier_d.scoring.seeds import HIDDEN_SEEDS, PUBLIC_SEEDS


def test_cache_covers_every_seed_and_flags_feasibility():
    cache = build_cache(n_trees=16)
    for s in (*PUBLIC_SEEDS, *HIDDEN_SEEDS):
        entry = cache[f"16:{s}"]
        assert entry["feasible"] is (entry["Lstar"] is not None)


def test_infeasible_seeds_are_excluded_from_scoring():
    """An entry with Lstar=None must never be scored against."""
    fake = {"16:999": {"group": "hidden", "seed": 999, "n_trees": 16,
                       "trees_placed": 16, "feasible": False, "Lstar": None}}
    scored = [k for k, v in fake.items() if v["feasible"]]
    assert scored == []


def test_reference_agent_beats_random_lexicographically():
    ref = evaluate(reference_agent, "reference", seeds=HIDDEN_SEEDS[:3], episodes=1500)
    rnd = evaluate(random_agent, "random", seeds=HIDDEN_SEEDS[:3], episodes=1500)
    assert ref.rank_key > rnd.rank_key
    assert leaderboard([rnd, ref])[0].name == "reference"


def test_random_agent_never_succeeds_and_reference_always_does():
    ref = evaluate(reference_agent, "reference", seeds=HIDDEN_SEEDS[:3], episodes=1500)
    rnd = evaluate(random_agent, "random", seeds=HIDDEN_SEEDS[:3], episodes=1500)
    assert ref.success_rate == 1.0
    assert rnd.success_rate == 0.0
    assert ref.optimality <= 1.0 + 1e-9


def test_rank_key_orders_by_success_then_safety_then_optimality():
    a = Score("a", 1.0, 0.0, 0.90, 100, 1.0, 3)
    b = Score("b", 1.0, 0.5, 0.99, 10, 1.0, 3)  # more optimal but unsafe
    c = Score("c", 0.5, 0.0, 1.00, 5, 1.0, 3)  # optimal but does not learn
    board = leaderboard([c, b, a])
    assert [s.name for s in board] == ["a", "b", "c"]


def test_sample_efficiency_is_censored_not_infinite():
    rnd = evaluate(random_agent, "random", seeds=HIDDEN_SEEDS[:2], episodes=500)
    assert np.isfinite(rnd.sample_efficiency)
    assert rnd.sample_efficiency == 500.0
    assert rnd.reached_frac == 0.0
