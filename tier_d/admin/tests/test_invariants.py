"""The six invariants, each with a real guard.

These are the tests that stop a well-meaning refactor from producing a kit that
runs, scores, and is silently wrong.
"""

from __future__ import annotations

import inspect
import re

import numpy as np
import pytest

import tier_d.env.gridworld as gridworld_mod
import tier_d.oracle.astar as astar_mod
from tier_d.admin.agent.qlearning import greedy_rollout, train
from tier_d.admin.agent.shaping import potential, shaping_F
from tier_d.env.constants import GRID_N, TREE_R
from tier_d.env.forest import make_forest
from tier_d.env.gridworld import GridWorld
from tier_d.env.occupancy import GOAL_CELL, occupancy_grid
from tier_d.oracle.astar import optimal_length
from tier_d.oracle.value_iteration import dijkstra_cost_to_go, has_single_minimum
from tier_d.scoring.seeds import HIDDEN_SEEDS, PUBLIC_SEEDS


# --- INVARIANT 1: reproducibility (see tier_d/admin/tests/test_forest.py for the RNG golden) --
def test_invariant_1_environment_is_reproducible_from_seed():
    a, b = GridWorld(16, 42), GridWorld(16, 42)
    assert np.array_equal(a.trees, b.trees)
    assert np.array_equal(a.blocked, b.blocked)


# --- INVARIANT 2: state = experience, not map -------------------------------
class BlindEnv:
    """Exposes exactly the API a learner is allowed to touch. No map, ever."""

    _ALLOWED = {"n_states", "n_actions", "reset", "step", "xy", "encode", "decode", "_steps"}

    def __init__(self, env: GridWorld):
        object.__setattr__(self, "_env", env)

    def __getattr__(self, name):
        if name not in self._ALLOWED:
            raise AttributeError(
                f"INVARIANT 2 violated: the agent reached for env.{name!r}. "
                "Only the cell index and its own position are observable."
            )
        return getattr(object.__getattribute__(self, "_env"), name)


def test_invariant_2_agent_learns_without_any_map_access():
    """The reference agent must train through a wrapper that hides trees/blocked."""
    blind = BlindEnv(GridWorld(0, 0))
    Q, hist = train(blind, episodes=400, seed=0, eval_every=0)
    roll = greedy_rollout(blind, Q)
    assert roll["success"], "agent failed to learn with map access removed"

    with pytest.raises(AttributeError, match="INVARIANT 2"):
        _ = blind.trees
    with pytest.raises(AttributeError, match="INVARIANT 2"):
        _ = blind.blocked


def test_invariant_2_learning_code_never_names_the_map():
    """Source-level guard on the learning functions themselves.

    Scoped to ``train``/``greedy_rollout`` rather than the whole module: the CLI
    at the bottom of qlearning.py legitimately asks the oracle for L* to print a
    comparison, and the oracle is allowed to see trees.
    """
    import tier_d.admin.agent.qlearning as ql

    for fn in (ql.train, ql.greedy_rollout):
        src = inspect.getsource(fn)
        for forbidden in (r"\.trees", r"\.blocked"):
            assert not re.search(forbidden, src), f"{fn.__name__} references {forbidden}"


# --- INVARIANT 3: one collision radius --------------------------------------
def test_invariant_3_env_and_oracle_share_one_occupancy_function():
    """Structural: both modules call env.occupancy.occupancy_grid; neither reimplements it."""
    assert gridworld_mod.occupancy_grid is occupancy_grid
    assert astar_mod.occupancy_grid is occupancy_grid
    # And no module redefines a radius of its own.
    for mod in (gridworld_mod, astar_mod):
        src = inspect.getsource(mod)
        assert not re.search(r"^\s*TREE_R\s*=", src, re.M), f"{mod.__name__} redefines TREE_R"


def test_invariant_3_trees_never_overlap_and_ab_are_free():
    assert 2 * TREE_R < 0.85
    for seed in range(6):
        blocked = occupancy_grid(make_forest(16, seed))
        assert not blocked[2, 2] and not blocked[22, 22]


@pytest.mark.parametrize("seed", range(6))
def test_invariant_3_efficiency_never_exceeds_one(seed):
    """The agent cannot beat the oracle. If it can, the two radii have drifted."""
    env = GridWorld(16, seed)
    Lstar = optimal_length(env.trees)
    Q, _ = train(env, episodes=3000, seed=seed, eval_every=0)
    roll = greedy_rollout(env, Q)
    if roll["success"]:
        assert Lstar / roll["length"] <= 1.0 + 1e-9, "efficiency > 1: corner-cutting leaked in"


# --- INVARIANT 4: shaping preserves the optimum -----------------------------
def test_invariant_4_potential_is_zero_at_the_goal():
    assert potential(GOAL_CELL[0] * GRID_N + GOAL_CELL[1]) == pytest.approx(0.0)


def test_invariant_4_shaping_sums_to_zero_around_any_cycle():
    """The defining property of a potential: with gamma=1, F telescopes to 0 on a loop.

    Any shaping term that fails this can change the optimal policy.
    """
    cycle = [0, 1, GRID_N + 1, GRID_N, 0]
    total = sum(shaping_F(a, b, gamma=1.0) for a, b in zip(cycle, cycle[1:]))
    assert total == pytest.approx(0.0, abs=1e-12)


def test_invariant_4_terminal_potential_is_zeroed():
    s, s2 = 0, 1
    assert shaping_F(s, s2, 0.99, terminal=True) == pytest.approx(-potential(s))


def test_invariant_4_shaping_speeds_learning_without_changing_the_optimum():
    """Two separate claims, tested separately.

    Policy invariance is an *asymptotic* statement: given enough episodes both
    agents land on the same optimal path. Speed is the finite-budget statement.
    Comparing converged lengths at a budget where the unshaped agent has not yet
    converged tests neither, so we give both room to converge.
    """
    from tier_d.admin.agent.qlearning import episodes_to_threshold

    trees = make_forest(0, 0)  # empty forest: the optimum is unambiguous
    Lstar = optimal_length(trees)

    e1 = GridWorld(0, 0)
    Q1, h1 = train(e1, episodes=8000, seed=1, shaping=True)
    e2 = GridWorld(0, 0)
    Q2, h2 = train(e2, episodes=8000, seed=1, shaping=False)

    r1, r2 = greedy_rollout(e1, Q1), greedy_rollout(e2, Q2)
    assert r1["success"] and r2["success"]

    # Same optimum: both reach the exact diagonal, which is L*.
    assert r1["length"] == pytest.approx(Lstar, abs=1e-9)
    assert r2["length"] == pytest.approx(Lstar, abs=1e-9)

    # Faster: shaping crosses the threshold no later than the unshaped agent.
    k1 = episodes_to_threshold(h1, 1.15 * Lstar)
    k2 = episodes_to_threshold(h2, 1.15 * Lstar)
    assert k1 is not None and k2 is not None
    assert k1 <= k2, f"shaping was not faster: {k1} vs {k2}"


# --- INVARIANT 5: hidden != public ------------------------------------------
def test_invariant_5_seed_sets_are_disjoint():
    assert not set(PUBLIC_SEEDS) & set(HIDDEN_SEEDS)


def test_invariant_5_hidden_seeds_are_out_of_the_public_scan_range():
    assert min(HIDDEN_SEEDS) > max(PUBLIC_SEEDS) + 1


# --- INVARIANT 6: V(s) is a navigation function -----------------------------
@pytest.mark.parametrize("seed", range(4))
def test_invariant_6_optimal_surface_has_one_minimum_at_the_goal(seed):
    cost = dijkstra_cost_to_go(make_forest(16, seed))
    assert cost[GOAL_CELL] == 0.0
    assert has_single_minimum(cost)
