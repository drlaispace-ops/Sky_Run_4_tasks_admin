"""Phase 2 checks: A*, feasibility, and the navigation function."""

from __future__ import annotations

import numpy as np
import pytest

from tier_d.env.constants import CELL, GRID_N
from tier_d.env.forest import make_forest
from tier_d.env.occupancy import START_CELL, cell_center
from tier_d.oracle.astar import is_feasible, optimal_length, optimal_path_xy, string_pull
from tier_d.oracle.value_iteration import (
    dijkstra_cost_to_go,
    has_single_minimum,
    optimal_efficiency_bound,
)

EMPTY = np.zeros((0, 2))


def wall_of_trees() -> np.ndarray:
    """A vertical palisade at x = 5.0 covering every cell centre of that column.

    Built by hand, not by make_forest: the rejection sampler enforces a minimum
    tree separation that makes a fully sealed wall impossible to generate.
    """
    col_x = 5.0
    return np.array([[col_x, (j + 0.5) * CELL] for j in range(GRID_N)])


def test_empty_forest_optimum_is_the_euclidean_distance():
    assert optimal_length(EMPTY) == pytest.approx(np.hypot(8.0, 8.0), abs=1e-6)


def test_blocked_corridor_is_infeasible_and_has_no_optimum():
    wall = wall_of_trees()
    assert is_feasible(wall) is False
    assert optimal_length(wall) is None
    assert optimal_path_xy(wall) is None


def test_open_forests_are_feasible():
    for seed in range(6):
        assert is_feasible(make_forest(16, seed))


def test_lstar_is_monotone_non_decreasing_in_tree_count():
    """More trees can never shorten the optimum (make_forest extends its prefix)."""
    prev = 0.0
    for n in (0, 4, 8, 12, 16, 20, 24):
        L = optimal_length(make_forest(n, 5))
        assert L is not None
        assert L >= prev - 1e-9, f"L* dropped at n={n}"
        prev = L


def test_dijkstra_agrees_with_astar_at_the_start_cell():
    """Two independent shortest-path routines, one graph. They must agree."""
    for seed in range(5):
        trees = make_forest(16, seed)
        assert optimal_efficiency_bound(trees) == pytest.approx(optimal_length(trees), abs=1e-9)


@pytest.mark.parametrize("seed", range(6))
def test_navigation_function_has_a_single_minimum(seed):
    """INVARIANT 6: -U* has one minimum, at the goal, and no local traps."""
    cost = dijkstra_cost_to_go(make_forest(16, seed))
    assert np.isfinite(cost[START_CELL])
    assert has_single_minimum(cost)


def test_string_pull_shortens_but_is_never_used_for_scoring():
    trees = make_forest(16, 0)
    path = optimal_path_xy(trees)
    pulled = string_pull(path, trees)

    def length(p):
        return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))

    assert length(pulled) <= length(path) + 1e-9
    assert len(pulled) <= len(path)
    # The scored L* is the grid cost, so it is unaffected by smoothing.
    assert optimal_length(trees) == pytest.approx(length(path), abs=1e-9)


def test_goal_cost_is_zero_and_blocked_cells_are_unreachable():
    trees = make_forest(16, 2)
    cost = dijkstra_cost_to_go(trees)
    from tier_d.env.occupancy import GOAL_CELL, occupancy_grid

    assert cost[GOAL_CELL] == 0.0
    assert np.all(np.isinf(cost[occupancy_grid(trees)]))
