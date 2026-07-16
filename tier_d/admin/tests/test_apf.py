"""APF + residual track. Baseline and foil, never a scoring target."""

from __future__ import annotations

import numpy as np
import pytest

from tier_d.admin.agent.qlearning import greedy_rollout, train
from tier_d.admin.apf.compare import find_trap_seeds
from tier_d.admin.apf.field import apf_flight, grad_U, potential_grid, repulsive_potential, total_potential
from tier_d.admin.apf.residual import apf_greedy_rollout, apf_prior, residual_rollout, train_residual
from tier_d.env.constants import GOAL_XY, TREE_R
from tier_d.env.forest import make_forest
from tier_d.env.gridworld import GridWorld
from tier_d.oracle.astar import is_feasible, optimal_length


def test_potential_is_minimal_at_the_goal_in_an_empty_forest():
    empty = np.zeros((0, 2))
    at_goal = total_potential(np.asarray(GOAL_XY), empty)
    elsewhere = total_potential(np.array([1.0, 1.0]), empty)
    assert at_goal < elsewhere
    assert at_goal == pytest.approx(0.0, abs=1e-9)


def test_repulsion_blows_up_at_the_tree_surface():
    trees = np.array([[5.0, 5.0]])
    near = repulsive_potential(np.array([5.0 + TREE_R + 0.01, 5.0]), trees)
    far = repulsive_potential(np.array([5.0 + TREE_R + 0.5, 5.0]), trees)
    assert near > far >= 0.0


def test_repulsion_is_inactive_beyond_its_cutoff():
    trees = np.array([[5.0, 5.0]])
    assert repulsive_potential(np.array([9.0, 9.0]), trees) == pytest.approx(0.0)


def test_gradient_points_downhill():
    """Finite differences must agree with the analytic gradient."""
    trees = make_forest(8, 0)
    x = np.array([4.0, 3.0])
    g = grad_U(x, trees)
    eps = 1e-6
    fd = np.array([
        (total_potential(x + np.array([eps, 0]), trees) - total_potential(x - np.array([eps, 0]), trees)) / (2 * eps),
        (total_potential(x + np.array([0, eps]), trees) - total_potential(x - np.array([0, eps]), trees)) / (2 * eps),
    ])
    assert np.allclose(g, fd, rtol=1e-3, atol=1e-4)


def test_apf_reaches_the_goal_on_an_empty_forest():
    r = apf_flight(np.zeros((0, 2)))
    assert r["success"] and not r["trapped"]


@pytest.mark.parametrize("seed", range(10))
def test_apf_never_flies_into_a_tree(seed):
    """A correct repulsive field fails by stalling, never by colliding."""
    trees = make_forest(16, seed)
    r = apf_flight(trees)
    assert not r["collided"]


def test_apf_is_a_fair_baseline_not_a_strawman():
    """It must succeed on a good share of forests, or the RL comparison is rigged."""
    outcomes = [apf_flight(make_forest(16, s))["success"] for s in range(20)]
    rate = float(np.mean(outcomes))
    assert 0.3 < rate < 0.9, f"APF success rate {rate:.2f} looks tuned"


def test_local_minimum_traps_exist():
    seeds = find_trap_seeds(16, range(20), limit=3)
    assert len(seeds) >= 1


def test_prior_is_bounded_cosine():
    prior = apf_prior(make_forest(8, 0))
    assert prior.shape[1] == 8
    assert np.all(prior >= -1.0 - 1e-9) and np.all(prior <= 1.0 + 1e-9)


def test_residual_starts_as_pure_apf():
    """With Q_res == 0 the policy is exactly argmax of the APF prior."""
    trees = make_forest(16, 0)
    env = GridWorld(16, 0)
    prior = apf_prior(trees)
    a = apf_greedy_rollout(env, prior)
    b = residual_rollout(env, np.zeros_like(prior), prior)
    assert a["success"] == b["success"]
    assert a["length"] == pytest.approx(b["length"])


def test_residual_and_qlearning_escape_a_trap_that_stops_apf():
    """Concept thesis #2, tested rather than asserted."""
    seeds = find_trap_seeds(16, range(20), limit=1)
    seed = seeds[0]
    trees = make_forest(16, seed)
    assert is_feasible(trees)
    assert apf_flight(trees)["trapped"]

    prior = apf_prior(trees)
    e1 = GridWorld(16, seed)
    Qr, _ = train_residual(e1, prior, episodes=3000, seed=seed)
    assert residual_rollout(e1, Qr, prior)["success"]

    e2 = GridWorld(16, seed)
    Q, _ = train(e2, episodes=3000, seed=seed, eval_every=0)
    roll = greedy_rollout(e2, Q)
    assert roll["success"]
    # And the map-blind learner still cannot beat the oracle.
    assert optimal_length(trees) / roll["length"] <= 1.0 + 1e-9


def test_potential_grid_is_finite_and_clipped():
    X, Y, U = potential_grid(make_forest(8, 0), n=40, clip=50.0)
    assert X.shape == Y.shape == U.shape == (40, 40)
    assert np.all(np.isfinite(U)) and U.max() <= 50.0
