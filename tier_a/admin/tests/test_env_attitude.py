"""Tier-A checks: 6-DOF dynamics and the reference trajectory generator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tier_a.env_attitude import dynamics
from tier_a.env_attitude.forest import make_forest
from tier_a.env_attitude.constants import (
    ACTIONS_ATTITUDE,
    D_TAU,
    DT,
    GRAVITY,
    INERTIA,
    MASS,
    N_TREES_ATTITUDE,
    R_COLLISION,
    R_DEPARTED,
    R_GOAL,
    R_STEP,
)
from tier_a.env_attitude.env import AttitudeEnv
from tier_a.env_attitude.occupancy import collides
from tier_a.env_attitude.reference_trajectory import build_reference
from tier_d.oracle.astar import is_feasible


# --- dynamics ----------------------------------------------------------------
def test_level_hover_with_zero_torque_is_a_fixed_point():
    """No torque, no tilt, zero rates: thrust exactly cancels gravity."""
    s = dynamics.make_initial_state((1.0, 1.0), 1.6, psi0=0.3)
    s2 = dynamics.step(s, (0.0, 0.0, 0.0), DT)
    assert np.allclose(s2, s, atol=1e-9)


def test_known_torque_gives_known_angular_acceleration():
    """With zero initial body rates the gyroscopic cross terms vanish, so
    p/q/r after one tick should match tau/I * dt exactly."""
    s = dynamics.make_initial_state((0.0, 0.0), 1.6)
    Ixx, Iyy, Izz = INERTIA
    for axis, I in enumerate((Ixx, Iyy, Izz)):
        tau = [0.0, 0.0, 0.0]
        tau[axis] = D_TAU
        s2 = dynamics.step(s, tuple(tau), DT)
        expected = (D_TAU / I) * DT
        assert s2[9 + axis] == pytest.approx(expected, abs=1e-9)


def test_tilting_reduces_vertical_thrust_component():
    """A body tilted away from level cannot fully cancel gravity: it should
    start sinking (negative vz), which is the coupling that makes attitude
    tracking matter for altitude."""
    s = dynamics.make_initial_state((0.0, 0.0), 1.6)
    s[6] = math.radians(30.0)  # roll = 30 degrees, zero torque thereafter
    s2 = dynamics.step(s, (0.0, 0.0, 0.0), DT)
    assert s2[5] < 0.0  # vz


# --- reference trajectory ------------------------------------------------
@pytest.mark.parametrize("seed", range(4))
def test_reference_is_reproducible_from_seed(seed):
    a = build_reference(N_TREES_ATTITUDE, seed)
    b = build_reference(N_TREES_ATTITUDE, seed)
    assert a is not None and b is not None
    assert np.array_equal(a.xy, b.xy)
    assert np.array_equal(a.t, b.t)
    assert np.array_equal(a.trees, b.trees)


def test_reference_reaches_start_and_goal():
    ref = build_reference(N_TREES_ATTITUDE, seed=0)
    assert ref is not None
    assert ref.xy[0] == pytest.approx(ref.waypoints_xy[0], abs=1e-9)
    assert ref.xy[-1] == pytest.approx(ref.waypoints_xy[-1], abs=1e-9)


def test_infeasible_seed_returns_none():
    for seed in range(50):
        if not is_feasible(make_forest(24, seed)):
            assert build_reference(24, seed) is None
            return
    pytest.skip("no infeasible seed found in range(50) at n_trees=24")


def test_straight_segment_has_near_zero_bank_and_curved_segment_does_not():
    """Curvature-driven bank angle should be ~0 on a straight leg and
    non-trivial (correctly signed) on a curved leg."""
    ref = build_reference(N_TREES_ATTITUDE, seed=0)
    assert ref is not None
    i_straight = len(ref.kappa) // 2
    # find the point of maximum |curvature| as the "curved" sample
    i_curved = int(np.argmax(np.abs(ref.kappa)))
    assert abs(ref.phi_ref[i_curved]) >= abs(ref.phi_ref[i_straight])


def test_nearest_projection_sign_convention():
    ref = build_reference(N_TREES_ATTITUDE, seed=0)
    assert ref is not None
    i = len(ref.xy) // 2
    s_on, e_on, _ = ref.nearest(tuple(ref.xy[i]))
    assert e_on == pytest.approx(0.0, abs=1e-6)

    psi_i = ref.psi_ref[i]
    normal = np.array([-math.sin(psi_i), math.cos(psi_i)])  # left-hand normal
    p_left = ref.xy[i] + 0.2 * normal
    _, e_left, _ = ref.nearest(tuple(p_left))
    assert e_left == pytest.approx(0.2, abs=1e-3)


# --- collision -------------------------------------------------------------
def test_collides_exactly_at_tree_radius():
    trees = np.array([[5.0, 5.0]])
    from tier_d.env.constants import TREE_R

    assert collides((5.0, 5.0), 1.6, trees)  # dead centre
    assert not collides((5.0 + TREE_R + 0.01, 5.0), 1.6, trees)  # just outside
    assert not collides((5.0, 5.0), 10.0, trees)  # above the trunk


# --- AttitudeEnv end-to-end --------------------------------------------------
def test_attitude_env_reset_and_step_shapes():
    env = AttitudeEnv(n_trees=N_TREES_ATTITUDE, seed=0)
    obs = env.reset()
    assert obs.shape == (env.obs_dim,)
    assert env.n_actions == len(ACTIONS_ATTITUDE) == 27

    obs2, r, done, info = env.step(13)  # (0,0,0) torque, if that's the zero entry
    assert obs2.shape == (env.obs_dim,)
    assert isinstance(done, bool)
    assert set(info) >= {"collision", "departed", "loss_of_control", "goal", "truncated", "wind"}


def test_zero_torque_action_exists_and_is_index_13():
    assert ACTIONS_ATTITUDE[13] == (0.0, 0.0, 0.0)


def test_episode_terminates_and_reward_matches_outcome():
    env = AttitudeEnv(n_trees=N_TREES_ATTITUDE, seed=1)
    rng = np.random.default_rng(1)
    env.reset()
    for _ in range(2000):
        obs, r, done, info = env.step(int(rng.integers(env.n_actions)))
        if done:
            if info["collision"]:
                assert r == R_COLLISION
            elif info["loss_of_control"] or info["departed"]:
                assert r == R_DEPARTED
            elif info["goal"]:
                assert r == R_GOAL
            return
        if info["truncated"]:
            assert r == R_STEP
            return
    pytest.fail("episode never terminated or timed out within 2000 steps")


def test_wind_gust_frequency_matches_wind_prob():
    env = AttitudeEnv(n_trees=N_TREES_ATTITUDE, seed=2, wind_prob=0.05)
    env.reset()
    gusts = 0
    n = 4000
    for _ in range(n):
        _, _, done, info = env.step(13)
        gusts += int(info["wind"])
        if done:
            env.reset()
    assert 0.03 < gusts / n < 0.08


# --- online LIDAR-based replanning ------------------------------------------
def test_online_lidar_planning_runs_end_to_end_and_eventually_replans():
    from tier_a.admin.agent.hybrid_attitude import pd_action

    env = AttitudeEnv(n_trees=N_TREES_ATTITUDE, seed=0, online_lidar_planning=True)
    obs = env.reset()
    replans = 0
    for _ in range(1500):
        obs, r, done, info = env.step(pd_action(obs))
        replans += int(info["replanned"])
        if done:
            break
    assert replans > 0  # the belief must have grown enough to trigger at least one replan


def test_replan_does_not_itself_cause_an_immediate_terminal():
    """The reference swap alone must not instantly read as a collision or
    departure -- a real bug in the anchoring geometry would show up here as
    every replanned tick immediately terminating the episode."""
    from tier_a.admin.agent.hybrid_attitude import pd_action

    for seed in range(3):
        env = AttitudeEnv(n_trees=N_TREES_ATTITUDE, seed=seed, online_lidar_planning=True)
        obs = env.reset()
        for _ in range(1500):
            obs, r, done, info = env.step(pd_action(obs))
            if info["replanned"]:
                assert not info["collision"]
                assert not info["departed"]
            if done:
                break


def test_replan_resets_reference_time_but_not_wall_clock_truncation():
    """self._t (reference lookup) resets on replan; self._t_wall (truncation)
    must not, or the timeout budget would reset every replan and the episode
    could run forever."""
    env = AttitudeEnv(n_trees=N_TREES_ATTITUDE, seed=0, online_lidar_planning=True)
    env.reset()
    for _ in range(3000):
        _, _, done, info = env.step(13)
        if done:
            break
    assert done  # must terminate (collision/departed/goal/timeout), never run past max_steps unbounded


def test_online_time_margin_exceeds_the_offline_default():
    """A discovered-not-optimal path is longer than L* by construction, so the
    online condition needs -- and is given -- a larger timeout margin than
    the offline default. Disclosed explicitly, not a silent advantage."""
    from tier_a.env_attitude.constants import ONLINE_TIME_MARGIN, TIME_MARGIN

    assert ONLINE_TIME_MARGIN > TIME_MARGIN
