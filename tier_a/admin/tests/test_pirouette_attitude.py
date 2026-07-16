"""Tier-A pirouette-guidance checks: PD stabilization + a tiny Q-learned
roll-target layer over the dense 64-ray LIDAR. Kept fast (small episode
budgets) per the kit's pytest time budget; the honest curriculum numbers are
documented in ``env_attitude/README.md``, not re-asserted here as a pass/fail
gate (mirrors ``tests/test_hybrid_attitude.py``'s own stated reasoning).
"""

from __future__ import annotations

import inspect
import math
import re
from pathlib import Path

import numpy as np
import pytest

import tier_a.admin.agent.pirouette_attitude as pa
from tier_a.env_attitude.scenarios_pirouette import build_pirouette_gate_env


def test_pirouette_state_is_always_in_range():
    rng = np.random.default_rng(0)
    for _ in range(200):
        obs = rng.normal(scale=0.5, size=10)
        lidar = rng.uniform(0.0, 2.5, size=64)
        phi_now = float(rng.uniform(-1.5, 1.5))
        s = pa.pirouette_state(obs, lidar, phi_now)
        assert 0 <= s < pa.N_PIROUETTE_STATES


def test_roll_targets_include_zero_and_the_pirouette_limit():
    assert 0.0 in pa.ROLL_TARGETS
    assert max(pa.ROLL_TARGETS) == pytest.approx(pa.PHI_MAX_PIROUETTE)
    assert min(pa.ROLL_TARGETS) == pytest.approx(-pa.PHI_MAX_PIROUETTE)
    assert len(pa.ROLL_TARGETS) == 7


def test_proactive_margin_penalty_is_zero_when_clear_and_negative_when_close():
    from tier_d.env.constants import TREE_R

    far = TREE_R + pa.presented_half_width(0.0) + pa.SAFE_MARGIN + 1.0
    assert pa.proactive_margin_penalty(0.0, far) == pytest.approx(0.0)
    at_threshold = TREE_R + pa.presented_half_width(0.0) + pa.SAFE_MARGIN
    assert pa.proactive_margin_penalty(0.0, at_threshold) == pytest.approx(0.0, abs=1e-9)
    close = TREE_R + pa.presented_half_width(0.0)
    assert pa.proactive_margin_penalty(0.0, close) < 0.0


def test_proactive_margin_penalty_accounts_for_current_roll():
    """The same sensed lidar_min should be penalised less (or not at all)
    when the drone is already rolled -- presented_half_width shrinks, so
    the real margin against that same obstacle grows."""
    from tier_d.env.constants import TREE_R

    lidar_min = TREE_R + pa.presented_half_width(0.0)  # exactly at the level-flight threshold
    penalty_level = pa.proactive_margin_penalty(0.0, lidar_min)
    penalty_rolled = pa.proactive_margin_penalty(math.radians(80), lidar_min)
    assert penalty_rolled > penalty_level  # less negative: more real margin at this roll


def test_pd_action_with_zero_bias_matches_hand_tuned_controller():
    from tier_a.env_attitude.constants import ACTIONS_ATTITUDE, D_TAU

    rng = np.random.default_rng(0)
    for _ in range(50):
        obs = rng.normal(scale=0.3, size=10)
        e_phi, e_theta, e_psi = obs[3], obs[4], obs[5]
        p, q, r = obs[6], obs[7], obs[8]
        cmd = (-(1.0 * e_phi + 0.3 * p), -(1.0 * e_theta + 0.3 * q), -(1.0 * e_psi + 0.3 * r))
        expected_tau = tuple(D_TAU if c > 0 else (-D_TAU if c < 0 else 0.0) for c in cmd)
        expected = ACTIONS_ATTITUDE.index(expected_tau)
        assert pa.pd_action(obs, phi_bias=0.0) == expected


def test_pd_action_bias_shifts_the_effective_target():
    changed = False
    for e_phi in np.linspace(-0.05, 0.05, 11):
        obs = np.array([0, 0, 0, e_phi, 0, 0, 0, 0, 0, 0], dtype=float)
        a_no_bias = pa.pd_action(obs, phi_bias=0.0)
        a_biased = pa.pd_action(obs, phi_bias=pa.ROLL_TARGETS[3])  # 60 deg
        if a_no_bias != a_biased:
            changed = True
    assert changed


def test_smoke_train_and_rollout_run_without_crashing():
    env = build_pirouette_gate_env(seed=0, stage="straight")
    Q, hist = pa.train_pirouette_guidance(env, episodes=100, seed=0)
    assert Q.shape == (pa.N_PIROUETTE_STATES, pa.N_ROLL_ACTIONS)
    roll = pa.greedy_rollout_pirouette(env, Q)
    assert roll["steps"] > 0
    assert 0.0 <= roll["min_lidar"] <= 2.5 + 1e-9


def test_q_init_warm_start_reuses_the_provided_table_shape():
    env = build_pirouette_gate_env(seed=0, stage="straight")
    Q_straight, _ = pa.train_pirouette_guidance(env, episodes=50, seed=0)
    env2 = build_pirouette_gate_env(seed=0, stage="angled")
    Q_warm, _ = pa.train_pirouette_guidance(env2, episodes=50, seed=0, Q_init=Q_straight)
    assert Q_warm.shape == Q_straight.shape
    assert not np.array_equal(Q_warm, np.zeros_like(Q_warm))


def test_recorded_rollout_trajectory_has_matching_length():
    env = build_pirouette_gate_env(seed=0, stage="straight")
    Q, _ = pa.train_pirouette_guidance(env, episodes=100, seed=0)
    roll = pa.greedy_rollout_pirouette(env, Q, record=True)
    traj = roll["trajectory"]
    assert len(traj["xyz"]) == len(traj["attitude"]) == roll["steps"] + 1


def test_pirouette_attitude_never_imports_torch():
    src = Path(pa.__file__).read_text()
    assert not re.search(r"^\s*(import torch|from torch)", src, re.M)


def test_learning_code_never_names_physical_state():
    """Mirrors tests/test_hybrid_attitude.py's own scope exactly:
    greedy_rollout_*'s optional record=True path legitimately uses
    env.xyz/.attitude for trajectory visualization (the class docstring's
    "viz/tests only" carve-out), so it's excluded here just as
    greedy_rollout_hybrid is excluded from the equivalent hybrid check."""
    for fn in (pa.train_pirouette_guidance, pa.pirouette_state):
        src = inspect.getsource(fn)
        for forbidden in (r"\.trees", r"\.xyz", r"\.attitude\b"):
            assert not re.search(forbidden, src), f"{fn.__name__} references {forbidden}"
