"""Tier-A hybrid-controller checks: PD stabilization + tiny Q-learned LIDAR
guidance layer. Kept fast (small episode budgets) per the kit's pytest time
budget; the honest convergence numbers (1/10 seeds succeeding cleanly) are
documented in ``env_attitude/README.md``, not re-asserted here as a pass/fail
gate (see ``tests/test_scoring_attitude.py``'s reasoning for the same call).
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest

import tier_a.admin.agent.hybrid_attitude as hy
from tier_a.env_attitude.scenarios import build_local_gauntlet_env


def test_guidance_state_is_always_in_range():
    rng = np.random.default_rng(0)
    for _ in range(200):
        obs = rng.normal(scale=0.5, size=10)
        lidar = rng.uniform(0.0, 2.5, size=12)
        s = hy.guidance_state(obs, lidar)
        assert 0 <= s < hy.N_GUIDANCE_STATES


def test_proximity_penalty_is_zero_when_clear_and_negative_when_close():
    assert hy.proximity_penalty(hy.PROX_SAFE) == pytest.approx(0.0)
    assert hy.proximity_penalty(hy.PROX_SAFE * 2) == pytest.approx(0.0)
    assert hy.proximity_penalty(0.0) < hy.proximity_penalty(hy.PROX_SAFE)


def test_pd_action_with_zero_bias_matches_hand_tuned_controller():
    """pd_action(obs, phi_bias=0.0) must reduce to exactly the same bang-bang
    choice as the plain hand-tuned PD used throughout this session's
    diagnostics -- the bias is additive, not a replacement."""
    from tier_a.env_attitude.constants import ACTIONS_ATTITUDE, D_TAU

    rng = np.random.default_rng(0)
    for _ in range(50):
        obs = rng.normal(scale=0.3, size=10)
        e_phi, e_theta, e_psi = obs[3], obs[4], obs[5]
        p, q, r = obs[6], obs[7], obs[8]
        cmd = (-(1.0 * e_phi + 0.3 * p), -(1.0 * e_theta + 0.3 * q), -(1.0 * e_psi + 0.3 * r))
        expected_tau = tuple(D_TAU if c > 0 else (-D_TAU if c < 0 else 0.0) for c in cmd)
        expected = ACTIONS_ATTITUDE.index(expected_tau)
        assert hy.pd_action(obs, phi_bias=0.0) == expected


def test_pd_action_bias_shifts_the_effective_target():
    """A nonzero phi_bias must change the chosen action for at least some
    small-error observations (otherwise the bias has no effect at all)."""
    changed = False
    for e_phi in np.linspace(-0.05, 0.05, 11):
        obs = np.array([0, 0, 0, e_phi, 0, 0, 0, 0, 0, 0], dtype=float)
        a_no_bias = hy.pd_action(obs, phi_bias=0.0)
        a_biased = hy.pd_action(obs, phi_bias=hy.BIAS_ANGLE)
        if a_no_bias != a_biased:
            changed = True
    assert changed


def test_smoke_train_and_rollout_run_without_crashing():
    env = build_local_gauntlet_env(seed=0)
    Q, hist = hy.train_hybrid_guidance(env, episodes=100, seed=0)
    assert Q.shape == (hy.N_GUIDANCE_STATES, hy.N_BIAS_ACTIONS)
    roll = hy.greedy_rollout_hybrid(env, Q)
    assert roll["steps"] > 0
    assert 0.0 <= roll["min_lidar"] <= 2.5 + 1e-9


def test_recorded_rollout_trajectory_has_matching_length():
    env = build_local_gauntlet_env(seed=0)
    Q, _ = hy.train_hybrid_guidance(env, episodes=100, seed=0)
    roll = hy.greedy_rollout_hybrid(env, Q, record=True)
    traj = roll["trajectory"]
    assert len(traj["xyz"]) == len(traj["attitude"]) == roll["steps"] + 1


def test_hybrid_attitude_never_imports_torch():
    src = Path(hy.__file__).read_text()
    assert not re.search(r"^\s*(import torch|from torch)", src, re.M)


def test_learning_code_never_names_physical_state():
    for fn in (hy.train_hybrid_guidance, hy.guidance_state):
        src = inspect.getsource(fn)
        for forbidden in (r"\.trees", r"\.xyz", r"\.attitude\b"):
            assert not re.search(forbidden, src), f"{fn.__name__} references {forbidden}"
