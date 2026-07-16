"""Tier-A tabular Q-learning checks: discretization, safety reward terms,
and a smoke training/rollout run. Mirrors ``tests/test_env_attitude.py``'s
style; kept fast (small episode budgets) per the kit's pytest time budget.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest

import tier_a.admin.agent.qtable_attitude as qt
from tier_a.env_attitude.scenarios import build_local_gauntlet_env


def test_discretize_is_always_in_range():
    rng = np.random.default_rng(0)
    for _ in range(200):
        obs = rng.normal(scale=0.5, size=10)
        lidar = rng.uniform(0.0, 2.5, size=12)
        s = qt.discretize(obs, lidar)
        assert 0 <= s < qt.N_STATES


def test_safety_bonus_is_zero_when_calm_and_clear():
    assert qt.safety_bonus(0.0, 0.0, 0.0, qt.PROX_SAFE) == pytest.approx(0.0)
    assert qt.safety_bonus(0.0, 0.0, 0.0, qt.PROX_SAFE * 2) == pytest.approx(0.0)


def test_safety_bonus_penalizes_spin_and_proximity():
    calm = qt.safety_bonus(0.0, 0.0, 0.0, qt.PROX_SAFE)
    spinning = qt.safety_bonus(3.0, 0.0, 0.0, qt.PROX_SAFE)
    assert spinning < calm

    clear = qt.safety_bonus(0.0, 0.0, 0.0, qt.PROX_SAFE)
    close = qt.safety_bonus(0.0, 0.0, 0.0, 0.0)
    assert close < clear


def test_shaping_terminal_potential_is_zeroed():
    obs = np.array([0.3, -0.2, 0.1, 0.05, -0.05, 0.02, 0.0, 0.0, 0.0, 0.0])
    assert qt.shaping_F(obs, obs, gamma=0.99, terminal=True) == pytest.approx(-qt.potential(obs))


def test_smoke_train_and_rollout_run_without_crashing():
    env = build_local_gauntlet_env(seed=0)
    Q, hist = qt.train_tabular_attitude(env, episodes=150, seed=0)
    assert Q.shape == (qt.N_STATES, env.n_actions)
    roll = qt.greedy_rollout_tabular(env, Q)
    assert roll["steps"] > 0
    assert roll["mean_rate"] >= 0.0
    assert 0.0 <= roll["min_lidar"] <= 2.5 + 1e-9


def test_recorded_rollout_trajectory_has_matching_length():
    env = build_local_gauntlet_env(seed=0)
    Q, _ = qt.train_tabular_attitude(env, episodes=150, seed=0)
    roll = qt.greedy_rollout_tabular(env, Q, record=True)
    traj = roll["trajectory"]
    assert len(traj["xyz"]) == len(traj["attitude"]) == roll["steps"] + 1


def test_agent_qtable_attitude_never_imports_torch():
    path = Path(qt.__file__)
    src = path.read_text()
    assert not re.search(r"^\s*(import torch|from torch)", src, re.M)


def test_learning_code_never_names_physical_state():
    """``greedy_rollout_tabular`` is deliberately excluded: its ``record=True``
    path touches ``.xyz``/``.attitude`` only to build a trajectory for
    visualization, never to choose an action (the action is always
    ``argmax(Q[s])`` from the discretized state) -- the same allowance the
    scaffold notebook's plotting cell gets in ``test_scaffold_attitude.py``.
    """
    for fn in (qt.train_tabular_attitude, qt.discretize):
        src = inspect.getsource(fn)
        for forbidden in (r"\.trees", r"\.xyz", r"\.attitude\b"):
            assert not re.search(forbidden, src), f"{fn.__name__} references {forbidden}"
