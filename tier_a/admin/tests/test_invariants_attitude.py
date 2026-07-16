"""Tier-A's adapted invariants, each with a real guard -- the same discipline
as ``tests/test_invariants.py``, applied to the continuous attitude-control
track.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest

import tier_a.env_attitude as env_attitude
from tier_a.admin.agent.qnet_attitude import greedy_rollout_attitude, potential, shaping_F, train_attitude
from tier_d.env.constants import TREE_R as BASE_TREE_R
from tier_a.env_attitude.constants import N_TREES_ATTITUDE, TREE_R as ATTITUDE_TREE_R
from tier_a.env_attitude.env import AttitudeEnv
from tier_a.env_attitude.reference_trajectory import build_reference
from tier_d.scoring.seeds import HIDDEN_SEEDS, PUBLIC_SEEDS


# --- INVARIANT 1: reproducibility -------------------------------------------
def test_reference_trajectory_is_reproducible_from_seed():
    a = build_reference(N_TREES_ATTITUDE, 42)
    b = build_reference(N_TREES_ATTITUDE, 42)
    assert np.array_equal(a.xy, b.xy)
    assert np.array_equal(a.t, b.t)
    assert np.array_equal(a.trees, b.trees)


def test_wind_gust_sequence_is_reproducible_from_seed():
    """wind_prob defaults to 0.0 (deterministic baseline) -- exercise the
    opt-in gust mechanism explicitly here rather than relying on the default."""
    e1 = AttitudeEnv(n_trees=N_TREES_ATTITUDE, seed=7, wind_prob=0.3)
    e2 = AttitudeEnv(n_trees=N_TREES_ATTITUDE, seed=7, wind_prob=0.3)
    e1.reset()
    e2.reset()
    winds1, winds2 = [], []
    for _ in range(200):
        _, _, done1, info1 = e1.step(13)
        _, _, done2, info2 = e2.step(13)
        winds1.append(info1["wind"])
        winds2.append(info2["wind"])
        if done1 or done2:
            break
    assert winds1 == winds2


def test_default_construction_is_deterministic_no_wind():
    """AttitudeEnv mirrors env.GridWorld: deterministic by default. Wind is
    an explicit, opt-in harder variant (env_wind's pattern), not the default."""
    env = AttitudeEnv(n_trees=N_TREES_ATTITUDE, seed=3)
    assert env.wind_prob == 0.0
    env.reset()
    for _ in range(200):
        _, _, done, info = env.step(13)
        assert not info["wind"]
        if done:
            break


# --- INVARIANT 2: state = experience, not map -------------------------------
class AttitudeBlindEnv:
    """Exposes exactly the API a learner is allowed to touch. No physical
    state (xyz/attitude/trees), ever."""

    _ALLOWED = {"n_actions", "obs_dim", "reset", "step", "lidar_scan", "lidar_scan_dense"}

    def __init__(self, env: AttitudeEnv):
        object.__setattr__(self, "_env", env)

    def __getattr__(self, name):
        if name not in self._ALLOWED:
            raise AttributeError(
                f"INVARIANT 2 (Tier-A) violated: the agent reached for env.{name!r}. "
                "Only the observation vector is observable."
            )
        return getattr(object.__getattribute__(self, "_env"), name)


def test_agent_trains_without_any_physical_state_access():
    blind = AttitudeBlindEnv(AttitudeEnv(n_trees=N_TREES_ATTITUDE, seed=0))
    qnet, hist = train_attitude(blind, episodes=5, seed=0, eval_every=0)
    greedy_rollout_attitude(blind, qnet, max_steps=50)

    with pytest.raises(AttributeError, match="INVARIANT 2"):
        _ = blind.trees
    with pytest.raises(AttributeError, match="INVARIANT 2"):
        _ = blind.xyz


def test_learning_code_never_names_physical_state():
    """Source-level guard, scoped to the learning functions themselves."""
    for fn in (train_attitude, greedy_rollout_attitude):
        src = inspect.getsource(fn)
        for forbidden in (r"\.trees", r"\.xyz", r"\.attitude\b"):
            assert not re.search(forbidden, src), f"{fn.__name__} references {forbidden}"


# --- INVARIANT 3: one collision radius --------------------------------------
def test_tree_r_matches_base_kit_exactly():
    assert ATTITUDE_TREE_R == BASE_TREE_R


def test_env_attitude_never_redefines_tree_r():
    """Structural: constants/occupancy must import TREE_R, never re-declare it."""
    import tier_a.env_attitude.constants as c_mod
    import tier_a.env_attitude.occupancy as o_mod

    for mod in (c_mod, o_mod):
        src = inspect.getsource(mod)
        assert not re.search(r"^\s*TREE_R\s*=\s*[\d.]", src, re.M), f"{mod.__name__} redefines TREE_R"


# --- INVARIANT 4: shaping preserves the optimum -----------------------------
def test_shaping_terminal_potential_is_zeroed():
    obs = np.array([0.3, -0.2, 0.1, 0.05, -0.05, 0.02, 0.0, 0.0, 0.0, 0.0])
    assert shaping_F(obs, obs, gamma=0.99, terminal=True) == pytest.approx(-potential(obs))


def test_shaping_sums_to_zero_around_any_cycle():
    """The defining property of a potential: with gamma=1, F telescopes to 0
    around a closed loop of non-terminal observations."""
    rng = np.random.default_rng(0)
    cycle = [rng.normal(scale=0.3, size=10) for _ in range(4)]
    cycle.append(cycle[0])
    total = sum(shaping_F(a, b, gamma=1.0, terminal=False) for a, b in zip(cycle, cycle[1:]))
    assert total == pytest.approx(0.0, abs=1e-9)


# --- INVARIANT 5: hidden != public -------------------------------------------
def test_hidden_and_public_seed_sets_are_disjoint():
    assert not (set(PUBLIC_SEEDS) & set(HIDDEN_SEEDS))


# --- INVARIANT 6: potential is a navigation function ------------------------
def test_potential_is_zero_only_at_zero_tracking_error():
    zero_obs = np.zeros(10)
    assert potential(zero_obs) == pytest.approx(0.0)

    rng = np.random.default_rng(0)
    for _ in range(20):
        obs = rng.normal(scale=0.5, size=10)
        if np.allclose(obs[[0, 1, 3, 4, 5]], 0.0):
            continue
        assert potential(obs) < 0.0


# --- INVARIANT 7 (Tier-A specific): torch stays confined --------------------
def test_env_attitude_never_imports_torch():
    pkg_dir = Path(env_attitude.__file__).parent
    for path in pkg_dir.glob("*.py"):
        src = path.read_text()
        assert not re.search(r"^\s*(import torch|from torch)", src, re.M), f"{path} imports torch"
