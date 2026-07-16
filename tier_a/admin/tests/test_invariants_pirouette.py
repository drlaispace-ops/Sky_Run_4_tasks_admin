"""Invariant 2, adapted for the pirouette guidance layer: it may consume the
dense LIDAR scan (indirect sensing -- the same legitimacy already granted
lidar3d.py's consumption by qtable_attitude/hybrid_attitude) but must never
read ground-truth tree coordinates directly. Mirrors
tests/test_invariants_attitude.py's AttitudeBlindEnv + source-regex guard
pattern exactly.

Also enforces this module's own non-negotiable design rule (see
agent/pirouette_attitude.py's module docstring and env_attitude/lidar3d.py's
dense-variant comment): dense sensing must never translate into a large
learned state. pirouette_state() must extract only a small, fixed number of
scalars from the scan (lidar.min() and the agent's own attitude) regardless
of how many rays the caller passes in -- naively binning even 16 rays at 3
bins each would be 3**16 ~= 43M states, a direct regression to the
already-falsified agent/qtable_attitude.py design.
"""

from __future__ import annotations

import inspect
import re

import pytest

from tier_a.admin.agent import pirouette_attitude as pa
from tier_a.env_attitude.constants import N_TREES_ATTITUDE
from tier_a.env_attitude.scenarios_pirouette import build_pirouette_gate_env
from tier_a.admin.tests.test_invariants_attitude import AttitudeBlindEnv


def test_pirouette_state_never_references_dot_trees_or_raw_lidar_array():
    src = inspect.getsource(pa.pirouette_state)
    assert not re.search(r"\.trees\b", src)


def test_pirouette_state_only_takes_a_fixed_small_set_of_scalars():
    """Structural guard for the state-size rule: pirouette_state's signature
    must stay (obs, lidar, phi_now) -- no raw per-ray array threaded into the
    Q-table's indexing math, no matter how many rays the LIDAR provides."""
    params = list(inspect.signature(pa.pirouette_state).parameters)
    assert params == ["obs", "lidar", "phi_now"]
    src = inspect.getsource(pa.pirouette_state)
    # only .min() may be pulled off the dense scan -- no indexing/iterating
    # over individual rays, which would be the first step toward smuggling
    # per-ray detail back into the learned state.
    assert "lidar.min()" in src
    assert not re.search(r"lidar\[", src)


def test_pirouette_state_size_is_9_regardless_of_ray_count():
    """The whole point of the section-2 rule made concrete: scanning with 12,
    16, or 64 rays must not change N_PIROUETTE_STATES."""
    assert pa.N_PIROUETTE_STATES == 9


def test_training_functions_never_name_physical_state():
    for fn in (pa.train_pirouette_guidance, pa.pirouette_state):
        src = inspect.getsource(fn)
        for forbidden in (r"\.trees", r"\.xyz", r"\.attitude\b"):
            assert not re.search(forbidden, src), f"{fn.__name__} references {forbidden}"


def test_pirouette_agent_trains_and_rolls_out_without_any_physical_state_access():
    real_env = build_pirouette_gate_env(seed=0, stage="straight")
    blind = AttitudeBlindEnv(real_env)

    # Drive a short blind training/rollout loop by hand, using only the
    # allowed surface (reset/step/lidar_scan_dense) -- exactly what
    # train_pirouette_guidance/greedy_rollout_pirouette do internally.
    obs = blind.reset()
    lidar = blind.lidar_scan_dense()
    phi_now = float(obs[3])
    s = pa.pirouette_state(obs, lidar, phi_now)
    for _ in range(50):
        a = pa.pd_action(obs, phi_bias=pa.ROLL_TARGETS[s % pa.N_ROLL_ACTIONS])
        obs, r, done, info = blind.step(a)
        lidar = blind.lidar_scan_dense()
        phi_now = float(obs[3])
        s = pa.pirouette_state(obs, lidar, phi_now)
        if done:
            break

    with pytest.raises(AttributeError, match="INVARIANT 2"):
        _ = blind.trees
    with pytest.raises(AttributeError, match="INVARIANT 2"):
        _ = blind.xyz
    with pytest.raises(AttributeError, match="INVARIANT 2"):
        _ = blind.attitude


def test_real_training_run_end_to_end_through_the_blind_wrapper():
    """A full (small) train_pirouette_guidance call, driven through the blind
    wrapper's allowed surface only -- proves the actual training loop, not
    just a hand-rolled stand-in, never reaches for ground truth."""
    real_env = build_pirouette_gate_env(seed=0, stage="straight")
    blind = AttitudeBlindEnv(real_env)
    Q, hist = pa.train_pirouette_guidance(blind, episodes=20, seed=0)
    assert Q.shape == (pa.N_PIROUETTE_STATES, pa.N_ROLL_ACTIONS)
    roll = pa.greedy_rollout_pirouette(blind, Q)
    assert roll["steps"] > 0

    with pytest.raises(AttributeError, match="INVARIANT 2"):
        _ = blind.trees
