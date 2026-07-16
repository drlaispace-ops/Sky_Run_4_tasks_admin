"""env_attitude/scenarios_pirouette.py: curriculum gate builders and the
gap_required_roll helper that ties each stage's clearance to real physics."""

from __future__ import annotations

import math

import pytest

from tier_d.env.constants import TREE_R
from tier_a.env_attitude.constants import H_BODY, PHI_MAX_PIROUETTE, R_BODY
from tier_a.env_attitude.scenarios_pirouette import (
    PIROUETTE_CLEARANCE_STAGES,
    build_pirouette_gate_env,
    gap_required_roll,
)


def test_straight_stage_needs_no_roll():
    assert gap_required_roll(PIROUETTE_CLEARANCE_STAGES["straight"]) == pytest.approx(0.0)


def test_angled_stage_needs_a_partial_roll():
    r = gap_required_roll(PIROUETTE_CLEARANCE_STAGES["angled"])
    assert r is not None
    assert math.radians(20) < r < math.radians(60)


def test_pirouette_stage_needs_a_steep_roll():
    r = gap_required_roll(PIROUETTE_CLEARANCE_STAGES["pirouette"])
    assert r is not None
    assert r > math.radians(60)
    assert r <= PHI_MAX_PIROUETTE


def test_a_gap_narrower_than_the_knife_edge_floor_is_impossible():
    impossible_clearance = TREE_R + H_BODY - 0.05
    assert gap_required_roll(impossible_clearance) is None


def test_a_gap_wider_than_the_level_footprint_needs_exactly_zero_roll():
    wide_clearance = TREE_R + R_BODY + 0.2
    assert gap_required_roll(wide_clearance) == pytest.approx(0.0)


def test_required_roll_increases_monotonically_as_the_gap_narrows():
    clearances = [0.70, 0.62, 0.58, 0.54, 0.50]
    rolls = [gap_required_roll(c) for c in clearances]
    assert all(r is not None for r in rolls)
    assert all(rolls[i] <= rolls[i + 1] + 1e-9 for i in range(len(rolls) - 1))


def test_build_pirouette_gate_env_defaults_to_body_aware_collision():
    env = build_pirouette_gate_env(seed=0, stage="pirouette")
    assert env.body_aware_collision is True
    assert len(env.trees) == 2


def test_build_pirouette_gate_env_respects_an_explicit_override():
    env = build_pirouette_gate_env(seed=0, stage="straight", body_aware_collision=False)
    assert env.body_aware_collision is False


@pytest.mark.parametrize("stage", list(PIROUETTE_CLEARANCE_STAGES))
def test_every_stage_builds_a_valid_env(stage):
    env = build_pirouette_gate_env(seed=0, stage=stage)
    obs = env.reset()
    assert obs.shape == (env.obs_dim,)
