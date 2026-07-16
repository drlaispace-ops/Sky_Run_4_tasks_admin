"""Orientation-aware collision (the "pirouette" mechanic): the drone's body
is a rollable ellipsoid (env_attitude/occupancy.py's presented_half_width,
collides_oriented), not a point -- rolling toward PHI_MAX_PIROUETTE shrinks
the width it presents to a gap, letting it thread a corridor narrower than
its level-flight footprint. Opt-in (AttitudeEnv(body_aware_collision=True));
every existing Tier-A result was measured against the original point-mass
collides() and stays unaffected by default.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tier_d.env.constants import TREE_R
from tier_a.env_attitude.constants import H_BODY, R_BODY
from tier_a.env_attitude.occupancy import collides, collides_oriented, presented_half_width
from tier_a.env_attitude.scenarios import build_local_gauntlet_env


def test_presented_half_width_is_r_body_at_level_and_h_body_at_knife_edge():
    assert presented_half_width(0.0) == pytest.approx(R_BODY)
    assert presented_half_width(math.pi / 2) == pytest.approx(H_BODY, abs=1e-9)


def test_presented_half_width_decreases_monotonically_with_roll():
    """The correct ellipse-projection formula, not the rotated-rectangle one
    (which has a spurious local maximum a few degrees off level) -- see the
    module docstring for the derivation."""
    angles = np.linspace(0, math.pi / 2, 50)
    widths = [presented_half_width(a) for a in angles]
    assert all(w2 <= w1 + 1e-12 for w1, w2 in zip(widths, widths[1:]))


def test_collides_oriented_with_zero_body_matches_plain_collides():
    trees = np.array([[5.0, 5.0]])
    for phi in (0.0, math.radians(45), math.radians(89)):
        for xy in [(5.0, 5.0), (5.3, 5.0), (5.5, 5.0)]:
            assert collides_oriented(xy, 1.6, phi, trees, r_body=0.0, h_body=0.0) == collides(xy, 1.6, trees)


def test_a_gate_too_narrow_when_level_is_passable_when_rolled():
    """The core physical claim: a gate narrower than TREE_R+R_BODY (level
    footprint) but wider than TREE_R+H_BODY (knife-edge footprint) collides
    at phi=0 and clears at phi close to 90 deg, with the SAME trees."""
    clearance = 0.5 * (TREE_R + R_BODY + TREE_R + H_BODY)  # midpoint of the two thresholds
    assert TREE_R + H_BODY < clearance < TREE_R + R_BODY

    env_level = build_local_gauntlet_env(seed=0, clearance=clearance, body_aware_collision=True)
    env_level.reset()
    env_level._state[0], env_level._state[1], env_level._state[6] = 5.0, 5.0, 0.0
    _, _, _, info_level = env_level.step(13)
    assert info_level["collision"] is True

    env_rolled = build_local_gauntlet_env(seed=0, clearance=clearance, body_aware_collision=True)
    env_rolled.reset()
    env_rolled._state[0], env_rolled._state[1], env_rolled._state[6] = 5.0, 5.0, math.radians(88)
    _, _, _, info_rolled = env_rolled.step(13)
    assert info_rolled["collision"] is False


def test_default_point_mass_collision_is_unaffected_by_body_aware_flag_default():
    """body_aware_collision defaults to False -- every prior Tier-A result
    was measured against the point-mass collides(), and must stay so."""
    clearance = 0.5 * (TREE_R + R_BODY + TREE_R + H_BODY)
    env = build_local_gauntlet_env(seed=0, clearance=clearance)  # body_aware_collision not passed
    assert env.body_aware_collision is False
    env.reset()
    env._state[0], env._state[1], env._state[6] = 5.0, 5.0, 0.0
    _, _, _, info = env.step(13)
    assert info["collision"] is False  # only needs TREE_R clearance, which this gate has


def test_phi_max_pirouette_is_looser_than_cruise_phi_max():
    from tier_a.env_attitude.constants import PHI_MAX, PHI_MAX_PIROUETTE
    assert PHI_MAX_PIROUETTE > PHI_MAX
