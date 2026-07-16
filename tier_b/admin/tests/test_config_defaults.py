"""config.yaml ↔ дефолти чистих модулів: одна правда у двох місцях, звірена тестом.
(У головному venv немає yaml — тест чесно скіпається; у conda `drones` він біжить.)"""

from __future__ import annotations

import pytest

yaml = pytest.importorskip("yaml")

from tier_b import load_config  # noqa: E402
from tier_b.envs import collision_ellipsoid, ctbr, forest_gen, lidar  # noqa: E402
from tier_b import rewards  # noqa: E402
from tier_b.envs import trajectory  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_body_semi_axes(cfg):
    assert tuple(cfg["body"]["semi_axes"]) == collision_ellipsoid.SEMI_AXES


def test_action_defaults(cfg):
    assert cfg["action"]["c_max_g"] == ctbr.C_MAX_G
    assert tuple(cfg["action"]["rate_limits"]) == tuple(ctbr.RATE_LIMITS)
    assert cfg["action"]["tau_c"] == ctbr.TAU_C


def test_lidar_defaults(cfg):
    assert cfg["lidar"]["n_rays"] == lidar.N_RAYS
    assert cfg["lidar"]["max_range"] == lidar.MAX_RANGE


def test_forest_defaults(cfg):
    f = cfg["forest"]
    assert tuple(f["gate_x"]) == forest_gen.GATE_X
    assert f["tree_radius"] == forest_gen.TREE_RADIUS
    assert f["tree_height"] == forest_gen.TREE_HEIGHT
    assert tuple(f["n_distractors"]) == forest_gen.N_DISTRACTORS
    assert f["corridor_clearance"] == forest_gen.CORRIDOR_CLEARANCE
    assert f["cruise_z"] == forest_gen.CRUISE_Z
    assert tuple(f["start"]) == forest_gen.START_XY
    assert tuple(f["finish"]) == forest_gen.FINISH_XY


def test_reward_defaults(cfg):
    r = cfg["reward"]
    assert r["gamma"] == rewards.GAMMA
    assert r["shaping_gamma"] == rewards.SHAPING_GAMMA
    assert r["k_prog"] == rewards.K_PROG
    assert r["k_gate"] == rewards.K_GATE
    assert r["collision_base"] == rewards.COLLISION_BASE
    assert r["collision_depth_scale"] == rewards.COLLISION_DEPTH_SCALE
    assert r["collision_depth_ref"] == rewards.COLLISION_DEPTH_REF
    assert r["z_min"] == rewards.Z_MIN
    assert r["k_alt"] == rewards.K_ALT
    assert r["k_omega"] == rewards.K_OMEGA
    assert r["step_cost"] == rewards.STEP_COST
    assert r["gate_bonus"] == rewards.GATE_BONUS
    assert r["finish_bonus"] == rewards.FINISH_BONUS


def test_curriculum_gaps_match_aviary(cfg):
    pytest.importorskip("gym_pybullet_drones")
    from tier_b.callbacks import N_LEVELS
    from tier_b.envs.pirouette_aviary import GAPS

    assert tuple(cfg["curriculum"]["gaps"]) == GAPS
    assert len(GAPS) == N_LEVELS
    # драбина строго спадна, хвіст — knife-edge геометрія Tier-A+
    assert all(a > b for a, b in zip(GAPS, GAPS[1:]))
    assert GAPS[-1] == 0.12


def test_waypoint_defaults(cfg):
    assert cfg["waypoints"]["pass_radius"] == trajectory.PASS_RADIUS
    assert cfg["jam"]["finish_zone"] == trajectory.FINISH_RADIUS
