"""ctbr.py: алокація, декодування, P-закон, сатурація — чиста математика без pybullet."""

from __future__ import annotations

import numpy as np
import pytest

from tier_b.envs.ctbr import (
    CTBRParams, SIGN_ROLL, SIGN_PITCH, SIGN_YAW,
    allocation_matrix, ctbr_to_rpm, decode_action, mix, rate_p_torques, rpm_from_thrusts,
)

P = CTBRParams()
HOVER_RPM = 14468.429183500699  # step-0 аудит CF2X


def test_allocation_matrix_invertible_and_consistent():
    a = allocation_matrix(P)
    assert np.linalg.cond(a) < 1e7
    ainv = np.linalg.inv(a)
    assert np.allclose(a @ ainv, np.eye(4), atol=1e-9)


def test_hover_thrust_gives_equal_motors_at_hover_rpm():
    f = mix(P.m * P.g, np.zeros(3), P)
    assert np.allclose(f, P.m * P.g / 4.0)
    rpm = rpm_from_thrusts(f, P)
    assert np.allclose(rpm, HOVER_RPM, rtol=1e-6)


def test_decode_action_range():
    c, w = decode_action(np.array([1.0, 0.0, 0.0, 0.0]), P)
    assert c == pytest.approx(2.0 * P.g)
    c, w = decode_action(np.array([-1.0, 1.0, -1.0, 0.5]), P)
    assert c == pytest.approx(0.0)
    assert np.allclose(w, [6.0, -6.0, 1.5])


def test_rate_p_law_sign_and_magnitude():
    tau = rate_p_torques(np.array([3.0, 0.0, 0.0]), np.zeros(3), P)
    assert tau[0] == pytest.approx(P.j_diag[0] / P.tau_c * 3.0)
    assert tau[1] == tau[2] == 0.0
    tau = rate_p_torques(np.zeros(3), np.array([0.0, 2.0, 0.0]), P)
    assert tau[1] < 0  # гасимо наявну швидкість


@pytest.mark.parametrize("axis,signs", [(0, SIGN_ROLL), (1, SIGN_PITCH), (2, SIGN_YAW)])
def test_mixer_motor_signs_match_cf2x_convention(axis, signs):
    tau = np.zeros(3)
    tau[axis] = 1e-4
    f = mix(P.m * P.g, tau, P)
    df = f - P.m * P.g / 4.0
    assert np.all(np.sign(df[np.abs(df) > 1e-12]) == signs[np.abs(df) > 1e-12])


def test_achieved_torque_equals_requested_when_unsaturated():
    tau_req = np.array([2e-4, -1e-4, 5e-6])
    f = mix(P.m * P.g, tau_req, P)
    achieved = allocation_matrix(P) @ f
    assert achieved[0] == pytest.approx(P.m * P.g)
    assert np.allclose(achieved[1:], tau_req, rtol=1e-9)


def test_saturation_keeps_thrust_and_scales_torque_collinearly():
    thrust = 2.0 * P.g * P.m  # c=2g: на мотор лишається 12.5% запасу
    tau_req = np.array([5e-3, 0.0, 0.0])  # свідомо завеликий торк
    f = mix(thrust, tau_req, P)
    assert np.all(f >= -1e-12) and np.all(f <= P.f_motor_max + 1e-12)
    achieved = allocation_matrix(P) @ f
    assert achieved[0] == pytest.approx(thrust, rel=1e-9)  # тяга-пріоритет
    alpha = achieved[1] / tau_req[0]
    assert 0.0 < alpha < 1.0
    assert np.allclose(achieved[1:], alpha * tau_req, atol=1e-12)


def test_full_pipeline_rpm_bounds():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = rng.uniform(-1, 1, 4)
        w = rng.uniform(-8, 8, 3)
        rpm = ctbr_to_rpm(a, w, P)
        assert rpm.shape == (4,)
        assert np.all(rpm >= 0.0) and np.all(rpm <= P.max_rpm + 1e-9)


def test_c_max_2g_is_within_motor_envelope():
    """T2W CF2X = 2.25 → c=2g досяжне з запасом ~12.5% (виміряно на step-0)."""
    f = mix(2.0 * P.g * P.m, np.zeros(3), P)
    assert np.all(f < P.f_motor_max)
    assert (P.f_motor_max * 4) / (P.m * P.g) == pytest.approx(2.25, rel=1e-3)
