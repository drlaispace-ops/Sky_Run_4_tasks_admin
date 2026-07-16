"""Phase-2 gate: відкритий контур CTBR у реальній фізиці PyBullet.

(a) c=g, ω_sp=0 → зависання (майже нульове прискорення)
(b) c=2g       → чистий підйом ≈ +g
(c) ω_x=3 рад/с → виміряна швидкість крену: правильний знак, ±20% за 0.3 с
(d) знакова матриця всіх трьох осей
(e) трекінг на межі ±6 рад/с — вирішує, чи потрібен substep-фолбек

Біжить лише у conda `drones` (importorskip)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pybullet")
pytest.importorskip("gym_pybullet_drones")

from tier_b.envs.pirouette_aviary import OBS_DIM, PirouetteAviary  # noqa: E402


@pytest.fixture()
def env():
    e = PirouetteAviary(level=0, master_seed=0)
    e.reset(seed=0)
    yield e
    e.close()


def _body_rates(e):
    import pybullet as p

    rot = np.array(p.getMatrixFromQuaternion(e.quat[0])).reshape(3, 3)
    return rot.T @ e.ang_v[0]


def _run(e, action, seconds):
    n = int(round(seconds * e.CTRL_FREQ))
    for _ in range(n):
        obs, r, term, trunc, info = e.step(np.array([action], dtype=np.float32))
        if term or trunc:
            break
    return obs


def test_obs_shape_and_api(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    out = env.step(np.zeros((1, 4), dtype=np.float32))
    assert len(out) == 5


def test_hover_at_c_equals_g(env):
    _run(env, [0.0, 0.0, 0.0, 0.0], 1.0)  # a0=0 → c=g
    v = np.linalg.norm(env.vel[0])
    assert v < 0.5, f"дрейф при c=g: |v|={v:.3f} м/с"
    assert abs(env.pos[0][2] - 1.0) < 0.3


def test_full_thrust_gives_net_g_up(env):
    t = 0.5
    _run(env, [1.0, 0.0, 0.0, 0.0], t)  # c=2g → чистий +g
    vz = env.vel[0][2]
    assert vz == pytest.approx(env.G * t, rel=0.2), f"vz={vz:.2f}, чекали ≈{env.G*t:.2f}"


def test_roll_rate_tracking_3rad(env):
    _run(env, [0.0, 0.5, 0.0, 0.0], 0.3)  # ω_x = 3 рад/с
    wx = _body_rates(env)[0]
    assert wx > 0, "знак крену неправильний"
    assert wx == pytest.approx(3.0, rel=0.2), f"трекінг ω_x: {wx:.2f} із 3.0"


@pytest.mark.parametrize("axis,idx,setpoint", [
    ("roll", 1, 3.0), ("pitch", 2, 3.0), ("yaw", 3, 1.5),
])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_sign_matrix_all_axes(axis, idx, setpoint, sign):
    e = PirouetteAviary(level=0, master_seed=0)
    e.reset(seed=0)
    a = [0.0, 0.0, 0.0, 0.0]
    a[idx] = sign * (setpoint / (6.0 if idx < 3 else 3.0))
    _run(e, a, 0.25)
    w = _body_rates(e)[idx - 1]
    e.close()
    assert np.sign(w) == sign, f"{axis}: команда {sign:+}, виміряно {w:+.2f}"
    assert abs(w) > 0.3 * setpoint


def test_rate_tracking_at_limit_6rad():
    """Гейт фолбека: якщо на ±6 рад/с похибка > 20% — вмикати substep-цикл."""
    e = PirouetteAviary(level=0, master_seed=0)
    e.reset(seed=0)
    _run(e, [0.3, 1.0, 0.0, 0.0], 0.3)  # ω_x = 6 рад/с, трохи більша тяга
    wx = _body_rates(e)[0]
    e.close()
    assert wx == pytest.approx(6.0, rel=0.2), \
        f"трекінг на межі: {wx:.2f} із 6.0 — потрібен substep-фолбек"


def test_reset_regenerates_forest_deterministically():
    e1 = PirouetteAviary(level=0, master_seed=7)
    e1.reset()
    trees_ep1 = e1.forest.trees.copy()
    e1.reset()
    trees_ep2 = e1.forest.trees.copy()
    e1.close()
    assert not np.array_equal(trees_ep1, trees_ep2)  # інший епізод → інший ліс

    e2 = PirouetteAviary(level=0, master_seed=7)
    e2.reset()
    e2.reset()
    assert np.array_equal(trees_ep2, e2.forest.trees)  # той самий сід → той самий
    e2.close()
