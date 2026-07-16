"""LIDAR у реальному PyBullet: відома геометрія → відомий hitFraction;
фільтр-маска ігнорує все, крім дерев. Біжить лише у conda `drones`."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pybullet")

from tier_b.envs.lidar import MAX_RANGE, N_RAYS, TREE_GROUP, scan  # noqa: E402


@pytest.fixture()
def client():
    import pybullet as p

    c = p.connect(p.DIRECT)
    yield c
    p.disconnect(c)


def _add_tree(client, x, y, r=0.2, h=4.0):
    import pybullet as p

    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h,
                                 physicsClientId=client)
    body = p.createMultiBody(0, col, -1, [x, y, h / 2], physicsClientId=client)
    p.setCollisionFilterGroupMask(body, -1, 1 | TREE_GROUP, -1,
                                  physicsClientId=client)
    return body


def test_forward_ray_measures_known_distance(client):
    _add_tree(client, 1.0, 0.0)
    frac = scan(client, [0, 0, 1.0], [0, 0, 0, 1])
    assert frac.shape == (N_RAYS,)
    # промінь 0 — уперед (+x): поверхня на 0.8 м → 0.8/4.0 = 0.2
    assert frac[0] == pytest.approx(0.8 / MAX_RANGE, abs=0.01)
    # промінь 4 — назад (−x): нічого
    assert frac[4] == pytest.approx(1.0)


def test_non_tree_bodies_are_invisible_to_rays(client):
    """Виміряний нюанс pybullet: маска променя фільтрує лише тіла з ЯВНО
    заданою групою; тіло за замовчуванням промінь влучає завжди. Тому
    "не-дерева" (дрон, земля) отримують явну групу 1 — і стають невидимі."""
    import pybullet as p

    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.3],
                                 physicsClientId=client)
    box = p.createMultiBody(0, col, -1, [1.0, 0.0, 1.0], physicsClientId=client)
    frac = scan(client, [0, 0, 1.0], [0, 0, 0, 1])
    assert frac[0] < 1.0  # група за замовчуванням: промінь влучає (нюанс!)

    p.setCollisionFilterGroupMask(box, -1, 1, -1, physicsClientId=client)
    frac = scan(client, [0, 0, 1.0], [0, 0, 0, 1])
    assert np.all(frac == 1.0)  # явна група 1 → невидимий для LIDAR


def test_rotation_carries_rays_with_the_body(client):
    import pybullet as p

    _add_tree(client, 1.0, 0.0)
    # yaw 90°: тілесний +x дивиться у світовий +y → уперед-промінь має промахнутись,
    # а тілесний −y (промінь 6) — дивитись у світовий +x і влучити
    quat = p.getQuaternionFromEuler([0, 0, np.pi / 2])
    frac = scan(client, [0, 0, 1.0], quat)
    assert frac[0] == pytest.approx(1.0)
    assert frac[6] == pytest.approx(0.8 / MAX_RANGE, abs=0.01)


def test_env_lidar_sees_gate_trees():
    pytest.importorskip("gym_pybullet_drones")
    from tier_b.envs.pirouette_aviary import PirouetteAviary

    e = PirouetteAviary(level=0, master_seed=0)
    obs, _ = e.reset(seed=0)
    lidar = obs[13:13 + N_RAYS]
    assert np.all((lidar >= 0.0) & (lidar <= 1.0))
    # телепорт впритул до першого дерева воріт — щось точно в полі зору
    import pybullet as p

    tx, ty, _ = e.forest.trees[0]
    p.resetBasePositionAndOrientation(e.DRONE_IDS[0], [tx - 0.8, ty, 1.0],
                                      [0, 0, 0, 1], physicsClientId=e.CLIENT)
    e._updateAndStoreKinematicInformation()
    frac = e._computeObs()[13:13 + N_RAYS]
    assert np.any(frac < 0.25), "дерево за 0.6 м не видно жодним променем"
    e.close()
