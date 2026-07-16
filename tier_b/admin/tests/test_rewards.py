"""rewards.py: інваріант 4, градуйований урон, анти-reward-hacking порядок."""

from __future__ import annotations

import numpy as np
import pytest

from tier_b.rewards import (
    GATE_BONUS, collision_penalty, omega_penalty, shaping, step_reward,
)


def test_shaping_zeroes_potential_at_terminal():
    assert shaping(phi_new=-3.7, phi_old=-5.0, gamma=0.99, terminal=True) == \
        pytest.approx(5.0)


def test_shaping_sums_to_zero_around_a_cycle():
    rng = np.random.default_rng(0)
    phis = list(rng.normal(size=6))
    phis.append(phis[0])
    total = sum(shaping(b, a, gamma=1.0, terminal=False)
                for a, b in zip(phis, phis[1:]))
    assert total == pytest.approx(0.0, abs=1e-12)


def test_collision_penalty_graduated():
    assert collision_penalty(0.0) == pytest.approx(-100.0)
    assert collision_penalty(0.04) == pytest.approx(-200.0)
    assert collision_penalty(1.0) == pytest.approx(-200.0)  # насичення
    assert collision_penalty(0.02) == pytest.approx(-150.0)
    d = np.linspace(0, 0.05, 20)
    p = [collision_penalty(x) for x in d]
    assert all(b <= a for a, b in zip(p, p[1:]))  # монотонно гіршає


def test_standstill_earns_exactly_zero_shaping():
    """Якір виміряного фарму (probe 300k): за γ<1 у формі shaping стояння
    далеко від цілі давало +0.58/крок; телескопічна форма (Φ′−Φ) дає рівно 0,
    і повний крок на місці — лише step_cost."""
    from tier_b.rewards import SHAPING_GAMMA, STEP_COST

    assert SHAPING_GAMMA == 1.0
    assert shaping(-12.0, -12.0, gamma=SHAPING_GAMMA, terminal=False) == 0.0
    r = step_reward(phi_prog_new=-12.0, phi_prog_old=-12.0,
                    phi_gate_new=-3.0, phi_gate_old=-3.0,
                    z=1.0, omega=(0, 0, 0), terminal=False)
    assert r == pytest.approx(STEP_COST)


def test_collision_dominates_max_potential_cashout():
    """Структурний якір виміряного smoke-хака: Φ(terminal)=0 обналічує
    −Φ(s_крашу) ≤ k_prog·14 + k_gate·4.2 ≈ 78 у будь-якій точці, де взагалі
    є дерево. |collision_base| мусить бути БІЛЬШИМ, інакше «розбитися об
    перше дерево» — локальний оптимум (виміряно: collision-rate ріс 9%→22%)."""
    from tier_b.rewards import COLLISION_BASE, K_GATE, K_PROG

    max_cashout = K_PROG * 14.0 + K_GATE * 4.2
    assert -COLLISION_BASE > max_cashout


def test_omega_burst_costs_less_than_gate_bonus():
    """Сплеск 6 рад/с упродовж 0.3 с (≈14.4 кроків при 48 Гц) ≪ бонус воріт —
    ω-штраф не може задушити сам маневр."""
    per_step = -omega_penalty([6.0, 0.0, 0.0])
    assert per_step * 15 < GATE_BONUS / 4


def _episode_return(steps):
    return sum(step_reward(**s) for s in steps)


def _leg(phi_from, phi_to, n, **flags):
    """n кроків рівномірного руху потенціалу; flags застосовуються до останнього."""
    out = []
    xs = np.linspace(phi_from, phi_to, n + 1)
    for i in range(n):
        s = dict(phi_prog_new=xs[i + 1], phi_prog_old=xs[i],
                 phi_gate_new=xs[i + 1] / 3, phi_gate_old=xs[i] / 3,
                 z=1.0, omega=(0.0, 0.0, 0.0), terminal=False)
        if i == n - 1:
            s.update(flags)
        out.append(s)
    return out


def test_anti_hacking_ordering_crash_lt_hover_lt_pass():
    """Ключова властивість: «підлетів і розбився» < «висів до jam» < «пролетів».
    Φ(terminal)=0 конфіскує накопичений потенціал на краші — банкувати shaping
    і розбиватись невигідно."""
    # проліт: потенціал −12 → 0, фініш+ворота
    ep_pass = _leg(-12.0, 0.0, 200, terminal=True, finished=True, gate_passed=True)
    # зависання: потенціал стоїть, 96 кроків, jam = truncation (terminal=False!)
    ep_hover = _leg(-12.0, -12.0, 96)
    # підліт до воріт і краш на них: прогрес до −9, КОВЗНА колізія (depth=0 —
    # найвигідніший для хакінгу випадок: максимальний cash-out, мінімальний штраф)
    ep_crash = _leg(-12.0, -9.0, 50, terminal=True, collided=True, depth=0.0)

    r_pass = _episode_return(ep_pass)
    r_hover = _episode_return(ep_hover)
    r_crash = _episode_return(ep_crash)
    assert r_crash < r_hover < r_pass


def test_jam_is_not_terminal_for_shaping():
    """jam → truncation: shaping НЕ занулює Φ′ (bootstrap зберігається)."""
    r_trunc = step_reward(phi_prog_new=-5.0, phi_prog_old=-5.0,
                          phi_gate_new=-2.0, phi_gate_old=-2.0,
                          z=1.0, omega=(0, 0, 0), terminal=False)
    # той самий стан як «терминал» дав би різко інший shaping
    r_term = step_reward(phi_prog_new=-5.0, phi_prog_old=-5.0,
                         phi_gate_new=-2.0, phi_gate_old=-2.0,
                         z=1.0, omega=(0, 0, 0), terminal=True)
    assert r_term != pytest.approx(r_trunc)


def test_altitude_floor_penalizes_below_z_min_only():
    r_ok = step_reward(phi_prog_new=0, phi_prog_old=0, phi_gate_new=0,
                       phi_gate_old=0, z=0.5, omega=(0, 0, 0), terminal=False)
    r_low = step_reward(phi_prog_new=0, phi_prog_old=0, phi_gate_new=0,
                        phi_gate_old=0, z=0.05, omega=(0, 0, 0), terminal=False)
    assert r_low < r_ok
