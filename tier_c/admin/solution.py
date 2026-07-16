# -*- coding: utf-8 -*-
"""
ОРГАНІЗАТОРСЬКИЙ РЕФЕРЕНС — НЕ роздавати учасникам хакатону.

Дзеркалить структуру/порядок кореневого solution.py (та сама «форма», яку
заповнюють учасники), але з РОБОЧОЮ реалізацією — зручно для `diff solution.py
admin/solution.py` при перевірці чи для тимчасової підміни
(скопіювати вміст сюди в solution.py), щоб перевірити, що решта проєкту
(sim_headless.py, blender_manual.py) і далі коректно працює з «еталонним»
розв'язком.

Плюс сюди ж перенесена математика потенціалу APF (attraction_potential/
repulsion_potential/potential_field_grid/blended_display_field) — вона малює
ЕТАЛОННЕ поле притягання/відштовхування для адмінської теплокарти в Blender
HUD і в HTML-звіті метрик (blender_manual.py._draw_apf_map,
metrics_report.py), тож теж не повинна потрапляти до учасників — інакше
Адмін-режим (пароль лежить прямо у вихідному коді) став би готовою підказкою
до правильного розв'язку."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from astar2d import build_occupancy_grid, world_to_cell, cell_to_world

Vec2 = Tuple[float, float]


# ═══════════════════════════ ШАР 1 — A*: глобальне планування шляху ═══════════════

_NEI = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]


def find_path(md, cfg, cell_size: float = 1.0) -> Optional[List[Vec2]]:
    """A* від md.start до md.checkpoints[0] в обхід перешкод, 8-зв'язність,
    октильна евристика — над сіткою зайнятості build_occupancy_grid."""
    import heapq
    b = cfg.bounds
    grid, cs, nx, ny = build_occupancy_grid(md, cfg, cell_size)

    sx, sy = world_to_cell(md.start[0], md.start[1], b, cs)
    gx, gy = world_to_cell(md.checkpoints[0][0], md.checkpoints[0][1], b, cs)

    def blocked(i, j):
        return not (0 <= i < nx and 0 <= j < ny) or grid[j][i]

    if blocked(sx, sy) or blocked(gx, gy):
        return None

    def h(i, j):
        dx, dy = abs(i - gx), abs(j - gy)
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

    open_heap = [(h(sx, sy), 0.0, (sx, sy))]
    came_from = {}
    g_score = {(sx, sy): 0.0}
    closed = set()

    while open_heap:
        _, g, (ci, cj) = heapq.heappop(open_heap)
        if (ci, cj) in closed:
            continue
        if (ci, cj) == (gx, gy):
            path_cells = [(ci, cj)]
            while (ci, cj) in came_from:
                ci, cj = came_from[(ci, cj)]
                path_cells.append((ci, cj))
            path_cells.reverse()
            return [cell_to_world(i, j, b, cs) for i, j in path_cells]
        closed.add((ci, cj))

        for di, dj, cost in _NEI:
            ni, nj = ci + di, cj + dj
            if blocked(ni, nj) or (ni, nj) in closed:
                continue
            if di != 0 and dj != 0 and blocked(ci + di, cj) and blocked(ci, cj + dj):
                continue
            ng = g + cost
            if ng < g_score.get((ni, nj), math.inf):
                g_score[(ni, nj)] = ng
                came_from[(ni, nj)] = (ci, cj)
                heapq.heappush(open_heap, (ng + h(ni, nj), ng, (ni, nj)))

    return None


# ═══════════════════ ШАР 2 — APF: реактивне уникнення перешкод ════════════════════

@dataclass
class APFParams:
    k_attract: float = 1.0
    k_repulse: float = 6.0
    influence_radius: float = 4.0
    lookahead: float = 3.0
    stuck_boost_factor: float = 3.0
    stuck_boost_duration: float = 3.0


def _attraction_force(pos: Vec2, target: Vec2, k: float) -> Vec2:
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    return (k * dx, k * dy)


def _repulsion_force(lidar, bin_angles, k: float, d0: float) -> Vec2:
    fx = fy = 0.0
    for d, ang in zip(lidar, bin_angles):
        if d < d0 and d > 1e-6:
            mag = k * (1.0 / d - 1.0 / d0) / (d * d)
            fx -= mag * math.cos(ang)
            fy -= mag * math.sin(ang)
    return (fx, fy)


def compute_desired_direction(pos: Vec2, lidar, bin_angles, tracker, stuck,
                              t: float, p: APFParams, boost_state: dict):
    """Ціль притягує (морквина на A*-шляху), кожен активний сектор лідара
    відштовхує; сума векторів → бажаний напрям. Застрягання → тимчасовий
    буст притягання (вихід із локального мінімуму)."""
    is_stuck_now = stuck.update(t, pos[0], pos[1])
    if is_stuck_now:
        boost_state["until_t"] = t + p.stuck_boost_duration
    boosted = t < boost_state.get("until_t", -math.inf)
    k_attract = p.k_attract * p.stuck_boost_factor if boosted else p.k_attract

    carrot = tracker.lookahead_point(pos, p.lookahead)
    fax, fay = _attraction_force(pos, carrot, k_attract)
    frx, fry = _repulsion_force(lidar, bin_angles, p.k_repulse, p.influence_radius)
    fx, fy = fax + frx, fay + fry

    norm = math.hypot(fx, fy)
    direction = (fx / norm, fy / norm) if norm > 1e-6 else (0.0, 0.0)
    return direction, boosted, carrot


# ═══════════════════════ ШАР 3 — детектор застрягання ═════════════════════════════

@dataclass
class StuckDetector:
    """Застрягання = розмах руху дрона (bounding-box діагональ) за ковзне
    вікно window_s менший за radius — див. admin-докстрінг у git-історії
    apf_controller.py для повного обґрунтування підходу."""
    radius: float = 2.2
    window_s: float = 3.0
    _history: List[Tuple[float, float, float]] = field(default_factory=list)

    def update(self, t: float, x: float, y: float) -> bool:
        self._history.append((t, x, y))
        self._history = [(ht, hx, hy) for (ht, hx, hy) in self._history
                         if t - ht <= self.window_s]
        ht0 = self._history[0][0]
        if t - ht0 < self.window_s * 0.9:
            return False
        xs = [hx for _ht, hx, _hy in self._history]
        ys = [hy for _ht, _hx, hy in self._history]
        span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        return span < self.radius


# ═══════════════════ ШАР 4 — кінематичний автопілот-виконавець ════════════════════

@dataclass
class AutopilotParams:
    max_speed: float = 3.0
    max_accel: float = 4.0
    alt_clearance: float = 2.5
    max_climb_rate: float = 2.0
    yaw_rate_max: float = 4.0
    yaw_tau: float = 0.20
    max_tilt: float = math.radians(30.0)
    pitch_visual_gain: float = 0.8
    roll_visual_gain: float = 0.8
    att_tau: float = 0.15
    proximity_slow_dist: float = 3.0
    proximity_min_scale: float = 0.25


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _clip(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return lo if v < lo else hi if v > hi else v


def step_autopilot(state, direction_xy: Vec2, terrain, dt: float,
                   p: AutopilotParams = AutopilotParams(), min_lidar: float = None):
    """Кінематичний крок: швидкість НАПРЯМУ переслідує бажаний напрям з
    обмеженим прискоренням; висота — contour-following над рельєфом; pitch/roll
    — лише ВІЗУАЛЬНІ, з результуючого прискорення заднім числом."""
    from kinematic_autopilot import AutopilotState

    dx, dy = direction_xy

    prox_factor = 1.0
    if min_lidar is not None:
        prox_factor = p.proximity_min_scale + (1.0 - p.proximity_min_scale) * _clip(
            min_lidar / p.proximity_slow_dist, 0.0, 1.0)
    target_speed = p.max_speed * prox_factor

    vx_des, vy_des = dx * target_speed, dy * target_speed
    dvx, dvy = vx_des - state.vx, vy_des - state.vy
    dv = math.hypot(dvx, dvy)
    max_dv = p.max_accel * dt
    if dv > max_dv and dv > 1e-9:
        scale = max_dv / dv
        dvx *= scale
        dvy *= scale
    ax, ay = dvx / dt, dvy / dt
    new_vx, new_vy = state.vx + dvx, state.vy + dvy
    new_x, new_y = state.x + new_vx * dt, state.y + new_vy * dt

    target_z = terrain.height_at(new_x, new_y) + p.alt_clearance
    dz = _clip(target_z - state.z, -p.max_climb_rate * dt, p.max_climb_rate * dt)
    new_z = state.z + dz
    new_vz = dz / dt

    speed = math.hypot(new_vx, new_vy)
    if speed > 0.05:
        target_yaw = math.atan2(new_vy, new_vx)
        err = _wrap_pi(target_yaw - state.yaw)
        max_dyaw = p.yaw_rate_max * dt
        a_yaw = 1.0 - math.exp(-dt / p.yaw_tau)
        new_yaw = state.yaw + _clip(err * a_yaw, -max_dyaw, max_dyaw)
    else:
        new_yaw = state.yaw

    fwd_x, fwd_y = math.cos(new_yaw), math.sin(new_yaw)
    right_x, right_y = math.sin(new_yaw), -math.cos(new_yaw)
    a_fwd = ax * fwd_x + ay * fwd_y
    a_lat = ax * right_x + ay * right_y
    target_pitch = -_clip(p.pitch_visual_gain * a_fwd / max(p.max_accel, 1e-6)) * p.max_tilt
    target_roll = _clip(p.roll_visual_gain * a_lat / max(p.max_accel, 1e-6)) * p.max_tilt
    a_att = 1.0 - math.exp(-dt / p.att_tau)
    new_pitch = state.pitch + (target_pitch - state.pitch) * a_att
    new_roll = state.roll + (target_roll - state.roll) * a_att

    return AutopilotState(new_x, new_y, new_z, new_vx, new_vy, new_vz, new_yaw, new_pitch, new_roll)


# ── Потенціал (для адмінської APF-карти — HUD/HTML, показує ЕТАЛОННЕ поле) ────────

def attraction_potential(pos: Vec2, target: Vec2, k: float) -> float:
    """U = 0.5·k·|target-pos|² — енергія, чий градієнт (зі знаком мінус) дає
    _attraction_force вище."""
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    return 0.5 * k * (dx * dx + dy * dy)


def repulsion_potential(dist: float, k: float, d0: float) -> float:
    """U = 0.5·k·(1/d − 1/d0)² для d<d0 (0 далі за d0)."""
    if dist >= d0:
        return 0.0
    d = max(dist, 1e-6)
    return 0.5 * k * (1.0 / d - 1.0 / d0) ** 2


def potential_field_grid(obs_x, obs_y, obs_r, target_xy: Vec2, bounds: float, n: int,
                          k_attract: float, k_repulse: float, d0: float,
                          d_floor: float = 0.35):
    """Векторизована n×n сітка (numpy) над [-bounds,bounds]². Повертає
    (xs, ys, U_attract, U_repulse) — ОКРЕМО, не сумою (див. blended_display_field)."""
    xs = np.linspace(-bounds, bounds, n)
    ys = np.linspace(-bounds, bounds, n)
    X, Y = np.meshgrid(xs, ys)
    U_attract = 0.5 * k_attract * ((X - target_xy[0]) ** 2 + (Y - target_xy[1]) ** 2)

    ox = np.asarray(obs_x, dtype=float)[None, None, :]
    oy = np.asarray(obs_y, dtype=float)[None, None, :]
    orad = np.asarray(obs_r, dtype=float)[None, None, :]
    d = np.sqrt((X[..., None] - ox) ** 2 + (Y[..., None] - oy) ** 2) - orad
    d = np.maximum(d, d_floor)
    active = d < d0
    U_repulse = np.where(active, 0.5 * k_repulse * (1.0 / d - 1.0 / d0) ** 2, 0.0).sum(axis=-1)

    return xs, ys, U_attract, U_repulse


def blended_display_field(U_attract, U_repulse, w_attract: float = 0.45,
                          w_repulse: float = 0.55, pct: float = 97.0):
    """Незалежна нормалізація 0..1 кожного шару (U_repulse — через percentile,
    щоб поодинокі піки не спалили контраст), тоді зважена сума."""
    def _norm(U, use_pct):
        lo = float(U.min())
        hi = float(np.percentile(U, pct)) if use_pct else float(U.max())
        if hi - lo < 1e-9:
            return np.zeros_like(U)
        return np.clip((U - lo) / (hi - lo), 0.0, 1.0)

    a = _norm(U_attract, use_pct=False)
    r = _norm(U_repulse, use_pct=True)
    return np.clip(w_attract * a + w_repulse * r, 0.0, 1.0)


# Пароль адмін-режиму Blender-гри (blender_manual.py імпортує його звідси;
# в учасницькій копії без admin/ адмін-режим недоступний узагалі).
ADMIN_PASSWORD = "2wsx3edc0"
