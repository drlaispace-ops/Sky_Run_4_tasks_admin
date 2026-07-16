"""forest_gen + trajectory: відтворюваність, геометрія воріт, трекер, потенціали."""

from __future__ import annotations

import numpy as np
import pytest

from tier_b.envs.forest_gen import (
    CORRIDOR_CLEARANCE, TREE_RADIUS, generate_forest, episode_seed, point_polyline_dist,
)
from tier_b.envs.trajectory import GateTracker, build_waypoints


def test_same_seed_same_forest_byte_for_byte():
    a = generate_forest(gap=0.30, seed=42)
    b = generate_forest(gap=0.30, seed=42)
    assert np.array_equal(a.trees, b.trees)
    assert np.array_equal(a.gate_centers, b.gate_centers)


def test_different_seed_different_forest():
    a = generate_forest(gap=0.30, seed=1)
    b = generate_forest(gap=0.30, seed=2)
    assert not np.array_equal(a.trees, b.trees)


def test_episode_seed_is_injective_over_reasonable_ranges():
    seen = {episode_seed(m, e) for m in range(3) for e in range(9999)}
    assert len(seen) == 3 * 9999


def test_gate_trees_geometry():
    gap = 0.26
    f = generate_forest(gap=gap, seed=7)
    for gi, (gx, cy) in enumerate(f.gate_centers):
        t1, t2 = f.trees[2 * gi], f.trees[2 * gi + 1]
        assert t1[0] == pytest.approx(gx) and t2[0] == pytest.approx(gx)
        # відстань між ЦЕНТРАМИ = gap + 2r → вільна щілина між поверхнями = gap
        assert abs(t2[1] - t1[1]) == pytest.approx(gap + 2 * TREE_RADIUS)
        assert (t1[1] + t2[1]) / 2 == pytest.approx(cy)


def test_distractors_stay_out_of_corridor():
    f = generate_forest(gap=0.30, seed=123)
    wps = build_waypoints(f)
    poly = wps[:, :2]
    for tree in f.trees[6:]:  # перші 6 — дерева воріт
        assert point_polyline_dist(tree[:2], poly) >= CORRIDOR_CLEARANCE - 1e-9


def test_waypoints_shape_and_height():
    f = generate_forest(gap=0.30, seed=5)
    wps = build_waypoints(f)
    assert wps.shape == (5, 3)
    assert np.all(wps[:, 2] == 1.0)


def _fly_straight_through(tracker: GateTracker, wps: np.ndarray, n_per_seg: int = 40):
    events = []
    for i in range(len(wps) - 1):
        for t in np.linspace(0, 1, n_per_seg, endpoint=False)[1:]:
            pos = wps[i] + t * (wps[i + 1] - wps[i])
            events.append(tracker.update(pos))
    events.append(tracker.update(wps[-1]))
    return events


def test_tracker_counts_gates_and_finish_on_center_flight():
    f = generate_forest(gap=0.30, seed=9)
    wps = build_waypoints(f)
    tr = GateTracker(f, wps)
    events = _fly_straight_through(tr, wps)
    assert tr.gates_passed == 3
    assert tr.finished
    assert sum(e["gate_passed"] for e in events) == 3
    assert sum(e["finished"] for e in events) == 1


def test_tracker_gives_no_gate_bonus_when_flying_around():
    f = generate_forest(gap=0.30, seed=9)
    wps = build_waypoints(f)
    tr = GateTracker(f, wps)
    gx, cy = f.gate_centers[0]
    # обхід: перетинаємо площину воріт на 2 м вище центру по y
    tr.update(np.array([gx - 0.5, cy + 2.0, 1.0]))
    ev = tr.update(np.array([gx + 0.5, cy + 2.0, 1.0]))
    assert not ev["gate_passed"]
    assert tr.gates_passed == 0


def test_remaining_length_decreases_along_route():
    f = generate_forest(gap=0.30, seed=11)
    wps = build_waypoints(f)
    tr = GateTracker(f, wps)
    prev = tr.remaining_length(wps[0])
    for i in range(len(wps) - 1):
        for t in np.linspace(0, 1, 20, endpoint=False)[1:]:
            pos = wps[i] + t * (wps[i + 1] - wps[i])
            tr.update(pos)
            cur = tr.remaining_length(pos)
            assert cur <= prev + 1e-6
            prev = cur


def test_phi_gate_targets_next_unpassed_gate():
    f = generate_forest(gap=0.30, seed=13)
    wps = build_waypoints(f)
    tr = GateTracker(f, wps)
    pos = wps[0]
    assert tr.phi_gate(pos) == pytest.approx(-np.linalg.norm(pos - wps[1]))
    _fly_straight_through(tr, wps[:3])  # пролетіли перші двоє воріт
    assert tr.gates_passed == 2
    p2 = wps[2]
    assert tr.phi_gate(p2) == pytest.approx(-np.linalg.norm(p2 - wps[3]))
