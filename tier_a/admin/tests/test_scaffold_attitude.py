"""Tier-A scaffold checks: the student notebook must have real holes -- and be
runnable. Mirrors ``tests/test_scaffold.py``'s exec-with-SOLUTION-substituted
pattern exactly, at the notebook's own smoke-config episode budget (150
episodes) so this stays fast, matching the kit's pytest time budget.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

NB = Path(__file__).resolve().parents[2] / "scaffold" / "starter_attitude.ipynb"

SOLUTION = {
    "        target = ...": "        target = r + gamma * target_net(s2).max(dim=1).values * (~terminal).float()",
    "    loss = ...": "    loss = ((q_sa - target) ** 2).mean()",
}


def _cells() -> list[str]:
    nb = json.loads(NB.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_is_valid_json_with_the_expected_structure():
    nb = json.loads(NB.read_text())
    assert nb["nbformat"] == 4
    assert any(c["cell_type"] == "markdown" for c in nb["cells"])


def test_the_holes_are_actually_holes():
    src = "\n".join(_cells())
    assert "TODO(1)" in src and "TODO(2)" in src
    assert "target = ..." in src, "the DQN Bellman target must be left unfilled"
    assert "loss = ..." in src, "the loss must be left unfilled"


def test_scaffold_never_hands_the_student_physical_state():
    """INVARIANT 2 (Tier-A): the physical state may be read only to render
    the 3D flight, never to learn. ``.xyz``/``.attitude`` appear only inside
    the plotting/rollout-for-viz cell, never inside the training loop."""
    ALLOWED = ("plot_flight_3d", "draw_attitude_gizmo", "rollout_with_pose", "xyzs")
    for cell in _cells():
        for line in cell.splitlines():
            if ".xyz" in line or ".attitude" in line or ".trees" in line:
                assert any(a in line for a in ALLOWED), f"scaffold leaks physical state: {line.strip()}"


def test_filled_in_notebook_trains_without_crashing():
    """Execs the notebook's own smoke config (150 episodes) with the two
    holes filled by the intended solution. This is a genuinely hard 6-DOF
    control problem trained from scratch in seconds -- the assertion is that
    the pipeline runs and produces sane numbers, NOT that it reaches the
    goal (see env_attitude/README.md for why success isn't guaranteed at the
    smoke budget)."""
    ns: dict = {}
    for cell in _cells():
        for hole, fix in SOLUTION.items():
            cell = cell.replace(hole, fix)
        exec(compile(cell, "<starter_attitude.ipynb>", "exec"), ns)
        if "qnet, curve = train_local(" in cell:
            break
    else:
        pytest.fail("training cell not found in the notebook")

    assert ns["roll"]["steps"] > 0
    assert ns["roll"]["tracking_rmse"] >= 0.0
    assert len(ns["curve"]) > 0
    assert all(np.isfinite(v) for v in ns["curve"][:, 1])
