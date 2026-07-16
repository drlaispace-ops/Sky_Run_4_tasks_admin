"""Phase 6: the student notebook must have real holes -- and be solvable.

Runs the notebook's own code cells (up to training) with the two ``# TODO``
lines filled in by the intended answer. If the scaffold's training loop, shaping
helper or environment API ever drift, this fails instead of the students.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

NB = Path(__file__).resolve().parents[2] / "scaffold" / "starter.ipynb"

SOLUTION = {
    "    target = ...": "    target = r if terminal else r + gamma * np.max(Q[s2])",
    "    Q[s, a] = ...": "    Q[s, a] = Q[s, a] + alpha * td",
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
    assert "target = ..." in src, "the Bellman target must be left unfilled"
    assert "Q[s, a] = ..." in src, "the update step must be left unfilled"


def test_scaffold_never_hands_the_student_the_map():
    """INVARIANT 2: the map may be read only to score or to draw, never to learn.

    ``optimal_length`` is the oracle (it is *supposed* to see trees), and the
    viz helpers need them to render circles. Anywhere else, a mention of
    ``.trees`` / ``.blocked`` in the students' notebook means the agent is being
    handed the answer.
    """
    ALLOWED = ("optimal_length", "plot_", "value_surface")
    for cell in _cells():
        for line in cell.splitlines():
            if ".trees" in line or ".blocked" in line:
                assert any(a in line for a in ALLOWED), f"scaffold leaks the map: {line.strip()}"


def test_filled_in_notebook_learns_to_reach_the_goal():
    ns: dict = {}
    for cell in _cells():
        for hole, fix in SOLUTION.items():
            cell = cell.replace(hole, fix)
        exec(compile(cell, "<starter.ipynb>", "exec"), ns)
        if "Q, curve = train(" in cell:
            break
    else:
        pytest.fail("training cell not found in the notebook")

    assert ns["roll"]["success"], "the intended solution does not reach the goal"
    assert ns["LSTAR"] / ns["roll"]["length"] <= 1.0 + 1e-9, "efficiency > 1"
    assert len(ns["curve"]) > 0
