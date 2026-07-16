#!/usr/bin/env python3
"""Regenerates tier_a/admin/solution_attitude.ipynb from scaffold/starter_attitude.ipynb by
filling in the two TODO holes with the intended answer -- the same
substitution ``tests/test_scaffold_attitude.py`` uses to validate the
scaffold, kept here so the organizer-facing solution notebook never drifts
from what the tests consider correct.

    python tier_a/admin/make_solution_attitude.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
STARTER = HERE.parent / "scaffold" / "starter_attitude.ipynb"
SOLUTION_NB = HERE / "solution_attitude.ipynb"

SOLUTION = {
    "        target = ...": "        target = r + gamma * target_net(s2).max(dim=1).values * (~terminal).float()",
    "    loss = ...": "    loss = ((q_sa - target) ** 2).mean()",
}


def main() -> None:
    nb = json.loads(STARTER.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        for hole, fix in SOLUTION.items():
            src = src.replace(hole, fix)
        cell["source"] = src.splitlines(keepends=True)

    SOLUTION_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {SOLUTION_NB}")


if __name__ == "__main__":
    main()
