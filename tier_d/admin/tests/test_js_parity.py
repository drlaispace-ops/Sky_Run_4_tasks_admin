"""INVARIANT 1 across the language boundary.

The Field Scanner (tier_d/viz/field_scanner.html) regenerates the forest in JavaScript.
If its ``mulberry32`` drifts from ``tier_d/env/forest.py`` by a single bit, the browser
demo and the Python kit show different forests for the same seed -- and every
cached L* silently becomes wrong for one of them.

Skipped when ``node`` is unavailable, so the suite still runs on the event M4s.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tier_d.env.forest import Mulberry32, make_forest

HTML = Path(__file__).resolve().parents[2] / "viz" / "field_scanner.html"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _js_core() -> str:
    """The RNG + forest section of the page, without any DOM code."""
    html = HTML.read_text()
    return html[html.index("function mulberry32") : html.index("// --- APF")]


def _run_node(script: str) -> dict:
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_html_declares_the_same_constants_as_python():
    from tier_d.env.constants import AB_CLEARANCE, BORDER_MARGIN, MIN_TREE_SEP, TREE_R

    src = _js_core()
    for name, val in (("BORDER_MARGIN", BORDER_MARGIN), ("AB_CLEARANCE", AB_CLEARANCE),
                      ("MIN_TREE_SEP", MIN_TREE_SEP), ("TREE_R", TREE_R)):
        m = re.search(rf"{name}\s*=\s*([0-9.]+)", src)
        assert m, f"{name} not found in field_scanner.html"
        assert float(m.group(1)) == pytest.approx(val), f"{name} drifted"


def test_js_mulberry32_matches_python_bit_for_bit():
    js = _run_node(_js_core() + """
        const r = mulberry32(0); const v = [];
        for (let i = 0; i < 8; i++) v.push(r());
        console.log(JSON.stringify({rng: v}));
    """)
    rng = Mulberry32(0)
    py = [rng() for _ in range(8)]
    assert py == js["rng"], "mulberry32 diverged between Python and JS"


def test_js_make_forest_matches_python_exactly():
    js = _run_node(_js_core() + """
        const out = {};
        for (const n of [0, 8, 16, 24])
          for (const s of [0, 1, 7, 13])
            out[n + ":" + s] = makeForest(n, s);
        console.log(JSON.stringify(out));
    """)
    for key, jstrees in js.items():
        n, s = map(int, key.split(":"))
        py = make_forest(n, s)
        jt = np.array(jstrees, dtype=float).reshape(-1, 2)
        assert jt.shape == py.shape, f"{key}: tree count differs"
        if len(py):
            assert np.array_equal(jt, py), f"{key}: coordinates differ"
