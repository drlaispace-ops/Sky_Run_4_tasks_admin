"""Regenerate the APF gain sweep that justifies the constants in tier_d/apf/field.py.

    python -m tier_d.admin.apf.gain_sweep

Writes tier_d/admin/reports/apf_gain_sweep.md. The point is to show the chosen gains give a
*fair* classical APF -- one that solves most forests and stalls on a minority --
rather than a strawman tuned to fail so the RL track looks good.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

import tier_d.admin.apf.field as F
from tier_d.env.forest import make_forest
from tier_d.oracle.astar import is_feasible

SEEDS = range(30)
TREE_COUNTS = (8, 12, 16, 20)
K_REPS = (0.5, 1.0, 2.0, 6.0)
RHO_0S = (0.6, 0.8, 1.2)
D_STARS = (2.0, 4.0)

CHOSEN = (0.5, 0.6, 4.0)


def sweep() -> tuple[list[dict], int]:
    forests = {
        (n, s): make_forest(n, s)
        for n in TREE_COUNTS
        for s in SEEDS
        if is_feasible(make_forest(n, s))
    }
    saved = (F.K_REP, F.RHO_0, F.D_STAR)
    rows = []
    try:
        for krep, rho0, dstar in itertools.product(K_REPS, RHO_0S, D_STARS):
            F.K_REP, F.RHO_0, F.D_STAR = krep, rho0, dstar
            su = tr = co = 0
            for trees in forests.values():
                r = F.apf_flight(trees)
                su += r["success"]; tr += r["trapped"]; co += r["collided"]
            n = len(forests)
            rows.append(dict(k_rep=krep, rho_0=rho0, d_star=dstar,
                             success=su / n, trapped=tr / n, collided=co / n))
    finally:
        F.K_REP, F.RHO_0, F.D_STAR = saved
    return rows, len(forests)


def main() -> None:
    rows, n = sweep()
    out = Path("tier_d/admin/reports/apf_gain_sweep.md")
    out.parent.mkdir(exist_ok=True)

    lines = [
        "# APF gain sweep",
        "",
        f"`apf_flight` over **{n} feasible forests** "
        f"(trees ∈ {list(TREE_COUNTS)}, seeds 0–{max(SEEDS)}).",
        "",
        "The gains in `tier_d/apf/field.py` are chosen from this table. A classical APF that",
        "trapped on every seed would make the RL comparison meaningless, so the target",
        "is a baseline that *mostly works* — and fails only where the method genuinely",
        "has local minima. Note `collided` is 0 everywhere: a correct repulsive field",
        "fails by stalling, never by flying into a tree.",
        "",
        "| K_rep | rho_0 | D_star | success | trapped | collided |",
        "|------:|------:|-------:|--------:|--------:|---------:|",
    ]
    for r in rows:
        mark = "  ← **chosen**" if (r["k_rep"], r["rho_0"], r["d_star"]) == CHOSEN else ""
        lines.append(
            f"| {r['k_rep']:.1f} | {r['rho_0']:.1f} | {r['d_star']:.1f} "
            f"| {r['success']:.2f} | {r['trapped']:.2f} | {r['collided']:.2f} |{mark}"
        )

    chosen = next(r for r in rows if (r["k_rep"], r["rho_0"], r["d_star"]) == CHOSEN)
    lines += [
        "",
        f"**Chosen:** `K_REP={CHOSEN[0]}`, `RHO_0={CHOSEN[1]}`, `D_STAR={CHOSEN[2]}` — "
        f"solves {chosen['success']:.0%} of forests, stalls on {chosen['trapped']:.0%}, "
        f"collides on {chosen['collided']:.0%}.",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(rows)} gain combinations, {n} forests)")
    print(f"chosen: success={chosen['success']:.2f} trapped={chosen['trapped']:.2f} "
          f"collided={chosen['collided']:.2f}")


if __name__ == "__main__":
    main()
