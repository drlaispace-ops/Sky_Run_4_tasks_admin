"""Offline reference-trajectory + feasibility cache for Tier-A. Never runs in
the hot loop during the event.

Writes ``scoring/cache_Lref_attitude.json``. Infeasible seeds are recorded
with ``"Lref": null`` so the harness can exclude them automatically, mirroring
``tier_d/scoring/precompute.py``'s ``cache_Lstar.json`` convention exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

from tier_a.env_attitude.constants import N_TREES_ATTITUDE
from tier_a.env_attitude.forest import make_forest
from tier_a.env_attitude.reference_trajectory import build_reference
from tier_d.oracle.astar import is_feasible
from tier_d.scoring.seeds import HIDDEN_SEEDS, PUBLIC_SEEDS

CACHE = Path(__file__).parent / "cache_Lref_attitude.json"


def build_cache_attitude(n_trees: int = N_TREES_ATTITUDE) -> dict:
    cache: dict[str, dict] = {}
    for group, seeds in (("public", PUBLIC_SEEDS), ("hidden", HIDDEN_SEEDS)):
        for s in seeds:
            trees = make_forest(n_trees, s)
            feasible = is_feasible(trees)
            ref = build_reference(n_trees, s) if feasible else None
            cache[f"{n_trees}:{s}"] = {
                "group": group,
                "seed": s,
                "n_trees": n_trees,
                "trees_placed": int(len(trees)),
                "feasible": feasible,
                "Lref": ref.S_total if ref else None,
                "Tref": ref.T_total if ref else None,
            }
    return cache


def load_cache_attitude() -> dict:
    if not CACHE.exists():
        raise FileNotFoundError(f"{CACHE} missing -- run: python -m tier_a.admin.scoring.precompute_attitude")
    return json.loads(CACHE.read_text())


def main() -> None:
    cache = build_cache_attitude()
    CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))
    feasible = sum(v["feasible"] for v in cache.values())
    print(f"wrote {CACHE} : {len(cache)} entries, {feasible} feasible, "
          f"{len(cache) - feasible} excluded as infeasible")
    for k, v in sorted(cache.items()):
        L = f"{v['Lref']:.3f}" if v["Lref"] else "INFEASIBLE"
        print(f"  {v['group']:>6} seed={v['seed']:<4} trees={v['trees_placed']:<3} Lref={L}")


if __name__ == "__main__":
    main()
