"""Offline L* + feasibility cache. Never runs in the hot loop during the event.

Writes ``tier_d/scoring/cache_Lstar.json``. Infeasible seeds are recorded with
``"Lstar": null`` so the harness can exclude them automatically instead of
silently scoring a team against an impossible map.
"""

from __future__ import annotations

import json
from pathlib import Path

from tier_d.env.forest import make_forest
from tier_d.oracle.astar import is_feasible, optimal_length
from tier_d.scoring.seeds import HIDDEN_SEEDS, N_TREES, PUBLIC_SEEDS

CACHE = Path(__file__).parent / "cache_Lstar.json"


def build_cache(n_trees: int = N_TREES) -> dict:
    cache: dict[str, dict] = {}
    for group, seeds in (("public", PUBLIC_SEEDS), ("hidden", HIDDEN_SEEDS)):
        for s in seeds:
            trees = make_forest(n_trees, s)
            feasible = is_feasible(trees)
            cache[f"{n_trees}:{s}"] = {
                "group": group,
                "seed": s,
                "n_trees": n_trees,
                "trees_placed": int(len(trees)),
                "feasible": feasible,
                "Lstar": optimal_length(trees) if feasible else None,
            }
    return cache


def load_cache() -> dict:
    if not CACHE.exists():
        raise FileNotFoundError(f"{CACHE} missing -- run: python -m tier_d.admin.scoring.precompute")
    return json.loads(CACHE.read_text())


def main() -> None:
    cache = build_cache()
    CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))
    feasible = sum(v["feasible"] for v in cache.values())
    print(f"wrote {CACHE} : {len(cache)} entries, {feasible} feasible, "
          f"{len(cache) - feasible} excluded as infeasible")
    for k, v in sorted(cache.items()):
        L = f"{v['Lstar']:.3f}" if v["Lstar"] else "INFEASIBLE"
        print(f"  {v['group']:>6} seed={v['seed']:<4} trees={v['trees_placed']:<3} L*={L}")


if __name__ == "__main__":
    main()
