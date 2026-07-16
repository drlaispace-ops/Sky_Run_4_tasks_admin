"""Train N maps, snapshot every 10 episodes, export a compact blob for the web lab.

    python -m tier_d.admin.viz.export_runs

Writes tier_d/admin/reports/runs.json. Layout:

  meta        grid geometry, action table, episode stride
  maps[]      per seed: trees, L*, A* path, blocked bitmask, per-frame metrics
  v / pol     gzip+base64 uint8 blobs, [map][frame][cell]
  paths       gzip+base64 uint16 blob + offsets (greedy path, variable length)

V is stored as the learnt value with obstacle cells filled from
``experienced_obstacle_value`` -- so a tree the agent never hit stays flat, and
the browser renders the same honest surface the report does.
"""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

import numpy as np

from tier_d.admin.agent.qlearning import experienced_obstacle_value, greedy_rollout, train
from tier_d.env.constants import ACTIONS, CELL, DOMAIN, GOAL_XY, GRID_N, R_COLLISION, R_GOAL, START_XY, TREE_R
from tier_d.env.gridworld import GridWorld
from tier_d.env.occupancy import GOAL_CELL, START_CELL
from tier_d.oracle.astar import astar, optimal_length
from tier_d.oracle.value_iteration import dijkstra_cost_to_go

DEMO_SEEDS = tuple(range(10))  # 10 maps. NOT the hidden set (101-110) -- invariant 5.
N_TREES = 16
EPISODES = 3000
STRIDE = 10  # snapshot every 10 episodes

OUT = Path("tier_d/admin/reports/runs.json")

V_LO, V_HI = float(R_COLLISION), float(R_GOAL)  # -100 .. +100


def _q8(V: np.ndarray) -> np.ndarray:
    """Quantise value to uint8 over the fixed [-100, +100] range (same scale for all maps)."""
    x = (np.clip(V, V_LO, V_HI) - V_LO) / (V_HI - V_LO)
    return np.round(x * 255).astype(np.uint8)


def _b64gz(buf: bytes) -> str:
    return base64.b64encode(gzip.compress(buf, 9)).decode()


def run_one(seed: int) -> dict:
    env = GridWorld(N_TREES, seed)
    Lstar = optimal_length(env.trees)
    Q, hist = train(env, episodes=EPISODES, seed=seed, snapshot_every=STRIDE, eval_every=0)

    coll = np.cumsum(np.array(hist.collision, dtype=int))
    succ = np.cumsum(np.array(hist.success, dtype=int))

    vs, pols, paths, frames = [], [], [], []
    for ep, _, Qsnap in hist.v_snapshots:
        V = Qsnap.max(axis=1).reshape(GRID_N, GRID_N).copy()
        peaks = experienced_obstacle_value(Qsnap, env.blocked)
        V[env.blocked] = peaks[env.blocked]
        V[GOAL_CELL] = R_GOAL
        vs.append(_q8(V).ravel())
        pols.append(Qsnap.argmax(axis=1).astype(np.uint8))

        roll = greedy_rollout(GridWorld(N_TREES, seed), Qsnap)
        cells = np.array(
            [r * GRID_N + c for r, c in
             ((int(y / CELL), int(x / CELL)) for x, y in roll["path"])],
            dtype=np.uint16,
        )
        paths.append(cells)

        learned = int((peaks[env.blocked] < -1.0).sum())
        frames.append({
            "ep": int(ep),
            "len": round(float(roll["length"]), 3) if roll["success"] else None,
            "ok": bool(roll["success"]),
            "hit": bool(roll["collision"]),
            "coll": int(coll[min(ep, len(coll) - 1)]),
            "succ": int(succ[min(ep, len(succ) - 1)]),
            "learned": learned,
        })

    blocked_bits = np.packbits(env.blocked.ravel()).tobytes()
    a_star = astar(env.blocked)
    return {
        "seed": seed,
        "trees": [[round(float(x), 4), round(float(y), 4)] for x, y in env.trees],
        "n_blocked": int(env.blocked.sum()),
        "Lstar": round(float(Lstar), 4),
        "astar": [int(r * GRID_N + c) for r, c in a_star[1]] if a_star else [],
        "blocked": base64.b64encode(blocked_bits).decode(),
        "frames": frames,
        "_v": np.concatenate(vs),
        "_pol": np.concatenate(pols),
        "_paths": paths,
    }


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    maps, V, POL, PATHS, offsets = [], [], [], [], [0]

    for seed in DEMO_SEEDS:
        r = run_one(seed)
        V.append(r.pop("_v"))
        POL.append(r.pop("_pol"))
        for p in r.pop("_paths"):
            PATHS.append(p)
            offsets.append(offsets[-1] + len(p))
        maps.append(r)
        ok = sum(f["ok"] for f in r["frames"])
        print(f"  seed {seed:>2}: {len(r['frames'])} frames · L*={r['Lstar']:.2f} "
              f"· greedy solves {ok}/{len(r['frames'])} snapshots")

    payload = {
        "meta": {
            "grid": GRID_N, "cell": CELL, "domain": DOMAIN, "tree_r": TREE_R,
            "start": list(START_XY), "goal": list(GOAL_XY),
            "start_cell": START_CELL[0] * GRID_N + START_CELL[1],
            "goal_cell": GOAL_CELL[0] * GRID_N + GOAL_CELL[1],
            "actions": [[dr, dc] for dr, dc, _ in ACTIONS],
            "episodes": EPISODES, "stride": STRIDE, "n_trees": N_TREES,
            "v_lo": V_LO, "v_hi": V_HI,
            "n_frames": len(maps[0]["frames"]),
        },
        "maps": maps,
        "v": _b64gz(np.concatenate(V).tobytes()),
        "pol": _b64gz(np.concatenate(POL).tobytes()),
        "paths": _b64gz(np.concatenate(PATHS).tobytes()),
        "path_offsets": offsets,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    kb = OUT.stat().st_size / 1024
    print(f"\nwrote {OUT} ({kb:.0f} KB) · {len(maps)} maps × {payload['meta']['n_frames']} frames")


if __name__ == "__main__":
    main()
