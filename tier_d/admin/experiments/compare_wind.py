#!/usr/bin/env python3
"""
Controlled experiment: baseline Q-learning vs Q-learning with wind.

This script trains the same agent on two environments:
  1. env.GridWorld (deterministic, baseline)
  2. env_wind.GridWorldWind (stochastic with 5% wind probability)

Both use identical hyperparameters, identical seeds, identical training loop.
The only difference is the environment's stochasticity.

Output: side-by-side comparison of learning curves, showing how wind affects:
  - Speed of convergence
  - Stability of learning
  - Final policy quality (sample efficiency)

    python tier_d/experiments/compare_wind.py --seed 0 --episodes 3000 --wind-prob 0.05
"""

from __future__ import annotations

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from tier_d.env.gridworld import GridWorld, cell_center
from tier_d.env_wind.gridworld import GridWorldWind
from tier_d.env.constants import N_ACTIONS, GRID_N
from tier_d.env.occupancy import GOAL_CELL, START_CELL
from tier_d.oracle.astar import optimal_length, astar
from tier_d.admin.agent.qlearning import train, greedy_rollout


def train_both(seed: int, n_trees: int = 16, episodes: int = 3000, wind_prob: float = 0.05,
               alpha: float = 0.15, gamma: float = 0.99, eps0: float = 1.0, eps_min: float = 0.05):
    """Train both baseline and wind agents with identical hyperparameters."""

    # Baseline
    env_base = GridWorld(n_trees, seed)
    Lstar = optimal_length(env_base.trees)
    trees_xy = env_base.trees.tolist()  # [(x,y), ...]
    print(f"Seed {seed}: L* = {Lstar:.3f}")
    print("Training baseline (deterministic)...", end=" ", flush=True)
    Q_base, curve_base = train(env_base, episodes=episodes, seed=seed, snapshot_every=1)
    print("done.")

    # Wind
    env_wind = GridWorldWind(n_trees, seed, wind_prob=wind_prob)
    print(f"Training with wind (p={wind_prob})...", end=" ", flush=True)
    Q_wind, curve_wind = train(env_wind, episodes=episodes, seed=seed, snapshot_every=1)
    print("done.")

    # Evaluate on greedy policies (deterministic test)
    print("Evaluating greedy policies...")
    roll_base = greedy_rollout(GridWorld(n_trees, seed), Q_base)
    roll_wind = greedy_rollout(GridWorld(n_trees, seed), Q_wind)

    # Compute A* path
    astar_result = astar(env_base.blocked)
    astar_xy = []
    if astar_result is not None:
        _, path_cells = astar_result
        astar_xy = [cell_center(rc[0], rc[1]) for rc in path_cells]

    goal_xy = cell_center(GOAL_CELL[0], GOAL_CELL[1])
    start_xy = cell_center(START_CELL[0], START_CELL[1])

    return {
        "seed": seed,
        "lstar": Lstar,
        "trees": trees_xy,
        "astar_path": astar_xy,
        "start": list(start_xy),
        "goal": list(goal_xy),
        "domain_size": 10.0,
        "baseline": {
            "Q": Q_base,
            "curve": curve_base,
            "rollout": roll_base,
            "success": roll_base["success"],
            "length": roll_base["length"] if roll_base["success"] else np.nan,
            "efficiency": (Lstar / roll_base["length"]) if roll_base["success"] else 0.0,
            "path": roll_base["path"].tolist() if "path" in roll_base else [],
        },
        "wind": {
            "Q": Q_wind,
            "curve": curve_wind,
            "rollout": roll_wind,
            "success": roll_wind["success"],
            "length": roll_wind["length"] if roll_wind["success"] else np.nan,
            "efficiency": (Lstar / roll_wind["length"]) if roll_wind["success"] else 0.0,
            "path": roll_wind["path"].tolist() if "path" in roll_wind else [],
        },
    }


def export_for_viz(all_results: dict, out_path: str = "tier_d/experiments/wind_comparison_data.json"):
    """Export results to JSON for interactive visualization.

    Converts numpy arrays to lists and filters data for web display.
    """
    exported = {}

    for seed_str, results in all_results.items():
        seed = int(seed_str)
        baseline_hist = results["baseline"]["curve"]  # History object
        wind_hist = results["wind"]["curve"]  # History object

        exported[seed_str] = {
            "lstar": float(results["lstar"]),
            "trees": results["trees"],
            "astar_path": results["astar_path"],
            "start": results["start"],
            "goal": results["goal"],
            "domain_size": results["domain_size"],
            "baseline": {
                "L": float(results["baseline"]["length"]),
                "success": bool(results["baseline"]["success"]),
                "efficiency": float(results["baseline"]["efficiency"]),
                "path": results["baseline"]["path"],
                "eval_ep": baseline_hist.eval_ep,
                "eval_len": [float(x) if not np.isinf(x) else None for x in baseline_hist.eval_len],
                "eval_success": baseline_hist.eval_success,
            },
            "wind": {
                "L": float(results["wind"]["length"]),
                "success": bool(results["wind"]["success"]),
                "efficiency": float(results["wind"]["efficiency"]),
                "path": results["wind"]["path"],
                "eval_ep": wind_hist.eval_ep,
                "eval_len": [float(x) if not np.isinf(x) else None for x in wind_hist.eval_len],
                "eval_success": wind_hist.eval_success,
            },
        }

    with open(out_path, "w") as f:
        json.dump(exported, f, indent=2)
    print(f"✓ Exported visualization data to {out_path}")


def _curve_array(hist) -> np.ndarray:
    """History -> Nx2 (episode, eval_len); inf (не долетів) -> nan для панелей."""
    return np.array(
        [[ep, (np.nan if np.isinf(L) else L)] for ep, L in zip(hist.eval_ep, hist.eval_len)],
        dtype=float,
    ).reshape(-1, 2)


def plot_comparison(results: dict, wind_prob: float, out_path: str = "tier_d/admin/reports/wind_comparison.png"):
    """Plot side-by-side comparison of learning curves."""
    seed = results["seed"]
    lstar = results["lstar"]
    curve_base = _curve_array(results["baseline"]["curve"])
    curve_wind = _curve_array(results["wind"]["curve"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Q-Learning: Deterministic vs Wind (seed {seed}, p_wind={wind_prob})", fontsize=13)

    # -- Panel 1: Learning curve (baseline) --
    ax = axes[0, 0]
    ok = ~np.isnan(curve_base[:, 1])
    ax.plot(curve_base[ok, 0], curve_base[ok, 1], color="#0969da", lw=2, label="baseline")
    ax.axhline(lstar, color="#8250df", ls="--", lw=1.3, label=f"L* = {lstar:.2f}")
    ax.axhline(1.15 * lstar, color="grey", ls=":", lw=1.0)
    ax.set_xlabel("episode"); ax.set_ylabel("path length")
    ax.set_title("Baseline (deterministic)")
    ax.legend(); ax.grid(alpha=0.3)
    if curve_base[ok, 1].size > 0:
        ax.set_ylim(lstar - 1, max(curve_base[ok, 1].max(), 1.15 * lstar) + 1)

    # -- Panel 2: Learning curve (wind) --
    ax = axes[0, 1]
    ok = ~np.isnan(curve_wind[:, 1])
    ax.plot(curve_wind[ok, 0], curve_wind[ok, 1], color="#d1260d", lw=2, label="with wind")
    ax.axhline(lstar, color="#8250df", ls="--", lw=1.3, label=f"L* = {lstar:.2f}")
    ax.axhline(1.15 * lstar, color="grey", ls=":", lw=1.0)
    ax.set_xlabel("episode"); ax.set_ylabel("path length")
    ax.set_title(f"With Wind (p={wind_prob})")
    ax.legend(); ax.grid(alpha=0.3)
    if curve_wind[ok, 1].size > 0:
        ax.set_ylim(lstar - 1, max(curve_wind[ok, 1].max(), 1.15 * lstar) + 1)

    # -- Panel 3: Overlay --
    ax = axes[1, 0]
    ok_base = ~np.isnan(curve_base[:, 1])
    ok_wind = ~np.isnan(curve_wind[:, 1])
    ax.plot(curve_base[ok_base, 0], curve_base[ok_base, 1], color="#0969da", lw=1.8, label="baseline", alpha=0.8)
    ax.plot(curve_wind[ok_wind, 0], curve_wind[ok_wind, 1], color="#d1260d", lw=1.8, label="wind", alpha=0.8)
    ax.axhline(lstar, color="#8250df", ls="--", lw=1.3)
    ax.axhline(1.15 * lstar, color="grey", ls=":", lw=1.0)
    ax.set_xlabel("episode"); ax.set_ylabel("path length")
    ax.set_title("Direct Comparison")
    ax.legend(); ax.grid(alpha=0.3)

    # -- Panel 4: Statistics --
    ax = axes[1, 1]
    ax.axis("off")

    base_info = results["baseline"]
    wind_info = results["wind"]

    stats = f"""
Baseline (deterministic):
  Success: {base_info["success"]}
  L = {base_info["length"]:.3f}
  L*/L = {base_info["efficiency"]:.3f}

Wind (p={wind_prob}):
  Success: {wind_info["success"]}
  L = {wind_info["length"]:.3f}
  L*/L = {wind_info["efficiency"]:.3f}

Difference:
  L delta = {wind_info["length"] - base_info["length"]:+.3f}
  Efficiency delta = {wind_info["efficiency"] - base_info["efficiency"]:+.3f}
    """
    ax.text(0.1, 0.5, stats, fontsize=11, family="monospace", verticalalignment="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved to {out_path}")
    return fig


def main():
    p = argparse.ArgumentParser(description="Compare deterministic vs wind-augmented Q-learning")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seeds", type=int, default=6, help="Number of seeds to run (0 to N-1)")
    p.add_argument("--trees", type=int, default=16)
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--wind-prob", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args()

    print("\n" + "="*70)
    print("Controlled Experiment: Deterministic vs Wind-Augmented Q-Learning")
    print("="*70)
    print()

    if a.seed is not None:
        # Single seed
        results = train_both(
            seed=a.seed,
            n_trees=a.trees,
            episodes=a.episodes,
            wind_prob=a.wind_prob,
            alpha=a.alpha,
            gamma=a.gamma
        )

        print()
        print(f"Results for seed {a.seed}:")
        print(f"  L* = {results['lstar']:.3f}")
        print(f"  Baseline L = {results['baseline']['length']:.3f}, L*/L = {results['baseline']['efficiency']:.3f}")
        print(f"  Wind L = {results['wind']['length']:.3f}, L*/L = {results['wind']['efficiency']:.3f}")
        print(f"  Delta L = {results['wind']['length'] - results['baseline']['length']:+.3f}")
        print()

        if not a.no_plot:
            plot_comparison(results, a.wind_prob)
    else:
        # Multiple seeds
        all_results = {}
        for seed in range(a.seeds):
            print(f"\n--- Seed {seed} ({seed+1}/{a.seeds}) ---")
            results = train_both(
                seed=seed,
                n_trees=a.trees,
                episodes=a.episodes,
                wind_prob=a.wind_prob,
                alpha=a.alpha,
                gamma=a.gamma
            )
            all_results[str(seed)] = results

            print(f"Results for seed {seed}:")
            print(f"  L* = {results['lstar']:.3f}")
            print(f"  Baseline L = {results['baseline']['length']:.3f}, L*/L = {results['baseline']['efficiency']:.3f}")
            print(f"  Wind L = {results['wind']['length']:.3f}, L*/L = {results['wind']['efficiency']:.3f}")
            print(f"  Delta L = {results['wind']['length'] - results['baseline']['length']:+.3f}")

        export_for_viz(all_results)

    print()
    print("="*70)
    print()


if __name__ == "__main__":
    main()
