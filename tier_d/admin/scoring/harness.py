"""Objective leaderboard on hidden seeds.

Takes an ``agent_factory(env, episodes, seed) -> (Q, history)`` and, per seed:
trains under a fixed episode budget, then measures

  sample_efficiency  episodes until the first successful path <= 1.15 * L*
  optimality         L* / L of the final greedy policy
  collision_rate     collisions of the *greedy* policy across eval rollouts
  success            greedy success rate on held-out seeds

Rank is lexicographic, exactly as the concept doc specifies:
    learned reliably -> safely -> optimally -> quickly

Infeasible seeds are dropped using the oracle cache, never scored.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from tier_d.admin.agent.qlearning import History, episodes_to_threshold, greedy_rollout, train
from tier_d.env.gridworld import GridWorld
from tier_d.admin.scoring.precompute import load_cache
from tier_d.scoring.seeds import HIDDEN_SEEDS, N_TREES, PUBLIC_SEEDS, TRAIN_BUDGET_EPISODES

AgentFactory = Callable[[GridWorld, int, int], tuple[np.ndarray, History]]

THRESHOLD_FACTOR = 1.15


@dataclass
class Score:
    name: str
    success_rate: float
    collision_rate: float
    optimality: float  # mean L*/L over solved seeds
    sample_efficiency: float  # mean episodes-to-threshold, censored at budget
    reached_frac: float  # fraction of seeds that ever crossed the threshold
    n_seeds: int

    @property
    def rank_key(self) -> tuple:
        """Lexicographic: learned -> safe -> optimal -> fast (higher is better)."""
        return (
            round(self.success_rate, 3),
            round(-self.collision_rate, 3),
            round(self.optimality, 3),
            -self.sample_efficiency,
        )

    def row(self) -> list:
        # Censored mean: seeds that never crossed count as the full budget. An
        # uncensored mean would be inf and erase every difference between agents.
        se = f"{self.sample_efficiency:.0f}" + ("" if self.reached_frac == 1.0
                                                else f" ({self.reached_frac:.0%})")
        return [self.name, f"{self.success_rate:.2f}", f"{self.collision_rate:.2f}",
                f"{self.optimality:.3f}", se, self.n_seeds]


def reference_agent(env: GridWorld, episodes: int, seed: int) -> tuple[np.ndarray, History]:
    return train(env, episodes=episodes, seed=seed, shaping=True)


def no_shaping_agent(env: GridWorld, episodes: int, seed: int) -> tuple[np.ndarray, History]:
    return train(env, episodes=episodes, seed=seed, shaping=False)


def random_agent(env: GridWorld, episodes: int, seed: int) -> tuple[np.ndarray, History]:
    """Untrained Q. The scoreboard must visibly discriminate against this."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(env.n_states, env.n_actions)) * 1e-6, History()


def evaluate(
    factory: AgentFactory,
    name: str,
    seeds=HIDDEN_SEEDS,
    n_trees: int = N_TREES,
    episodes: int = TRAIN_BUDGET_EPISODES,
) -> Score:
    cache = load_cache()
    successes, collisions, effs, samples, reached = [], [], [], [], []

    for s in seeds:
        entry = cache.get(f"{n_trees}:{s}")
        if entry is None or not entry["feasible"]:
            continue  # infeasible seeds are excluded automatically
        Lstar = entry["Lstar"]

        env = GridWorld(n_trees, s)
        Q, hist = factory(env, episodes, s)
        roll = greedy_rollout(env, Q)

        successes.append(roll["success"])
        collisions.append(roll["collision"])
        if roll["success"]:
            effs.append(Lstar / roll["length"])
        k = episodes_to_threshold(hist, THRESHOLD_FACTOR * Lstar) if hist.length else None
        samples.append(episodes if k is None else k)  # right-censored at the budget
        reached.append(k is not None)

    return Score(
        name=name,
        success_rate=float(np.mean(successes)) if successes else 0.0,
        collision_rate=float(np.mean(collisions)) if collisions else 0.0,
        optimality=float(np.mean(effs)) if effs else 0.0,
        sample_efficiency=float(np.mean(samples)) if samples else float(episodes),
        reached_frac=float(np.mean(reached)) if reached else 0.0,
        n_seeds=len(successes),
    )


def leaderboard(scores: list[Score]) -> list[Score]:
    return sorted(scores, key=lambda s: s.rank_key, reverse=True)


def main() -> None:
    p = argparse.ArgumentParser(description="AI SkyRun scoring harness")
    p.add_argument("--public", action="store_true", help="score on PUBLIC seeds (self-check)")
    p.add_argument("--episodes", type=int, default=TRAIN_BUDGET_EPISODES)
    p.add_argument("--csv", type=Path, default=Path("tier_d/admin/reports/leaderboard.csv"))
    args = p.parse_args()

    seeds = PUBLIC_SEEDS if args.public else HIDDEN_SEEDS
    label = "PUBLIC" if args.public else "HIDDEN"

    entries = [
        (reference_agent, "reference (Q-learning + shaping)"),
        (no_shaping_agent, "ablation (no shaping)"),
        (random_agent, "baseline (random / untrained)"),
    ]
    scores = [evaluate(f, n, seeds=seeds, episodes=args.episodes) for f, n in entries]
    board = leaderboard(scores)

    hdr = ["agent", "success", "collision", "optimality", "episodes_to_1.15L*", "seeds"]
    widths = [34, 8, 10, 11, 19, 6]
    print(f"\n=== AI SkyRun leaderboard — {label} seeds "
          f"({len(seeds)} requested, budget {args.episodes} episodes) ===\n")
    print("".join(h.ljust(w) for h, w in zip(hdr, widths)))
    print("-" * sum(widths))
    for s in board:
        print("".join(str(c).ljust(w) for c, w in zip(s.row(), widths)))
    print("\nrank = success -> (1-collision) -> optimality -> sample efficiency")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(s.row() for s in board)
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
