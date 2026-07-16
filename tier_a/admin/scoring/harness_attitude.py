"""Objective leaderboard on hidden seeds for Tier-A (attitude tracking).

Takes an ``agent_factory(env, episodes, seed) -> (qnet, history)`` and, per
seed: trains under a fixed episode budget, then measures the greedy policy's

    success_rate      reached the final waypoint in time, no crash/departure
    collision_rate    flew into a tree trunk
    departed_rate     exceeded the cross-track/altitude/attitude tolerance
    tracking_rmse     RMS(e_lat, e_alt) over the rollout -- replaces L*/L,
                      since there is no single scalar path length for
                      continuous tracking
    sample_efficiency episodes until the greedy policy first reaches the goal

Rank is lexicographic, the same shape as ``scoring/harness.py``:
    learned reliably -> safely -> accurately -> quickly

Infeasible seeds are dropped using the reference cache, never scored. NOTE:
this is a genuinely hard from-scratch 6-DOF control problem -- do not expect
success_rate near the base kit's; the ranking still meaningfully discriminates
a trained agent from an untrained one on collision/departed rate and tracking
RMSE even when few or no rollouts reach the goal outright (see
``env_attitude/README.md``).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from tier_a.admin.agent.qnet_attitude import HistoryAttitude, QNet, greedy_rollout_attitude, train_attitude
from tier_a.env_attitude.constants import N_TREES_ATTITUDE
from tier_a.env_attitude.env import AttitudeEnv
from tier_a.admin.scoring.precompute_attitude import load_cache_attitude
from tier_d.scoring.seeds import HIDDEN_SEEDS, PUBLIC_SEEDS

AttitudeAgentFactory = Callable[[AttitudeEnv, int, int], tuple[QNet, HistoryAttitude]]

TRAIN_BUDGET_EPISODES_ATTITUDE = 1200


@dataclass
class ScoreAttitude:
    name: str
    success_rate: float
    collision_rate: float
    departed_rate: float
    tracking_rmse: float
    sample_efficiency: float
    n_seeds: int

    @property
    def rank_key(self) -> tuple:
        """Lexicographic: learned -> safe -> accurate -> fast (higher is better)."""
        return (
            round(self.success_rate, 3),
            round(-(self.collision_rate + self.departed_rate), 3),
            round(-self.tracking_rmse, 3),
            -self.sample_efficiency,
        )

    def row(self) -> list:
        return [
            self.name, f"{self.success_rate:.2f}", f"{self.collision_rate:.2f}",
            f"{self.departed_rate:.2f}", f"{self.tracking_rmse:.3f}",
            f"{self.sample_efficiency:.0f}", self.n_seeds,
        ]


def reference_agent(env: AttitudeEnv, episodes: int, seed: int) -> tuple[QNet, HistoryAttitude]:
    return train_attitude(env, episodes=episodes, seed=seed, shaping=True)


def no_shaping_agent(env: AttitudeEnv, episodes: int, seed: int) -> tuple[QNet, HistoryAttitude]:
    return train_attitude(env, episodes=episodes, seed=seed, shaping=False)


def random_agent(env: AttitudeEnv, episodes: int, seed: int) -> tuple[QNet, HistoryAttitude]:
    """Untrained net. The scoreboard must visibly discriminate against this."""
    torch.manual_seed(seed)
    qnet = QNet(env.obs_dim, env.n_actions)
    return qnet, HistoryAttitude()


def evaluate_attitude(
    factory: AttitudeAgentFactory,
    name: str,
    seeds=HIDDEN_SEEDS,
    n_trees: int = N_TREES_ATTITUDE,
    episodes: int = TRAIN_BUDGET_EPISODES_ATTITUDE,
) -> ScoreAttitude:
    cache = load_cache_attitude()
    successes, collisions, departs, rmses, samples = [], [], [], [], []

    for s in seeds:
        entry = cache.get(f"{n_trees}:{s}")
        if entry is None or not entry["feasible"]:
            continue  # infeasible seeds are excluded automatically

        env = AttitudeEnv(n_trees=n_trees, seed=s)
        qnet, hist = factory(env, episodes, s)
        roll = greedy_rollout_attitude(env, qnet)

        successes.append(roll["success"])
        collisions.append(roll["collision"])
        departs.append(roll["departed"])
        rmses.append(roll["tracking_rmse"])

        k = None
        for ep, ok in zip(hist.eval_ep, hist.eval_success):
            if ok:
                k = ep
                break
        samples.append(episodes if k is None else k)  # right-censored at the budget

    return ScoreAttitude(
        name=name,
        success_rate=float(np.mean(successes)) if successes else 0.0,
        collision_rate=float(np.mean(collisions)) if collisions else 0.0,
        departed_rate=float(np.mean(departs)) if departs else 0.0,
        tracking_rmse=float(np.mean(rmses)) if rmses else float("inf"),
        sample_efficiency=float(np.mean(samples)) if samples else float(episodes),
        n_seeds=len(successes),
    )


def leaderboard(scores: list[ScoreAttitude]) -> list[ScoreAttitude]:
    return sorted(scores, key=lambda s: s.rank_key, reverse=True)


def main() -> None:
    p = argparse.ArgumentParser(description="AI SkyRun Tier-A (attitude) scoring harness")
    p.add_argument("--public", action="store_true", help="score on PUBLIC seeds (self-check)")
    p.add_argument("--episodes", type=int, default=TRAIN_BUDGET_EPISODES_ATTITUDE)
    p.add_argument("--csv", type=Path, default=Path("tier_a/admin/reports/leaderboard_attitude.csv"))
    args = p.parse_args()

    seeds = PUBLIC_SEEDS if args.public else HIDDEN_SEEDS
    label = "PUBLIC" if args.public else "HIDDEN"

    entries = [
        (reference_agent, "reference (DQN + shaping)"),
        (no_shaping_agent, "ablation (no shaping)"),
        (random_agent, "baseline (untrained net)"),
    ]
    scores = [evaluate_attitude(f, n, seeds=seeds, episodes=args.episodes) for f, n in entries]
    board = leaderboard(scores)

    hdr = ["agent", "success", "collision", "departed", "tracking_rmse", "ep_to_goal", "seeds"]
    widths = [28, 8, 10, 10, 14, 11, 6]
    print(f"\n=== AI SkyRun Tier-A leaderboard -- {label} seeds "
          f"({len(seeds)} requested, budget {args.episodes} episodes) ===\n")
    print("".join(h.ljust(w) for h, w in zip(hdr, widths)))
    print("-" * sum(widths))
    for s in board:
        print("".join(str(c).ljust(w) for c, w in zip(s.row(), widths)))
    print("\nrank = success -> (1-collision-departed) -> tracking accuracy -> sample efficiency")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(s.row() for s in board)
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
