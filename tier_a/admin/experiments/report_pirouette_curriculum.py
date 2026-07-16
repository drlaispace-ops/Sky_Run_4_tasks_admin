#!/usr/bin/env python3
"""
Honest multi-seed curriculum report for the pirouette guidance layer
(agent/pirouette_attitude.py): trains the "pirouette" stage (the one that
actually needs a near-full knife-edge roll) two ways per seed --

  (a) curriculum transfer: straight -> angled -> pirouette, warm-starting
      each stage's 9x7 Q-table from the previous stage's result
  (b) from scratch: pirouette trained alone, same episode budget, no
      warm start

-- across N independent seeds, and reports the raw success/collision/departed
counts for both, plus each seed's "first clean greedy success" checkpoint --
the literal artifact of "episode 1 crashes, episode 2 crashes, ... episode N
starts evading," made visible rather than only claimed.

    python experiments/report_pirouette_curriculum.py --seeds 10

Transfer is a hypothesis being tested here, not an assumed fact -- report
whichever condition actually does better, honestly, even if that's neither,
or if both stay at 0/N (a legitimate, documented outcome per this repo's own
established culture: env_attitude/README.md already reports agent/hybrid_attitude.py's
1/10 and agent/qtable_attitude.py's 3/61 plainly, not dressed up).
"""

from __future__ import annotations

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

import tier_a.admin.agent.pirouette_attitude as pa
from tier_a.env_attitude.scenarios_pirouette import build_pirouette_gate_env


def train_one_seed(seed: int, episodes_straight: int, episodes_angled: int,
                   episodes_pirouette: int, checkpoint_every: int) -> dict:
    env_s = build_pirouette_gate_env(seed=seed, stage="straight")
    Q_s, _ = pa.train_pirouette_guidance(env_s, episodes=episodes_straight, seed=seed)

    env_a = build_pirouette_gate_env(seed=seed, stage="angled")
    Q_a, _ = pa.train_pirouette_guidance(env_a, episodes=episodes_angled, seed=seed, Q_init=Q_s)

    env_p_transfer = build_pirouette_gate_env(seed=seed, stage="pirouette")
    Q_transfer, hist_transfer = pa.train_pirouette_guidance(
        env_p_transfer, episodes=episodes_pirouette, seed=seed, Q_init=Q_a, eval_every=checkpoint_every,
    )

    env_p_scratch = build_pirouette_gate_env(seed=seed, stage="pirouette")
    Q_scratch, hist_scratch = pa.train_pirouette_guidance(
        env_p_scratch, episodes=episodes_pirouette, seed=seed, eval_every=checkpoint_every,
    )

    def summarize(env, Q, hist):
        roll = pa.greedy_rollout_pirouette(env, Q)
        first_success_ep = next((ep for ep, ok in zip(hist.eval_ep, hist.eval_success) if ok), None)
        return {
            "final_success": bool(roll["success"]),
            "final_collision": bool(roll["collision"]),
            "final_departed": bool(roll["departed"]),
            "first_success_ep": first_success_ep,
            "eval_ep": [int(e) for e in hist.eval_ep],
            "eval_success": [bool(x) for x in hist.eval_success],
        }

    return {
        "seed": seed,
        "transfer": summarize(env_p_transfer, Q_transfer, hist_transfer),
        "scratch": summarize(env_p_scratch, Q_scratch, hist_scratch),
    }


def outcome(cond: dict) -> str:
    if cond["final_success"]:
        return "success"
    if cond["final_collision"]:
        return "collision"
    if cond["final_departed"]:
        return "departed"
    return "timeout"


def plot_report(results: list[dict], out_path: str = "tier_a/admin/reports/tier_a_pirouette_curriculum.png") -> None:
    seeds = [r["seed"] for r in results]
    n = len(seeds)
    colors = {"success": "#1a7f37", "collision": "#cf222e", "departed": "#bf8700", "timeout": "#57606a"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    x = np.arange(n)
    w = 0.35
    t_colors = [colors[outcome(r["transfer"])] for r in results]
    s_colors = [colors[outcome(r["scratch"])] for r in results]
    ax.bar(x - w / 2, [1] * n, w, color=t_colors, edgecolor="black", linewidth=0.5, label="transfer")
    ax.bar(x + w / 2, [1] * n, w, color=s_colors, edgecolor="black", linewidth=0.5, hatch="//", label="scratch")
    ax.set_xticks(x); ax.set_xticklabels(seeds)
    ax.set_xlabel("seed"); ax.set_yticks([])
    ax.set_title("Final greedy outcome per seed")
    ax.legend(fontsize=8)

    ax = axes[1]
    for i, r in enumerate(results):
        fe_t = r["transfer"]["first_success_ep"]
        fe_s = r["scratch"]["first_success_ep"]
        if fe_t is not None:
            ax.scatter(fe_t, i - 0.12, color=colors["success"], marker="o", s=40)
        if fe_s is not None:
            ax.scatter(fe_s, i + 0.12, color=colors["success"], marker="^", s=40)
    ax.set_yticks(range(n)); ax.set_yticklabels(seeds)
    ax.set_xlabel("episode of first clean greedy success"); ax.set_ylabel("seed")
    ax.set_title("First success checkpoint (o=transfer, ^=scratch)")
    ax.grid(alpha=0.2)

    fig.suptitle("Pirouette guidance: curriculum transfer vs. from-scratch, N seeds", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description="Honest multi-seed pirouette curriculum report")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--episodes-straight", type=int, default=500)
    p.add_argument("--episodes-angled", type=int, default=1000)
    p.add_argument("--episodes-pirouette", type=int, default=3000)
    p.add_argument("--checkpoint-every", type=int, default=150)
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args()

    results = []
    print(f"{'seed':>4} | {'transfer':>10} {'first_ep':>9} | {'scratch':>10} {'first_ep':>9}")
    for seed in range(a.seeds):
        r = train_one_seed(seed, a.episodes_straight, a.episodes_angled, a.episodes_pirouette, a.checkpoint_every)
        results.append(r)
        t, s = r["transfer"], r["scratch"]
        print(f"{seed:>4} | {outcome(t):>10} {str(t['first_success_ep']):>9} | "
              f"{outcome(s):>10} {str(s['first_success_ep']):>9}")

    n = len(results)
    n_transfer = sum(r["transfer"]["final_success"] for r in results)
    n_scratch = sum(r["scratch"]["final_success"] for r in results)
    print()
    print(f"Final success rate: transfer {n_transfer}/{n}   scratch {n_scratch}/{n}")

    with open("tier_a/admin/reports/tier_a_pirouette_seeds_data.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote tier_a/admin/reports/tier_a_pirouette_seeds_data.json")

    if not a.no_plot:
        plot_report(results)


if __name__ == "__main__":
    main()
