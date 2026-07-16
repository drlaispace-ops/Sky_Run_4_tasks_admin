#!/usr/bin/env python3
"""
Controlled experiment: today's offline, ground-truth-A*-baked reference vs
the new online, LIDAR-discovered replanning reference (env_attitude/online_planner.py).

Both conditions fly the SAME cross-track PD controller: the existing
hand-tuned attitude PD (agent.hybrid_attitude.pd_action) plus a P+D term on
e_lat/e_lat_dot (obs[0] and its finite-difference rate) fed in as phi_bias --
NOT the hybrid/tabular guidance layers, which study a different,
already-answered question (LIDAR-reactive nudging). Using a fixed-gain
cross-track PD isolates exactly: how well does a controller that actively
holds the reference line follow one that's fixed in advance vs. one that's
discovered-and-occasionally-jumps.

    python experiments/compare_online_planning.py --seeds 6

Earlier revision of this script used PLAIN pd_action (phi_bias=0, attitude
tracking only, no lateral feedback at all) and found 0/6 success in BOTH
conditions on the full 8-tree lap -- a floor effect that made success rate
uninformative. A small offline prototype (see conversation/session notes)
found this was because plain PD never looks at e_lat, so any perturbation
away from the reference line is never corrected. Adding the missing P+D term
here (found and validated on the local two-tree gauntlet: 10/10 success
recovering from up to a 0.5m lateral perturbation, vs 0/10 for plain PD
beyond ~0.2m) is expected to raise both conditions off the floor and let
success rate actually discriminate offline vs. online again.
"""

from __future__ import annotations

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import argparse

import matplotlib.pyplot as plt
import numpy as np

import math

from tier_a.admin.agent.hybrid_attitude import pd_action
from tier_a.env_attitude.constants import DT, N_TREES_ATTITUDE
from tier_a.env_attitude.env import AttitudeEnv
from tier_d.oracle.astar import optimal_length

K_P_LAT = 0.6
K_D_LAT = 0.6
CAP_DEG = 20.0


def cross_track_bias(e_lat: float, e_lat_prev: float, dt: float = DT,
                      k_p: float = K_P_LAT, k_d: float = K_D_LAT, cap_deg: float = CAP_DEG) -> float:
    """P+D on cross-track error (obs[0]) -- see the module docstring: plain
    pd_action never looks at e_lat at all, so any lateral perturbation is
    never corrected. This is the minimal fix, not a new agent module."""
    e_lat_dot = (e_lat - e_lat_prev) / dt
    return float(np.clip(k_p * e_lat + k_d * e_lat_dot, -math.radians(cap_deg), math.radians(cap_deg)))


def run_rollout(env: AttitudeEnv, max_steps: int = 3000) -> dict:
    obs = env.reset()
    sq_err = []
    replans = 0
    path_len = 0.0
    prev_xy = env.xyz[:2].copy()
    e_lat_prev = float(obs[0])
    info = {"goal": False, "collision": False, "departed": False, "loss_of_control": False, "truncated": False}
    for i in range(max_steps):
        bias = cross_track_bias(float(obs[0]), e_lat_prev)
        e_lat_prev = float(obs[0])
        a = pd_action(obs, phi_bias=bias)
        obs, r, done, info = env.step(a)
        if info.get("replanned"):
            replans += 1
        xy = env.xyz[:2]
        path_len += float(np.hypot(*(xy - prev_xy)))
        prev_xy = xy.copy()
        sq_err.append(float(obs[0]) ** 2)
        if done:
            break
    return {
        "success": bool(info["goal"]),
        "collision": bool(info["collision"]),
        "departed": bool(info["departed"] or info["loss_of_control"]),
        "timeout": bool(info["truncated"] and not (info["goal"] or info["collision"])),
        "steps": i + 1,
        "path_length": path_len,
        "tracking_rmse": float(np.sqrt(np.mean(sq_err))) if sq_err else 0.0,
        "replans": replans,
    }


def compare_seed(seed: int, n_trees: int = N_TREES_ATTITUDE) -> dict:
    env_offline = AttitudeEnv(n_trees=n_trees, seed=seed)
    lstar = optimal_length(env_offline.trees)
    offline = run_rollout(env_offline)
    offline["efficiency"] = lstar / offline["path_length"] if offline["success"] and offline["path_length"] > 0 else float("nan")

    env_online = AttitudeEnv(n_trees=n_trees, seed=seed, online_lidar_planning=True)
    online = run_rollout(env_online)
    online["efficiency"] = lstar / online["path_length"] if online["success"] and online["path_length"] > 0 else float("nan")

    return {"seed": seed, "lstar": lstar, "offline": offline, "online": online}


def outcome(r: dict) -> str:
    if r["success"]:
        return "success"
    if r["collision"]:
        return "collision"
    if r["departed"]:
        return "departed"
    return "timeout"


def plot_comparison(results: list[dict], out_path: str = "tier_a/admin/reports/tier_a_online_planning_comparison.png") -> None:
    seeds = [r["seed"] for r in results]
    n = len(seeds)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    outcome_colors = {"success": "#1a7f37", "collision": "#cf222e", "departed": "#bf8700", "timeout": "#57606a"}

    ax = axes[0]
    x = np.arange(n)
    w = 0.35
    off_steps = [r["offline"]["steps"] for r in results]
    on_steps = [r["online"]["steps"] for r in results]
    off_colors = [outcome_colors[outcome(r["offline"])] for r in results]
    on_colors = [outcome_colors[outcome(r["online"])] for r in results]
    ax.bar(x - w / 2, off_steps, w, color=off_colors, edgecolor="black", linewidth=0.5, label="offline")
    ax.bar(x + w / 2, on_steps, w, color=on_colors, edgecolor="black", linewidth=0.5, hatch="//", label="online")
    ax.set_xticks(x); ax.set_xticklabels(seeds)
    ax.set_xlabel("seed"); ax.set_ylabel("steps survived")
    ax.set_title("Steps survived (color = outcome)")
    ax.legend(fontsize=8)

    ax = axes[1]
    off_rmse = [r["offline"]["tracking_rmse"] for r in results]
    on_rmse = [r["online"]["tracking_rmse"] for r in results]
    ax.bar(x - w / 2, off_rmse, w, color="#57606a", label="offline")
    ax.bar(x + w / 2, on_rmse, w, color="#57606a", alpha=0.5, hatch="//", label="online")
    ax.set_xticks(x); ax.set_xticklabels(seeds)
    ax.set_xlabel("seed"); ax.set_ylabel("tracking RMSE, e_lat (m)")
    ax.set_title("Cross-track tracking error")
    ax.legend(fontsize=8)

    ax = axes[2]
    on_replans = [r["online"]["replans"] for r in results]
    ax.bar(x, on_replans, color="#0969da")
    ax.set_xticks(x); ax.set_xticklabels(seeds)
    ax.set_xlabel("seed"); ax.set_ylabel("replans triggered")
    ax.set_title("Online: replan count")

    fig.suptitle("Offline (ground-truth A*) vs Online (LIDAR-discovered) reference -- PD + cross-track P+D, no guidance layer", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description="Compare offline ground-truth vs online LIDAR-discovered Tier-A reference")
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--trees", type=int, default=N_TREES_ATTITUDE)
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args()

    results = []
    print(f"{'seed':>4} {'L*':>6} | {'offline':>10} {'steps':>6} {'rmse':>6} | {'online':>10} {'steps':>6} {'rmse':>6} {'replans':>7}")
    for seed in range(a.seeds):
        r = compare_seed(seed, n_trees=a.trees)
        results.append(r)
        off, on = r["offline"], r["online"]
        print(f"{seed:>4} {r['lstar']:>6.2f} | {outcome(off):>10} {off['steps']:>6} {off['tracking_rmse']:>6.3f} "
              f"| {outcome(on):>10} {on['steps']:>6} {on['tracking_rmse']:>6.3f} {on['replans']:>7}")

    n = len(results)
    off_success = sum(r["offline"]["success"] for r in results)
    on_success = sum(r["online"]["success"] for r in results)
    off_rmse_mean = float(np.mean([r["offline"]["tracking_rmse"] for r in results]))
    on_rmse_mean = float(np.mean([r["online"]["tracking_rmse"] for r in results]))
    off_steps_mean = float(np.mean([r["offline"]["steps"] for r in results]))
    on_steps_mean = float(np.mean([r["online"]["steps"] for r in results]))

    print()
    print(f"Success rate:      offline {off_success}/{n}   online {on_success}/{n}")
    print(f"Mean tracking RMSE: offline {off_rmse_mean:.3f}   online {on_rmse_mean:.3f}")
    print(f"Mean steps survived: offline {off_steps_mean:.1f}   online {on_steps_mean:.1f}")

    if not a.no_plot:
        plot_comparison(results)


if __name__ == "__main__":
    main()
