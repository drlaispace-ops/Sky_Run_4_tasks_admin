#!/usr/bin/env python3
"""Лідерборд tier_b на ПРИХОВАНИХ сідах (організаторський; conda-only).

Оцінює чекпоінт(и) команд на HIDDEN_SEEDS: по одному детермінованому епізоду
на master-сід. Ранжування лексикографічне: success-rate, потім середня
кількість пройдених воріт, потім середній return.

УВАГА: тренованих моделей у репозиторії немає (tier_b/runs/ у .gitignore) —
команда здає свій final_model.zip, шлях передається через --model.

    conda run -n drones python -m tier_b.admin.scoring.harness_b \
        --model runs/sac/seed0/final_model.zip --algo sac --level 5
    conda run -n drones python -m tier_b.admin.scoring.harness_b \
        --model ... --public          # самоперевірка на PUBLIC_SEEDS
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402

from tier_b.eval import rollout_episode  # noqa: E402
from tier_b.scoring.seeds import HIDDEN_SEEDS, PUBLIC_SEEDS  # noqa: E402


def evaluate_on_seeds(model_path: str, algo: str, level: int,
                      seeds: tuple[int, ...]) -> list[dict]:
    from stable_baselines3 import PPO, SAC

    from tier_b.envs.pirouette_aviary import PirouetteAviary

    model = (SAC if algo == "sac" else PPO).load(model_path, device="cpu")
    predict = lambda obs: model.predict(obs, deterministic=True)[0]  # noqa: E731

    results = []
    for seed in seeds:
        env = PirouetteAviary(level=level, master_seed=seed)
        r = rollout_episode(env, predict)
        env.close()
        results.append({"seed": seed, **r})
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="шлях до final_model.zip команди")
    ap.add_argument("--algo", choices=("ppo", "sac"), default=None)
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--public", action="store_true",
                    help="PUBLIC_SEEDS замість прихованих (самоперевірка)")
    ap.add_argument("--csv", type=pathlib.Path,
                    default=pathlib.Path("tier_b/admin/reports/leaderboard_b.csv"))
    args = ap.parse_args()

    seeds = PUBLIC_SEEDS if args.public else HIDDEN_SEEDS
    if not seeds:
        raise SystemExit("HIDDEN_SEEDS порожні — це учнівська копія без tier_b/admin/.")
    algo = args.algo or ("sac" if "sac" in args.model.lower() else "ppo")

    results = evaluate_on_seeds(args.model, algo, args.level, seeds)
    n = len(results)
    sr = sum(r["success"] for r in results) / n
    gates = float(np.mean([r["gates_passed"] for r in results]))
    ret = float(np.mean([r["return"] for r in results]))

    label = "PUBLIC" if args.public else "HIDDEN"
    print(f"{args.model}  [{algo}, L{args.level}, {label}]")
    print(f"  success-rate : {sr:.2f}  ({sum(r['success'] for r in results)}/{n})")
    print(f"  mean gates   : {gates:.2f}")
    print(f"  mean return  : {ret:.1f}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.csv.exists()
    with open(args.csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["model", "algo", "level", "seed_set",
                        "success_rate", "mean_gates", "mean_return"])
        w.writerow([args.model, algo, args.level, label,
                    f"{sr:.3f}", f"{gates:.3f}", f"{ret:.2f}"])
    print(f"дописано {args.csv}")


if __name__ == "__main__":
    main()
