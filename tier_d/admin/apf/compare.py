"""Head-to-head: classical APF vs APF+residual vs map-blind Q-learning.

This is the organizer's central validation of concept thesis #2 -- "RL removes
the local minimum by itself". It is *searched*, not asserted: we look for seeds
where continuous APF provably stalls, then check what each planner does there.

Fairness note, restated because it matters: APF and APF+residual read tree
coordinates. Q-learning reads only its cell index. The map-blind planner is the
one that is handicapped, and it is the one that wins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tier_d.admin.agent.qlearning import greedy_rollout, train
from tier_d.admin.apf.field import apf_flight
from tier_d.admin.apf.residual import apf_greedy_rollout, apf_prior, residual_rollout, train_residual
from tier_d.env.forest import make_forest
from tier_d.env.gridworld import GridWorld
from tier_d.oracle.astar import is_feasible, optimal_length


@dataclass
class Row:
    seed: int
    n_trees: int
    Lstar: float
    apf_success: bool
    apf_trapped: bool
    apf_stall_xy: tuple[float, float] | None
    res_success: bool
    res_len: float
    ql_success: bool
    ql_len: float

    @property
    def ql_efficiency(self) -> float:
        return self.Lstar / self.ql_len if self.ql_success else 0.0

    @property
    def res_efficiency(self) -> float:
        return self.Lstar / self.res_len if self.res_success else 0.0


def evaluate_seed(seed: int, n_trees: int, episodes: int = 3000) -> Row | None:
    trees = make_forest(n_trees, seed)
    if not is_feasible(trees):
        return None
    Lstar = optimal_length(trees)

    apf = apf_flight(trees)

    prior = apf_prior(trees)
    env = GridWorld(n_trees, seed)
    Qr, _ = train_residual(env, prior, episodes=episodes, seed=seed)
    res = residual_rollout(env, Qr, prior)

    env2 = GridWorld(n_trees, seed)
    Q, _ = train(env2, episodes=episodes, seed=seed)
    ql = greedy_rollout(env2, Q)

    return Row(
        seed=seed,
        n_trees=n_trees,
        Lstar=Lstar,
        apf_success=apf["success"],
        apf_trapped=apf["trapped"],
        apf_stall_xy=None if apf["stall_xy"] is None else (
            round(float(apf["stall_xy"][0]), 2),
            round(float(apf["stall_xy"][1]), 2),
        ),
        res_success=res["success"],
        res_len=res["length"],
        ql_success=ql["success"],
        ql_len=ql["length"],
    )


def find_trap_seeds(n_trees: int = 16, seeds=range(30), limit: int = 5) -> list[int]:
    """Seeds where continuous APF stalls in a local minimum short of the goal."""
    out = []
    for s in seeds:
        trees = make_forest(n_trees, s)
        if not is_feasible(trees):
            continue
        if apf_flight(trees)["trapped"]:
            out.append(s)
        if len(out) >= limit:
            break
    return out


def main() -> None:
    trap_seeds = find_trap_seeds(16, range(30), limit=6)
    print(f"APF-trapping seeds (16 trees): {trap_seeds}\n")
    print(f"{'seed':>4} {'L*':>6} | {'APF':>8} {'stall':>14} | {'APF+res':>8} {'eff':>5} "
          f"| {'Q-learn':>8} {'eff':>5}")
    print("-" * 72)

    rows = []
    for s in trap_seeds:
        r = evaluate_seed(s, 16)
        if r is None:
            continue
        rows.append(r)
        apf_txt = "trapped" if r.apf_trapped else ("goal" if r.apf_success else "fail")
        stall = "" if r.apf_stall_xy is None else str(r.apf_stall_xy)
        print(f"{r.seed:>4} {r.Lstar:6.2f} | {apf_txt:>8} {stall:>14} "
              f"| {'goal' if r.res_success else 'fail':>8} {r.res_efficiency:5.3f} "
              f"| {'goal' if r.ql_success else 'fail':>8} {r.ql_efficiency:5.3f}")

    if rows:
        print("-" * 72)
        print(f"On seeds where classical APF is trapped: "
              f"APF+residual reaches goal {sum(r.res_success for r in rows)}/{len(rows)}, "
              f"Q-learning {sum(r.ql_success for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
