"""Reference tabular Q-learning. Ours, not the students'.

Validates the environment, the scoring harness and the visualiser. The Bellman
update is the hole the students fill in ``tier_d/scaffold/starter.ipynb``.

INVARIANT 2 is enforced structurally: the only thing this file reads from the
environment is ``env.state`` (an int) and the step tuple. It never touches
``env.trees`` or ``env.blocked``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from tier_d.admin.agent.shaping import shaping_F
from tier_d.env.constants import ACTIONS, N_ACTIONS  # noqa: F401
from tier_d.env.gridworld import GridWorld

# Plot helpers moved to the student-visible scaffold (starter.ipynb imports
# them); re-exported here so organizer code and tests keep working.
from tier_d.scaffold.qtools import (  # noqa: F401
    experienced_obstacle_value,
    greedy_arrows,
    value_surface,
)


@dataclass
class History:
    """Per-episode training log; the visualiser and harness both read this."""

    length: list[float] = field(default_factory=list)  # world-units travelled
    steps: list[int] = field(default_factory=list)
    success: list[bool] = field(default_factory=list)
    collision: list[bool] = field(default_factory=list)
    ret: list[float] = field(default_factory=list)
    epsilon: list[float] = field(default_factory=list)
    # (episode, V=max_a Q, full Q). The full Q is needed to read back what the
    # agent learned about obstacle cells, which max_a Q cannot express.
    v_snapshots: list[tuple[int, np.ndarray, np.ndarray]] = field(default_factory=list)

    # Periodic *greedy* evaluation. Sample efficiency is read off this curve,
    # not off the exploration episodes above: with eps floored at 0.05 the
    # exploring agent keeps taking random detours long after it has learned,
    # which would hide the moment it actually became good.
    eval_ep: list[int] = field(default_factory=list)
    eval_len: list[float] = field(default_factory=list)
    eval_success: list[bool] = field(default_factory=list)


def greedy_rollout(env: GridWorld, Q: np.ndarray, max_steps: int = 400) -> dict:
    """Run argmax_a Q(s,a) with no exploration. Returns path + outcome."""
    s = env.reset()
    path = [env.xy]
    length = 0.0
    seen = {s}
    for _ in range(max_steps):
        a = int(np.argmax(Q[s]))
        s, _, done, info = env.step(a)
        path.append(env.xy)
        length += info["moved"]
        if info["collision"]:
            return {"success": False, "collision": True, "length": length, "path": np.array(path)}
        if info["goal"]:
            return {"success": True, "collision": False, "length": length, "path": np.array(path)}
        if s in seen and info["moved"] == 0.0:
            break  # greedy cycle against a wall
        seen.add(s)
    return {"success": False, "collision": False, "length": length, "path": np.array(path)}


def train(
    env: GridWorld,
    episodes: int = 3000,
    alpha: float = 0.15,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_frac: float = 0.6,
    shaping: bool = True,
    seed: int = 0,
    snapshot_every: int = 0,
    optimistic: float = 0.0,
    eval_every: int = 25,
) -> tuple[np.ndarray, History]:
    """Tabular Q-learning with optional potential-based shaping.

    ``eps`` decays linearly to ``eps_end`` over the first ``eps_decay_frac`` of
    training, then holds. Truncation (time limit) bootstraps; true terminals do
    not -- conflating the two is the classic silent value-leak.
    """
    rng = np.random.default_rng(seed)
    Q = np.full((env.n_states, N_ACTIONS), optimistic, dtype=np.float64)
    hist = History()
    decay_eps = max(1, int(episodes * eps_decay_frac))

    for ep in range(episodes):
        eps = max(eps_end, eps_start + (eps_end - eps_start) * ep / decay_eps)
        s = env.reset()
        total, length = 0.0, 0.0

        while True:
            a = rng.integers(N_ACTIONS) if rng.random() < eps else int(np.argmax(Q[s]))
            s2, r, done, info = env.step(int(a))
            length += info["moved"]
            total += r

            terminal = info["collision"] or info["goal"]
            r_train = r + (shaping_F(s, s2, gamma, terminal) if shaping else 0.0)

            # Bootstrap through truncation, not through a real terminal.
            target = r_train if terminal else r_train + gamma * float(np.max(Q[s2]))
            Q[s, a] += alpha * (target - Q[s, a])

            s = s2
            if done:
                break

        hist.length.append(length)
        hist.steps.append(env._steps)
        hist.success.append(bool(info["goal"]))
        hist.collision.append(bool(info["collision"]))
        hist.ret.append(total)
        hist.epsilon.append(eps)
        if snapshot_every and (ep % snapshot_every == 0 or ep == episodes - 1):
            hist.v_snapshots.append((ep, Q.max(axis=1).copy(), Q.copy()))
        if eval_every and (ep % eval_every == 0 or ep == episodes - 1):
            roll = greedy_rollout(env, Q)
            hist.eval_ep.append(ep)
            hist.eval_len.append(roll["length"] if roll["success"] else float("inf"))
            hist.eval_success.append(roll["success"])

    return Q, hist


def episodes_to_threshold(hist: History, threshold: float) -> int | None:
    """Sample efficiency: first episode at which the *greedy* path is <= threshold.

    Falls back to the exploration curve only if no greedy evaluations were run.
    """
    if hist.eval_ep:
        for ep, ok, L in zip(hist.eval_ep, hist.eval_success, hist.eval_len):
            if ok and L <= threshold:
                return ep
        return None
    for i, (ok, L) in enumerate(zip(hist.success, hist.length)):
        if ok and L <= threshold:
            return i
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Reference tabular Q-learning for AI SkyRun")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trees", type=int, default=16)
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--no-shaping", action="store_true")
    args = p.parse_args()

    from tier_d.oracle.astar import optimal_length

    env = GridWorld(args.trees, args.seed)
    Lstar = optimal_length(env.trees)
    Q, hist = train(
        env,
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        eps_start=args.eps,
        shaping=not args.no_shaping,
        seed=args.seed,
    )
    roll = greedy_rollout(env, Q)
    thr = None if Lstar is None else episodes_to_threshold(hist, 1.15 * Lstar)

    print(f"seed={args.seed} trees={len(env.trees)} shaping={not args.no_shaping}")
    print(f"  L* (oracle)        : {Lstar:.3f}" if Lstar else "  L*: infeasible")
    print(f"  L  (greedy policy) : {roll['length']:.3f}  success={roll['success']}")
    if Lstar and roll["success"]:
        print(f"  efficiency L*/L    : {Lstar / roll['length']:.3f}")
    print(f"  episodes to 1.15*L*: {thr}")
    print(f"  final success rate : {np.mean(hist.success[-50:]):.2f} (last 50 eps)")


if __name__ == "__main__":
    main()
