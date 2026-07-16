"""The training film: one map, one snapshot every 10 episodes.

    python -m tier_d.admin.viz.training_film                 # seed 0, mp4 + gif
    python -m tier_d.admin.viz.training_film --seed 3 --no-gif

Four panels, all driven by the same Q snapshot:

  1. -V(s) surface, seen from above. Obstacle height is what the agent learnt by
     crashing, so untouched trees stay flat and peaks rise as it explores.
  2. The greedy policy field, plus the path it currently flies.
  3. Greedy path length against L*.
  4. Cumulative collisions and how many obstacle cells the agent has met.

Panels 3 and 4 are the honest counterweight to panel 1: the surface looks
finished long before the policy is any good.
"""

from __future__ import annotations

import argparse

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from tier_d.admin.agent.qlearning import experienced_obstacle_value, greedy_arrows, greedy_rollout, train
from tier_d.env.constants import DOMAIN, GRID_N, R_COLLISION, R_GOAL
from tier_d.env.gridworld import GridWorld
from tier_d.env.occupancy import GOAL_CELL, cell_centers
from tier_d.oracle.astar import optimal_length, optimal_path_xy
from tier_d.viz.style import APF, GRIDC, INK, MUTED, ORACLE, RL, TRAP, draw_endpoints, draw_forest, setup_domain

STRIDE = 10
EPISODES = 3000


def collect(seed: int, n_trees: int = 16):
    env = GridWorld(n_trees, seed)
    Lstar = optimal_length(env.trees)
    Q, hist = train(env, episodes=EPISODES, seed=seed, snapshot_every=STRIDE, eval_every=0)

    coll = np.cumsum(np.array(hist.collision, dtype=int))
    frames = []
    for ep, _, Qs in hist.v_snapshots:
        V = Qs.max(axis=1).reshape(GRID_N, GRID_N).copy()
        peaks = experienced_obstacle_value(Qs, env.blocked)
        V[env.blocked] = peaks[env.blocked]
        V[GOAL_CELL] = R_GOAL
        roll = greedy_rollout(GridWorld(n_trees, seed), Qs)
        frames.append({
            "ep": ep, "V": V, "Q": Qs,
            "path": roll["path"], "ok": roll["success"],
            "len": roll["length"] if roll["success"] else np.nan,
            "coll": int(coll[min(ep, len(coll) - 1)]),
            "learned": int((peaks[env.blocked] < -1.0).sum()),
        })
    return env, Lstar, frames


def render(seed: int, out_mp4: str, out_gif: str | None, fps: int = 14) -> None:
    env, Lstar, frames = collect(seed)
    gx, gy = cell_centers()
    n_obst = int(env.blocked.sum())
    apath = optimal_path_xy(env.trees)
    eps = np.array([f["ep"] for f in frames])
    lens = np.array([f["len"] for f in frames], dtype=float)
    colls = np.array([f["coll"] for f in frames])
    learned = np.array([f["learned"] for f in frames])
    top = float(np.nanmax(lens[np.isfinite(lens)])) if np.isfinite(lens).any() else Lstar * 2

    fig = plt.figure(figsize=(13.6, 3.7))
    # right=0.945 leaves room for the twin y-axis label on the last panel,
    # which is otherwise clipped off the canvas.
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.05, 1.05], wspace=0.36,
                          left=0.035, right=0.945, bottom=0.16, top=0.84)
    ax_v, ax_p, ax_l, ax_c = (fig.add_subplot(gs[i]) for i in range(4))

    def frame(i):
        f = frames[i]
        for a in (ax_v, ax_p, ax_l, ax_c):
            a.clear()

        # 1. the surface, from above
        setup_domain(ax_v, "−V(s): піки там, де боляче")
        ax_v.imshow(-f["V"], origin="lower", extent=(0, DOMAIN, 0, DOMAIN),
                    cmap=cm.viridis, vmin=-R_GOAL, vmax=-R_COLLISION, zorder=1,
                    interpolation="nearest")
        draw_forest(ax_v, env.trees, alpha=0.30)
        draw_endpoints(ax_v, size=55)

        # 2. policy + current greedy path
        setup_domain(ax_p, "жадібна політика")
        u, v = greedy_arrows(f["Q"])
        free = ~env.blocked
        ax_p.quiver(gx[free], gy[free], u[free], v[free], color=MUTED, alpha=0.55,
                    scale=44, width=0.0034, zorder=3)
        draw_forest(ax_p, env.trees)
        if apath is not None:
            ax_p.plot(apath[:, 0], apath[:, 1], color=ORACLE, lw=1.8, ls="--", zorder=4)
        p = f["path"]
        ax_p.plot(p[:, 0], p[:, 1], color=RL if f["ok"] else TRAP, lw=2.4, zorder=5)
        draw_endpoints(ax_p, size=55)

        # 3. path length vs L*
        ax_l.axhline(Lstar, color=ORACLE, ls="--", lw=1.3)
        ax_l.axhline(1.15 * Lstar, color=MUTED, ls=":", lw=1.0)
        m = np.isfinite(lens[: i + 1])
        ax_l.plot(eps[: i + 1][m], lens[: i + 1][m], color=RL, lw=1.7)
        if f["ok"]:
            ax_l.scatter([f["ep"]], [f["len"]], s=26, c=RL, zorder=4)
        ax_l.set_xlim(0, EPISODES); ax_l.set_ylim(Lstar - 1, top + 1)
        ax_l.set_xlabel("епізод"); ax_l.set_ylabel("довжина шляху")
        ax_l.set_title(f"L*={Lstar:.2f}   L=" + (f"{f['len']:.2f}" if f["ok"] else "—"), pad=6)
        ax_l.grid(True, color=GRIDC, lw=0.5)
        if not np.isfinite(lens[: i + 1]).any():
            ax_l.text(EPISODES * .5, (Lstar + top) / 2, "жадібна політика\nще не доходить до B",
                      ha="center", va="center", color=MUTED, fontsize=8)

        # 4. what it has actually experienced
        ax_c.plot(eps[: i + 1], colls[: i + 1], color=TRAP, lw=1.7, label="колізій (сумарно)")
        ax_c.set_xlim(0, EPISODES); ax_c.set_ylim(0, max(colls.max(), 1) * 1.05)
        ax_c.set_xlabel("епізод"); ax_c.set_ylabel("колізій", color=TRAP)
        ax_c.grid(True, color=GRIDC, lw=0.5)
        ax2 = ax_c.twinx()
        ax2.plot(eps[: i + 1], learned[: i + 1], color=APF, lw=1.7)
        ax2.set_ylim(0, n_obst * 1.05)
        ax2.set_ylabel(f"дерев вивчено / {n_obst}", color=APF, fontsize=8, labelpad=2)
        ax2.tick_params(labelsize=7)
        ax_c.set_title(f"колізій {f['coll']}   ·   дерев вивчено {f['learned']}/{n_obst}", pad=6)

        fig.suptitle(f"seed {seed} · епізод {f['ep']:>4} / {EPISODES}"
                     f"    (кадр кожні {STRIDE} епізодів)",
                     fontsize=11, color=INK, y=0.975)
        return []

    anim = FuncAnimation(fig, frame, frames=len(frames), interval=1000 // fps)
    anim.save(out_mp4, writer=FFMpegWriter(fps=fps, bitrate=2600))
    print(f"  wrote {out_mp4}  ({len(frames)} frames, every {STRIDE} episodes)")

    if out_gif:
        # GIF cannot carry 301 frames at a sane size; take every 3rd.
        keep = list(range(0, len(frames), 3))
        anim2 = FuncAnimation(fig, frame, frames=keep, interval=1000 // 10)
        anim2.save(out_gif, writer=PillowWriter(fps=10))
        print(f"  wrote {out_gif}  ({len(keep)} frames, every {STRIDE * 3} episodes)")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-gif", action="store_true")
    a = p.parse_args()
    render(a.seed, f"tier_d/admin/reports/training_seed{a.seed}.mp4",
           None if a.no_gif else f"tier_d/admin/reports/training_seed{a.seed}.gif")


if __name__ == "__main__":
    main()
