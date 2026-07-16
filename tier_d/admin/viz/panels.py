"""The remaining report panels: APF, LIDAR, learning curves, reward anatomy."""

from __future__ import annotations

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

from tier_d.admin.agent.shaping import potential
from tier_d.admin.apf.field import potential_grid
from tier_d.env.constants import DOMAIN, GRID_N, LIDAR_RANGE, R_COLLISION, R_GOAL, R_STEP
from tier_d.env.lidar import endpoints, scan
from tier_d.env.occupancy import cell_centers
from tier_d.viz.style import (
    APF,
    GOAL,
    GRIDC,
    INK,
    LIDARC,
    MUTED,
    ORACLE,
    RESIDUAL,
    RL,
    TRAP,
    draw_endpoints,
    draw_forest,
    draw_occupancy,
    setup_domain,
)


# --- APF -------------------------------------------------------------------
def plot_apf_surface_3d(ax, trees: np.ndarray, stall_xy=None, clip: float = 40.0) -> None:
    X, Y, U = potential_grid(trees, n=160, clip=clip)
    ax.plot_surface(X, Y, U, cmap=cm.inferno, linewidth=0, antialiased=True, alpha=0.95)
    if stall_xy is not None:
        ax.scatter(*stall_xy, clip, s=60, c=TRAP, marker="v", depthshade=False, zorder=10)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("U(x,y)")
    ax.set_box_aspect((1, 1, 0.55))
    ax.view_init(elev=48, azim=-120)
    ax.grid(False)
    ax.set_title("APF potential U = U_att + U_rep\n(hand-built, needs the map)", pad=2)


def plot_apf_trap_2d(ax, trees, apf_path, rl_path, res_path=None, stall_xy=None,
                     oracle_path=None, title="") -> None:
    setup_domain(ax, title)
    X, Y, U = potential_grid(trees, n=240, clip=30.0)
    ax.contourf(X, Y, U, levels=28, cmap="inferno", alpha=0.30, zorder=0)
    ax.contour(X, Y, U, levels=16, colors="white", linewidths=0.3, alpha=0.5, zorder=1)
    draw_forest(ax, trees)
    if oracle_path is not None:
        ax.plot(oracle_path[:, 0], oracle_path[:, 1], color=ORACLE, lw=1.8, ls="--",
                zorder=4, label="A* optimum")
    ax.plot(apf_path[:, 0], apf_path[:, 1], color=APF, lw=2.6, zorder=5, label="APF (−∇U)")
    if res_path is not None:
        ax.plot(res_path[:, 0], res_path[:, 1], color=RESIDUAL, lw=2.2, ls="-.",
                zorder=5, label="APF + residual")
    ax.plot(rl_path[:, 0], rl_path[:, 1], color=RL, lw=2.6, zorder=6, label="Q-learning (map-blind)")
    if stall_xy is not None:
        ax.scatter(*stall_xy, s=150, c=TRAP, marker="v", zorder=8, edgecolors="white",
                   linewidths=1.2, label="APF local minimum")
    draw_endpoints(ax)
    ax.legend(loc="upper left", fontsize=7)


# --- LIDAR -----------------------------------------------------------------
def plot_lidar_scans(ax, trees: np.ndarray, path: np.ndarray, every: int = 4, title="") -> None:
    setup_domain(ax, title)
    draw_forest(ax, trees)
    pts = path[::every]
    for p in pts:
        d = scan(p, trees)
        e = endpoints(p, d)
        for k, (ex, ey) in enumerate(e):
            hit = d[k] < LIDAR_RANGE - 1e-6
            ax.plot([p[0], ex], [p[1], ey], color=LIDARC, lw=0.45,
                    alpha=0.75 if hit else 0.22, zorder=3)
            if hit:
                ax.scatter(ex, ey, s=3.5, c=TRAP, zorder=4)
    ax.plot(path[:, 0], path[:, 1], color=RL, lw=2.4, zorder=5, label="flown trajectory")
    ax.scatter(pts[:, 0], pts[:, 1], s=12, c=INK, zorder=6, label="scan pose")
    draw_endpoints(ax)
    ax.legend(loc="upper left", fontsize=7)


def plot_lidar_returns(ax, trees: np.ndarray, path: np.ndarray) -> None:
    """Range image along the trajectory: what a Tier 3 state would actually see."""
    from tier_d.env.lidar import scan_path

    S = scan_path(path, trees)
    im = ax.imshow(S.T, aspect="auto", origin="lower", cmap="magma_r",
                   extent=(0, len(path), 0, 360), vmin=0, vmax=LIDAR_RANGE)
    ax.set_xlabel("step along trajectory")
    ax.set_ylabel("ray bearing (deg)")
    ax.set_title("LIDAR range image (16 rays)", pad=6)
    plt.colorbar(im, ax=ax, label="distance", fraction=0.046, pad=0.03)


# --- Learning curves -------------------------------------------------------
def plot_learning_curves(axes, hist_shaped, hist_plain, Lstar: float) -> None:
    ax1, ax2 = axes

    def smooth(x, k=51):
        x = np.asarray(x, dtype=float)
        if len(x) < k:
            return x
        return np.convolve(x, np.ones(k) / k, mode="valid")

    n_ep = max(hist_shaped.eval_ep[-1], hist_plain.eval_ep[-1])
    top = Lstar + 12

    ax1.axhline(Lstar, color=ORACLE, ls="--", lw=1.4, label=f"L* = {Lstar:.2f}")
    ax1.axhline(1.15 * Lstar, color=MUTED, ls=":", lw=1.0, label="1.15·L* threshold")
    series = ((hist_shaped, RL, "with shaping", -0.35, "right"),
              (hist_plain, APF, "no shaping", 0.35, "left"))
    for h, c, lbl, dy, ha in series:
        ep = np.array(h.eval_ep)
        L = np.array(h.eval_len, dtype=float)
        ok = np.isfinite(L)
        # Before the first success the greedy policy has no path at all. Draw that
        # stretch flat along the top rather than silently cropping the x-axis.
        ax1.plot(ep[~ok], np.full(int((~ok).sum()), top), color=c, lw=1.0, alpha=0.30)
        ax1.plot(ep[ok], L[ok], color=c, lw=1.7, label=lbl)
        if ok.any():
            first = int(ep[ok][0])
            ax1.axvline(first, color=c, ls=":", lw=0.9, alpha=0.6)
            # Offset the two labels sideways or they land on top of each other.
            ax1.annotate(f"first path @{first}", (first, top - 3.0 + dy * 5),
                         color=c, fontsize=6.5, ha=ha, va="top",
                         xytext=(4 if ha == "left" else -4, 0), textcoords="offset points")

    ax1.text(n_ep * 0.02, top - 0.35, "greedy policy reaches no goal",
             fontsize=6.5, color=MUTED, va="top")
    ax1.set_xlabel("episode"); ax1.set_ylabel("greedy path length")
    ax1.set_title("Greedy policy length vs episodes", pad=6)
    ax1.set_xlim(0, n_ep)
    ax1.set_ylim(Lstar - 1, top + 0.6)
    ax1.grid(True, color=GRIDC, lw=0.5)
    ax1.legend(fontsize=7.5, loc="center left")

    for h, c, lbl in ((hist_shaped, RL, "with shaping"), (hist_plain, APF, "no shaping")):
        ax2.plot(smooth(h.success), color=c, lw=1.5, label=lbl)
    ax2.set_xlabel("episode"); ax2.set_ylabel("success rate (51-ep mean)")
    ax2.set_title("Exploration success rate", pad=6)
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, color=GRIDC, lw=0.5)
    ax2.legend(fontsize=7.5)


# --- Reward + shaping anatomy ---------------------------------------------
def plot_reward_anatomy(axes) -> None:
    ax1, ax2, ax3 = axes

    names = ["step", "collision", "goal"]
    vals = [R_STEP, R_COLLISION, R_GOAL]
    colors = [MUTED, TRAP, RESIDUAL]
    bars = ax1.bar(names, vals, color=colors, width=0.55)
    ax1.axhline(0, color=INK, lw=0.8)
    ax1.set_title("Reward function r(s,a)", pad=6)
    ax1.set_ylabel("reward")
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + (5 if v > 0 else -12),
                 f"{v:+.0f}", ha="center", fontsize=8.5)
    ax1.grid(True, axis="y", color=GRIDC, lw=0.5)

    gx, gy = cell_centers()
    phi = np.array([[potential(r * GRID_N + c) for c in range(GRID_N)] for r in range(GRID_N)])
    im = ax2.imshow(phi, origin="lower", extent=(0, DOMAIN, 0, DOMAIN), cmap="viridis")
    ax2.set_title("Shaping potential  Φ(s) = −dist(s, B)", pad=6)
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)

    # F telescopes to zero on a closed loop -- the reason the optimum survives.
    gamma = 1.0
    loop = [0, 1, GRID_N + 1, GRID_N, 0]
    F = [gamma * potential(b) - potential(a) for a, b in zip(loop, loop[1:])]
    cum = np.cumsum([0.0] + F)
    ax3.step(range(len(cum)), cum, where="post", color=RL, lw=2.0)
    ax3.scatter(range(len(cum)), cum, s=22, c=RL, zorder=3)
    ax3.axhline(0, color=TRAP, ls="--", lw=1.2)
    ax3.set_title("Σ F around a closed loop = 0\n(γ=1) ⇒ optimal policy unchanged", pad=6)
    ax3.set_xlabel("step around the loop"); ax3.set_ylabel("cumulative F")
    ax3.grid(True, color=GRIDC, lw=0.5)


def plot_forest_and_paths(ax, trees, blocked, oracle_path, rl_path, title="") -> None:
    setup_domain(ax, title)
    draw_occupancy(ax, blocked)
    draw_forest(ax, trees)
    if oracle_path is not None:
        ax.plot(oracle_path[:, 0], oracle_path[:, 1], color=ORACLE, lw=2.2, ls="--",
                zorder=4, label="A* optimum (L*)")
    ax.plot(rl_path[:, 0], rl_path[:, 1], color=RL, lw=2.4, zorder=5, label="learned policy (L)")
    draw_endpoints(ax)
    ax.legend(loc="upper left", fontsize=7.5)
