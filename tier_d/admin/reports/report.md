# AI SkyRun — organizer validation report

Everything below was produced by `python -m viz.report`, `python -m scoring.harness`
and `python -m apf.gain_sweep` on this machine. No number is hand-written.

Reproduce: `pytest && python -m scoring.precompute && python -m viz.report`

---

## Verdict

| Question the organizer must answer | Answer | Evidence |
|---|---|---|
| Does the environment reproduce byte-for-byte? | **Yes**, and across Python↔JS | `tests/test_js_parity.py` |
| Can the agent beat the oracle (cheat)? | **No.** `η = L*/L ≤ 1.000` always | `test_invariant_3_efficiency_never_exceeds_one` |
| Does shaping change the optimum? | **No.** Both reach exactly `L*` | `test_invariant_4_...` |
| Does shaping speed learning? | **Yes.** 1990 vs 2492 episodes | leaderboard, below |
| Does the scoreboard discriminate? | **Yes.** 1.00 vs 0.00 success | leaderboard, below |
| Is `V(s)` a navigation function? | **Yes.** Single minimum, at the goal | `has_single_minimum` on all seeds |
| Does RL remove APF's local minimum? | **Yes.** Demonstrated, not asserted | `06_apf_vs_rl.png` |
| Can a student read the map? | **No.** Blocked structurally | `BlindEnv` in `test_invariants.py` |

**125 tests, all passing, in 11.5 s.** Full training run: 3000 episodes in ~0.2 s.

---

## 1. Environment and oracle · `01_forest_and_paths.png`

Seed 0, 16 trees. `L* = 12.7196` (A* over the shared occupancy grid).
The learned greedy policy flies `L = 14.8451`, so `η = L*/L = 0.857`.

Empty forest sanity check: `L* = 11.3137` = the exact Euclidean diagonal
`√(8²+8²)`, matching the build plan's expected `11.31`.

`string_pull` smoothing is rendered for the eye but **never scored** — smoothing
would put `η = 1.0` permanently out of reach.

## 2. Learning curves · `02_learning_curves.png`

Sample efficiency is read off the **greedy** evaluation curve, not the
exploration episodes: with `ε` floored at 0.05 the exploring agent keeps taking
random detours long after it has learned, which hides the moment it got good.

On seed 0, the greedy policy first reaches B at episode **2225** with shaping
and **2400** without.

## 3. Reward and shaping · `03_reward_and_shaping.png`

`r = −1` step, `−100` collision (terminal), `+100` goal (terminal).

Shaping is potential-based: `F = γ·Φ(s′) − Φ(s)` with `Φ(s) = −dist(s, B)`.
The third panel shows the property that makes it safe — **`Σ F` around any closed
loop is exactly 0** (at `γ=1`), so no cycle can be made profitable and the
optimal policy is untouched (Ng–Harada–Russell 1999).

`Φ(terminal) := 0`. Without that the guarantee is void; it is enforced in
`agent/shaping.py` and tested.

## 4. The value surface · `04_value_surface.png`, `value_surface.gif`

The peaks are **grown from reward, not stamped from the map**. A tree cell's
height is `min` over the learned `Q(neighbour, action_into_tree)` — i.e. the pain
it actually caused. A tree the agent never hit stays flat:

| episode | obstacle cells learned about | mean learned depth |
|---:|---:|---:|
| 50 | 10 / 50 | −3.8 |
| 300 | 36 / 50 | −17.8 |
| 1500 | 50 / 50 | −41.0 |
| 3000 | 50 / 50 | −52.0 |

This is INVARIANT 2 made visible, and it is why the GIF is the recruiting asset:
the surface is not revealed, it is *built*.

## 5. Navigation function · `05_navigation_function.png`

Learned `−V(s)` beside the oracle's `U*(s)` (Dijkstra cost-to-go). Both
normalised, because they measure different things and only the shape compares.

`has_single_minimum(U*)` holds on every seed tested: every reachable non-goal
cell has a strictly cheaper neighbour. One minimum, at the goal, no traps —
the Rimon–Koditschek property the concept promises.

## 6. APF vs RL · `06_apf_vs_rl.png`  ← the central argument

Classical APF, given the gains in `tier_d/reports/apf_gain_sweep.md`, solves **66%** of
feasible forests, stalls on **34%**, and **collides on 0%** — it fails by
stalling, which is exactly the Koren & Borenstein (1991) local-minimum defect.

On seed 0 it stalls at **(5.16, 6.20)**, boxed in by a tree cluster.
On the same forest:

| planner | sees the map? | outcome | `η = L*/L` |
|---|---|---|---|
| APF (`−∇U`) | **yes** | trapped at (5.16, 6.20) | — |
| APF + learned residual | **yes** | reaches B | 0.863 |
| Q-learning | **no** | reaches B | 0.857 |

Read that column order carefully. The two planners that get the tree coordinates
are the ones that need them; the **map-blind** learner solves it anyway. Across
the first six trapping seeds, residual and Q-learning each reach the goal 6/6.

> The APF track **violates INVARIANT 2 on purpose**. It is cartography, kept as a
> controlled foil. It must never be scored against student submissions.

## 7. LIDAR · `07_lidar.png`

Analytic ray-vs-circle, closed form, no physics engine. 16 rays, 3.0 m range.
Left: scan fans along the flown trajectory. Right: the range image a Tier 3
continuous-state agent would actually consume.

The tabular core never calls this. It exists for rendering and for Tier 3.

## 8. 3D flight · `08_flight_3d.png`, `flight_3d.gif`

Webots is not installed on this machine, so the learned greedy policy is flown
through a matplotlib 3D scene: real cylinders, the real trajectory, live LIDAR.

`webots/bridge.py` holds the policy→waypoint conversion a Webots controller
would consume and is unit-tested **without importing Webots** (118 control ticks
from A to B). The bridge follows waypoints rather than raw 8-way action vectors —
steering on the action vector cannot home onto a cell centre, and inside the goal
cell the terminal `Q` row is all zeros, so `argmax` would fly the drone back out.

---

## Leaderboard — HIDDEN seeds (10 seeds, 3000-episode budget)

| agent | success | collision | optimality | episodes to 1.15·L* |
|---|---|---|---|---|
| reference (Q-learning + shaping) | 1.00 | 0.00 | 0.937 | **1990** (90% reach) |
| ablation (no shaping) | 1.00 | 0.00 | 0.916 | 2492 (80% reach) |
| baseline (random / untrained) | 0.00 | 0.00 | 0.000 | 3000 (0% reach) |

Rank is lexicographic: learned → safe → optimal → fast.
Sample efficiency is **right-censored at the budget**; an uncensored mean would
be `inf` and erase every difference between agents.

Both trained agents reach 1.00 success and 0.00 collision, so the ranking is
decided on optimality then sample efficiency — precisely the intended behaviour.

---

## Known knife-edges (for the organizer, not the students)

1. **Suicide is nearly rational.** With `γ = 0.99`, wandering forever is worth
   `−1/(1−γ) = −100`, exactly the collision penalty. The `+100` goal breaks the
   tie, and the agent converges — but the margin is zero. If you raise
   `max_steps` or lower `γ`, re-check that collision rate stays at 0.00.

2. **`TREE_R = 0.40` vs `CELL = 0.40`.** Occupancy is by cell *centre*, so a tree
   blocks between one and four cells. The rejection sampler's `MIN_TREE_SEP = 0.85`
   then makes a *sealed* wall impossible to generate — no `make_forest` seed has
   ever been infeasible. `is_feasible` is therefore vacuous on generated forests,
   though it correctly rejects a hand-built palisade (`tests/test_oracle.py`).
   Keep the check; it costs nothing and guards against future constant changes.

3. **Seed 0 converges to `η = 0.857`**, just outside the `1.15·L*` bar. The bar is
   still the right one — 90% of hidden seeds cross it, which is what makes the
   metric discriminating rather than saturated.

4. **`HIDDEN_SEEDS` currently live in `scoring/seeds.py`**, which ships with the
   kit. Before the event, move them to a file that is not distributed.
