# AI SkyRun — organizer kit

Grid-world RL hackathon kit: seeded forest MDP, A* oracle, reference tabular
Q-learning with potential-based shaping, an APF baseline, a `V(s)` visualiser, a
scoring harness on hidden seeds, and a student scaffold.

Pure Python + NumPy + Matplotlib. The core is **tabular** — no neural nets, no GPU.
A 3000-episode training run takes ~0.2 s; the full test suite ~1 min.

## Tiers

Each tier is self-contained (code, tests, reports, `CONCEPT.md`) and split into a
student part and an organizer part:

| Folder | Track | Stack |
|---|---|---|
| [`tier_d/`](tier_d/) | the base **2D** competition kit described below: grid MDP, A* oracle, tabular Q-learning, APF foil, scoring, scaffold | main venv |
| [`tier_a/`](tier_a/) | research: 6DOF attitude control along a reference path — ends in the measured fixed-thrust physics veto | main venv (+optional torch) |
| [`tier_b/`](tier_b/) | research: CTBR pirouette in gym-pybullet-drones, PPO vs SAC | conda env `drones` |
| [`tier_c/`](tier_c/) | realistic FPV drone in Blender: A* + APF + kinematic autopilot, manual & autonomous modes | Blender (bundled python) |

## Admin/student split

Every tier keeps its organizer materials in `tier_<x>/admin/` — reference
agents/solutions, solution notebooks, scoring harnesses and caches, **hidden
seeds**, tests, generated reports. Everything outside `admin/` is safe to hand
to teams.

**The student handout = this repo minus every `*/admin/` folder (and,
optionally, `prompts/`).** That is the whole packaging rule: after deleting
`*/admin/`, the kit still runs — public seeds work, `validate_team*` scripts
give a friendly message until the team fills its solution stub, and
`tier_b/rewards.py` transparently switches to the student implementation.

AI-assistant system prompts for the event live in [`prompts/`](prompts/):
one universal protocol + one card per tier.

---

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Verify: `python -c "import numpy, matplotlib; print('ok')"`

## Run

### Core

```bash
pytest                                               # full suite (all tiers)
pytest tier_d/admin/tests/test_invariants.py -v      # the six invariants alone

python -m tier_d.admin.scoring.precompute            # L* + feasibility cache (offline)
python -m tier_d.admin.scoring.harness               # leaderboard on HIDDEN seeds
python -m tier_d.admin.scoring.harness --public      # self-check on PUBLIC seeds
python -m tier_d.admin.agent.qlearning --seed 0 --trees 16 --episodes 3000
```

### Visualisation & demos

```bash
python -m tier_d.admin.viz.report                    # all figures + GIFs -> tier_d/admin/reports/
python -m tier_d.admin.viz.report --fast             # skip the animations
python -m tier_d.admin.viz.training_film --seed 0    # mp4 + gif, one frame per 10 episodes
python -m tier_d.admin.viz.export_runs               # train 10 maps -> tier_d/admin/reports/runs.json
python -m tier_d.admin.viz.build_lab                 # -> tier_d/admin/reports/lab.html (interactive)
```

### Experiments (wind & ablations)

```bash
python tier_d/admin/experiments/compare_wind.py --seed 0  # deterministic vs 5% wind gusts
python tier_d/admin/experiments/compare_wind.py --seed 0 --wind-prob 0.10
```

`training_film` needs `ffmpeg` for the MP4; pass `--no-gif` to skip the slow GIF.
`build_lab` inflates its data in-browser with `DecompressionStream`, so it needs a
2023-or-later browser.

Open `tier_d/viz/field_scanner.html` in a browser for the Tier 0 APF demo. Its forest is
bit-identical to Python's (`tier_d/admin/tests/test_js_parity.py` enforces it; needs `node`).

## Layout (tier_d)

| Path | What |
|---|---|
| `tier_d/env/` | `constants.py` (all geometry), `forest.py` (mulberry32), `occupancy.py` (the one collision test), `gridworld.py` (the MDP), `lidar.py` |
| `tier_d/oracle/` | `astar.py` (L*, feasibility), `value_iteration.py` (navigation function, V*) |
| `tier_d/scoring/` | `seeds.py` — PUBLIC seeds (+ guarded import of hidden ones from admin/) |
| `tier_d/scaffold/` | `starter.ipynb` (two `# TODO` holes), `team_solution.py` (student export stub), `qtools.py` (plot helpers), `README.md`, `validate_team.py` |
| `tier_d/viz/` | student-visible: `style.py`, `forest3d.py`, `value_surface.py`, Field Scanner |
| `tier_d/webots/` | `bridge.py` — policy→waypoint, no simulator import |
| `tier_d/admin/agent/` | `shaping.py`, `qlearning.py` — the reference agent |
| `tier_d/admin/apf/` | the discarded track, kept as a foil (reads the map on purpose) |
| `tier_d/admin/scoring/` | `harness.py`, `precompute.py`, `cache_Lstar.json`, `hidden_seeds.py` |
| `tier_d/admin/tests/` | the full test suite, incl. `test_scaffold.py` (contains the TODO answers) |
| `tier_d/admin/viz/` | report/animation/artifact builders |
| `tier_d/admin/reports/` | generated: figures, GIFs, `facts.json`, `leaderboard.csv`, `report.md`, … |

## Read first

**For teams:**
- `tier_d/scaffold/README.md` — RL concepts, the invariants teams must respect
- `tier_d/scaffold/GAMEPLAY_GUIDE.md` — diagnostic checklist, Tier 1–3 approaches, FAQ
- `tier_d/scaffold/validate_team.py` — self-check on PUBLIC_SEEDS before final

**For organizers:**
- `CLAUDE.md` — the six correctness invariants and phase gating
- `tier_d/admin/LAUNCH.md` — pre-event checklist, merge-freeze, hidden seeds discipline
- `tier_d/admin/ORGANIZERS_GUIDE.md` — 3-day schedule, judging rubric, troubleshooting
- `tier_d/admin/reports/report.md` — validation summary: what was checked, what passed
- `prompts/README.md` — AI-assistant prompts: universal protocol + per-tier cards

## Not done

**Phase 7 (Webots) is not verified end-to-end** — Webots is not installed on this
machine. `tier_d/webots/bridge.py` contains the policy→waypoint logic and is unit-tested
without importing the simulator; the 3D flight is rendered in Matplotlib instead.
Writing the `.wbt` world and the controller is the remaining work.

**Phase 8 (packaging)** — versions are not frozen and no local pip mirror exists.
Run `pip freeze > requirements.lock` only after a successful M4 dry-run.

Hidden seeds now live in `tier_*/admin/scoring/hidden_seeds.py`; the student
handout is produced by deleting `*/admin/` (see “Admin/student split” above).
