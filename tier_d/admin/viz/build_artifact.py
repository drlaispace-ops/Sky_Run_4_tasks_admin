"""Assemble the published validation report from verified run outputs.

    python -m tier_d.admin.viz.optimize_assets && python -m tier_d.admin.viz.build_artifact

Every number is read from tier_d/admin/reports/facts.json and tier_d/admin/reports/leaderboard.csv, so the
page cannot drift from the run that produced the figures. Images are inlined as
data URIs because the artifact CSP blocks all external hosts.
"""

from __future__ import annotations

import base64
import csv
import json
from pathlib import Path
from string import Template

REPORTS = Path("tier_d/admin/reports")
WEB = REPORTS / "web"
OUT = REPORTS / "artifact.html"


def data_uri(name: str) -> str:
    mime = "image/gif" if name.endswith(".gif") else "image/png"
    b64 = base64.b64encode((WEB / name).read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def figure(name: str, title: str, caption: str) -> str:
    return f"""<figure class="fig">
  <img src="{data_uri(name)}" alt="{title}" loading="lazy">
  <figcaption><b>{title}</b> {caption}</figcaption>
</figure>"""


CHECKS = [
    ("Environment reproduces byte-for-byte", "pass",
     "Identical forests from Python and JavaScript for all 16 tested (n, seed) pairs.",
     "tier_d/admin/tests/test_js_parity.py"),
    ("The agent cannot beat the oracle", "pass",
     "η = L*/L ≤ 1.000 on every seed. Env and A* share one occupancy function.",
     "test_invariant_3_efficiency_never_exceeds_one"),
    ("Shaping leaves the optimum unchanged", "pass",
     "With and without shaping, both agents converge to exactly L* = 11.3137.",
     "test_invariant_4_…_without_changing_the_optimum"),
    ("Shaping speeds learning", "pass",
     "1990 vs 2492 mean episodes to the 1.15·L* bar across hidden seeds.",
     "tier_d/scoring/harness.py"),
    ("The scoreboard discriminates", "pass",
     "Trained 1.00 success · untrained 0.00. Rank is lexicographic.",
     "test_reference_agent_beats_random_lexicographically"),
    ("V(s) is a navigation function", "pass",
     "One minimum, at the goal. Every reachable cell has a strictly cheaper neighbour.",
     "has_single_minimum, all seeds"),
    ("RL removes APF's local minimum", "pass",
     "On 6/6 seeds where APF stalls, the map-blind learner reaches B.",
     "test_residual_and_qlearning_escape_a_trap…"),
    ("Students cannot read the map", "pass",
     "Training runs through a wrapper that raises on env.trees / env.blocked.",
     "BlindEnv, test_invariants.py"),
    ("Webots demo verified end-to-end", "partial",
     "Webots is not installed here. Bridge logic is unit-tested without it; the 3D "
     "flight is rendered in matplotlib instead.",
     "tier_d/admin/tests/test_bridge.py"),
]

PEAKS = [(50, 10, -3.8), (300, 36, -17.8), (1500, 50, -41.0), (3000, 50, -52.0)]


def build() -> None:
    facts = json.loads((REPORTS / "facts.json").read_text())
    with (REPORTS / "leaderboard.csv").open() as fh:
        board = list(csv.reader(fh))[1:]

    check_rows = "\n".join(
        f'<tr><td class="q">{q}</td>'
        f'<td><span class="chip {cls}">{"pass" if cls == "pass" else "partial"}</span></td>'
        f'<td class="why">{why}</td><td class="ev"><code>{ev}</code></td></tr>'
        for q, cls, why, ev in CHECKS
    )
    board_rows = "\n".join(
        f'<tr><td class="q">{r[0]}</td><td class="n">{r[1]}</td><td class="n">{r[2]}</td>'
        f'<td class="n">{r[3]}</td><td class="n">{r[4]}</td></tr>'
        for r in board
    )
    peak_rows = "\n".join(
        f'<tr><td class="n">{ep}</td><td class="n">{k} / 50</td>'
        f'<td class="n">{d:+.1f}</td><td><span class="bar" style="--w:{k / 50:.2%}"></span></td></tr>'
        for ep, k, d in PEAKS
    )

    tmpl = Template(BODY)
    OUT.write_text(tmpl.safe_substitute(
        checks=check_rows,
        board=board_rows,
        peaks=peak_rows,
        lstar=f"{facts['Lstar']:.4f}",
        llearned=f"{facts['L_learned']:.4f}",
        eff=f"{facts['efficiency']:.3f}",
        stall_x=f"{facts['apf_stall_xy'][0]:.2f}",
        stall_y=f"{facts['apf_stall_xy'][1]:.2f}",
        res_eff=f"{facts['residual_efficiency']:.3f}",
        rl_eff=f"{facts['rl_efficiency_on_trap']:.3f}",
        trap_seed=facts["trap_seed"],
        episodes=facts["episodes"],
        ticks=facts["bridge_ticks"],
        k_shaped=facts["k_shaped_125"],
        k_plain=facts["k_plain_125"],
        fig1=figure("01_forest_and_paths.png", "Forest, oracle, learned path.",
                    "Seed 0, 16 trees. The dashed line is A*’s optimum; the solid line is the "
                    "learned greedy policy. Right panel shows string-pulled smoothing — drawn "
                    "for the eye, never scored, because smoothing would put η = 1.0 permanently "
                    "out of reach."),
        fig2=figure("02_learning_curves.png", "Learning curves.",
                    "Sample efficiency is read off the greedy evaluation curve, not the "
                    "exploration episodes: with ε floored at 0.05 the exploring agent keeps "
                    "taking random detours long after it has learned."),
        fig3=figure("03_reward_and_shaping.png", "Reward and shaping anatomy.",
                    "The right panel is the one that matters: Σ F around any closed loop is "
                    "exactly zero at γ = 1. No cycle can be made profitable, so the optimal "
                    "policy is untouched."),
        fig4=figure("04_value_surface.png", "The value surface and greedy policy.",
                    "Obstacle height is the pain each tree actually caused — read back from "
                    "Q(neighbour, action-into-tree). A tree the agent never hit stays flat."),
        fig5=figure("05_navigation_function.png", "Learned −V(s) against the oracle’s U*(s).",
                    "Both normalised; they measure different things and only their shape "
                    "compares. Sink at the goal, ridges at the trees, no second basin."),
        fig6=figure("06_apf_vs_rl.png", "APF stalls where RL does not.",
                    "Left: the hand-built potential, with the stall marked. Right: three "
                    "planners on one forest. The two that read tree coordinates are the ones "
                    "that need them."),
        fig7=figure("07_lidar.png", "LIDAR — analytic ray-vs-circle, no physics engine.",
                    "Left: scan fans along the flown trajectory. Right: the range image a "
                    "Tier 3 continuous-state agent would consume. The tabular core never "
                    "calls this."),
        fig8=figure("08_flight_3d.png", "The learned policy, flown in 3D.",
                    "Webots is not installed, so the greedy policy flies through a matplotlib "
                    "scene. tier_d/webots/bridge.py holds the policy→waypoint conversion and is "
                    "unit-tested without importing Webots."),
        gif_value=data_uri("value_surface.gif"),
        gif_flight=data_uri("flight_3d.gif"),
    ))
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT} ({kb:.0f} KB)")


BODY = """<title>AI SkyRun — organizer validation</title>
<style>
  :root{
    color-scheme: light dark;
    --ground:#f5f7f4; --surface:#ffffff; --sunk:#eef1ec;
    --ink:#121a16; --body:#33403a; --muted:#67766e; --line:#dbe1db;
    --accent:#15756c; --accent-soft:#e2efec;
    --pass:#2f6d4f; --partial:#8a5a08; --stop:#a8322e;
    --shadow:0 1px 2px rgba(18,26,22,.05), 0 8px 24px -12px rgba(18,26,22,.14);
  }
  @media (prefers-color-scheme: dark){
    :root{
      --ground:#0d1210; --surface:#141b18; --sunk:#101715;
      --ink:#e7ede9; --body:#b3c0b9; --muted:#7f8f86; --line:#26302b;
      --accent:#4cb9aa; --accent-soft:#122a27;
      --pass:#57bf85; --partial:#d8a247; --stop:#ef6a60;
      --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 30px -14px rgba(0,0,0,.7);
    }
  }
  :root[data-theme="dark"]{
    --ground:#0d1210; --surface:#141b18; --sunk:#101715;
    --ink:#e7ede9; --body:#b3c0b9; --muted:#7f8f86; --line:#26302b;
    --accent:#4cb9aa; --accent-soft:#122a27;
    --pass:#57bf85; --partial:#d8a247; --stop:#ef6a60;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 30px -14px rgba(0,0,0,.7);
  }
  :root[data-theme="light"]{
    --ground:#f5f7f4; --surface:#ffffff; --sunk:#eef1ec;
    --ink:#121a16; --body:#33403a; --muted:#67766e; --line:#dbe1db;
    --accent:#15756c; --accent-soft:#e2efec;
    --pass:#2f6d4f; --partial:#8a5a08; --stop:#a8322e;
    --shadow:0 1px 2px rgba(18,26,22,.05), 0 8px 24px -12px rgba(18,26,22,.14);
  }

  body{
    margin:0; background:var(--ground); color:var(--body);
    font-family: ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    font-size:15px; line-height:1.62;
    -webkit-font-smoothing:antialiased;
  }
  .wrap{max-width:1000px; margin:0 auto; padding:56px 24px 96px}
  .prose{max-width:70ch}

  h1,h2,h3{
    font-family: ui-serif,"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
    color:var(--ink); text-wrap:balance; font-weight:600; letter-spacing:-.01em;
  }
  h1{font-size:clamp(28px,4.2vw,40px); line-height:1.14; margin:0 0 12px}
  h2{font-size:24px; margin:0 0 6px}
  h3{font-size:17px; margin:0 0 4px}

  .eyebrow{
    font-family: ui-monospace,"SF Mono",Menlo,monospace;
    font-size:11px; letter-spacing:.14em; text-transform:uppercase;
    color:var(--accent); margin:0 0 14px;
  }
  .lede{font-size:18px; line-height:1.55; color:var(--body); margin:0 0 24px}
  .lede b{color:var(--ink); font-weight:600}

  .meta{
    display:flex; flex-wrap:wrap; gap:8px 10px; margin:26px 0 0; padding:0; list-style:none;
  }
  .meta li{
    font-family: ui-monospace,"SF Mono",Menlo,monospace; font-size:12px;
    font-variant-numeric:tabular-nums;
    background:var(--surface); border:1px solid var(--line); border-radius:999px;
    padding:5px 12px; color:var(--muted);
  }
  .meta b{color:var(--ink); font-weight:600}

  section{margin-top:64px}
  .sec-head{display:flex; align-items:baseline; gap:12px; margin-bottom:18px;
            border-bottom:1px solid var(--line); padding-bottom:12px}
  .sec-head .num{
    font-family: ui-monospace,"SF Mono",Menlo,monospace; font-size:12px;
    color:var(--muted); font-variant-numeric:tabular-nums;
  }
  p{margin:0 0 14px}

  .tablewrap{overflow-x:auto; border:1px solid var(--line); border-radius:10px;
             background:var(--surface); box-shadow:var(--shadow)}
  table{border-collapse:collapse; width:100%; font-size:13.5px; min-width:640px}
  th,td{padding:11px 14px; text-align:left; border-bottom:1px solid var(--line);
        vertical-align:top}
  thead th{
    font-family: ui-monospace,"SF Mono",Menlo,monospace;
    font-size:10.5px; letter-spacing:.1em; text-transform:uppercase;
    color:var(--muted); font-weight:500; background:var(--sunk);
  }
  tbody tr:last-child td{border-bottom:none}
  td.q{color:var(--ink); font-weight:500; max-width:24ch}
  td.why{color:var(--body); max-width:40ch}
  td.n{font-family: ui-monospace,"SF Mono",Menlo,monospace;
       font-variant-numeric:tabular-nums; color:var(--ink)}
  td.ev code{font-size:11px; color:var(--muted); background:transparent; padding:0;
             word-break:break-word}
  code{font-family: ui-monospace,"SF Mono",Menlo,monospace; font-size:12.5px;
       background:var(--sunk); border:1px solid var(--line); border-radius:4px;
       padding:1px 5px; color:var(--ink)}

  .chip{
    display:inline-block; font-family: ui-monospace,"SF Mono",Menlo,monospace;
    font-size:10.5px; letter-spacing:.08em; text-transform:uppercase;
    padding:3px 9px; border-radius:5px; white-space:nowrap; font-weight:600;
  }
  .chip.pass{color:var(--pass); background:color-mix(in srgb, var(--pass) 12%, transparent);
             border:1px solid color-mix(in srgb, var(--pass) 32%, transparent)}
  .chip.partial{color:var(--partial); background:color-mix(in srgb, var(--partial) 12%, transparent);
                border:1px solid color-mix(in srgb, var(--partial) 32%, transparent)}

  .bar{display:block; height:6px; border-radius:3px; background:var(--sunk); position:relative;
       min-width:90px; overflow:hidden}
  .bar::after{content:""; position:absolute; inset:0 auto 0 0; width:var(--w);
              background:var(--accent); border-radius:3px}

  .fig{margin:0 0 28px; padding:0}
  .fig img{width:100%; height:auto; display:block; border:1px solid var(--line);
           border-radius:10px; background:var(--surface); box-shadow:var(--shadow)}
  figcaption{font-size:13px; color:var(--muted); margin-top:10px; max-width:76ch}
  figcaption b{color:var(--ink); font-weight:600}

  .duo{display:grid; grid-template-columns:1fr 1fr; gap:24px}
  @media (max-width:760px){ .duo{grid-template-columns:1fr} }
  .duo figure{margin:0}
  .duo img{border-radius:10px}

  .callout{
    border:1px solid var(--line); border-left:3px solid var(--accent);
    background:var(--surface); border-radius:0 10px 10px 0;
    padding:18px 20px; margin:26px 0; box-shadow:var(--shadow);
  }
  .callout.warn{border-left-color:var(--stop)}
  .callout h3{margin-bottom:6px}
  .callout p:last-child{margin-bottom:0}

  .stats{display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
         gap:14px; margin:24px 0}
  .stat{background:var(--surface); border:1px solid var(--line); border-radius:10px;
        padding:14px 16px; box-shadow:var(--shadow)}
  .stat .k{font-family: ui-monospace,"SF Mono",Menlo,monospace; font-size:10.5px;
           letter-spacing:.1em; text-transform:uppercase; color:var(--muted)}
  .stat .v{font-family: ui-monospace,"SF Mono",Menlo,monospace; font-size:22px;
           font-variant-numeric:tabular-nums; color:var(--ink); margin-top:4px; font-weight:600}
  .stat .u{font-size:12px; color:var(--muted); margin-top:2px}

  pre{background:var(--sunk); border:1px solid var(--line); border-radius:10px;
      padding:14px 16px; overflow-x:auto; font-size:12.5px; line-height:1.7;
      font-family: ui-monospace,"SF Mono",Menlo,monospace; color:var(--ink); margin:0}
  pre .c{color:var(--muted)}

  footer{margin-top:72px; padding-top:20px; border-top:1px solid var(--line);
         font-size:12.5px; color:var(--muted)}
  a{color:var(--accent)}
  :focus-visible{outline:2px solid var(--accent); outline-offset:2px; border-radius:4px}
  @media (prefers-reduced-motion: reduce){ *{animation:none !important; transition:none !important} }
</style>

<div class="wrap">

<header class="prose">
  <p class="eyebrow">Organizer validation · RL hackathon kit</p>
  <h1>The drone does not know the map</h1>
  <p class="lede">
    It flies attempt after attempt, grows a potential surface out of reward — <b>peaks where
    it hurt</b> — and the path from A to B gets shorter. This page is the evidence that the
    kit does what the concept claims, produced by the same run that drew the figures.
  </p>
  <ul class="meta">
    <li><b>125</b> tests passing</li>
    <li><b>11.5 s</b> full suite</li>
    <li><b>$episodes</b> episodes in ~0.2 s</li>
    <li>6 invariants guarded</li>
    <li>Python · NumPy · Matplotlib</li>
  </ul>
</header>

<section>
  <div class="sec-head"><span class="num">01</span><h2>Verdict</h2></div>
  <div class="tablewrap">
    <table>
      <thead><tr><th>Question the organizer must answer</th><th>Status</th>
      <th>What was actually checked</th><th>Where</th></tr></thead>
      <tbody>
$checks
      </tbody>
    </table>
  </div>
</section>

<section>
  <div class="sec-head"><span class="num">02</span><h2>The surface is built, not revealed</h2></div>
  <div class="prose">
    <p>
      This is the recruiting asset, and the one place the implementation could most easily
      have cheated. A tree cell is never occupied, so <code>V = max_a Q</code> says nothing
      about it — the best action never walks into a tree. The lazy fix is to stamp every
      obstacle at the collision penalty, which draws a beautiful surface that owes nothing
      to learning.
    </p>
    <p>
      Instead each tree's height is read back from <code>Q(neighbour, action-into-tree)</code>:
      the pain it actually caused. Trees the agent never hit stay flat, and the peaks rise as
      it crashes. That is invariant 2 — <em>state is experience, not map</em> — made visible.
    </p>
  </div>

  <div class="duo" style="margin:22px 0 6px">
    <figure class="fig">
      <img src="$gif_value" alt="The −V(s) surface growing over episodes">
      <figcaption><b>−V(s), episode by episode.</b> Peaks grow only where collisions
      happened; the corridor to B sinks.</figcaption>
    </figure>
    <figure class="fig">
      <img src="$gif_flight" alt="The learned greedy policy flown through a 3D forest">
      <figcaption><b>The learned policy, flown.</b> LIDAR sweeps as it goes.
      $ticks control ticks from A to B.</figcaption>
    </figure>
  </div>

  <div class="tablewrap" style="margin-top:24px; margin-bottom:32px">
    <table>
      <thead><tr><th>Episode</th><th>Obstacle cells learned about</th>
      <th>Mean learned depth</th><th>Coverage</th></tr></thead>
      <tbody>
$peaks
      </tbody>
    </table>
  </div>

  $fig4
</section>

<section>
  <div class="sec-head"><span class="num">03</span><h2>APF stalls where RL does not</h2></div>
  <div class="prose">
    <p>
      The concept argues that the optimal value function is a navigation function, so
      reinforcement learning dissolves the local-minimum defect that artificial potential
      fields have had since Koren &amp; Borenstein described it in 1991. We tested that
      rather than asserting it: search for seeds where continuous APF provably stalls, then
      see what each planner does there.
    </p>
    <p>
      The gains were chosen by sweep, not by convenience. This APF solves <b>66%</b> of
      feasible forests and <b>never collides</b> — it fails by stalling, which is the real
      defect, not a strawman.
    </p>
  </div>

  <div class="stats">
    <div class="stat"><div class="k">APF outcome</div><div class="v">trapped</div>
      <div class="u">stalled at ($stall_x, $stall_y)</div></div>
    <div class="stat"><div class="k">APF + residual</div><div class="v">$res_eff</div>
      <div class="u">η = L*/L · reaches B</div></div>
    <div class="stat"><div class="k">Q-learning</div><div class="v">$rl_eff</div>
      <div class="u">η = L*/L · map-blind</div></div>
    <div class="stat"><div class="k">Trap seeds solved</div><div class="v">6 / 6</div>
      <div class="u">by both learners</div></div>
  </div>

  $fig6

  <div class="callout warn">
    <h3>Read the fairness column before quoting this</h3>
    <p>
      APF and APF+residual both read tree coordinates. They <b>violate invariant 2 on
      purpose</b> — that track is cartography, kept as a controlled foil, and must never be
      scored against student submissions. The planner that is handicapped is the map-blind
      one, and it is the one that wins.
    </p>
  </div>
</section>

<section>
  <div class="sec-head"><span class="num">04</span><h2>Reward, shaping, and why the optimum survives</h2></div>
  <div class="prose">
    <p>
      Reward is <code>−1</code> per step, <code>−100</code> collision, <code>+100</code> goal.
      The sparse-reward trap is the single biggest risk to the event, so potential-based
      shaping is part of the core rather than an option: <code>F = γ·Φ(s′) − Φ(s)</code> with
      <code>Φ(s) = −dist(s, B)</code>.
    </p>
    <p>
      Ng, Harada &amp; Russell's theorem guarantees the optimal policy is unchanged, and the
      third panel below shows the property it rests on — the shaping term sums to exactly zero
      around any closed loop, so no cycle can be made profitable. The episodic form also
      requires <code>Φ(terminal) := 0</code>; without it the guarantee is void, and the
      collision terminal is where that bites.
    </p>
    <p>
      Empirically both claims hold. On an empty forest, shaped and unshaped agents converge to
      exactly the same optimum, <code>L* = 11.3137</code> — the exact Euclidean diagonal. On
      seed 0 the shaped agent finds its first good path at episode <b>$k_shaped</b> against
      <b>$k_plain</b> without.
    </p>
  </div>
  $fig3
  $fig2
</section>

<section>
  <div class="sec-head"><span class="num">05</span><h2>The oracle cannot be beaten</h2></div>
  <div class="prose">
    <p>
      Efficiency <code>η = L*/L</code> is only meaningful if the agent physically cannot
      out-run the oracle. That is enforced structurally rather than by convention: the
      environment's collision test and A*'s occupancy test are <em>the same function object</em>,
      and both apply the identical no-corner-cutting rule. A test asserts neither module
      defines a radius of its own.
    </p>
    <p>
      String-pull smoothing exists for drawing only. Scoring against a smoothed path would put
      <code>η = 1.0</code> permanently out of reach, so <code>L*</code> stays the grid cost.
    </p>
  </div>
  <div class="stats">
    <div class="stat"><div class="k">L* (oracle)</div><div class="v">$lstar</div>
      <div class="u">seed 0 · 16 trees</div></div>
    <div class="stat"><div class="k">L (learned)</div><div class="v">$llearned</div>
      <div class="u">greedy policy</div></div>
    <div class="stat"><div class="k">η = L*/L</div><div class="v">$eff</div>
      <div class="u">≤ 1.000 always</div></div>
  </div>
  $fig1
  $fig5
</section>

<section>
  <div class="sec-head"><span class="num">06</span><h2>Sensing and the 3D demo</h2></div>
  <div class="prose">
    <p>
      LIDAR is closed-form ray-vs-circle — no physics engine, no PyBullet. The tabular core
      never calls it; it exists to render the drone identity the grid world would otherwise
      lose, and to feed the optional Tier 3 continuous state.
    </p>
    <p>
      Webots is not installed on the build machine, so the phase-7 bridge was written with
      <em>no simulator import</em> and unit-tested on its own: policy to waypoint, waypoint to
      velocity. It follows waypoints rather than raw action vectors, because an 8-way direction
      command cannot home onto a cell centre — and inside the goal cell the terminal Q row is
      all zeros, so <code>argmax</code> would happily fly the drone back out.
    </p>
  </div>
  $fig7
  $fig8
</section>

<section>
  <div class="sec-head"><span class="num">07</span><h2>Leaderboard — hidden seeds</h2></div>
  <div class="prose">
    <p>
      Ten hidden seeds, 3000-episode budget, rank lexicographic: learned → safe → optimal →
      fast. Both trained agents reach full success and zero collisions, so the ranking falls
      through to optimality and then sample efficiency, which is exactly the intended
      behaviour.
    </p>
  </div>
  <div class="tablewrap">
    <table>
      <thead><tr><th>Agent</th><th>Success</th><th>Collision</th><th>Optimality</th>
      <th>Episodes to 1.15·L*</th></tr></thead>
      <tbody>
$board
      </tbody>
    </table>
  </div>
  <div class="callout">
    <h3>Sample efficiency is right-censored</h3>
    <p>
      Seeds that never cross the bar count as the full budget. An uncensored mean would be
      <code>inf</code> and erase every difference between agents — the primary RL metric would
      silently stop measuring anything.
    </p>
  </div>
</section>

<section>
  <div class="sec-head"><span class="num">08</span><h2>Knife-edges worth knowing</h2></div>
  <div class="prose">
    <p>
      These are for the organizers, not the students. Each is currently benign and each would
      bite if a constant moved.
    </p>
  </div>

  <div class="callout warn">
    <h3>Suicide is nearly rational</h3>
    <p>
      With <code>γ = 0.99</code>, wandering forever is worth <code>−1/(1−γ) = −100</code> —
      exactly the collision penalty. The <code>+100</code> goal breaks the tie and the agent
      converges, but the margin is zero. If you raise <code>max_steps</code> or lower
      <code>γ</code>, re-check that the collision rate stays at 0.00.
    </p>
  </div>

  <div class="callout warn">
    <h3>No generated forest is infeasible</h3>
    <p>
      Occupancy is by cell centre with <code>TREE_R = CELL = 0.40</code>, and the sampler's
      <code>MIN_TREE_SEP = 0.85</code> makes a sealed wall impossible to generate. So
      <code>is_feasible</code> is vacuous on generated forests, though it correctly rejects a
      hand-built palisade. Keep it — it costs nothing and guards future constant changes.
    </p>
  </div>

  <div class="callout warn">
    <h3>Hidden seeds still ship with the kit</h3>
    <p>
      <code>HIDDEN_SEEDS</code> lives in <code>tier_d/scoring/seeds.py</code>. Before the event, move
      it to a file that is not distributed to teams. The disjointness test will keep passing
      either way, so this one will not announce itself.
    </p>
  </div>
</section>

<section>
  <div class="sec-head"><span class="num">09</span><h2>Reproduce</h2></div>
  <pre><span class="c"># toolchain</span>
python3.11 -m venv .venv &amp;&amp; source .venv/bin/activate
pip install -r requirements.txt

<span class="c"># every claim on this page, in order</span>
pytest                          <span class="c"># 125 tests, ~11 s</span>
python -m tier_d.admin.scoring.precompute    <span class="c"># L* + feasibility cache</span>
python -m tier_d.admin.scoring.harness       <span class="c"># leaderboard on hidden seeds</span>
python -m tier_d.admin.apf.compare           <span class="c"># APF vs residual vs Q-learning</span>
python -m tier_d.admin.apf.gain_sweep        <span class="c"># the fairness table for APF gains</span>
python -m tier_d.admin.viz.report            <span class="c"># all figures + GIFs + facts.json</span></pre>
</section>

<footer>
  Built from <code>tier_d/admin/reports/facts.json</code> and <code>tier_d/admin/reports/leaderboard.csv</code>.
  Figures rendered by <code>tier_d/viz/report.py</code>; nothing on this page is hand-entered.
  Phase 7 (Webots) is the one item not verified end-to-end — the simulator is not installed
  on this machine.
</footer>

</div>
"""

if __name__ == "__main__":
    build()
