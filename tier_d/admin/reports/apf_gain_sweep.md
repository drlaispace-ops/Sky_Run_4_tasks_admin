# APF gain sweep

`apf_flight` over **120 feasible forests** (trees ∈ [8, 12, 16, 20], seeds 0–29).

The gains in `apf/field.py` are chosen from this table. A classical APF that
trapped on every seed would make the RL comparison meaningless, so the target
is a baseline that *mostly works* — and fails only where the method genuinely
has local minima. Note `collided` is 0 everywhere: a correct repulsive field
fails by stalling, never by flying into a tree.

| K_rep | rho_0 | D_star | success | trapped | collided |
|------:|------:|-------:|--------:|--------:|---------:|
| 0.5 | 0.6 | 2.0 | 0.57 | 0.42 | 0.00 |
| 0.5 | 0.6 | 4.0 | 0.66 | 0.34 | 0.00 |  ← **chosen**
| 0.5 | 0.8 | 2.0 | 0.51 | 0.49 | 0.00 |
| 0.5 | 0.8 | 4.0 | 0.59 | 0.41 | 0.00 |
| 0.5 | 1.2 | 2.0 | 0.57 | 0.42 | 0.00 |
| 0.5 | 1.2 | 4.0 | 0.62 | 0.38 | 0.00 |
| 1.0 | 0.6 | 2.0 | 0.51 | 0.49 | 0.00 |
| 1.0 | 0.6 | 4.0 | 0.55 | 0.45 | 0.00 |
| 1.0 | 0.8 | 2.0 | 0.45 | 0.55 | 0.00 |
| 1.0 | 0.8 | 4.0 | 0.49 | 0.51 | 0.00 |
| 1.0 | 1.2 | 2.0 | 0.44 | 0.56 | 0.00 |
| 1.0 | 1.2 | 4.0 | 0.55 | 0.45 | 0.00 |
| 2.0 | 0.6 | 2.0 | 0.46 | 0.54 | 0.00 |
| 2.0 | 0.6 | 4.0 | 0.49 | 0.51 | 0.00 |
| 2.0 | 0.8 | 2.0 | 0.36 | 0.64 | 0.00 |
| 2.0 | 0.8 | 4.0 | 0.45 | 0.55 | 0.00 |
| 2.0 | 1.2 | 2.0 | 0.38 | 0.62 | 0.00 |
| 2.0 | 1.2 | 4.0 | 0.44 | 0.56 | 0.00 |
| 6.0 | 0.6 | 2.0 | 0.34 | 0.66 | 0.00 |
| 6.0 | 0.6 | 4.0 | 0.40 | 0.60 | 0.00 |
| 6.0 | 0.8 | 2.0 | 0.29 | 0.71 | 0.00 |
| 6.0 | 0.8 | 4.0 | 0.33 | 0.68 | 0.00 |
| 6.0 | 1.2 | 2.0 | 0.21 | 0.79 | 0.00 |
| 6.0 | 1.2 | 4.0 | 0.30 | 0.70 | 0.00 |

**Chosen:** `K_REP=0.5`, `RHO_0=0.6`, `D_STAR=4.0` — solves 66% of forests, stalls on 34%, collides on 0%.
