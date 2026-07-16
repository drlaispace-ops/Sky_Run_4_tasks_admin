"""ОРГАНІЗАТОРСЬКИЙ РЕФЕРЕНС reward-ів — НЕ роздавати учасникам.

Reward-члени. PURE NUMPY, усі коефіцієнти приходять параметрами
(джерело — config.yaml; дефолти тут дзеркалять його для тестів без yaml).

Компроміси кожного члена (вимога ТЗ):
  • shaping (k_prog, k_gate): potential-based (γΦ′−Φ, Φ(terminal)=0 — інваріант 4
    кіта) — НЕ зсуває оптимум; сума по замкненому циклу = 0, тож «нафармити»
    прогрес кружлянням неможливо.
  • collision: base + масштаб глибиною проникнення — лобовий вліт без спроби
    маневру = повний штраф, «майже проскочив» = менший. Φ(terminal)=0 означає,
    що на краші агент ВТРАЧАЄ весь накопичений потенціал — банкувати shaping
    і вигідно розбиватися не можна.
  • step_cost: анти-зависання. Разом із jam-обривом БЕЗ бонуса робить
    «зависнути подалі від дерев» строго гіршим за будь-який проліт.
  • k_omega·‖ω‖²: проти тріпотіння; свідомо малий, щоб не задушити маневр
    (сплеск 6 рад/с на 0.3 с коштує ≈5 ≪ бонус воріт 25).
  • k_alt: підлога по висоті — м'який штраф, не терміналь (дає шанс вирівнятись).
"""

from __future__ import annotations

# --- дефолти = config.yaml["reward"] -----------------------------------------
GAMMA = 0.99          # γ алгоритму (PPO/SAC)
# виміряна корекція №2 (probe 300k): у формі γΦ′−Φ з γ<1 стояння далеко від
# цілі «заробляє» −(1−γ)Φ > 0 (+0.58/крок) — агент завис на 532-крокових
# епізодах із 0 воріт. Телескопічна форма Φ′−Φ дає за стояння рівно 0.
SHAPING_GAMMA = 1.0
K_PROG = 5.0
K_GATE = 2.0
# виміряна корекція (smoke 50k, див. config.yaml): base мусить перевищувати
# максимальний cash-out потенціалу (~+70), інакше «розбитись об перше дерево»
# стає локальним оптимумом і collision-rate РОСТЕ з тренуванням
COLLISION_BASE = -100.0
COLLISION_DEPTH_SCALE = -100.0
COLLISION_DEPTH_REF = 0.04
Z_MIN = 0.15
K_ALT = 10.0
K_OMEGA = 0.01
STEP_COST = -0.02
GATE_BONUS = 25.0
FINISH_BONUS = 100.0


def shaping(phi_new: float, phi_old: float, gamma: float = GAMMA,
            terminal: bool = False) -> float:
    """F = γ·Φ(s′) − Φ(s), і Φ(терминал) ≡ 0 — БЕЗ цього гарантія
    Ng–Harada–Russell не діє на терміналі колізії (інваріант 4)."""
    phi_new_eff = 0.0 if terminal else gamma * phi_new
    return phi_new_eff - phi_old


def collision_penalty(depth: float, base: float = COLLISION_BASE,
                      scale: float = COLLISION_DEPTH_SCALE,
                      ref: float = COLLISION_DEPTH_REF) -> float:
    """Градуйований урон: 0 < depth → base + scale·min(1, depth/ref) (обидва від'ємні)."""
    return base + scale * min(1.0, max(0.0, depth) / ref)


def altitude_penalty(z: float, z_min: float = Z_MIN, k_alt: float = K_ALT) -> float:
    return -k_alt * max(0.0, z_min - z)


def omega_penalty(omega, k_omega: float = K_OMEGA) -> float:
    ox, oy, oz = float(omega[0]), float(omega[1]), float(omega[2])
    return -k_omega * (ox * ox + oy * oy + oz * oz)


def step_reward(phi_prog_new: float, phi_prog_old: float,
                phi_gate_new: float, phi_gate_old: float,
                z: float, omega, terminal: bool,
                collided: bool = False, depth: float = 0.0,
                gate_passed: bool = False, finished: bool = False,
                gamma: float = SHAPING_GAMMA, k_prog: float = K_PROG,
                k_gate: float = K_GATE, step_cost: float = STEP_COST,
                gate_bonus: float = GATE_BONUS,
                finish_bonus: float = FINISH_BONUS, **kw) -> float:
    """Повна винагорода кроку. terminal=True лише для СПРАВЖНІХ терміналів
    (колізія/фініш) — jam і таймаут це truncation, там terminal=False
    і bootstrap зберігається (дисципліна terminal-vs-truncated кіта)."""
    r = step_cost
    r += k_prog * shaping(phi_prog_new, phi_prog_old, gamma, terminal)
    r += k_gate * shaping(phi_gate_new, phi_gate_old, gamma, terminal)
    r += altitude_penalty(z, kw.get("z_min", Z_MIN), kw.get("k_alt", K_ALT))
    r += omega_penalty(omega, kw.get("k_omega", K_OMEGA))
    if gate_passed:
        r += gate_bonus
    if collided:
        r += collision_penalty(depth, kw.get("collision_base", COLLISION_BASE),
                               kw.get("collision_depth_scale", COLLISION_DEPTH_SCALE),
                               kw.get("collision_depth_ref", COLLISION_DEPTH_REF))
    if finished:
        r += finish_bonus
    return float(r)
