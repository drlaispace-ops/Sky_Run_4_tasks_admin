#!/usr/bin/env python3
"""Regenerates tier_b/admin/solution_rewards.py from scaffold/student_rewards.py
by filling the two TODO holes with the intended answer — the same substitution
``admin/tests/test_scaffold_b.py`` uses to validate the scaffold, kept here so
the organizer-facing solution never drifts from what the tests consider correct.

    python tier_b/admin/make_solution.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).parent
STARTER = HERE.parent / "scaffold" / "student_rewards.py"
SOLUTION_PY = HERE / "solution_rewards.py"

SOLUTION = {
    '    # TODO(команда): реалізуйте потенціал-орієнтований shaping.\n'
    '    raise NotImplementedError("shaping: заповніть тіло (див. інваріант 4)")':
        '    phi_new_eff = 0.0 if terminal else gamma * phi_new\n'
        '    return phi_new_eff - phi_old',

    '    # TODO(команда): зберіть повну винагороду кроку з членів вище.\n'
    '    raise NotImplementedError("step_reward: зберіть винагороду з членів вище")':
        '    r = step_cost\n'
        '    r += k_prog * shaping(phi_prog_new, phi_prog_old, gamma, terminal)\n'
        '    r += k_gate * shaping(phi_gate_new, phi_gate_old, gamma, terminal)\n'
        '    r += altitude_penalty(z, kw.get("z_min", Z_MIN), kw.get("k_alt", K_ALT))\n'
        '    r += omega_penalty(omega, kw.get("k_omega", K_OMEGA))\n'
        '    if gate_passed:\n'
        '        r += gate_bonus\n'
        '    if collided:\n'
        '        r += collision_penalty(depth, kw.get("collision_base", COLLISION_BASE),\n'
        '                               kw.get("collision_depth_scale", COLLISION_DEPTH_SCALE),\n'
        '                               kw.get("collision_depth_ref", COLLISION_DEPTH_REF))\n'
        '    if finished:\n'
        '        r += finish_bonus\n'
        '    return float(r)',
}

HEADER = (
    "# ЗГЕНЕРОВАНО make_solution.py — заповнений student_rewards.py.\n"
    "# НЕ редагувати вручну; еталон звіряється тестами з rewards_solution.py.\n"
)


def build_solution_source() -> str:
    src = STARTER.read_text()
    for hole, answer in SOLUTION.items():
        if hole not in src:
            raise SystemExit(f"scaffold hole not found (drifted?):\n{hole}")
        src = src.replace(hole, answer)
    return HEADER + src


def main() -> None:
    SOLUTION_PY.write_text(build_solution_source())
    print(f"написано {SOLUTION_PY}")


if __name__ == "__main__":
    main()
