"""Приховані seed-и фінального заліку. INVARIANT 5: ніколи не в тренуванні.

Цей файл живе в ``tier_d/admin/`` і НЕ потрапляє в учнівську роздачу.
``tier_d/scoring/seeds.py`` імпортує його через guarded import: без admin/
``HIDDEN_SEEDS`` порожній, і публічний кіт лишається робочим.
"""

from __future__ import annotations

HIDDEN_SEEDS: tuple[int, ...] = (101, 102, 103, 104, 105, 106, 107, 108, 109, 110)
