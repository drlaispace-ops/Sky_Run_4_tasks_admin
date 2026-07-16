"""Приховані master-сіди фінального заліку tier_b. INVARIANT 5.

Цей файл живе в ``tier_b/admin/`` і НЕ потрапляє в учнівську роздачу.
Діапазон 950000+ не перетинається ні з тренувальними сідами (малі числа),
ні з PUBLIC_SEEDS (900000–900007).
"""

from __future__ import annotations

HIDDEN_SEEDS: tuple[int, ...] = tuple(range(950_000, 950_010))
