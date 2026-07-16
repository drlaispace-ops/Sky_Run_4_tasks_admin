# Tier-D: звіт про повний прогін (2026-07-15)

Повний прогін базового 2D-кіта у головному venv одразу після реструктуризації
(переїзд у `tier_d/`). Кожен пункт — фактичний запуск цього дня.

## 1. Тести

```bash
pytest tier_d/tests
```

**125 passed** — інваріанти, ліс/RNG (включно з JS-паритетом через node),
оракул, Q-learning, shaping, LIDAR, APF, scoring, scaffold, webots-міст.

## 2. Кеш L* та лідерборд

```bash
python -m tier_d.scoring.precompute
python -m tier_d.scoring.harness            # приховані сіди
python -m tier_d.scoring.harness --public   # самоперевірка
```

Приховані сіди (10, бюджет 3000 епізодів):

| агент | success | collision | optimality (L*/L) | епізодів до 1.15·L* |
|---|---|---|---|---|
| reference (Q-learning + shaping) | 1.00 | 0.00 | 0.937 | 1990 (90%) |
| ablation (no shaping) | 1.00 | 0.00 | 0.916 | 2492 (80%) |
| baseline (random) | 0.00 | 0.00 | 0.000 | 3000 (0%) |

Публічні сіди: reference 0.945 / 1781 еп. (88%), ablation 0.935 / 2362.
Висновок незмінний: shaping не змінює якість збіжної політики (інваріант 4),
але даруює ~20–25% sample efficiency.

## 3. APF-порівняння

```bash
python -m tier_d.apf.compare
```

Trap-сіди (16 дерев): [0, 3, 10, 11, 13, 14]. На всіх шести класичний APF
застряг (без жодної колізії — коректне відштовхувальне поле фейлиться
стоянням), а APF+residual і map-blind Q-learning пройшли **6/6 і 6/6**
(η 0.86–1.00). Гейни APF — з gain sweep по 120 лісах (success 0.66 /
trapped 0.34 / collided 0.00) — чесний суперник, не солом'яне опудало.

## 4. Вітряна абляція (6 сідів)

```bash
python tier_d/experiments/compare_wind.py --seeds 6
```

Обидві умови 6/6 успіхів; ціна стійкості вимірна:
середня η **0.938 (без вітру) проти 0.866 (вітер 5%)**.
Найгірший сід (4): 0.953 → 0.801.

## 5. Скаффолд

```bash
python tier_d/scaffold/validate_team.py
```

**8/8 публічних сідів, 100%** — еталонне рішення проходить конвеєр команди.

## 6. Фігури

```bash
python -m tier_d.viz.report --fast
```

Всі 8 фігур регенеровано у `tier_d/reports/` (01–08), плюс
`wind_comparison.png`. `facts.json` оновлено: seed 0 → L*=12.72, L=14.85,
η=0.857; shaped/plain до 1.15·L*: 2225/2400; navigation function —
один мінімум: true.

## Знайдено і виправлено під час прогону

`tier_d/experiments/compare_wind.py::plot_comparison` падав з
`TypeError: 'History' object is not subscriptable` — код панелей очікував
масив Nx2, а `train()` повертає об'єкт `History` (інтерфейс з часом змінився,
а скрипт не покривався тестами — числа він друкував правильно, ламався лише
графік). Додано `_curve_array()` (History → Nx2, inf→nan). Після фіксу всі
чотири панелі малюються, `wind_comparison.png` записано.

## Підсумок

Базовий кіт повністю відтворюваний після реструктуризації: 125 тестів,
лідерборд, обидва контрольовані експерименти і скаффолд дають задокументовані
результати. Один знайдений баг — у нетестованому шарі візуалізації
експерименту; ядро (env/oracle/agent/scoring) не мало жодного відхилення.
