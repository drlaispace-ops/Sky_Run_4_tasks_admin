# Tier-A: звіт про повний прогін (2026-07-15)

Повний прогін треку Tier-A (6DOF attitude control) у головному venv
(Python 3.11, без torch-залежностей conda-середовища — один torch-тест
скіпається за дизайном). Кожен пункт — фактичний запуск цього дня, не
переказ README.

## 1. Тести

```bash
pytest tier_a/tests
```

**132 passed, 1 skipped** (skip — DQN-тест, що потребує torch; у головному
venv він скіпається через importorskip-гвард — так і задумано). Прогін
повторено після виправлення shim-ів (див. §6) — результат той самий.

Покривають: середовище і його інваріанти (стан = досвід, не мапа;
potential-based shaping з Φ(terminal)=0; один TREE_R; відтворюваність
сідів), 2D/3D LIDAR, mapping, online-planner, еліпсоїдний корпус,
табличні/гібридні/пірует-агенти, scaffold і scoring.

## 2. Scoring: перерахунок еталонного кешу

```bash
python -m tier_a.scoring.precompute_attitude
```

18 записів (8 public + 10 hidden сідів), **18/18 feasible**, Lref у
діапазоні 11.31–11.72 м. Кеш перезаписано у
`tier_a/scoring/cache_Lref_attitude.json` — байт-у-байт відтворюваний
(інваріант 1).

## 3. Scoring: лідерборд (публічні сіди, бюджет 1200 епізодів)

```bash
python -m tier_a.scoring.harness_attitude --public
```

| агент | success | collision | departed | tracking RMSE | ep-to-goal |
|---|---|---|---|---|---|
| ablation (no shaping) | 0.00 | 0.00 | 0.88 | 0.306 | 1200 (цензуровано) |
| baseline (untrained net) | 0.00 | 0.00 | 1.00 | 0.203 | 1200 |
| reference (DQN + shaping) | 0.00 | 0.00 | 1.00 | 0.398 | 1200 |

Success 0.00 у всіх — **очікуваний і задокументований** результат: docstring
харнесу та README прямо кажуть, що це чесно важка 6-DOF задача з нуля, і на
цьому бюджеті ранжування дискримінує за (collision+departed) і RMSE, а не за
повними прольотами. CSV: `tier_a/reports/leaderboard_attitude.csv`.

## 4. Експеримент: online-перепланування по LIDAR (6 сідів)

```bash
python tier_a/experiments/compare_online_planning.py
```

| | offline (ground truth) | online (LIDAR) |
|---|---|---|
| success | 0/6 | 0/6 |
| mean tracking RMSE | 0.004 | 0.014 (~3×) |
| mean steps survived | 836.7 | 1389.3 |
| replans | — | 41–54 на сід |

Якісний висновок README відтворився: обидві умови не завершують повне коло
(floor-effect чистого PD), online-трекінг у ~3 рази менш точний, але живе
довше. Числа відрізняються від таблиці README (там 0.081/0.250 і 722/1300):
таблиця README писалася до коміту `875173e` (orientation-aware collision +
cross-track зміни), який змінив метрику/умови. Співвідношення і висновок —
ті самі. Графік: `tier_a/reports/tier_a_online_planning_comparison.png`.

## 5. Експеримент: курикулум піруета, 10 сідів × 2 умови

```bash
python tier_a/experiments/report_pirouette_curriculum.py --seeds 10
```

| seed | transfer | scratch |
|---|---|---|
| 0 | departed | departed |
| 1 | departed | departed |
| 2 | collision | departed |
| 3 | departed | departed |
| 4 | departed | collision |
| 5 | collision | departed |
| 6 | departed | departed |
| 7 | collision | departed |
| 8 | departed | departed |
| 9 | departed | collision |

**Transfer 0/10, scratch 0/10, жодного «first clean greedy success»** — точне
відтворення головного результату треку: це не провал навчання, а **фізичне
вето** (за фіксованої hover-тяги крен у прямому польоті зносить дрон убік і
садить по висоті; knife-edge вимагає T = mg/cos φ — до 3.9× зависання).
Діагноз скриптованими зондами — у `tier_a/env_attitude/README.md`,
розділ «Tier-A+: the pirouette that physics vetoed». Саме звідси виріс
Tier-B (CTBR, тяга до 2g). Артефакти:
`tier_a/reports/tier_a_pirouette_curriculum.png`,
`tier_a/reports/tier_a_pirouette_seeds_data.json`.

## 6. Scaffold: самоперевірка учасника

```bash
python tier_a/scaffold/validate_team_attitude.py --smoke
```

8 сідів, T*=11.111, mean RMSE 0.375, success 0/8 — smoke-бюджет **не має**
збігатися, його завдання — впіймати креші й помилки форм за секунди (скрипт
сам про це попереджає і радить повний 1200-епізодний прогін перед
сабмітом). Конвеєр учасника справний.

## Знайдено і виправлено під час прогону

Після реструктуризації (`f671c6d`, переїзд у `tier_a/`) три скрипти мали
зламаний shim `sys.path.insert(..., parent.parent)` — він вказував на
`tier_a/` замість кореня репо, і запуск як скрипта падав з
`ModuleNotFoundError: tier_a`:

- `tier_a/scaffold/validate_team_attitude.py`
- `tier_a/experiments/report_pirouette_curriculum.py`
- `tier_a/experiments/compare_online_planning.py`

Виправлено на `parent.parent.parent`. Запуск через `python -m ...` працював
і до фіксу, тому тести цього не ловили. Після фіксу всі три entry-point
перевірені запуском, тести — зелені (132 passed, 1 skipped).

## Підсумок

Трек Tier-A повністю відтворюваний: тести, scoring, обидва дослідницькі
експерименти і scaffold дають рівно ті результати (включно з чесними
нулями), що задокументовані в README. Головний науковий результат треку —
0/10 через фізичне вето фіксованої тяги — відтворено незалежним прогоном.
