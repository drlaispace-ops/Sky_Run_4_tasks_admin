# Послідовність запуску AI SkyRun

Що вже зроблено, що лишилось, і в якому порядку це робити.
Кожен крок закінчується перевіркою — не переходь далі, доки вона не пройшла.

---

## Стан на зараз

| Фаза | Статус | Перевірка |
|---|---|---|
| 0 Тулчейн | ✅ | `.venv`, `pytest` бачить тести |
| 1 Середовище (MDP) | ✅ | `tier_d/tests/test_env.py`, `test_forest.py` |
| 2 Оракул (L*, feasibility) | ✅ | `tier_d/tests/test_oracle.py` |
| 3 Еталонний Q-learning + shaping | ✅ | `tier_d/tests/test_qlearning.py` |
| 3b APF + residual (контрприклад) | ✅ | `tier_d/tests/test_apf.py` |
| 4 Візуалізатор V(s) | ✅ | `tier_d/reports/04_value_surface.png`, `.gif`, `lab.html` |
| 5 Scoring-harness | ✅ | `python -m tier_d.scoring.harness` |
| 6 Скаффолд для учнів | ✅ | `tier_d/tests/test_scaffold.py` |
| 7 Webots-міст | ⚠️ частково | логіка в тестах; симулятора немає |
| 8 Пакування / M4 dry-run | ❌ | — |

**125 тестів зелені за ~11 с.** Навчання 3000 епізодів — 0.2 с.

---

## A. Локальна перевірка (5 хв, робиться зараз)

```bash
cd "AI SkyRun"
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt

pytest                        # 1. 125 тестів мають бути зелені
python -m tier_d.scoring.precompute  # 2. кеш L* + feasibility
python -m tier_d.scoring.harness     # 3. табло на прихованих seed-ах
python -m tier_d.viz.report          # 4. усі фігури + GIF
python -m tier_d.apf.compare         # 5. APF vs residual vs Q-learning
```

Готово, коли: `pytest` зелений, у табло reference має `success 1.00`,
а random — `0.00`.

## B. Методичні матеріали (для команд)

Повинні бути в роздатці:
- `tier_d/scaffold/README.md` — теорія RL
- `tier_d/scaffold/GAMEPLAY_GUIDE.md` — діагностика, Tier 2–3 ідеї, чек-лист
- `tier_d/scaffold/starter.ipynb` — три TODO для заповнення
- `tier_d/scaffold/validate_team.py` — самоперевірка на PUBLIC_SEEDS

## B1. Демо-матеріали (за потреби, для організаторів)

```bash
python -m tier_d.viz.training_film --seed 0   # плівка навчання (mp4 + gif)
python -m tier_d.viz.export_runs              # навчити 10 карт -> runs.json
python -m tier_d.viz.build_lab                # інтерактивна лабораторія
python -m tier_d.viz.optimize_assets && python -m tier_d.viz.build_artifact
open tier_d/viz/field_scanner.html            # Tier 0, APF у браузері
```

---

## C. Що лишилось ДО події

> **Дивись також:** `ORGANIZERS_GUIDE.md` — 3-денна розписання змагання, рубрика журі, діагностика на дні подієї.

### C1. Дисципліна seed-ів · **обовʼязково, інваріант 5**

`HIDDEN_SEEDS` зараз лежить у `tier_d/scoring/seeds.py`, який їде разом з китом до команд.

- [ ] Винести `HIDDEN_SEEDS` у файл, якого немає в роздатці (напр. `tier_d/scoring/_hidden.py`,
      додати в `.gitignore` роздаткової гілки).
- [ ] Перегенерувати фінальні набори `PUBLIC_SEEDS` / `HIDDEN_SEEDS`.
- [ ] `python -m tier_d.scoring.precompute` — перерахувати кеш.
- [ ] Вручну переглянути прихований набір на баланс складності (`L*` у межах 11.3–13.0).
- [ ] Перевірити, що `is_feasible` не викинув жодного seed-а (зараз їх нуль — див. нижче).

> Тест на неперетин наборів пройде **в обох випадках**, тож цей крок сам про себе не нагадає.

### C2. Webots (фаза 7)

Симулятора на цій машині немає, тому міст написаний **без жодного імпорту Webots**
і протестований окремо (`tier_d/tests/test_bridge.py`, 118 тіків A→B).

- [ ] Встановити Webots на M4.
- [ ] Написати `.wbt` світ: ліс із того самого seed (`env.forest.make_forest`),
      квадрокоптер, сталa висота.
- [ ] Контролер: щотік викликає `webots.bridge.next_waypoint(Q, xy)`,
      віддає результат у ваш готовий PID.
- [ ] Перевірка: одна демо-політика долітає A→B на спільному seed.

Якщо Webots не встигаєте — **демо вже є**: `tier_d/reports/flight_3d.gif` і 3D-сцена
в `tier_d/viz/forest3d.py`. Це не блокер.

### C3. Пакування (фаза 8)

- [ ] **M4 dry-run end-to-end** за розділом A вище. Заміряти час.
- [ ] `pip freeze > requirements.lock` — **тільки після** успішного dry-run.
- [ ] Локальне дзеркало pip-пакетів (жодного `pip` з мережі під час події).
- [ ] Перевірити, що `node` є на M4 — інакше `tier_d/tests/test_js_parity.py` просто
      пропуститься (skip), і розбіжність JS↔Python не буде спіймана.
- [ ] Прогнати `pytest` повністю на M4.

### C4. Матеріали події

- [ ] Рубрика журі (30%) — критерії з `tier_d/scaffold/README.md` §8.
- [ ] Фінальне табло (70%) — `python -m tier_d.scoring.harness` + CSV.
- [ ] Кікоф-лекція: `tier_d/reports/03_reward_and_shaping.png`, `value_surface.gif`,
      `06_apf_vs_rl.png` (чому RL, а не APF).
- [ ] Tier 0 на перший вечір: `tier_d/viz/field_scanner.html`.
- [ ] Роздатка командам: `tier_d/scaffold/starter.ipynb` + `tier_d/scaffold/README.md` + `tier_d/env/`,
      `tier_d/oracle/`, `tier_d/viz/` — **без** `tier_d/agent/qlearning.py` (це відповідь) і **без** hidden seeds.

---

## D. Що НЕ треба роздавати командам

**Правило одне: роздатка = репозиторій мінус усі `*/admin/` теки (та, опційно,
`prompts/` — командам дається лише SYSTEM_PROMPT.md + картка їхнього треку).**

Усе чутливе вже фізично перенесено в `tier_<x>/admin/`:

| Було | Тепер | Чому не роздавати |
|---|---|---|
| `tier_d/agent/qlearning.py` | `tier_d/admin/agent/` | еталонне рішення — відповідь на TODO у ноутбуці |
| hidden-сіди в `scoring/seeds.py` | `tier_d/admin/scoring/hidden_seeds.py` | інваріант 5 (у роздатці `HIDDEN_SEEDS = ()`) |
| `tier_d/apf/` | `tier_d/admin/apf/` | читає координати дерев; спокуса «half-RL» (§8.6 концепту) |
| `tier_d/reports/` | `tier_d/admin/reports/` | містить розвʼязані карти й табло |
| `tier_d/tests/` | `tier_d/admin/tests/` | `test_scaffold.py` містить готові відповіді на TODO |

Після `rm -rf */admin` кіт лишається робочим: env/oracle/scaffold цілі,
`validate_team.py` дружньо просить заповнити `team_solution.py`,
`tier_b/rewards.py` прозоро перемикається на учнівський reward.

---

## E. Три речі, що зламають подію, якщо їх не перевірити

1. **Розріджена винагорода.** Shaping увімкнено за замовчуванням у скаффолді
   (`USE_SHAPING = True`). Якщо хтось його вимкне «щоб чесніше» — команда не навчиться
   і згорить. Це не читерство, це частина кору.

2. **Setup-фрикція.** Заморожені версії + локальне дзеркало (C3). Без цього перша
   година події піде на `pip`.

3. **Розбіжність радіусів.** Будь-яка зміна `TREE_R`, `CELL`, `GRID_N` у
   `tier_d/env/constants.py` **інвалідує `tier_d/scoring/cache_Lstar.json`**. Після будь-якої такої
   зміни — обовʼязково `python -m tier_d.scoring.precompute`, інакше `efficiency` рахується
   проти старого оракула.

---

## F. Відомі гострі кути (для організаторів)

**Самогубство майже раціональне.** При `γ = 0.99` нескінченне блукання коштує
`−1/(1−γ) = −100` — рівно стільки ж, скільки колізія. Розвʼязує нічию лише `+100`
за ціль. Якщо піднімете `max_steps` або знизите `γ` — переперевірте, що
`collision rate` лишився `0.00`.

**Жоден згенерований ліс не є непрохідним.** `TREE_R = CELL = 0.40`, а
`MIN_TREE_SEP = 0.85`, тож суцільну стіну сітка згенерувати не може. `is_feasible`
на згенерованих лісах завжди `True` (на рукотворному частоколі — коректно `False`).
Фільтр лишаємо: він нічого не коштує й ловить майбутні зміни констант.

**Seed 0 сходиться до `η = 0.857`** — трохи за межею порогу `1.15·L*`. Поріг
правильний: його долають 90% прихованих seed-ів, тому метрика розрізняє команди,
а не насичується.
