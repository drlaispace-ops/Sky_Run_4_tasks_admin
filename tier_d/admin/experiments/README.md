# Experiments

Controlled comparisons and ablations to understand RL behavior.

## Wind Comparison

**Purpose:** Show how stochasticity (wind gusts) affects learning dynamics.

**Setup:**
- `env.GridWorld` (deterministic, baseline)
- `env_wind.GridWorldWind` (5% action replacement per step)
- Same hyperparameters, same seed, identical training loop

**Run:**
```bash
python tier_d/experiments/compare_wind.py --seed 0 --episodes 3000 --wind-prob 0.05
```

**Output:**
- Side-by-side learning curves (deterministic vs wind)
- Statistics: success rate, path length, optimality
- Saved to `tier_d/reports/wind_comparison.png`

**Expected results:**
1. **Baseline** learns faster (deterministic transitions)
2. **Wind** converges slower (5% noise makes policy less certain)
3. **Both** eventually reach the goal, but wind's path is longer (robustness cost)
4. **Efficiency gap** shows the cost of uncertainty in RL

**Interpretation:**
- Without wind, agent memorizes the optimal path
- With wind, agent learns a **robust policy** that works despite disturbances
- This is closer to real robotics, where uncertainty is inevitable

## Other experiments (future)

- `LIDAR_comparison.py` — cell index vs 16-ray LIDAR as state
- `DQL_vision.py` — function approximation: pixels → actions
- `Curriculum.py` — start position closer to goal, then move away
