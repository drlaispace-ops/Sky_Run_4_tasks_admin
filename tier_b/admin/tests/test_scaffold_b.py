"""Scaffold-тести tier_b (numpy-only, без conda).

Гарантії:
  (a) діри в scaffold/student_rewards.py НЕ заповнені (роздача чиста);
  (b) заповнений make_solution-ом стаб чисельно збігається з референсом
      admin/rewards_solution.py на входах, що покривають усі гілки;
  (c) у scaffold/student_config.yaml TODO-ключі гіперпараметрів = null.
"""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

TIER_B = Path(__file__).resolve().parents[2]
STARTER = TIER_B / "scaffold" / "student_rewards.py"
STUDENT_CFG = TIER_B / "scaffold" / "student_config.yaml"


def _load_module(name: str, source: str):
    path = Path(__file__).parent / f"_{name}_gen.py"
    path.write_text(source)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        path.unlink(missing_ok=True)


def test_holes_are_unfilled():
    src = STARTER.read_text()
    assert src.count("# TODO(команда)") == 2, "scaffold must keep exactly two TODO holes"
    assert "raise NotImplementedError" in src

    student = _load_module("student_rewards_check", src)
    with pytest.raises(NotImplementedError):
        student.shaping(1.0, 0.0)
    with pytest.raises(NotImplementedError):
        student.step_reward(1.0, 0.0, 1.0, 0.0, 1.0, (0, 0, 0), False)


def test_filled_scaffold_matches_reference():
    from tier_b.admin.make_solution import build_solution_source
    from tier_b.admin import rewards_solution as ref

    filled = _load_module("solution_rewards_check", build_solution_source())

    rng = np.random.default_rng(7)
    flags = list(itertools.product([False, True], repeat=4))  # terminal/collided/gate/finished
    for terminal, collided, gate_passed, finished in flags:
        for _ in range(8):
            args = dict(
                phi_prog_new=float(rng.uniform(-12, 12)),
                phi_prog_old=float(rng.uniform(-12, 12)),
                phi_gate_new=float(rng.uniform(-6, 6)),
                phi_gate_old=float(rng.uniform(-6, 6)),
                z=float(rng.uniform(0.0, 2.0)),
                omega=rng.uniform(-6, 6, size=3),
                terminal=terminal,
                collided=collided,
                depth=float(rng.uniform(0.0, 0.1)),
                gate_passed=gate_passed,
                finished=finished,
            )
            assert filled.step_reward(**args) == pytest.approx(ref.step_reward(**args))
    # shaping alone, including the Φ(terminal)=0 clause
    assert filled.shaping(3.0, 1.0, 0.99, terminal=True) == pytest.approx(
        ref.shaping(3.0, 1.0, 0.99, terminal=True))
    assert filled.shaping(3.0, 1.0, 0.99, terminal=False) == pytest.approx(
        ref.shaping(3.0, 1.0, 0.99, terminal=False))


def test_student_config_has_null_todo_hparams():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(STUDENT_CFG.read_text())
    ref = yaml.safe_load((TIER_B / "config.yaml").read_text())

    null_expected = {
        "ppo": ["learning_rate", "n_steps", "batch_size", "ent_coef",
                "log_std_init", "net_arch", "total_steps"],
        "sac": ["learning_rate", "batch_size", "tau", "learning_starts",
                "net_arch", "total_steps"],
    }
    for algo, keys in null_expected.items():
        for k in keys:
            assert cfg[algo][k] is None, f"{algo}.{k} must be null (a TODO) in student_config.yaml"
    # non-hyperparameter sections must match the reference config exactly
    for section in ("sim", "action", "body", "lidar", "forest", "curriculum",
                    "reward", "jam", "waypoints", "bounds", "eval"):
        assert cfg[section] == ref[section], f"student_config section {section} drifted"


def test_public_hidden_seeds_disjoint():
    from tier_b.scoring.seeds import PUBLIC_SEEDS, HIDDEN_SEEDS
    assert len(PUBLIC_SEEDS) >= 8
    if HIDDEN_SEEDS:  # organizer copy only
        assert not set(PUBLIC_SEEDS) & set(HIDDEN_SEEDS)
        assert min(HIDDEN_SEEDS) >= 950_000
