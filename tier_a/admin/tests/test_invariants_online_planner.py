"""Invariant 2, adapted for the online LIDAR-based planner: the planner and
mapping code may consume LIDAR ranges (indirect sensing -- the same
legitimacy this repo already grants ``lidar3d.py``'s consumption by the
tabular/hybrid agents) but must never read ground-truth tree coordinates
directly. Mirrors ``tests/test_invariants_attitude.py``'s ``AttitudeBlindEnv``
+ source-regex guard pattern exactly, applied to the new planning modules.
"""

from __future__ import annotations

import inspect
import re

import pytest

from tier_a.env_attitude import mapping, online_planner
from tier_a.env_attitude.constants import N_TREES_ATTITUDE
from tier_a.env_attitude.env import AttitudeEnv


def test_planner_and_mapping_modules_never_reference_dot_trees():
    for mod in (mapping, online_planner):
        src = inspect.getsource(mod)
        assert not re.search(r"\.trees\b", src), f"{mod.__name__} references .trees"


def test_planner_functions_never_accept_a_trees_argument():
    fns = (
        online_planner.OnlinePlanner.maybe_replan,
        online_planner.OnlinePlanner._anchor_waypoints,
        online_planner.OnlinePlanner._path_invalidated,
        mapping.BeliefMap.update_from_scan,
    )
    for fn in fns:
        assert "trees" not in inspect.signature(fn).parameters, fn.__qualname__


class BlindPlannerEnv:
    """Exposes only what the online-planning stack is allowed to touch --
    no ``.trees``, ``.xyz``, ``.attitude``, ``.reference``."""

    _ALLOWED = {"n_actions", "obs_dim", "reset", "step", "lidar_scan", "online_lidar_planning"}

    def __init__(self, env: AttitudeEnv):
        object.__setattr__(self, "_env", env)

    def __getattr__(self, name):
        if name not in self._ALLOWED:
            raise AttributeError(
                f"INVARIANT 2 (online planner) violated: reached for env.{name!r}. "
                "Only the observation vector and LIDAR-derived belief are observable."
            )
        return getattr(object.__getattribute__(self, "_env"), name)


def test_online_planning_runs_end_to_end_without_ever_touching_ground_truth():
    real_env = AttitudeEnv(n_trees=N_TREES_ATTITUDE, seed=0, online_lidar_planning=True)
    blind = BlindPlannerEnv(real_env)
    blind.reset()
    for _ in range(300):
        _, _, done, _ = blind.step(13)
        if done:
            break

    with pytest.raises(AttributeError, match="INVARIANT 2"):
        _ = blind.trees
    with pytest.raises(AttributeError, match="INVARIANT 2"):
        _ = blind.xyz
    with pytest.raises(AttributeError, match="INVARIANT 2"):
        _ = blind.reference
