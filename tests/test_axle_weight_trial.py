"""CPU strict-profile/fake-config checks; not native physics or behavior."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from training import tron1_axle_weight_trial as trial
from training import tron1_tight_axle_tracking as tracking

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("weight_parent_contract", ROOT / "tests/test_ground_load_substeps_trial.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def profile():
    return trial.load_axle_weight_profile(ROOT / "config/car_drift_axle_weight_task.json")


def test_weight_only_and_nonmutating_validation():
    p, b = profile(), trial.baseline_tight_profile()
    before = copy.deepcopy(p)
    assert {k for k in p if p[k] != b[k]} == {"description", trial.AXLE_KEY}
    assert b[trial.AXLE_KEY] == {"weight": 1.0, "std": .2}
    result = trial.validate_axle_weight_profile(p)
    result[trial.AXLE_KEY]["weight"] = 1.0
    assert p == before
    assert p["wheel_ground_load_reserve_sampling"] == "all_substeps_max_cost"
    with pytest.raises(ValueError): trial.previous.validate_tight_axle_profile(p)


@pytest.mark.parametrize("key", sorted(set(profile()) - {"description", trial.AXLE_KEY}))
def test_held_field_change_rejected(key):
    p = profile(); p[key] = "modified"
    with pytest.raises(ValueError): trial.validate_axle_weight_profile(p)


@pytest.mark.parametrize("settings", [None, {}, {"std": .2}, {"weight": 4, "std": .2},
    {"weight": True, "std": .2}, {"weight": 1.0, "std": .2}, {"weight": 4.0, "std": .4},
    {"weight": 4.0, "std": float("nan")}, {"weight": 4.0, "std": .2, "spin": False}])
def test_invalid_settings_rejected(settings):
    p = profile(); p[trial.AXLE_KEY] = settings
    with pytest.raises(ValueError): trial.validate_axle_weight_profile(p)


def test_original_contract_invariants_and_nonmutation():
    p = profile(); original = fixtures.parent_contract()
    current = copy.deepcopy(original); current["task_profile"] = p
    before = copy.deepcopy(current)
    result = trial.validate_axle_weight_contract(current, original, p)
    assert result == current == before and result is not current
    for field in set(original) - {"task_profile"}:
        bad = copy.deepcopy(current); bad[field] = "modified"
        with pytest.raises(ValueError): trial.validate_axle_weight_contract(bad, original, p)


def test_only_live_weight_replaced_without_mutating_held_term():
    previous = SimpleNamespace(func=tracking.wheel_axle_orientation_tracking, weight=1.0,
                               params={"command_name": "motion", "std": .2})
    unrelated = SimpleNamespace(func=object(), weight=-5.0, params={"std": .02})
    cfg = SimpleNamespace(rewards=SimpleNamespace(drift_wheel_axle_orientation=previous,
        drift_wheel_vertical_velocity=unrelated), sim=object(), scene=object(), events=object())
    identities = cfg.sim, cfg.scene, cfg.events
    with patch.object(trial.previous, "configure_tight_axle_trial", return_value=cfg) as held:
        assert trial.configure_axle_weight_trial(cfg, profile()) is cfg
        assert held.call_count == 1
        assert held.call_args.args[1] == trial.baseline_tight_profile()
    current = cfg.rewards.drift_wheel_axle_orientation
    assert current is not previous and current.func is previous.func
    assert type(current.weight) is float and current.weight == 4.0
    assert current.params == previous.params == {"command_name": "motion", "std": .2}
    assert current.params is not previous.params and previous.weight == 1.0
    assert cfg.rewards.drift_wheel_vertical_velocity is unrelated
    assert (cfg.sim, cfg.scene, cfg.events) == identities


@pytest.mark.parametrize("field,value", [("func", object()), ("weight", 2.0),
    ("weight", 1), ("params", {"command_name": "motion", "std": .4})])
def test_bad_held_live_term_refused(field, value):
    previous = SimpleNamespace(func=tracking.wheel_axle_orientation_tracking, weight=1.0,
                               params={"command_name": "motion", "std": .2})
    setattr(previous, field, value)
    cfg = SimpleNamespace(rewards=SimpleNamespace(drift_wheel_axle_orientation=previous))
    with patch.object(trial.previous, "configure_tight_axle_trial", return_value=cfg):
        with pytest.raises(ValueError): trial.configure_axle_weight_trial(cfg, profile())
