"""Pure CPU trial022 single-factor profile/contract/live-term checks."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from training import tron1_load_weight_trial as trial
from training import tron1_worst_ground_load_substeps as worst

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("load_weight_parent_contract", ROOT / "tests/test_ground_load_substeps_trial.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def profile():
    return trial.load_load_weight_profile(ROOT / "config/car_drift_load_weight_task.json")


def test_only_load_weight_changes_and_validation_does_not_mutate():
    current, held = profile(), trial.baseline_axle_mid_profile()
    before = copy.deepcopy(current)
    assert {key for key in current if current[key] != held[key]} == {"description", trial.LOAD_KEY}
    assert held[trial.LOAD_KEY] == {"weight": -1.0, "scale_fraction": .20}
    assert current["wheel_axle_orientation_tracking"] == held["wheel_axle_orientation_tracking"] == {"weight": 2.0, "std": .2}
    result = trial.validate_load_weight_profile(current)
    result[trial.LOAD_KEY]["weight"] = 0.0
    assert current == before
    with pytest.raises(ValueError): trial.previous.validate_axle_mid_profile(current)


def test_cache_uses_frozen_sampler_metadata_while_live_reward_changes():
    value = profile()
    before = copy.deepcopy(value)
    cached = trial.held_sampler_cache_profile(value)
    assert cached == trial.baseline_axle_mid_profile()
    assert cached[trial.LOAD_KEY] == {"weight": -1.0, "scale_fraction": .20}
    assert value[trial.LOAD_KEY] == {"weight": -2.0, "scale_fraction": .20}
    assert value == before and cached is not value
    cached[trial.LOAD_KEY]["weight"] = 0.0
    assert trial.baseline_axle_mid_profile()[trial.LOAD_KEY]["weight"] == -1.0


@pytest.mark.parametrize("key", sorted(set(profile()) - {"description", trial.LOAD_KEY}))
def test_other_profile_field_change_refused(key):
    value = profile(); value[key] = "changed"
    with pytest.raises(ValueError): trial.validate_load_weight_profile(value)


@pytest.mark.parametrize("settings", [None, {}, {"weight": -2},
    {"weight": -2, "scale_fraction": .20}, {"weight": True, "scale_fraction": .20},
    {"weight": -1.0, "scale_fraction": .20}, {"weight": -2.0, "scale_fraction": .25},
    {"weight": -2.0, "scale_fraction": float("nan")},
    {"weight": -2.0, "scale_fraction": .20, "other": False}])
def test_nonexact_settings_refused(settings):
    value = profile(); value[trial.LOAD_KEY] = settings
    with pytest.raises(ValueError): trial.validate_load_weight_profile(value)


def test_original_contract_and_nonmutation():
    value, original = profile(), fixtures.parent_contract()
    current = copy.deepcopy(original); current["task_profile"] = value
    before = copy.deepcopy(current)
    result = trial.validate_load_weight_contract(current, original, value)
    assert result == current == before and result is not current
    for field in set(original) - {"task_profile"}:
        wrong = copy.deepcopy(current); wrong[field] = "modified"
        with pytest.raises(ValueError): trial.validate_load_weight_contract(wrong, original, value)


def test_only_live_load_weight_replaced():
    load = SimpleNamespace(func=worst.substep_worst_wheel_ground_load_reserve_penalty,
                           weight=-1.0, params={"scale_fraction": .20})
    axle = SimpleNamespace(func=object(), weight=2.0, params={"std": .2})
    cfg = SimpleNamespace(rewards=SimpleNamespace(drift_wheel_ground_load_reserve=load,
                                                   drift_wheel_axle_orientation=axle), scene=object())
    with patch.object(trial.previous, "configure_axle_mid_trial", return_value=cfg) as held:
        assert trial.configure_load_weight_trial(cfg, profile()) is cfg
        assert held.call_args.args[1] == trial.baseline_axle_mid_profile()
    new = cfg.rewards.drift_wheel_ground_load_reserve
    assert new is not load and new.func is load.func and new.weight == -2.0
    assert type(new.weight) is float and load.weight == -1.0
    assert new.params == load.params and new.params is not load.params
    assert cfg.rewards.drift_wheel_axle_orientation is axle


@pytest.mark.parametrize("field,value", [("func", object()), ("weight", -2.0),
    ("weight", -1), ("params", {"scale_fraction": .25})])
def test_nonheld_live_load_term_refused(field, value):
    load = SimpleNamespace(func=worst.substep_worst_wheel_ground_load_reserve_penalty,
                           weight=-1.0, params={"scale_fraction": .20})
    setattr(load, field, value)
    cfg = SimpleNamespace(rewards=SimpleNamespace(drift_wheel_ground_load_reserve=load))
    with patch.object(trial.previous, "configure_axle_mid_trial", return_value=cfg):
        with pytest.raises(ValueError): trial.configure_load_weight_trial(cfg, profile())
