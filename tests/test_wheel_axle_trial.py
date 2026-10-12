"""Pure prospective profile/contract checks; no Isaac, actor or integration."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from training import tron1_wheel_axle_trial as trial

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("axle_parent_contract_fixture", ROOT / "tests/test_ground_load_substeps_trial.py")
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)


def new_profile():
    return json.loads((ROOT / "config/car_drift_wheel_axle_task.json").read_text())


def new_contract():
    value = fixture_module.parent_contract()
    value["task_profile"] = new_profile()
    return value


def test_actual_profile_is_exact_one_added_objective():
    value = new_profile()
    before = copy.deepcopy(value)
    checked = trial.validate_wheel_axle_profile(value)
    assert checked == value and checked is not value and value == before
    assert trial.load_wheel_axle_profile(ROOT / "config/car_drift_wheel_axle_task.json") == value
    checked[trial.AXLE_KEY]["weight"] = 2.0
    assert value == before
    baseline = trial.baseline_dense_profile()
    removed = copy.deepcopy(value)
    removed.pop(trial.AXLE_KEY)
    removed["description"] = baseline["description"]
    assert removed == baseline


@pytest.mark.parametrize("key", sorted(set(new_profile()) - {"description", trial.AXLE_KEY}))
def test_each_baseline_field_cannot_be_removed(key):
    value = new_profile()
    value.pop(key)
    with pytest.raises(ValueError):
        trial.validate_wheel_axle_profile(value)


@pytest.mark.parametrize("key", ["wheel_vertical_velocity_sampling", "wheel_support_velocity", "terrain", "unknown"])
def test_no_second_factor_or_alias(key):
    value = new_profile()
    value[key] = "new"
    with pytest.raises(ValueError):
        trial.validate_wheel_axle_profile(value)


@pytest.mark.parametrize("settings", [None, {}, {"weight": 1, "std": .4}, {"weight": 1.0, "std": True},
                                     {"weight": 2.0, "std": .4}, {"weight": 1.0, "std": .1},
                                     {"weight": 1.0, "std": float("nan")}, {"weight": 1.0, "std": .4, "sampling": "all"}])
def test_no_unsealed_weight_scale_or_mode(settings):
    value = new_profile()
    value[trial.AXLE_KEY] = settings
    with pytest.raises(ValueError):
        trial.validate_wheel_axle_profile(value)


@pytest.mark.parametrize("description", [None, False, 1, "", "  "])
def test_description_must_be_nonempty_string(description):
    value = new_profile()
    value["description"] = description
    with pytest.raises(ValueError):
        trial.validate_wheel_axle_profile(value)


@pytest.mark.parametrize("mutation", ["sideslip", "load", "sampling", "gain", "friction", "vz"])
def test_baseline_science_is_held(mutation):
    value = new_profile()
    if mutation == "sideslip":
        value["reward_weights"]["sideslip"] = 4.0
    elif mutation == "load":
        value["wheel_ground_load_reserve"]["weight"] = -2.0
    elif mutation == "sampling":
        value["wheel_ground_load_reserve_sampling"] = "last_substep"
    elif mutation == "gain":
        value["leg_kd"] = 12.0
    elif mutation == "friction":
        value["contact_friction"] = .2
    else:
        value["wheel_vertical_velocity_weight"] = -4.0
    with pytest.raises(ValueError):
        trial.validate_wheel_axle_profile(value)


def test_contract_only_profile_changes_and_full_spin_tracking_stays_off():
    value, original = new_contract(), fixture_module.parent_contract()
    before, original_before = copy.deepcopy(value), copy.deepcopy(original)
    checked = trial.validate_wheel_axle_contract(value, original, new_profile())
    assert checked == value and checked is not value
    assert value == before and original == original_before
    assert checked["wheel_orientation_tracking"] is False
    assert checked["wheel_angle_observation"] is False
    assert (checked["actor_dim"], checked["critic_dim"], checked["action_dim"]) == (51, 120, 8)


@pytest.mark.parametrize("key", sorted(trial._dense().NOMINAL_CONTRACT_KEYS - {"task_profile"}))
def test_each_nonprofile_contract_field_immutable(key):
    value = new_contract()
    value[key] = "changed"
    with pytest.raises(ValueError):
        trial.validate_wheel_axle_contract(value, fixture_module.parent_contract(), new_profile())


def test_strict_json_duplicate_nonfinite_and_nonobject(tmp_path):
    path = tmp_path / "profile.json"
    for data in ('{"schema_version":1,"schema_version":1}', '{"x":NaN}', '[]'):
        path.write_text(data)
        with pytest.raises(ValueError):
            trial.load_wheel_axle_profile(path)


def test_frozen_dependency_guard_rejects_different_expected_sha(monkeypatch):
    monkeypatch.setattr(trial, "DENSE_HELPER_SHA256", "0" * 64)
    with pytest.raises(RuntimeError):
        trial.baseline_dense_profile()


def test_validation_module_has_no_eager_simulator_or_tensor_import():
    tree = ast.parse((ROOT / "training/tron1_wheel_axle_trial.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not name.name.startswith(("torch", "isaac", "omni")) for name in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not node.module.startswith(("torch", "isaac", "omni"))
