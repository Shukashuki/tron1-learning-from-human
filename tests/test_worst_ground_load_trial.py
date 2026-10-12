import copy
import importlib.util
from pathlib import Path

import pytest

from training import tron1_worst_ground_load_trial as trial

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("worst_parent_contract", ROOT / "tests/test_ground_load_substeps_trial.py")
fixtures = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)


def profile():
    return trial.load_worst_ground_load_profile(ROOT / "config/car_drift_worst_ground_load_task.json")


def test_only_aggregation_and_description_change():
    p = profile(); held = trial.baseline_axle_profile()
    differences = {key for key in p if p[key] != held[key]}
    assert differences == {"description", "wheel_ground_load_reserve_sampling"}
    assert p["wheel_ground_load_reserve"]["weight"] == -1.0
    assert p["wheel_axle_orientation_tracking"] == {"weight": 1.0, "std": .4}


@pytest.mark.parametrize("key", sorted(set(profile()) - {"description", trial.SAMPLING_KEY}))
def test_no_other_field_can_change(key):
    p = profile(); p[key] = "modified"
    with pytest.raises(ValueError): trial.validate_worst_ground_load_profile(p)


@pytest.mark.parametrize("mode", [None, True, "all_substeps_mean_cost", "sum_of_wheel_maxima", "last_substep"])
def test_wrong_aggregation_rejected(mode):
    p = profile(); p[trial.SAMPLING_KEY] = mode
    with pytest.raises(ValueError): trial.validate_worst_ground_load_profile(p)


def test_original_nominal_contract_held():
    p = profile(); original = fixtures.parent_contract()
    current = copy.deepcopy(original); current["task_profile"] = p
    assert trial.validate_worst_ground_load_contract(current, original, p) == current
    for field in set(original) - {"task_profile"}:
        bad = copy.deepcopy(current); bad[field] = "modified"
        with pytest.raises(ValueError): trial.validate_worst_ground_load_contract(bad, original, p)
