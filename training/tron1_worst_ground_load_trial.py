"""Prospective trial018: only mean-to-worst ground-load aggregation may change.

No original017 source/profile is modified. This configuration is not a runtime
fixture, trained policy or behavior result. Keep original004 resume and gates.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
from pathlib import Path

from training import tron1_wheel_axle_trial as axle

ROOT = Path(__file__).resolve().parents[1]
BASE_PROFILE_FILE = "config/car_drift_wheel_axle_task.json"
BASE_PROFILE_SHA256 = "1db17ea1bb8820396e37bdf7043b6852151fda2b500c69f5d9a7716283932f78"
BASE_HELPER_SHA256 = "92301d90d844c3b3db0aa82e336abf7823855cde2ec21b01b639924be36e8421"
SAMPLING_KEY = "wheel_ground_load_reserve_sampling"
SAMPLING_MODE = "all_substeps_max_cost"


def require(condition, message):
    if not condition:
        raise ValueError("Trial018 worst ground load: " + message)


def baseline_axle_profile():
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    require(digest(axle.__file__) == BASE_HELPER_SHA256
            and digest(ROOT / BASE_PROFILE_FILE) == BASE_PROFILE_SHA256,
            "Frozen017 baseline source/profile changed")
    return axle.load_wheel_axle_profile(ROOT / BASE_PROFILE_FILE)


def validate_worst_ground_load_profile(profile):
    baseline = baseline_axle_profile()
    axle._dense()._json_value(profile)
    require(type(profile) is dict and set(profile) == set(baseline), "Exact017 profile fields required")
    require(profile[SAMPLING_KEY] == SAMPLING_MODE and type(profile[SAMPLING_KEY]) is str,
            "Only max of four instantaneous two-wheel sums")
    require(type(profile["description"]) is str and bool(profile["description"].strip()), "Nonempty description")
    normalized = copy.deepcopy(profile)
    normalized[SAMPLING_KEY] = baseline[SAMPLING_KEY]
    normalized["description"] = baseline["description"]
    require(axle._dense()._same_typed(normalized, baseline), "No other017 scientific field may change")
    return copy.deepcopy(profile)


def load_worst_ground_load_profile(path):
    return validate_worst_ground_load_profile(axle._dense()._read_json(path))


def validate_worst_ground_load_contract(contract, original_contract, profile):
    checked = validate_worst_ground_load_profile(profile)
    require(type(contract) is dict and axle._dense()._same_typed(contract.get("task_profile"), checked),
            "Actual018 contract/profile mismatch")
    normalized = copy.deepcopy(contract)
    normalized["task_profile"] = baseline_axle_profile()
    axle.validate_wheel_axle_contract(normalized, original_contract, normalized["task_profile"])
    return copy.deepcopy(contract)


def configure_worst_ground_load_trial(cfg, profile):
    checked = validate_worst_ground_load_profile(profile)
    axle.configure_wheel_axle_trial(cfg, baseline_axle_profile())
    mean = importlib.import_module("training.tron1_ground_load_substeps")
    worst = importlib.import_module("training.tron1_worst_ground_load_substeps")
    previous = cfg.rewards.drift_wheel_ground_load_reserve
    require(previous.func is mean.substep_wheel_ground_load_reserve_penalty
        and axle._dense()._same_typed(previous.weight, checked["wheel_ground_load_reserve"]["weight"])
        and axle._dense()._same_typed(previous.params, {"scale_fraction": .20}), "Held017 original term required")
    replacement = copy.copy(previous)
    replacement.params = copy.deepcopy(previous.params)
    replacement.func = worst.substep_worst_wheel_ground_load_reserve_penalty
    cfg.rewards.drift_wheel_ground_load_reserve = replacement
    return cfg
