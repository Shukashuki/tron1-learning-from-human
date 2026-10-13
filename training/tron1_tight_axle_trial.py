"""Strict trial019 single-factor configuration, based on immutable trial018.

Only wheel_axle_orientation_tracking.std changes .4 -> .2 rad. Validation
normalizes a private copy for baseline comparison, never a saved result. It
does not launch Isaac/Torch, compute rewards, or alter historical modules.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from training import tron1_worst_ground_load_trial as baseline

ROOT = Path(__file__).resolve().parents[1]
BASE_PROFILE_FILE = "config/car_drift_worst_ground_load_task.json"
BASE_PROFILE_SHA256 = "ad76a604264c142761175cad3c887100874394fed041e3b0a25b6b32df41d026"
BASE_HELPER_SHA256 = "6fae1d687c1e4c4520831d0d8496c9779e48c4d0e48de83d79b3cb4ed2a5b7cf"
AXLE_KEY = "wheel_axle_orientation_tracking"
SETTINGS = {"weight": 1.0, "std": .2}


def require(condition, message):
    if not condition:
        raise ValueError("Trial019 tight axle: " + message)


def baseline_worst_profile():
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    require(digest(baseline.__file__) == BASE_HELPER_SHA256
            and digest(ROOT / BASE_PROFILE_FILE) == BASE_PROFILE_SHA256,
            "Frozen018 source/profile changed")
    return baseline.load_worst_ground_load_profile(ROOT / BASE_PROFILE_FILE)


def validate_tight_axle_profile(profile):
    held = baseline_worst_profile()
    dense = baseline.axle._dense()
    dense._json_value(profile)
    require(type(profile) is dict and set(profile) == set(held), "Exact018 fields required")
    require(dense._same_typed(profile[AXLE_KEY], SETTINGS), "Only weight1.0/std.2 supported")
    require(type(profile["description"]) is str and bool(profile["description"].strip()),
            "Nonempty description required")
    normalized = copy.deepcopy(profile)
    normalized[AXLE_KEY] = held[AXLE_KEY]
    normalized["description"] = held["description"]
    require(dense._same_typed(normalized, held), "No other018 scientific field may change")
    return copy.deepcopy(profile)


def load_tight_axle_profile(path):
    return validate_tight_axle_profile(baseline.axle._dense()._read_json(path))


def validate_tight_axle_contract(contract, original_contract, profile):
    checked = validate_tight_axle_profile(profile)
    require(type(contract) is dict
            and baseline.axle._dense()._same_typed(contract.get("task_profile"), checked),
            "Actual019 contract/profile mismatch")
    normalized = copy.deepcopy(contract)
    normalized["task_profile"] = baseline_worst_profile()
    baseline.validate_worst_ground_load_contract(normalized, original_contract, normalized["task_profile"])
    return copy.deepcopy(contract)


def configure_tight_axle_trial(cfg, profile):
    """Install held018, then replace only the axle callable/angular scale."""
    checked = validate_tight_axle_profile(profile)
    baseline.configure_worst_ground_load_trial(cfg, baseline_worst_profile())
    from training import tron1_wheel_axle_tracking as old
    from training import tron1_tight_axle_tracking as new
    new._guard()
    previous = getattr(cfg.rewards, old.TERM_NAME)
    require(previous.func is old.wheel_axle_orientation_tracking
            and baseline.axle._dense()._same_typed(previous.weight, 1.0)
            and baseline.axle._dense()._same_typed(previous.params, {"command_name": "motion", "std": .4}),
            "Held018 original axle term required")
    replacement = copy.copy(previous)
    replacement.params = copy.deepcopy(previous.params)
    replacement.params["std"] = checked[AXLE_KEY]["std"]
    replacement.func = new.wheel_axle_orientation_tracking
    setattr(cfg.rewards, old.TERM_NAME, replacement)
    return cfg
