"""Strict trial021: only the existing019 axle reward weight changes 1 -> 2.

The directed-angle function, .2-rad width and LAST targets are unchanged.
Imports/profile/contract validation never start Isaac/Torch or compute a reward.
Baseline comparison uses a private copy, never a projected saved record.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from training import tron1_tight_axle_trial as previous

ROOT = Path(__file__).resolve().parents[1]
BASE_PROFILE_FILE = "config/car_drift_tight_axle_production_task.json"
BASE_PROFILE_SHA256 = "734e4f0cee7eef0aaec71fc413cdb31752043b8f10df1c854cdbee11ccfb9eb6"
BASE_HELPER_SHA256 = "aa3a652472f456590633433573b3868a9dd234ebce0734d3ffc12a9feb89ef0b"
AXLE_SOURCE_SHA256 = "b6e388e389762c8c3ad542fd1bcdbdeaf29a33e320badfd96c87f5d19f309520"
AXLE_KEY = "wheel_axle_orientation_tracking"
SETTINGS = {"weight": 2.0, "std": .2}


def require(condition, message):
    if not condition:
        raise ValueError("Trial021 axle weight: " + message)


def _dense():
    return previous.baseline.axle._dense()


def baseline_tight_profile():
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    require(digest(previous.__file__) == BASE_HELPER_SHA256
            and digest(ROOT / BASE_PROFILE_FILE) == BASE_PROFILE_SHA256,
            "Frozen019 source/profile changed")
    return previous.load_tight_axle_profile(ROOT / BASE_PROFILE_FILE)


def validate_axle_mid_profile(profile):
    held = baseline_tight_profile()
    _dense()._json_value(profile)
    require(type(profile) is dict and set(profile) == set(held), "Exact019 fields required")
    require(_dense()._same_typed(profile[AXLE_KEY], SETTINGS), "Only weight2.0/std.2 supported")
    require(type(profile["description"]) is str and bool(profile["description"].strip()),
            "Nonempty description required")
    normalized = copy.deepcopy(profile)
    normalized[AXLE_KEY]["weight"] = held[AXLE_KEY]["weight"]
    normalized["description"] = held["description"]
    require(_dense()._same_typed(normalized, held), "No other019 value or type may change")
    return copy.deepcopy(profile)


def load_axle_mid_profile(path):
    return validate_axle_mid_profile(_dense()._read_json(path))


def validate_axle_mid_contract(contract, original_contract, profile):
    checked = validate_axle_mid_profile(profile)
    require(type(contract) is dict and _dense()._same_typed(contract.get("task_profile"), checked),
            "Actual021 contract/profile mismatch")
    normalized = copy.deepcopy(contract)
    normalized["task_profile"] = baseline_tight_profile()
    previous.validate_tight_axle_contract(normalized, original_contract, normalized["task_profile"])
    return copy.deepcopy(contract)


def configure_axle_mid_trial(cfg, profile):
    """Install held019, then replace only the positive axle reward weight."""
    checked = validate_axle_mid_profile(profile)
    previous.configure_tight_axle_trial(cfg, baseline_tight_profile())
    from training import tron1_tight_axle_tracking as axle
    require(hashlib.sha256(Path(axle.__file__).read_bytes()).hexdigest() == AXLE_SOURCE_SHA256,
            "Frozen019 axle reward source changed")
    term = getattr(cfg.rewards, axle.TERM_NAME)
    require(term.func is axle.wheel_axle_orientation_tracking
            and _dense()._same_typed(term.weight, 1.0)
            and _dense()._same_typed(term.params, {"command_name": "motion", "std": .2}),
            "Held019 function, weight and params required")
    replacement = copy.copy(term)
    replacement.params = copy.deepcopy(term.params)
    replacement.weight = checked[AXLE_KEY]["weight"]
    setattr(cfg.rewards, axle.TERM_NAME, replacement)
    return cfg
