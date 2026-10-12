"""Prospective trial017: isolate a spin-invariant wheel-axle tracking reward.

Relative to completed016, add only an explicit two-wheel axle-direction
objective using the SAME yaw-reanchored robot kinematic reference. Knee
orientation already contains this information; this is task reweighting, not
new mocap, a car-wheel teacher, or full wheel-quaternion tracking.

No Isaac/Torch import, application, sampler, reset or reward computation occurs
on import or profile/contract validation. Use the explicit configure function
after the frozen environment factory, never a historical generic entrypoint.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AXLE_KEY = "wheel_axle_orientation_tracking"
SETTINGS = {"weight": 1.0, "std": 0.4}
BASELINE016_PROFILE_FILE = "config/car_drift_dense_ground_load_task.json"
BASELINE016_PROFILE_SHA256 = "989215d55b548e245d6add928411d9e6fa16852a21f6863a777179014e3230b5"
DENSE_HELPER_SHA256 = "1917e3147783cddc47d8f8aeae366a46f224dd1b4091cc695a7327eded0b5b3f"
DENSE_SAMPLER_SHA256 = "4df069ea80e37cf7575346201687be934927434db45ea143f5ce797d4979ecd2"


def _require(condition, message):
    if not condition:
        raise ValueError("Trial017 wheel axle: " + message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dense():
    for name, expected in (("training/tron1_ground_load_substeps_trial.py", DENSE_HELPER_SHA256),
                           ("training/tron1_ground_load_substeps.py", DENSE_SAMPLER_SHA256),
                           (BASELINE016_PROFILE_FILE, BASELINE016_PROFILE_SHA256)):
        if _sha(ROOT / name) != expected:
            raise RuntimeError("Frozen baseline016 dependency changed: " + name)
    return importlib.import_module("training.tron1_ground_load_substeps_trial")


def baseline_dense_profile():
    """Return the exact source-pinned016 profile with its original metadata."""
    return _dense().load_dense_load_profile(ROOT / BASELINE016_PROFILE_FILE)


def validate_wheel_axle_profile(profile):
    dense = _dense()
    _require(type(profile) is dict, "Profile object required")
    dense._json_value(profile)
    baseline = baseline_dense_profile()
    _require(set(profile) == set(baseline) | {AXLE_KEY}, "Exact016 fields plus one axle objective required")
    _require(dense._same_typed(profile[AXLE_KEY], SETTINGS), "Only fixed weight1.0/std0.4 is supported")
    _require(type(profile["description"]) is str and bool(profile["description"].strip()),
             "Nonempty descriptive metadata required")
    normalized = copy.deepcopy(profile)
    normalized.pop(AXLE_KEY)
    normalized["description"] = baseline["description"]
    _require(dense._same_typed(normalized, baseline),
             "No016 reward, load sampling, gain, scale, material or clock field may change")
    return copy.deepcopy(profile)


def load_wheel_axle_profile(path):
    """Strict JSON/profile check, before any application or output directory."""
    dense = _dense()
    return validate_wheel_axle_profile(dense._read_json(path))


def validate_wheel_axle_contract(contract, original_contract, profile):
    """Only task_profile differs from a frozen004/013 nominal contract.

    Full wheel-body quaternion tracking remains false. The explicit axle-only
    reward is disclosed in task_profile; actor/critic/action fields do not change.
    Export/runtime extras require separate provenance checks.
    """
    dense = _dense()
    checked = validate_wheel_axle_profile(profile)
    _require(type(contract) is dict and type(original_contract) is dict, "Nominal contract objects required")
    dense._json_value(contract)
    dense._json_value(original_contract)
    _require(dense._same_typed(contract.get("task_profile"), checked), "Contract/profile mismatch")
    normalized = copy.deepcopy(contract)
    normalized["task_profile"] = baseline_dense_profile()
    dense.validate_dense_load_contract(normalized, original_contract, normalized["task_profile"])
    return copy.deepcopy(contract)


def configure_wheel_axle_trial(cfg, profile):
    """Enable held016 dense load cost, then add exactly one axle reward.

    Called after make_env_cfg with baseline_dense_profile(). No original
    source is modified or monkeypatched, and the same native sampler is used.
    Runtime reward-ledger, reset, device and performance checks remain required.
    """
    checked = validate_wheel_axle_profile(profile)
    dense = _dense()
    dense.configure_dense_ground_load(cfg, baseline_dense_profile())
    axle = importlib.import_module("training.tron1_wheel_axle_tracking")
    axle.configure_wheel_axle_tracking(cfg, checked[AXLE_KEY])
    return cfg
