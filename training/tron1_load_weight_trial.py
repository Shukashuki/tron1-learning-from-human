"""Strict trial022: only the existing trial021 ground-load cost weight changes -1 to -2.

The worst-of-four native-substep sampler, exponential scale, axle reward,
reference, control and contact model are unchanged. Import is simulation-free.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from training import tron1_axle_mid_trial as previous

ROOT = Path(__file__).resolve().parents[1]
BASE_PROFILE_FILE = "config/car_drift_axle_mid_task.json"
BASE_PROFILE_SHA256 = "7b69b62dc1ea38e5adc4cea7aef96b56cd279943f02de6ee8d68159805fb3483"
BASE_HELPER_SHA256 = "d7c849556d6f38e9f56046c9913bb260a196b141db37ab1e516d38f2d4631bc4"
LOAD_SOURCE_SHA256 = "9e3cf953d1b5210e53530480610c1e338e7eeb65f1b009ad41271dc942f91363"
LOAD_KEY = "wheel_ground_load_reserve"
SETTINGS = {"weight": -2.0, "scale_fraction": .20}


def require(ok, message):
    if not ok:
        raise ValueError("Trial022 ground-load weight: " + message)


def baseline_axle_mid_profile():
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    require(digest(previous.__file__) == BASE_HELPER_SHA256
            and digest(ROOT / BASE_PROFILE_FILE) == BASE_PROFILE_SHA256,
            "Frozen021 source/profile changed")
    return previous.load_axle_mid_profile(ROOT / BASE_PROFILE_FILE)


def validate_load_weight_profile(profile):
    held = baseline_axle_mid_profile()
    dense = previous._dense()
    dense._json_value(profile)
    require(type(profile) is dict and set(profile) == set(held), "Exact021 fields required")
    require(dense._same_typed(profile[LOAD_KEY], SETTINGS), "Only weight-2.0/scale.20 supported")
    require(type(profile["description"]) is str and bool(profile["description"].strip()),
            "Nonempty description required")
    normalized = copy.deepcopy(profile)
    normalized[LOAD_KEY]["weight"] = held[LOAD_KEY]["weight"]
    normalized["description"] = held["description"]
    require(dense._same_typed(normalized, held), "No other021 value or type may change")
    return copy.deepcopy(profile)


def load_load_weight_profile(path):
    return validate_load_weight_profile(previous._dense()._read_json(path))


def held_sampler_cache_profile(profile):
    """Keep the frozen max-of-four sampler's cache metadata at its -1 baseline.

    The cache stores mg, sensors and a scale; it never multiplies by this weight.
    Only RewardManager uses the new -2 weight. This separation must be explicit
    because the frozen sampler rejects a cache initialized with any other label.
    """
    validate_load_weight_profile(profile)
    return baseline_axle_mid_profile()


def validate_load_weight_contract(contract, original_contract, profile):
    checked = validate_load_weight_profile(profile)
    require(type(contract) is dict and previous._dense()._same_typed(contract.get("task_profile"), checked),
            "Actual022 contract/profile mismatch")
    normalized = copy.deepcopy(contract)
    normalized["task_profile"] = baseline_axle_mid_profile()
    previous.validate_axle_mid_contract(normalized, original_contract, normalized["task_profile"])
    return copy.deepcopy(contract)


def configure_load_weight_trial(cfg, profile):
    """Install held021, then replace only the negative load-cost term weight."""
    checked = validate_load_weight_profile(profile)
    previous.configure_axle_mid_trial(cfg, baseline_axle_mid_profile())
    from training import tron1_worst_ground_load_substeps as worst
    require(hashlib.sha256(Path(worst.__file__).read_bytes()).hexdigest() == LOAD_SOURCE_SHA256,
            "Frozen worst-substep load-cost source changed")
    term = cfg.rewards.drift_wheel_ground_load_reserve
    require(term.func is worst.substep_worst_wheel_ground_load_reserve_penalty
            and previous._dense()._same_typed(term.weight, -1.0)
            and previous._dense()._same_typed(term.params, {"scale_fraction": .20}),
            "Held021 worst-substep function, weight and scale required")
    replacement = copy.copy(term)
    replacement.params = copy.deepcopy(term.params)
    replacement.weight = checked[LOAD_KEY]["weight"]
    cfg.rewards.drift_wheel_ground_load_reserve = replacement
    return cfg
