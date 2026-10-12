"""Strict, explicit trial016 force-cost quadrature configuration.

Relative to completed013, only the ground-load reserve sampling changes from
the last substep to the mean of four instantaneous *costs*.  Weight -1/r .20,
last-substep wheel-LINK-vz, sideslip2, controller, clocks and observations stay
fixed.  This module does not install a sampler or initialize the reserve cache.

Call ``configure_dense_ground_load`` AFTER the frozen environment factory and
BEFORE constructing the environment.  The historical factory, reserve helper,
trainer and evaluator are not patched.  Runtime readiness and behavior require
separate prospective smoke/provenance/assessment evidence.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLING_KEY = "wheel_ground_load_reserve_sampling"
SAMPLING_MODE = "all_substeps_mean_cost"
GROUND_LOAD_SUBSTEP_SOURCE_SHA256 = "4df069ea80e37cf7575346201687be934927434db45ea143f5ce797d4979ecd2"
BASELINE013_PROFILE_FILE = "config/car_drift_ground_load_task.json"
BASELINE013_PROFILE_SHA256 = "174a0543dfbf3c65bdd54867ab8e9b591047c8a572b0212eaaddbcd0573f2064"
PARENT004_PROFILE_FILE = "config/car_drift_contact_task.json"
PARENT004_PROFILE_SHA256 = "325cf279c699769694fb4934852ce85753b459e09670376b8c2d62d94f7d4e3c"
HISTORICAL_SOURCE_SHA256 = {
    "training/tron1_tracking.py": "0a29fa27c1d65c0b9add0886221491e549553e3529d6970f1ec5ba35e2532817",
    "training/tron1_drift.py": "6d5f7974602adec9313a5e4feceb8e3caab392ca99351ac4a5f717e57fe1cf8a",
    "training/tron1_domain_randomization.py": "3f3a0b62c85a0af496ae9e9e0d75dda85c3ec30da587e4a47969843f5b1b1dca",
    "training/tron1_contact_slip.py": "52ef38de9cece19a8ee02c120cb85d472ffd831e4587e5f0d1f340974ab699f4",
    "training/tron1_runtime_device.py": "2a1460d1b48e884fc5608e0365edc43b815d1fc74c8439945d4e2cc18027f775",
    "training/tron1_terrain.py": "dec819a3c3c526fce706fb6a5a6491388298884b32bfe73a5c331b166f976ebf",
    "training/tron1_ground_load_reserve.py": "761eccede3c0d90d744a4d9c45c9671b318426ab9bc6f9762ee7cacb3c14496d",
    "training/tron1_substep_rewards.py": "aeec31437226984a326a263109aaadebca004a6ef333f25a1d3377688aa276a5",
    "training/tron1_resume_lr.py": "596f1e2a455a4bba5644965985791aec9d4a934b5079932e6c6c4a208a1a3064",
    "scripts/train_tracking.py": "ea9e4cd14f25ff3ba5202b9ba43ac0f08320418e2311cbae205c8449f3a1da9c",
    "scripts/train_tracking_lr_sync.py": "9bd544b82e211633bb37c3fd8a658448d6d0e9b90f380f01a78dd9e8de02b8cb",
    "scripts/eval_tracking.py": "fd90089810d3d14bfd96210723afdd3a36113a19f667ec60b2df617dd032f992",
}
NOMINAL_CONTRACT_KEYS = frozenset({
    "policy_fps", "physics_dt", "decimation", "actor_dim", "critic_dim", "action_dim",
    "leg_joint_names", "wheel_joint_names", "observed_joint_velocity_order", "body_names",
    "orientation_body_names", "actor_terms", "raw_action_clip", "leg_position_scale",
    "leg_position_offset", "wheel_torque_scale_nm", "leg_kp", "leg_kd",
    "leg_torque_limit_nm", "wheel_torque_limit_nm", "actuator_model",
    "leg_saturation_effort_nm", "wheel_saturation_effort_nm", "leg_motor_velocity_limit_rad_s",
    "wheel_motor_velocity_limit_rad_s", "solver_joint_velocity_limit_rad_s",
    "motor_velocity_limit_semantics", "legacy_joint_friction_coefficient",
    "legacy_joint_friction_startup_readback_required", "rigid_body_max_angular_speed_rad_s",
    "rigid_body_angular_speed_config_units", "quaternion_order", "root_prescribed_during_steps",
    "wheel_orientation_tracking", "wheel_angle_observation", "reference_state_initialization_training_only",
    "training_full_clip_start_fraction", "deterministic_evaluation_start_frame",
    "motion_end_hidden_teleport", "domain_randomization", "actuator_limits_provenance",
    "hardware_ready", "task_profile",
})


def _require(condition, message):
    if not condition:
        raise ValueError("Trial016 dense ground load: " + message)


def _json_value(value):
    if value is None or type(value) in (bool, str, int):
        return
    if type(value) is float:
        _require(math.isfinite(value), "Finite JSON numbers required")
        return
    if type(value) is list:
        for item in value:
            _json_value(item)
        return
    if type(value) is dict:
        _require(all(type(key) is str for key in value), "Exact string object keys required")
        for item in value.values():
            _json_value(item)
        return
    raise ValueError("Trial016 dense ground load: Only exact JSON types are supported")


def _same_typed(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(_same_typed(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(_same_typed(a, b) for a, b in zip(left, right))
    return left == right


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _historical_sources():
    for name, expected in HISTORICAL_SOURCE_SHA256.items():
        if _sha256(ROOT / name) != expected:
            raise RuntimeError("Historical source changed: " + name)


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        _require(key not in value, "Duplicate JSON key: " + key)
        value[key] = item
    return value


def _invalid_constant(value):
    raise ValueError("Trial016 dense ground load: Nonfinite JSON constant: " + value)


def _read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique_object,
                      parse_constant=_invalid_constant)


def _pinned_profile(name, expected):
    _historical_sources()
    if _sha256(ROOT / name) != expected:
        raise RuntimeError("Historical profile changed: " + name)
    value = _read_json(ROOT / name)
    _json_value(value)
    return value


def validate_dense_load_profile(profile, baseline013):
    """Return a private copy; add only mean-cost mode and descriptive metadata."""
    _require(type(profile) is dict and type(baseline013) is dict, "Profile objects required")
    _json_value(profile)
    _json_value(baseline013)
    pinned = _pinned_profile(BASELINE013_PROFILE_FILE, BASELINE013_PROFILE_SHA256)
    _require(_same_typed(baseline013, pinned), "Baseline must be the exact frozen013 profile")
    _require(set(profile) == set(pinned) | {SAMPLING_KEY}, "Exact baseline fields plus one sampling key")
    _require(type(profile[SAMPLING_KEY]) is str and profile[SAMPLING_KEY] == SAMPLING_MODE,
             "Only explicit all_substeps_mean_cost is supported")
    _require(type(profile["description"]) is str and bool(profile["description"].strip()),
             "Nonempty descriptive metadata required")
    normalized = copy.deepcopy(profile)
    normalized.pop(SAMPLING_KEY)
    normalized["description"] = pinned["description"]
    _require(_same_typed(normalized, pinned),
             "Only reserve quadrature and description may change; no velocity/sideslip/weight/scale change")
    return copy.deepcopy(profile)


def load_dense_load_profile(path):
    """Read a prospective profile against byte-pinned013, without simulator imports."""
    baseline = _pinned_profile(BASELINE013_PROFILE_FILE, BASELINE013_PROFILE_SHA256)
    return validate_dense_load_profile(_read_json(path), baseline)


def validate_dense_load_contract(contract, original_contract, profile):
    """Typed nominal-only contract; caller separately seals its reference file.

    Export/runtime fields are deliberately not accepted here.  The producer
    checks those fields separately against the actual new checkpoint/export.
    """
    _require(type(contract) is dict and type(original_contract) is dict, "Nominal contract objects required")
    _json_value(contract)
    _json_value(original_contract)
    checked = validate_dense_load_profile(profile, _pinned_profile(
        BASELINE013_PROFILE_FILE, BASELINE013_PROFILE_SHA256))
    _require(set(contract) == set(original_contract) == NOMINAL_CONTRACT_KEYS,
             "Exact frozen nominal contract fields required; no export or telemetry aliases")
    base_profile = original_contract["task_profile"]
    pinned013 = _pinned_profile(BASELINE013_PROFILE_FILE, BASELINE013_PROFILE_SHA256)
    pinned004 = _pinned_profile(PARENT004_PROFILE_FILE, PARENT004_PROFILE_SHA256)
    _require(_same_typed(base_profile, pinned013) or _same_typed(base_profile, pinned004),
             "Reference profile must be exact completed013 or original004")
    _require(_same_typed(contract["task_profile"], checked), "Contract must carry the validated new profile")
    held = {"policy_fps": 50.0, "physics_dt": .005, "decimation": 4,
            "actor_dim": 51, "critic_dim": 120, "action_dim": 8,
            "leg_kp": 500.0, "leg_kd": 10.0, "leg_position_scale": 1.0,
            "leg_torque_limit_nm": 80.0, "wheel_torque_limit_nm": 12.0,
            "wheel_torque_scale_nm": 12.0, "raw_action_clip": [-1.0, 1.0],
            "actuator_model": "dc_motor", "hardware_ready": False,
            "root_prescribed_during_steps": False, "wheel_angle_observation": False,
            "wheel_orientation_tracking": False, "motion_end_hidden_teleport": False}
    _require(all(_same_typed(original_contract[k], v) for k, v in held.items()),
             "Original 500/10, 51/120/8, 200/50Hz and nominal control semantics required")
    normalized = copy.deepcopy(contract)
    normalized["task_profile"] = copy.deepcopy(base_profile)
    _require(_same_typed(normalized, original_contract), "No typed contract change outside task_profile")
    return copy.deepcopy(contract)


def _reward_term(rewards, name, function, weight, params):
    term = getattr(rewards, name, None)
    _require(term is not None and getattr(term, "func", None) is function,
             "Original reward function identity required: " + name)
    _require(_same_typed(getattr(term, "weight", None), weight)
             and _same_typed(getattr(term, "params", None), params),
             "Original typed weight/params required: " + name)
    return term


def configure_dense_ground_load(cfg, profile):
    """Replace exactly one private reward term after the frozen cfg factory.

    No cache initialization, environment construction, sampler installation or
    RewardManager computation occurs.  All other reward objects are untouched.
    """
    checked = validate_dense_load_profile(profile, _pinned_profile(
        BASELINE013_PROFILE_FILE, BASELINE013_PROFILE_SHA256))
    drift = importlib.import_module("training.tron1_drift")
    reserve = importlib.import_module("training.tron1_ground_load_reserve")
    rewards = getattr(cfg, "rewards", None)
    _require(rewards is not None, "Configured reward object required")
    functions = {"sideslip": drift.sideslip_tracking, "yaw_rate": drift.yaw_rate_tracking,
                 "speed": drift.speed_tracking, "grounded": drift.grounded_reward, "tilt": drift.tilt_penalty}
    for name, function in functions.items():
        params = {"std": checked["speed_std_m_s"]} if name == "speed" else {}
        _reward_term(rewards, "drift_" + name, function, checked["reward_weights"][name], params)
    _reward_term(rewards, "drift_wheel_vertical_velocity", drift.wheel_vertical_velocity_penalty,
                 checked["wheel_vertical_velocity_weight"], {"std": checked["wheel_vertical_velocity_std_m_s"]})
    _reward_term(rewards, "drift_leg_target_rate", drift.leg_target_rate_penalty,
                 checked["leg_target_rate_weight"], {})
    settings = checked["wheel_ground_load_reserve"]
    previous = _reward_term(rewards, "drift_wheel_ground_load_reserve", reserve.wheel_ground_load_reserve_penalty,
                            settings["weight"], {"scale_fraction": settings["scale_fraction"]})
    if _sha256(ROOT / "training/tron1_ground_load_substeps.py") != GROUND_LOAD_SUBSTEP_SOURCE_SHA256:
        raise RuntimeError("Explicit mean-cost sampler source changed")
    dense = importlib.import_module("training.tron1_ground_load_substeps")
    function = getattr(dense, "substep_wheel_ground_load_reserve_penalty", None)
    _require(callable(function) and getattr(dense, "MODE", None) == SAMPLING_MODE,
             "Explicit mean-cost reward function/module required")
    replacement = copy.copy(previous)
    replacement.func = function
    replacement.params = copy.deepcopy(previous.params)
    rewards.drift_wheel_ground_load_reserve = replacement
    return cfg
