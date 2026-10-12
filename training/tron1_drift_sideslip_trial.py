"""Pure trial015 single-factor checks and explicit, read-only fixture readback.

This helper is not called by the production trainer. It changes no reward,
teacher, observation, action, physics, reset, optimizer or RNG state. The
configuration readback cannot prove reward execution or full-training values.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
from pathlib import Path

from training import tron1_drift as drift


REWARD_SOURCE_SHA256 = {
    "training/tron1_drift.py": "6d5f7974602adec9313a5e4feceb8e3caab392ca99351ac4a5f717e57fe1cf8a",
    "training/tron1_substep_rewards.py": "aeec31437226984a326a263109aaadebca004a6ef333f25a1d3377688aa276a5",
}
HELD_DRIFT_WEIGHTS = {"sideslip": 4., "yaw_rate": 1., "speed": 3., "grounded": .4, "tilt": -.2}
PROFILE_KEYS = frozenset({"schema_version", "task", "contact_friction", "env_spacing_m",
    "reward_weights", "speed_std_m_s", "action_rate_weight", "leg_kd", "mujoco_contact_solref",
    "leg_target_rate_weight", "wheel_vertical_velocity_weight", "wheel_vertical_velocity_std_m_s",
    "wheel_vertical_velocity_sampling", "description"})
LIVE_AUDIT_KEYS = frozenset({"schema_version", "status", "evidence_scope", "source_sha256",
    "active_reward_terms", "wheel_vz_capture_exception", "reward_compute_calls",
    "environment_reset_calls", "live_full_training_readback", "limitations"})


def _require(condition, message):
    if not condition:
        raise ValueError("Trial015 sideslip: " + message)


def _json_value(value, label="configuration"):
    """Strict JSON types; bool and int/float substitutions remain distinct."""
    if value is None or type(value) in (bool, str, int):
        return
    if type(value) is float:
        _require(math.isfinite(value), "Nonfinite " + label)
        return
    if type(value) is list:
        for item in value:
            _json_value(item, label)
        return
    if type(value) is dict:
        _require(all(type(key) is str for key in value), "String keys required: " + label)
        for item in value.values():
            _json_value(item, label)
        return
    raise ValueError("Trial015 sideslip: Unsupported JSON type in " + label)


def _same_typed(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(_same_typed(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(_same_typed(a, b) for a, b in zip(left, right))
    return left == right


def _profile(value, label):
    _require(type(value) is dict, label + " must be an object")
    _json_value(value, label)
    _require(set(value) == PROFILE_KEYS or set(value) == PROFILE_KEYS - {"wheel_vertical_velocity_sampling"},
             "Exact original/new profile fields required")
    out = drift.validate_task_profile(value)
    _require(type(out.get("description")) is str and bool(out["description"].strip()),
             label + " requires descriptive metadata")
    _require("wheel_ground_load_reserve" not in out, "No simultaneous reserve objective")
    return out


def _weight(profile, expected, label):
    value = profile["reward_weights"]["sideslip"]
    _require(type(value) in (int, float) and value == expected, label)


def normalize_sideslip_profile_for_sampling(profile):
    """Return a COPY for the unchanged trial014 sampling validator.

    Caller must separately validate against its sealed baseline/parent. Only
    sideslip is normalized, preserving all other typed values and description.
    This never mutates a module, caller profile or runtime reward configuration.
    """
    out = _profile(profile, "New profile")
    _weight(out, 4., "Only explicit sideslip weight4 may be normalized")
    _require(out.get("wheel_vertical_velocity_sampling") == "all_substeps_mean",
             "Held all-four sampling required")
    out["reward_weights"]["sideslip"] = 2.0
    return out


def validate_sideslip_profile(profile, baseline014):
    """Exactly weight2->4 and metadata description, with typed equality elsewhere."""
    out, baseline = _profile(profile, "New profile"), _profile(baseline014, "Baseline014")
    _weight(out, 4., "New sideslip weight must be4")
    _weight(baseline, 2., "Baseline014 sideslip weight must be2")
    _require(baseline.get("wheel_vertical_velocity_sampling") == "all_substeps_mean",
             "Baseline014 must retain all-four sampling")
    _require(set(out) == set(baseline), "No new or missing profile fields")
    compare = copy.deepcopy(out)
    compare["description"] = baseline["description"]
    compare["reward_weights"]["sideslip"] = copy.deepcopy(baseline["reward_weights"]["sideslip"])
    _require(_same_typed(compare, baseline), "Only reward_weights.sideslip and description may differ")
    return out


def validate_original_parent_profile(profile, parent004):
    """Allow held all-four sampling plus weight4/description relative to parent004."""
    out, parent = _profile(profile, "New profile"), _profile(parent004, "Original004")
    _weight(parent, 2., "Original004 sideslip weight must be2")
    _require("wheel_vertical_velocity_sampling" not in parent,
             "Original004 must have its original implicit last-substep profile")
    normalized = normalize_sideslip_profile_for_sampling(out)
    _require(set(normalized) == set(parent) | {"wheel_vertical_velocity_sampling"},
             "Only held sampling may add a parent profile field")
    normalized.pop("wheel_vertical_velocity_sampling")
    normalized["description"] = parent["description"]
    normalized["reward_weights"]["sideslip"] = copy.deepcopy(parent["reward_weights"]["sideslip"])
    _require(_same_typed(normalized, parent), "Original004 differs beyond held sampling/weight/description")
    return out


def validate_sideslip_contract(contract, original_contract, profile):
    """Exact typed nominal contract outside its explicit task_profile exception."""
    _require(type(contract) is dict and type(original_contract) is dict, "Contract objects required")
    _json_value(contract, "New contract")
    _json_value(original_contract, "Original contract")
    out = copy.deepcopy(contract)
    checked_profile = _profile(profile, "New profile")
    _require(set(out) == set(original_contract) and "task_profile" in original_contract,
             "Exact original contract keys required")
    original_profile = _profile(original_contract["task_profile"], "Original contract profile")
    if "wheel_vertical_velocity_sampling" in original_profile:
        validate_sideslip_profile(checked_profile, original_profile)
    else:
        validate_original_parent_profile(checked_profile, original_profile)
    _require(_same_typed(out.get("task_profile"), checked_profile), "Contract profile differs from validated profile")
    compare = copy.deepcopy(out)
    compare["task_profile"] = copy.deepcopy(original_contract["task_profile"])
    _require(_same_typed(compare, original_contract), "No nominal contract change outside task_profile")
    return out


def _function_name(function):
    _require(callable(function), "Reward function must be callable")
    if inspect.isfunction(function) or inspect.ismethod(function) or inspect.isclass(function):
        module, qualname = function.__module__, function.__qualname__
    else:
        module, qualname = type(function).__module__, type(function).__qualname__
    _require(type(module) is str and type(qualname) is str and module and qualname,
             "Reward callable must have a stable qualified identity")
    return module + "." + qualname


def _parameters(value):
    """Copy JSON parameters, preserving tuple/slice/SceneEntityCfg types.

    No repr, config method, scene resolution, tensor conversion or arbitrary
    property is invoked. Unknown objects fail closed rather than serializing a
    memory address. SceneEntityCfg stored fields use its public __dict__ only.
    """
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float:
        _require(math.isfinite(value), "Nonfinite live parameter")
        return value
    if type(value) is list:
        return [_parameters(x) for x in value]
    if type(value) is tuple:
        return {"__type__": "tuple", "items": [_parameters(x) for x in value]}
    if type(value) is slice:
        return {"__type__": "slice", "start": _parameters(value.start),
                "stop": _parameters(value.stop), "step": _parameters(value.step)}
    if type(value) is dict:
        _require(all(type(k) is str for k in value), "Live parameter keys must be strings")
        return {k: _parameters(v) for k, v in value.items()}
    cls = type(value)
    if cls.__name__ == "SceneEntityCfg" and cls.__module__ in (
        "isaaclab.managers.scene_entity_cfg", "isaaclab.managers"):
        fields = vars(value)
        _require(all(type(k) is str and not k.startswith("_") for k in fields),
                 "Only public stored SceneEntityCfg fields supported")
        return {"__config_type__": cls.__module__ + "." + cls.__qualname__,
                "fields": {k: _parameters(v) for k, v in fields.items()}}
    raise ValueError("Trial015 sideslip: Unsupported live reward parameter type")


def _source_readback():
    root = Path(__file__).resolve().parents[1]
    actual = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in REWARD_SOURCE_SHA256}
    _require(actual == REWARD_SOURCE_SHA256, "Original reward/sampler source changed")
    return actual


def validate_live_reward_audit(audit, *, wheel_vz_capture_expected=True):
    """Pure validation of snapshot metadata, not fresh runtime/execution proof."""
    _require(type(audit) is dict and set(audit) == LIVE_AUDIT_KEYS, "Exact live reward audit schema required")
    _json_value(audit, "Live reward audit")
    _require(type(wheel_vz_capture_expected) is bool, "Explicit boolean capture expectation required")
    _require(type(audit["schema_version"]) is int and audit["schema_version"] == 1
             and audit["status"] == "live_reward_configuration_verified"
             and audit["evidence_scope"] == "read_only_current_reward_configuration_not_execution_or_full_training",
             "Read-only reward configuration evidence scope required")
    _require(_same_typed(audit["source_sha256"], REWARD_SOURCE_SHA256), "Original reward/sampler source seals required")
    for key in ("reward_compute_calls", "environment_reset_calls"):
        _require(type(audit[key]) is int and audit[key] == 0, "No execution in reward configuration readback")
    _require(audit["live_full_training_readback"] is False, "No full-training live readback claim")
    _require(type(audit["limitations"]) is list and len(audit["limitations"]) > 0
             and all(type(item) is str and item for item in audit["limitations"]), "Explicit readback limitations required")
    terms = audit["active_reward_terms"]
    _require(type(terms) is list and bool(terms), "All active reward term metadata required")
    by_name = {}
    for term in terms:
        _require(type(term) is dict and set(term) == {"name", "function", "weight", "params"},
                 "Exact active reward term schema required")
        _require(type(term["name"]) is str and term["name"] and term["name"] not in by_name,
                 "Unique named active reward terms required")
        _require(type(term["function"]) is str and term["function"] and " " not in term["function"]
                 and "0x" not in term["function"] and "." in term["function"], "Stable qualified reward function required")
        _require(type(term["weight"]) in (int, float) and math.isfinite(term["weight"])
                 and type(term["params"]) is dict, "Finite numeric weight and copied params required")
        by_name[term["name"]] = term
    function_names = {"sideslip": "sideslip_tracking", "yaw_rate": "yaw_rate_tracking",
                      "speed": "speed_tracking", "grounded": "grounded_reward", "tilt": "tilt_penalty"}
    for name, weight in HELD_DRIFT_WEIGHTS.items():
        term = by_name.get("drift_" + name)
        _require(term is not None and term["function"] == "training.tron1_drift." + function_names[name]
                 and term["weight"] == weight, "Held named drift metadata differs: " + name)
        _require(_same_typed(term["params"], {"std": .1} if name == "speed" else {}),
                 "Held named drift parameter metadata differs: " + name)
    production_name = "training.tron1_substep_rewards.substep_wheel_vertical_velocity_penalty"
    vz = by_name.get("drift_wheel_vertical_velocity")
    _require(vz is not None and vz["weight"] == -5. and _same_typed(vz["params"], {"std_m_s": .02})
             and "drift_wheel_ground_load_reserve" not in by_name, "Held all-four cost only; no reserve objective")
    exception = audit["wheel_vz_capture_exception"]
    _require(type(exception) is dict and set(exception) == {"enabled", "reward_name", "configured_function",
        "production_function", "identity_matches_explicit_capture_function", "original_call_count_proved_by_configuration"},
        "Exact wheel-vz capture exception schema required")
    _require(exception["enabled"] is wheel_vz_capture_expected
             and exception["identity_matches_explicit_capture_function"] is wheel_vz_capture_expected
             and exception["reward_name"] == "drift_wheel_vertical_velocity"
             and exception["configured_function"] == vz["function"]
             and exception["production_function"] == production_name
             and exception["original_call_count_proved_by_configuration"] is False,
             "Explicit fixture capture identity and non-execution scope required")
    if wheel_vz_capture_expected:
        _require(vz["function"] != production_name
                 and vz["function"] not in {"training.tron1_drift." + value for value in function_names.values()},
                 "Capture identity cannot replace another named drift function")
    else:
        _require(vz["function"] == production_name, "Original production wheel-vz function metadata required")
    return copy.deepcopy(audit)


def live_reward_configuration(env, *, wheel_vz_capture_function=None):
    """Read active RewardManager configs exactly once; never compute any reward.

    The isolated fixture can explicitly pass its wheel-vz capture callable.
    Only that same-object wheel-vz wrapper is permitted as an exception; this
    proves configuration identity, NOT that the wrapper executed original once.
    No production trainer hook, state write or full-run readback is introduced.
    """
    sources = _source_readback()
    from training import tron1_substep_rewards as sampler
    manager = env.reward_manager
    names = manager.active_terms
    _require(type(names) in (list, tuple) and all(type(name) is str and name for name in names)
             and len(set(names)) == len(names), "Unique active reward names required")
    _require(callable(manager.get_term_cfg), "Actual get_term_cfg API required")
    configs, terms = {}, []
    for name in names:
        cfg = manager.get_term_cfg(name)
        _require(type(cfg.params) is dict, "Actual reward params object required: " + name)
        weight = cfg.weight
        _require(type(weight) in (int, float) and math.isfinite(weight), "Finite nonboolean reward weight: " + name)
        configs[name] = cfg
        terms.append({"name": name, "function": _function_name(cfg.func),
                      "weight": weight, "params": _parameters(cfg.params)})
    functions = {"sideslip": drift.sideslip_tracking, "yaw_rate": drift.yaw_rate_tracking,
                 "speed": drift.speed_tracking, "grounded": drift.grounded_reward, "tilt": drift.tilt_penalty}
    for name, expected_weight in HELD_DRIFT_WEIGHTS.items():
        cfg = configs.get("drift_" + name)
        _require(cfg is not None and cfg.func is functions[name] and cfg.weight == expected_weight,
                 "Held named drift function/weight differs: " + name)
        expected_params = {"std": .1} if name == "speed" else {}
        _require(_same_typed(cfg.params, expected_params), "Held named drift params differ: " + name)
    vz = configs.get("drift_wheel_vertical_velocity")
    _require(vz is not None and vz.weight == -5. and _same_typed(vz.params, {"std_m_s": .02}),
             "Held all-four wheel-vz weight/params required")
    production = sampler.substep_wheel_vertical_velocity_penalty
    captured = wheel_vz_capture_function is not None
    if captured:
        _require(callable(wheel_vz_capture_function) and wheel_vz_capture_function is not production
                 and wheel_vz_capture_function not in functions.values()
                 and vz.func is wheel_vz_capture_function, "Only explicit same-identity wheel-vz capture exception allowed")
    else:
        _require(vz.func is production, "Original production all-four wheel-vz function required")
    _require("drift_wheel_ground_load_reserve" not in configs, "No live simultaneous reserve objective")
    result = {
        "schema_version": 1, "status": "live_reward_configuration_verified",
        "evidence_scope": "read_only_current_reward_configuration_not_execution_or_full_training",
        "source_sha256": sources, "active_reward_terms": terms,
        "wheel_vz_capture_exception": {
            "enabled": captured, "reward_name": "drift_wheel_vertical_velocity",
            "configured_function": _function_name(vz.func), "production_function": _function_name(production),
            "identity_matches_explicit_capture_function": captured,
            "original_call_count_proved_by_configuration": False,
        },
        "reward_compute_calls": 0, "environment_reset_calls": 0, "live_full_training_readback": False,
        "limitations": [
            "Only current configuration is read; no reward evaluation, actor, optimizer, reset or RNG call is made.",
            "An explicit wheel-vz capture identity is fixture-only; its execution/original-call count requires separately pinned source and raw evidence.",
            "Fixture configuration/source readback is not independently archived full-training live weights or tensor values.",
            "Signed reference-beta matching does not enforce bearing contact, patch slip or continuous duration; original behavior gates remain necessary.",
            "The original full-quaternion reward frame and yaw-plane assessor convention remain different when tilted.",
        ],
    }
    json.dumps(result, allow_nan=False)
    return validate_live_reward_audit(result, wheel_vz_capture_expected=captured)
