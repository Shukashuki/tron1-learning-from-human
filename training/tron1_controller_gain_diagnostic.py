"""Pure opt-in zero-shot common-leg-Kp deployment diagnostic provenance.

Never changes a manifest, simulator, policy, tensor, observation, or training.
The caller must first perform ordinary original-profile/manifest checks, then
explicitly request this bounded diagnostic. Export-only metadata may be added
later; original nominal controller fields cannot be silently overridden.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re

from training.tron1_drift import validate_task_profile
from training.tron1_physics_clock import validate_physics_clock

SCOPE = "zero_shot_common_leg_kp"
EVAL_NOMINAL_DOMAIN = "disabled during evaluation; fixed actuator parameters"
NOMINAL_DOMAINS = {"disabled for initial pilot", "disabled; nominal actuator parameters", EVAL_NOMINAL_DOMAIN}
EXPORT_METADATA_KEYS = frozenset({
    "actor_file", "actor_includes_observation_normalizer", "actor_output_is_unclipped",
    "actor_observation_groups", "export_validation_max_abs_error", "actual_joint_order",
    "actual_body_order", "reference_file_sha256", "checkpoint_sha256",
    "default_joint_position_rad", "training_domain_randomization",
})
CONTROLLER_KEYS = (
    "policy_fps", "physics_dt", "decimation", "actor_dim", "critic_dim", "action_dim",
    "leg_joint_names", "wheel_joint_names", "observed_joint_velocity_order", "body_names",
    "orientation_body_names", "actor_terms", "raw_action_clip", "leg_position_scale",
    "leg_position_offset", "wheel_torque_scale_nm", "leg_kp", "leg_kd", "leg_torque_limit_nm",
    "wheel_torque_limit_nm", "actuator_model", "leg_saturation_effort_nm", "wheel_saturation_effort_nm",
    "leg_motor_velocity_limit_rad_s", "wheel_motor_velocity_limit_rad_s", "solver_joint_velocity_limit_rad_s",
    "motor_velocity_limit_semantics", "legacy_joint_friction_coefficient",
    "legacy_joint_friction_startup_readback_required", "rigid_body_max_angular_speed_rad_s",
    "rigid_body_angular_speed_config_units", "quaternion_order", "root_prescribed_during_steps",
    "wheel_orientation_tracking", "wheel_angle_observation", "reference_state_initialization_training_only",
    "deterministic_evaluation_start_frame", "motion_end_hidden_teleport",
)


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _gain(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 100. <= value <= 500. or not math.isfinite(value):
        raise ValueError("Deployment leg_kp must be finite, nonboolean, and within[100,500]")
    return value


def _nominal_only(contract):
    if not isinstance(contract, dict):
        raise ValueError("Original nominal deployment contract required")
    if any(key in contract for key in ("physics_clock_refinement", "domain_parameters", "controller_gain_diagnostic")):
        raise ValueError("Gain diagnostic cannot combine domain draws, clock refinement, or nested diagnostics")
    domain = contract.get("domain_randomization")
    if not isinstance(domain, str) or domain not in NOMINAL_DOMAINS:
        raise ValueError("Gain diagnostic requires ordinary fixed nominal actuators")


def _controller(contract):
    if not isinstance(contract, dict) or any(key not in contract for key in CONTROLLER_KEYS):
        raise ValueError("Complete original controller contract required")
    for key, expected in (("actor_dim", 51), ("critic_dim", 120), ("action_dim", 8)):
        if type(contract[key]) is not int or contract[key] != expected:
            raise ValueError("Diagnostic requires the original51/120/8 interface")
    validate_physics_clock(contract)
    for key in ("leg_kp", "leg_kd", "leg_position_scale", "wheel_torque_scale_nm", "leg_torque_limit_nm",
                "wheel_torque_limit_nm", "leg_saturation_effort_nm", "wheel_saturation_effort_nm",
                "leg_motor_velocity_limit_rad_s", "wheel_motor_velocity_limit_rad_s", "solver_joint_velocity_limit_rad_s"):
        value = contract[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0. < value < 1e8 or not math.isfinite(value):
            raise ValueError(f"Invalid original controller {key}")
    if contract["leg_position_offset"] != "reference_frame_0_joint_pos":
        raise ValueError("Original reference-frame target offset required")
    if contract["actuator_model"] != "dc_motor":
        raise ValueError("Diagnostic requires the original explicit DC-motor controller")


def prepare_deployment_gain_diagnostic(original_profile, nominal_contract, training_manifest,
                                      deployment_leg_kp, *, reference_sha256):
    """Return effective profile and unsealed audit; never mutate original inputs.

    Root must insert original_manifest_sha256/checkpoint_sha256 before applying
    and validating stored provenance. No checkpoint is loaded by this helper.
    """
    profile = validate_task_profile(original_profile)
    gain = _gain(deployment_leg_kp)
    reference_sha256 = _sha(reference_sha256, "reference_sha256")
    if not isinstance(training_manifest, dict) or not isinstance(training_manifest.get("contract"), dict):
        raise ValueError("Original training manifest with controller contract required")
    trained = training_manifest["contract"]
    _controller(trained)
    _controller(nominal_contract)
    _nominal_only(nominal_contract)
    if "controller_gain_diagnostic" in trained or "physics_clock_refinement" in trained:
        raise ValueError("Original unmodified training contract required")
    if (training_manifest.get("task_profile") != profile or trained.get("task_profile") != profile
            or nominal_contract.get("task_profile") != profile):
        raise ValueError("Original task/material/reward profile must exactly match training")
    if training_manifest.get("motion_sha256") != reference_sha256:
        raise ValueError("Reference differs from the original training manifest")
    # The minimum controller schema is necessary but not exhaustive: factory
    # fields such as full-clip sampling and hardware readiness stay unchanged.
    # Training DR is deliberately separate from ordinary nominal evaluation.
    for key, value in nominal_contract.items():
        if key != "domain_randomization" and (key not in trained or value != trained[key]):
            raise ValueError(f"Original nominal controller differs from training: {key}")
    old_kp, kd = profile.get("leg_kp", 500.), profile.get("leg_kd", 10.)
    if nominal_contract["leg_kp"] != old_kp or nominal_contract["leg_kd"] != kd:
        raise ValueError("Original profile/controller leg gain mismatch")
    effective = copy.deepcopy(profile)
    effective["leg_kp"] = gain
    effective = validate_task_profile(effective)
    audit = {
        "schema_version": 1, "scope": SCOPE, "additional_training_updates": 0,
        "original_training_task_profile": copy.deepcopy(profile),
        "original_training_contract": copy.deepcopy(trained),
        "original_nominal_contract": copy.deepcopy(nominal_contract),
        "trained_leg_kp": old_kp, "deployment_leg_kp": gain, "unchanged_leg_kd": kd,
        "reference_sha256": reference_sha256,
        "original_training_contract_sha256": _canonical_sha(trained),
        "original_training_task_profile_sha256": _canonical_sha(profile),
        "effective_profile_sha256": _canonical_sha(effective),
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_at_deployment_gain_claimed": False, "hardware_ready": False,
        "statistical_robustness_claimed": False,
    }
    return effective, audit


def apply_deployment_gain_diagnostic_to_contract(nominal_contract, audit):
    """Deep-copy effective controller; sealed file hashes are supplied by root."""
    if not isinstance(audit, dict) or audit.get("scope") != SCOPE or type(audit.get("schema_version")) is not int or audit["schema_version"] != 1:
        raise ValueError("Explicit controller gain diagnostic audit required")
    if nominal_contract != audit.get("original_nominal_contract"):
        raise ValueError("Apply only to the inspected original nominal contract")
    _nominal_only(nominal_contract)
    _sha(audit.get("original_manifest_sha256"), "original_manifest_sha256")
    _sha(audit.get("checkpoint_sha256"), "checkpoint_sha256")
    out = copy.deepcopy(nominal_contract)
    out["leg_kp"] = _gain(audit.get("deployment_leg_kp"))
    out["task_profile"] = copy.deepcopy(audit["original_training_task_profile"])
    out["task_profile"]["leg_kp"] = out["leg_kp"]
    out["task_profile"] = validate_task_profile(out["task_profile"])
    if _canonical_sha(out["task_profile"]) != audit.get("effective_profile_sha256"):
        raise ValueError("Effective diagnostic profile hash differs")
    out["controller_gain_diagnostic"] = copy.deepcopy(audit)
    return out


def validate_deployment_gain_diagnostic(effective_contract, training_manifest, *,
                                      original_manifest_sha256, checkpoint_sha256):
    """Strict reusable stored-provenance validation, including enriched exports.

    Known ordinary export metadata is not controller state. Every original
    nominal key remains fixed except leg_kp, effective profile and the known
    nominal-evaluation domain string. Draws/refinement cannot be combined.
    """
    if not isinstance(effective_contract, dict) or not isinstance(effective_contract.get("controller_gain_diagnostic"), dict):
        raise ValueError("Stored explicit controller gain diagnostic required")
    audit = effective_contract["controller_gain_diagnostic"]
    if type(audit.get("schema_version")) is not int or audit["schema_version"] != 1:
        raise ValueError("Invalid diagnostic schema")
    if type(audit.get("additional_training_updates")) is not int or audit["additional_training_updates"] != 0:
        raise ValueError("Zero-shot diagnostic must have exactly zero additional training updates")
    for field, actual in (("original_manifest_sha256", original_manifest_sha256), ("checkpoint_sha256", checkpoint_sha256)):
        if _sha(audit.get(field), field) != _sha(actual, field):
            raise ValueError(f"Diagnostic {field} differs from actual input")
    profile, expected_audit = prepare_deployment_gain_diagnostic(
        audit.get("original_training_task_profile"), audit.get("original_nominal_contract"),
        training_manifest, audit.get("deployment_leg_kp"), reference_sha256=audit.get("reference_sha256"))
    if set(audit) != set(expected_audit) | {"original_manifest_sha256", "checkpoint_sha256"}:
        raise ValueError("Unknown or missing diagnostic metadata field")
    for key, expected in expected_audit.items():
        if audit.get(key) != expected or (isinstance(expected, bool) and type(audit.get(key)) is not bool):
            raise ValueError(f"Stored diagnostic metadata differs: {key}")
    original = audit["original_nominal_contract"]
    if set(effective_contract) - (set(original) | EXPORT_METADATA_KEYS | {"controller_gain_diagnostic"}):
        raise ValueError("Unknown export field or undeclared deployment override")
    candidate = {key: value for key, value in effective_contract.items() if key != "controller_gain_diagnostic"}
    _nominal_only(candidate)
    _controller(candidate)
    if candidate.get("task_profile") != profile or _canonical_sha(profile) != audit["effective_profile_sha256"]:
        raise ValueError("Effective deployment task/material/reward profile differs")
    for key, old in original.items():
        expected = audit["deployment_leg_kp"] if key == "leg_kp" else (profile if key == "task_profile" else old)
        if key == "domain_randomization":
            if candidate.get(key) not in (old, EVAL_NOMINAL_DOMAIN):
                raise ValueError("Only ordinary nominal-evaluation domain metadata may change")
        elif candidate.get(key) != expected:
            raise ValueError(f"Unexpected deployment override: {key}")
    if candidate.get("reference_file_sha256", audit["reference_sha256"]) != audit["reference_sha256"]:
        raise ValueError("Exported reference hash differs")
    if candidate.get("checkpoint_sha256", audit["checkpoint_sha256"]) != audit["checkpoint_sha256"]:
        raise ValueError("Exported checkpoint hash differs")
    if ("training_domain_randomization" in candidate
            and candidate["training_domain_randomization"] != training_manifest.get("domain_randomization_profile")):
        raise ValueError("Exported training domain profile differs from the original manifest")
    return copy.deepcopy(audit)
