"""Pure opt-in zero-shot common-leg-Kd deployment provenance.

Only damping changes; the original policy, manifest and controller stay saved
unchanged. This bounded diagnostic does not establish closed-loop stability.
The published Kp validator is reused read-only and pinned as a dependency.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

from training import tron1_controller_gain_diagnostic as original_validation
from training.tron1_drift import validate_task_profile

SCOPE = "zero_shot_common_leg_kd"
ORIGINAL_VALIDATION_SHA256 = "96ef3cba1761248062f12912b110ec3a000819f644a0030a5a89e28551197b4a"


def _dependency_sha():
    actual = hashlib.sha256(Path(original_validation.__file__).read_bytes()).hexdigest()
    if actual != ORIGINAL_VALIDATION_SHA256:
        raise ValueError("Immutable original-controller validation dependency changed")
    return actual


def _damping(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 5. <= value <= 13. or not math.isfinite(value):
        raise ValueError("Deployment leg_kd must be finite, nonboolean, and within[5,13]")
    return value


def _clock(contract):
    original_validation._controller(contract)
    if (contract["physics_dt"], contract["decimation"], contract["policy_fps"]) != (.005, 4, 50):
        raise ValueError("Damping diagnostic requires original200Hz physics/50Hz policy clock")


def prepare_deployment_gain_diagnostic(original_profile, nominal_contract, training_manifest,
                                      deployment_leg_kd, *, reference_sha256):
    """Return deep-copied effective profile and unsealed zero-update audit.

    Root adds only original_manifest_sha256/checkpoint_sha256 before applying.
    No checkpoint, tensor, app, observation or simulation state is accessed.
    """
    dependency_sha = _dependency_sha()
    damping = _damping(deployment_leg_kd)
    profile = validate_task_profile(original_profile)
    _, audit = original_validation.prepare_deployment_gain_diagnostic(
        profile, nominal_contract, training_manifest, profile.get("leg_kp", 500.),
        reference_sha256=reference_sha256)
    _clock(nominal_contract)
    effective = copy.deepcopy(profile)
    effective["leg_kd"] = damping
    effective = validate_task_profile(effective)
    trained_kp = audit.pop("trained_leg_kp")
    trained_kd = audit.pop("unchanged_leg_kd")
    del audit["deployment_leg_kp"]
    audit.update({
        "scope": SCOPE, "trained_leg_kd": trained_kd,
        "deployment_leg_kd": damping, "unchanged_leg_kp": trained_kp,
        "effective_profile_sha256": original_validation._canonical_sha(effective),
        "original_validation_dependency_sha256": dependency_sha,
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "global_stability_claimed": False,
    })
    return effective, audit


def apply_deployment_gain_diagnostic_to_contract(nominal_contract, audit):
    """Return effective damping contract without modifying original inputs."""
    _dependency_sha()
    if (not isinstance(audit, dict) or audit.get("scope") != SCOPE
            or type(audit.get("schema_version")) is not int or audit["schema_version"] != 1):
        raise ValueError("Explicit controller damping diagnostic audit required")
    if nominal_contract != audit.get("original_nominal_contract"):
        raise ValueError("Apply only to the inspected original nominal contract")
    original_validation._nominal_only(nominal_contract)
    _clock(nominal_contract)
    for field in ("original_manifest_sha256", "checkpoint_sha256"):
        original_validation._sha(audit.get(field), field)
    out = copy.deepcopy(nominal_contract)
    out["leg_kd"] = _damping(audit.get("deployment_leg_kd"))
    out["task_profile"] = copy.deepcopy(audit["original_training_task_profile"])
    out["task_profile"]["leg_kd"] = out["leg_kd"]
    out["task_profile"] = validate_task_profile(out["task_profile"])
    if original_validation._canonical_sha(out["task_profile"]) != audit.get("effective_profile_sha256"):
        raise ValueError("Effective damping profile hash differs")
    out["controller_gain_diagnostic"] = copy.deepcopy(audit)
    return out


def validate_deployment_gain_diagnostic(effective_contract, training_manifest, *,
                                      original_manifest_sha256, checkpoint_sha256):
    """Strict stored proof: only Kd/profile/known nominal DR metadata differ.

    Known ordinary export additions are permitted; unknown overrides and
    success claims are rejected. Material/reward/motor/clock fields stay fixed.
    """
    _dependency_sha()
    if not isinstance(effective_contract, dict) or not isinstance(effective_contract.get("controller_gain_diagnostic"), dict):
        raise ValueError("Stored explicit controller damping diagnostic required")
    audit = effective_contract["controller_gain_diagnostic"]
    if type(audit.get("schema_version")) is not int or audit["schema_version"] != 1:
        raise ValueError("Invalid diagnostic schema")
    if type(audit.get("additional_training_updates")) is not int or audit["additional_training_updates"] != 0:
        raise ValueError("Zero-shot diagnostic must have exactly zero additional training updates")
    for field, actual in (("original_manifest_sha256", original_manifest_sha256), ("checkpoint_sha256", checkpoint_sha256)):
        if original_validation._sha(audit.get(field), field) != original_validation._sha(actual, field):
            raise ValueError(f"Diagnostic {field} differs from actual input")
    profile, expected_audit = prepare_deployment_gain_diagnostic(
        audit.get("original_training_task_profile"), audit.get("original_nominal_contract"),
        training_manifest, audit.get("deployment_leg_kd"), reference_sha256=audit.get("reference_sha256"))
    if set(audit) != set(expected_audit) | {"original_manifest_sha256", "checkpoint_sha256"}:
        raise ValueError("Unknown or missing diagnostic metadata field")
    for key, expected in expected_audit.items():
        if audit.get(key) != expected or (isinstance(expected, bool) and type(audit.get(key)) is not bool):
            raise ValueError(f"Stored diagnostic metadata differs: {key}")
    original = audit["original_nominal_contract"]
    if set(effective_contract) - (set(original) | original_validation.EXPORT_METADATA_KEYS | {"controller_gain_diagnostic"}):
        raise ValueError("Unknown export field or undeclared deployment override")
    candidate = {key: value for key, value in effective_contract.items() if key != "controller_gain_diagnostic"}
    original_validation._nominal_only(candidate)
    _clock(candidate)
    if candidate.get("task_profile") != profile or original_validation._canonical_sha(profile) != audit["effective_profile_sha256"]:
        raise ValueError("Effective damping task/material/reward profile differs")
    for key, old in original.items():
        expected = audit["deployment_leg_kd"] if key == "leg_kd" else (profile if key == "task_profile" else old)
        if key == "domain_randomization":
            if candidate.get(key) not in (old, original_validation.EVAL_NOMINAL_DOMAIN):
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
