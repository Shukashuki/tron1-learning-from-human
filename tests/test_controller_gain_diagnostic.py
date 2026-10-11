"""Pure bounded zero-shot controller provenance; no Isaac/app/tensor import."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT));sys.path.insert(0, str(ROOT / "scripts"))
from eval_tracking_mujoco import read_contract
from training.tron1_drift import apply_task_profile_to_contract
from training.tron1_controller_gain_diagnostic import (
    apply_deployment_gain_diagnostic_to_contract, prepare_deployment_gain_diagnostic,
    validate_deployment_gain_diagnostic,
)

REF, MANIFEST, CHECKPOINT = "a" * 64, "b" * 64, "c" * 64


def inputs():
    profile = json.loads((ROOT / "config/car_drift_contact_task.json").read_text())
    nominal = apply_task_profile_to_contract(read_contract(), profile)
    manifest = {"contract": copy.deepcopy(nominal), "task_profile": copy.deepcopy(profile), "motion_sha256": REF}
    manifest["contract"]["domain_randomization"] = {"training_only": "unchanged distribution metadata"}
    manifest["domain_randomization_profile"] = copy.deepcopy(manifest["contract"]["domain_randomization"])
    return profile, nominal, manifest


def prepare(profile=None, nominal=None, manifest=None, gain=400.):
    p, c, m = inputs()
    return prepare_deployment_gain_diagnostic(p if profile is None else profile, c if nominal is None else nominal,
        m if manifest is None else manifest, gain, reference_sha256=REF)


def sealed():
    profile, nominal, manifest = inputs()
    effective_profile, audit = prepare_deployment_gain_diagnostic(profile, nominal, manifest, 400., reference_sha256=REF)
    audit.update(original_manifest_sha256=MANIFEST, checkpoint_sha256=CHECKPOINT)
    contract = apply_deployment_gain_diagnostic_to_contract(nominal, audit)
    return effective_profile, contract, manifest


def validate(contract, manifest):
    return validate_deployment_gain_diagnostic(contract, manifest,
        original_manifest_sha256=MANIFEST, checkpoint_sha256=CHECKPOINT)


def test_effective_profile_and_contract_change_only_kp_with_deepcopied_originals():
    profile, nominal, manifest = inputs();before = copy.deepcopy((profile, nominal, manifest))
    effective, audit = prepare_deployment_gain_diagnostic(profile, nominal, manifest, 400., reference_sha256=REF)
    assert effective == {**profile, "leg_kp": 400.}
    assert audit["original_training_contract"] == manifest["contract"]
    assert audit["original_nominal_contract"] == nominal and audit["trained_leg_kp"] == 500.
    audit.update(original_manifest_sha256=MANIFEST, checkpoint_sha256=CHECKPOINT)
    out = apply_deployment_gain_diagnostic_to_contract(nominal, audit)
    assert {k: v for k, v in out.items() if k != "controller_gain_diagnostic"} == {**nominal, "leg_kp": 400., "task_profile": effective}
    assert (profile, nominal, manifest) == before
    out["controller_gain_diagnostic"]["original_training_contract"]["leg_joint_names"][0] = "private copy"
    assert (profile, nominal, manifest) == before


@pytest.mark.parametrize("gain", [100, 400, 500, 100., 400., 500.])
def test_bounded_explicit_gains_are_valid(gain):
    assert prepare(gain=gain)[0]["leg_kp"] == gain


@pytest.mark.parametrize("gain", [None, True, False, "400", [], 99.999, 500.001, -1., float("nan"), float("inf"), 10 ** 500])
def test_bad_gain_types_nonfinite_or_out_of_bounds_rejected(gain):
    with pytest.raises(ValueError): prepare(gain=gain)


@pytest.mark.parametrize("manifest", [None, {}, [], True, {"contract": None}])
def test_manifest_required_and_malformed_rejected(manifest):
    profile, nominal, _ = inputs()
    with pytest.raises(ValueError):
        prepare_deployment_gain_diagnostic(profile, nominal, manifest, 400., reference_sha256=REF)


@pytest.mark.parametrize("key,value", [("leg_kp", 250.), ("leg_kd", 30.), ("physics_dt", .0025), ("decimation", 8),
    ("actor_dim", 50), ("critic_dim", True), ("leg_position_offset", "zero"), ("leg_position_scale", .5),
    ("wheel_torque_scale_nm", 6.), ("leg_torque_limit_nm", 40.), ("wheel_motor_velocity_limit_rad_s", 80.)])
def test_original_nominal_control_must_match_training(key, value):
    profile, nominal, manifest = inputs();nominal[key] = value
    with pytest.raises(ValueError):
        prepare_deployment_gain_diagnostic(profile, nominal, manifest, 400., reference_sha256=REF)


@pytest.mark.parametrize("key,value", [("training_full_clip_start_fraction", .5), ("hardware_ready", True),
    ("actuator_limits_provenance", "invented limits"), ("made_up_nominal_key", "undeclared")])
def test_all_original_factory_fields_must_match_training(key, value):
    profile, nominal, manifest = inputs();nominal[key] = value
    with pytest.raises(ValueError, match="differs from training"):
        prepare_deployment_gain_diagnostic(profile, nominal, manifest, 400., reference_sha256=REF)


@pytest.mark.parametrize("change", ["manifest_profile", "trained_profile", "nominal_profile", "input_reward", "reference", "missing_controller"])
def test_original_material_reward_reference_and_contract_cannot_be_bypassed(change):
    profile, nominal, manifest = inputs()
    if change == "manifest_profile": manifest["task_profile"]["contact_friction"] = .2
    elif change == "trained_profile": manifest["contract"]["task_profile"]["reward_weights"]["speed"] = 2.
    elif change == "nominal_profile": nominal["task_profile"]["speed_std_m_s"] = .2
    elif change == "input_reward": profile["reward_weights"]["speed"] = 2.
    elif change == "reference": manifest["motion_sha256"] = "d" * 64
    else: del manifest["contract"]["leg_position_offset"]
    with pytest.raises(ValueError):
        prepare_deployment_gain_diagnostic(profile, nominal, manifest, 400., reference_sha256=REF)


def test_stored_validator_accepts_json_roundtrip_and_ordinary_export_fields():
    profile, contract, manifest = sealed()
    contract.update(domain_randomization="disabled during evaluation; fixed actuator parameters",
        actor_file="actor_normalized.pt", actor_includes_observation_normalizer=True, actor_output_is_unclipped=True,
        actor_observation_groups=["policy"], export_validation_max_abs_error=1e-7,
        reference_file_sha256=REF, checkpoint_sha256=CHECKPOINT, default_joint_position_rad=[0.] * 8,
        actual_joint_order=contract["observed_joint_velocity_order"], actual_body_order=contract["body_names"],
        training_domain_randomization=manifest["domain_randomization_profile"])
    replay = json.loads(json.dumps(contract, allow_nan=False))
    audit = validate(replay, manifest)
    assert audit == contract["controller_gain_diagnostic"] and audit is not replay["controller_gain_diagnostic"]
    assert profile == replay["task_profile"]


@pytest.mark.parametrize("key,value", [("leg_kp", 500.), ("leg_kd", 20.), ("physics_dt", .0025),
    ("wheel_torque_scale_nm", 11.), ("domain_parameters", {}), ("physics_clock_refinement", {}),
    ("domain_randomization", "random draws enabled"), ("reference_file_sha256", "d" * 64), ("checkpoint_sha256", "d" * 64),
    ("training_domain_randomization", {"mode": "forged"}), ("terrain", {"type": "changed floor"}),
    ("undeclared_torque_override", 0.), ("reference_sha256", REF)])
def test_stored_validator_rejects_other_controller_domain_clock_or_reference_overrides(key, value):
    _, contract, manifest = sealed();contract[key] = value
    with pytest.raises(ValueError):validate(contract, manifest)


def test_stored_validator_rejects_unknown_audit_claim_even_when_expected_fields_are_intact():
    _, contract, manifest = sealed();contract["controller_gain_diagnostic"]["hardware_success"] = True
    with pytest.raises(ValueError, match="Unknown or missing diagnostic"):
        validate(contract, manifest)


def test_exported_training_dr_uses_manifest_profile_not_contract_description():
    _, contract, manifest = sealed()
    manifest["domain_randomization_profile"] = {"mode": "manifest distribution"}
    contract["training_domain_randomization"] = copy.deepcopy(manifest["contract"]["domain_randomization"])
    with pytest.raises(ValueError, match="training domain profile"):
        validate(contract, manifest)
    contract["training_domain_randomization"] = manifest["domain_randomization_profile"]
    assert validate(contract, manifest)["additional_training_updates"] == 0


@pytest.mark.parametrize("key,value", [("scope", "trained at400"), ("schema_version", True),
    ("additional_training_updates", True), ("additional_training_updates", 1), ("trained_leg_kp", 250.),
    ("deployment_leg_kp", True), ("unchanged_leg_kd", 30.), ("effective_profile_sha256", "d" * 64),
    ("original_manifest_sha256", "d" * 64), ("checkpoint_sha256", "d" * 64), ("helper_sha256", "d" * 64),
    ("hardware_ready", True), ("statistical_robustness_claimed", True)])
def test_stored_validator_rejects_forged_diagnostic_metadata(key, value):
    _, contract, manifest = sealed();contract["controller_gain_diagnostic"][key] = value
    with pytest.raises(ValueError):validate(contract, manifest)


def test_effective_reward_or_contact_changes_rejected():
    for key, value in (("contact_friction", .2), ("speed_std_m_s", .2), ("mujoco_contact_solref", [.02, 1.])):
        _, contract, manifest = sealed();contract["task_profile"][key] = value
        with pytest.raises(ValueError):validate(contract, manifest)


def test_apply_requires_sealed_actual_manifest_checkpoint_hashes():
    _, nominal, _ = inputs();_, audit = prepare()
    with pytest.raises(ValueError):apply_deployment_gain_diagnostic_to_contract(nominal, audit)


def test_actual004_manifest_factory_contract_and_enriched_export_are_compatible():
    path = ROOT / "outputs/drift-sim2sim-20261011/trial004/training/manifest.json"
    if not path.exists():pytest.skip("Private real004evidence notinstalled")
    manifest = json.loads(path.read_text());profile = manifest["task_profile"]
    ordinary_export = json.loads((path.parent.parent / "isaac/policy_contract.json").read_text())
    nominal = apply_task_profile_to_contract(read_contract(), profile)
    effective, audit = prepare_deployment_gain_diagnostic(profile, nominal, manifest, 400., reference_sha256=manifest["motion_sha256"])
    manifest_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint_sha = ordinary_export["checkpoint_sha256"]
    audit.update(original_manifest_sha256=manifest_sha, checkpoint_sha256=checkpoint_sha)
    out = apply_deployment_gain_diagnostic_to_contract(nominal, audit)
    out.update({key: value for key, value in ordinary_export.items() if key not in nominal})
    out["domain_randomization"] = ordinary_export["domain_randomization"]
    roundtrip = json.loads(json.dumps(out, allow_nan=False))
    assert validate_deployment_gain_diagnostic(roundtrip, manifest, original_manifest_sha256=manifest_sha,
        checkpoint_sha256=checkpoint_sha)["deployment_leg_kp"] == effective["leg_kp"] == 400.
