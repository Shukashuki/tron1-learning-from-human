"""Pure zero-shot damping provenance; published Kp helper remains immutable."""
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
from training import tron1_controller_damping_diagnostic as damping

REF, MANIFEST, CHECKPOINT = "a" * 64, "b" * 64, "c" * 64


def inputs():
    profile = json.loads((ROOT / "config/car_drift_contact_task.json").read_text())
    nominal = apply_task_profile_to_contract(read_contract(), profile)
    manifest = {"contract": copy.deepcopy(nominal), "task_profile": copy.deepcopy(profile), "motion_sha256": REF,
                "domain_randomization_profile": {"training_only": "original distribution"}}
    manifest["contract"]["domain_randomization"] = copy.deepcopy(manifest["domain_randomization_profile"])
    return profile, nominal, manifest


def prepare(gain=12.):
    profile, nominal, manifest = inputs()
    return damping.prepare_deployment_gain_diagnostic(profile, nominal, manifest, gain, reference_sha256=REF)


def sealed():
    profile, nominal, manifest = inputs()
    effective, audit = damping.prepare_deployment_gain_diagnostic(profile, nominal, manifest, 12., reference_sha256=REF)
    audit.update(original_manifest_sha256=MANIFEST, checkpoint_sha256=CHECKPOINT)
    return effective, damping.apply_deployment_gain_diagnostic_to_contract(nominal, audit), manifest


def validate(contract, manifest):
    return damping.validate_deployment_gain_diagnostic(contract, manifest,
        original_manifest_sha256=MANIFEST, checkpoint_sha256=CHECKPOINT)


def test_only_damping_changes_inputs_unchanged_no_optional_kp_inserted():
    profile, nominal, manifest = inputs();before = copy.deepcopy((profile, nominal, manifest))
    effective, audit = damping.prepare_deployment_gain_diagnostic(profile, nominal, manifest, 12., reference_sha256=REF)
    assert effective == {**profile, "leg_kd": 12.} and "leg_kp" not in effective
    assert (audit["trained_leg_kd"], audit["deployment_leg_kd"], audit["unchanged_leg_kp"]) == (10., 12., 500.)
    assert audit["scope"] == "zero_shot_common_leg_kd" and audit["additional_training_updates"] == 0
    assert audit["global_stability_claimed"] is False and audit["hardware_ready"] is False
    audit.update(original_manifest_sha256=MANIFEST, checkpoint_sha256=CHECKPOINT)
    contract = damping.apply_deployment_gain_diagnostic_to_contract(nominal, audit)
    assert {k: v for k, v in contract.items() if k != "controller_gain_diagnostic"} == {
        **nominal, "leg_kd": 12., "task_profile": effective}
    contract["controller_gain_diagnostic"]["original_training_contract"]["leg_joint_names"][0] = "private"
    assert (profile, nominal, manifest) == before


@pytest.mark.parametrize("gain", [5, 10, 12, 13, 5., 12., 13.])
def test_bounded_explicit_damping_valid(gain):
    assert prepare(gain)[0]["leg_kd"] == gain


@pytest.mark.parametrize("gain", [None, True, False, "12", [], 4.999, 13.001, 30., -1., float("nan"), float("inf"), 10 ** 500])
def test_bad_damping_is_rejected_not_clipped(gain):
    with pytest.raises(ValueError):prepare(gain)


@pytest.mark.parametrize("manifest", [None, {}, [], True, {"contract": None}])
def test_original_manifest_required(manifest):
    profile, nominal, _ = inputs()
    with pytest.raises(ValueError):
        damping.prepare_deployment_gain_diagnostic(profile, nominal, manifest, 12., reference_sha256=REF)


@pytest.mark.parametrize("key,value", [("leg_kp", 400.), ("leg_kd", 12.), ("physics_dt", .0025), ("decimation", 8),
    ("actor_dim", 50), ("critic_dim", True), ("leg_position_offset", "zero"), ("leg_position_scale", .5),
    ("wheel_torque_scale_nm", 6.), ("leg_torque_limit_nm", 40.), ("wheel_motor_velocity_limit_rad_s", 80.),
    ("actuator_model", "implicit"), ("training_full_clip_start_fraction", .5), ("hardware_ready", True),
    ("made_up_nominal_key", "undeclared")])
def test_original_controller_must_match_training_for_all_nominal_fields(key, value):
    profile, nominal, manifest = inputs();nominal[key] = value
    with pytest.raises(ValueError):
        damping.prepare_deployment_gain_diagnostic(profile, nominal, manifest, 12., reference_sha256=REF)


def test_matching_refined_clock_still_forbidden_for_this200hz_diagnostic():
    profile, nominal, manifest = inputs()
    nominal.update(physics_dt=.0025, decimation=8)
    manifest["contract"].update(physics_dt=.0025, decimation=8)
    with pytest.raises(ValueError, match="200Hz"):
        damping.prepare_deployment_gain_diagnostic(profile, nominal, manifest, 12., reference_sha256=REF)


@pytest.mark.parametrize("change", ["manifest_profile", "trained_profile", "input_reward", "reference", "missing_controller"])
def test_original_profile_reference_material_checks_not_bypassed(change):
    profile, nominal, manifest = inputs()
    if change == "manifest_profile": manifest["task_profile"]["contact_friction"] = .2
    elif change == "trained_profile": manifest["contract"]["task_profile"]["reward_weights"]["speed"] = 2.
    elif change == "input_reward": profile["reward_weights"]["speed"] = 2.
    elif change == "reference": manifest["motion_sha256"] = "d" * 64
    else: del manifest["contract"]["leg_kd"]
    with pytest.raises(ValueError):
        damping.prepare_deployment_gain_diagnostic(profile, nominal, manifest, 12., reference_sha256=REF)


def test_ordinary_export_and_json_roundtrip_accepted():
    effective, contract, manifest = sealed()
    contract.update(domain_randomization=damping.original_validation.EVAL_NOMINAL_DOMAIN,
        actor_file="actor_normalized.pt", actor_includes_observation_normalizer=True, actor_output_is_unclipped=True,
        actor_observation_groups=["policy"], export_validation_max_abs_error=1e-7,
        reference_file_sha256=REF, checkpoint_sha256=CHECKPOINT, default_joint_position_rad=[0.] * 8,
        actual_joint_order=contract["observed_joint_velocity_order"], actual_body_order=contract["body_names"],
        training_domain_randomization=manifest["domain_randomization_profile"])
    replay = json.loads(json.dumps(contract, allow_nan=False))
    audited = validate(replay, manifest)
    assert audited == contract["controller_gain_diagnostic"] and audited is not replay["controller_gain_diagnostic"]
    assert replay["task_profile"] == effective and replay["leg_kp"] == 500.


@pytest.mark.parametrize("key,value", [("leg_kd", 10.), ("leg_kp", 400.), ("physics_dt", .0025),
    ("wheel_torque_scale_nm", 11.), ("domain_parameters", {}), ("physics_clock_refinement", {}),
    ("domain_randomization", "random draws enabled"), ("reference_file_sha256", "d" * 64),
    ("checkpoint_sha256", "d" * 64), ("training_domain_randomization", {"mode": "forged"}),
    ("terrain", {"type": "different floor"}), ("reference_sha256", REF)])
def test_other_deployment_overrides_are_rejected(key, value):
    _, contract, manifest = sealed();contract[key] = value
    with pytest.raises(ValueError):validate(contract, manifest)


@pytest.mark.parametrize("key,value", [("scope", "zero_shot_common_leg_kp"), ("schema_version", True),
    ("additional_training_updates", True), ("additional_training_updates", 1), ("trained_leg_kd", 12.),
    ("deployment_leg_kd", True), ("unchanged_leg_kp", 400.), ("effective_profile_sha256", "d" * 64),
    ("original_manifest_sha256", "d" * 64), ("checkpoint_sha256", "d" * 64), ("helper_sha256", "d" * 64),
    ("original_validation_dependency_sha256", "d" * 64), ("hardware_ready", True),
    ("statistical_robustness_claimed", True), ("global_stability_claimed", True),
    ("training_at_deployment_gain_claimed", True), ("unknown_hardware_success", True)])
def test_forged_audit_metadata_rejected(key, value):
    _, contract, manifest = sealed();contract["controller_gain_diagnostic"][key] = value
    with pytest.raises(ValueError):validate(contract, manifest)


def test_effective_profile_only_damping_not_reward_friction_stiffness():
    for key, value in (("contact_friction", .2), ("speed_std_m_s", .2), ("mujoco_contact_solref", [.02, 1.]), ("leg_kp", 500.)):
        _, contract, manifest = sealed();contract["task_profile"][key] = value
        with pytest.raises(ValueError):validate(contract, manifest)


def test_apply_requires_original_contract_and_actual_file_hashes():
    _, nominal, _ = inputs();_, audit = prepare()
    with pytest.raises(ValueError):damping.apply_deployment_gain_diagnostic_to_contract(nominal, audit)
    audit.update(original_manifest_sha256=MANIFEST, checkpoint_sha256=CHECKPOINT)
    nominal["leg_kp"] = 400.
    with pytest.raises(ValueError):damping.apply_deployment_gain_diagnostic_to_contract(nominal, audit)


def test_immutable_kp_dependency_unchanged_and_bad_pin_rejected(monkeypatch):
    digest = hashlib.sha256(Path(damping.original_validation.__file__).read_bytes()).hexdigest()
    assert digest == damping.ORIGINAL_VALIDATION_SHA256
    assert prepare()[1]["original_validation_dependency_sha256"] == digest
    monkeypatch.setattr(damping, "ORIGINAL_VALIDATION_SHA256", "d" * 64)
    with pytest.raises(ValueError, match="dependency changed"):prepare()


def test_real004_manifest_complete_enriched_contract_roundtrip():
    path = ROOT / "outputs/drift-sim2sim-20261011/trial004/training/manifest.json"
    if not path.exists():pytest.skip("Private original004 evidence not installed")
    manifest = json.loads(path.read_text());profile = manifest["task_profile"]
    ordinary = json.loads((path.parent.parent / "isaac/policy_contract.json").read_text())
    nominal = apply_task_profile_to_contract(read_contract(), profile)
    effective, audit = damping.prepare_deployment_gain_diagnostic(profile, nominal, manifest, 12., reference_sha256=manifest["motion_sha256"])
    manifest_sha = hashlib.sha256(path.read_bytes()).hexdigest();checkpoint_sha = ordinary["checkpoint_sha256"]
    audit.update(original_manifest_sha256=manifest_sha, checkpoint_sha256=checkpoint_sha)
    contract = damping.apply_deployment_gain_diagnostic_to_contract(nominal, audit)
    contract.update({key: value for key, value in ordinary.items() if key not in nominal})
    contract["domain_randomization"] = ordinary["domain_randomization"]
    replay = json.loads(json.dumps(contract, allow_nan=False))
    result = damping.validate_deployment_gain_diagnostic(replay, manifest,
        original_manifest_sha256=manifest_sha, checkpoint_sha256=checkpoint_sha)
    assert result["deployment_leg_kd"] == effective["leg_kd"] == 12. and replay["leg_kp"] == 500.
    assert "leg_kp" not in effective
