"""Pure offline LR-context fixtures; none is a training/physics result."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import publish_car_drift_lr_sync as publish


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return path


def lr_audit():
    return {"schema_version": 1, "scope": "adaptive_ppo_learning_rate_scalar_from_loaded_optimizer",
        "status": "restored_from_loaded_optimizer", "scalar_before": .001,
        "loaded_group_learning_rates": [1.e-5], "scalar_after": 1.e-5,
        "touched_field": "algorithm.learning_rate", "optimizer_group_learning_rates_unchanged": True,
        "adaptive_schedule_preserved": True, "expected_scalar_before": .001, "expected_loaded_lr": 1.e-5,
        "multi_gpu": False}


@pytest.fixture
def context(tmp_path, monkeypatch):
    """Generated JSON/tokens, no actor or checkpoint is opened/deserialized."""
    root = tmp_path / "repository"
    trial = root / "outputs/drift-sim2sim-20261011/trial012"
    for relative in publish.BASE_SOURCE_PINS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(ROOT / relative)
    sources = dict(publish.BASE_SOURCE_PINS)
    for relative in publish.NEW_SOURCE_PINS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("GENERATED SOURCE FIXTURE: " + relative).encode())
        sources[relative] = publish.base.sha256(path)
    profile = publish.base.read_json(ROOT / "config/car_drift_contact_task.json")
    profile_path = root / "config/car_drift_contact_task.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_bytes((ROOT / "config/car_drift_contact_task.json").read_bytes())
    baseline = {"contract": {"task_profile": profile}, "domain_randomization_profile": {"mode": "random", "seed": 42}}
    baseline_path = write(root / "outputs/drift-sim2sim-20261011/trial004/training/manifest.json", baseline)
    monkeypatch.setattr(publish, "PARENT_MANIFEST_SHA256", publish.base.sha256(baseline_path))
    runner_path = write(trial / "training/runner_config.json", {"num_steps_per_env": 24})
    monkeypatch.setattr(publish, "RUNNER_SHA256", publish.base.sha256(runner_path))
    protocol = publish.base.read_json(ROOT / "config/car_drift_lr_sync_trial.json")
    protocol["training"]["parent_manifest_sha256"] = publish.PARENT_MANIFEST_SHA256
    protocol["training"]["runner_config_sha256"] = publish.RUNNER_SHA256
    protocol_path = write(root / "config/car_drift_lr_sync_trial.json", protocol)
    checkpoint = trial / "training/model_final.pt"
    checkpoint.write_bytes(b"GENERATED final checkpoint token, NEVER DESERIALIZED")
    actor = trial / "isaac/actor_normalized.pt"
    actor.parent.mkdir(parents=True, exist_ok=True)
    actor.write_bytes(b"GENERATED actor token, NEVER EXECUTED")
    cp_sha, actor_sha = publish.base.sha256(checkpoint), publish.base.sha256(actor)
    contract = {"task_profile": profile, "checkpoint_sha256": cp_sha,
                "reference_file_sha256": publish.REFERENCE_PINS["motion_sha256"]}
    audit = lr_audit()
    material = {"leg_pd_gain_readback": {"kp_range": [500., 500.], "kd_range": [10., 10.]}}
    manifest = {"source_sha256": sources, "versions": {"rsl-rl-lib": "3.0.1"},
        "resume_learning_rate_audit": audit,
        "resume_source_audit": {"schema_version": 1, "rsl_rl_version": "3.0.1",
            "source_root": "/data/private-account/conda/rsl_rl", "expected_source_sha256": publish.RSL_SOURCE_PINS,
            "actual_source_sha256": publish.RSL_SOURCE_PINS, "verified_before_runner_load": True},
        "contract": baseline["contract"], "domain_randomization_profile": baseline["domain_randomization_profile"],
        "task_profile_sha256": publish.PROFILE_SHA256, "task_profile": profile, "num_envs": 2048,
        "iterations_requested": 600, "seed": 42, "smoke_steps_passed": 8,
        "resume_sha256": publish.PARENT_CP_SHA256, "contact_material_audit": material}
    manifest.update(motion_sha256=publish.REFERENCE_PINS["motion_sha256"], asset_sha256=publish.REFERENCE_PINS["robot_asset_sha256"])
    records = {"manifest": manifest, "contract": contract,
        "isaac_report": {"task_profile": profile, "checkpoint_sha256": cp_sha, "contact_material_audit": material,
                         "asset_root_file_sha256": publish.REFERENCE_PINS["robot_asset_sha256"]},
        "mujoco_report": {"contract": contract, "model_sha256": publish.REFERENCE_PINS["mujoco_model_sha256"]},
        "runner": {"num_steps_per_env": 24},
        "run_status": {"status": "training_completed", "num_envs": 2048, "iterations_requested": 600,
            "training_environment_steps": 29_491_200, "last_iteration": 4595,
            "checkpoint": "/data/private-account/run/outputs/trial012/training/model_final.pt"}}
    filenames = {"manifest": "training/manifest.json", "contract": "isaac/policy_contract.json",
        "isaac_report": "isaac/report.json", "mujoco_report": "mujoco/report.json", "run_status": "training/run_status.json"}
    files = {key: write(trial / relative, records[key]) for key, relative in filenames.items()}
    files.update(actor=actor, final_checkpoint=checkpoint, runner=runner_path)
    for relative in publish.FINAL_INPUTS:
        path = trial / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("GENERATED archive token: " + relative).encode())
    hashes = {key: publish.base.sha256(path) for key, path in files.items()}
    hashes.update(motion=publish.REFERENCE_PINS["motion_sha256"], robot_reference=publish.REFERENCE_PINS["robot_reference_sha256"])
    completion = {"status": "training_completion_verified", "final_checkpoint_sha256": cp_sha,
        "training_budget": {"updates": 600, "num_envs": 2048, "rollout_steps_per_update": 24,
            "seed": 42, "additional_environment_steps": 29_491_200, "saved_iteration": 4595}}
    for name, iteration, step, count in (("parent_checkpoint_health", 3996, 80060, 145072128),
                                        ("final_checkpoint_health", 4595, 92060, 174563328)):
        completion[name] = {"saved_iteration": iteration, "model_tensor_count": 25, "optimizer_tensor_count": 51,
            "adam_parameter_state_count": 17, "every_model_optimizer_tensor_finite": True,
            "adam_parameter_steps": [step] * 17,
            "normalizer_counts": {"actor_obs_normalizer.count": count, "critic_obs_normalizer.count": count}}
    checks = {k: True for k in ("all_model_optimizer_tensors_finite", "exported_actor_normalizer_bit_exact_checkpoint",
        "all_recorded_actor_actions_replay", "same_contract_profile_both_engines", "all_assessor_provenance_checks_pass")}
    checks.update({"GENERATED_check_" + str(i): True for i in range(10)})
    proof = {"status": "deployment_provenance_pass", "checks": checks,
        "input_sha256": {relative: publish.base.sha256(trial / relative) for relative in publish.FINAL_INPUTS},
        "checkpoint_tensor_health": {"model_state_dict": {"tensor_count": 25, "all_finite": True},
                                    "optimizer_state_dict": {"tensor_count": 51, "all_finite": True}},
        "actor_export": {"actor_sha256": actor_sha, "includes_normalizer": True, "tensors_compared_bit_exact": 12,
            "recorded_pre_action_rows_checked": 288, "baseline_actor_equivalence_required": False}}
    for engine in ("isaac", "mujoco"):
        proof[engine + "_actor_replay"] = {"rows": 288, "predeclared_absolute_tolerance": 1.e-5,
                                          "max_abs_raw_error": 3.e-7, "max_abs_clipped_error": 3.e-7}
    dedicated = write(trial / "training/resume_learning_rate_audit.json", audit)
    completion["input_sha256"] = {"training/" + name: publish.base.sha256(trial / "training" / name)
        for name in ("manifest.json", "run_status.json", "runner_config.json", "model_final.pt", "resume_learning_rate_audit.json")}
    completion["input_sha256"].update({"repository/" + key: value for key, value in sources.items()})
    completion["input_sha256"]["repository/config/car_drift_lr_sync_trial.json"] = publish.base.sha256(protocol_path)
    files["lr_sync_protocol"], hashes["lr_sync_protocol"] = protocol_path, publish.base.sha256(protocol_path)
    complete_path = write(trial / "diagnostics/training_completion_verification.json", completion)
    proof["input_sha256"].update({"training/resume_learning_rate_audit.json": publish.base.sha256(dedicated),
        "diagnostics/training_completion_verification.json": publish.base.sha256(complete_path),
        "repository/config/car_drift_lr_sync_trial.json": publish.base.sha256(protocol_path)})
    proof["input_sha256"].update({"repository/" + key: value for key, value in sources.items()})
    proof_path = write(trial / "diagnostics/deployment_provenance_audit.json", proof)
    return {"records": records, "files": files, "hashes": hashes, "protocol": protocol, "lr_audit": audit,
        "completion": completion, "proof": proof, "repo_root": root, "expected_source_sha256": sources,
        "trial": trial, "protocol_path": protocol_path, "proof_path": proof_path}


def validate(c):
    keys = ("records", "files", "hashes", "protocol", "lr_audit", "completion", "proof", "repo_root", "expected_source_sha256")
    return publish.validate_context(**{k: c[k] for k in keys})


def test_explicit_single_factor_context(context):
    result = validate(context)
    assert result["restoration"] == lr_audit()
    assert result["legacy_publication_route_is_lr_aware"] is False
    assert result["behavior_gates_changed"] is False
    assert result["learned_actor_equals_baseline_required"] is False


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("scalar_before", True), ("scalar_before", .01),
    ("scalar_after", 1.e-4), ("scalar_after", float("nan")), ("scalar_after", float("inf")),
    ("loaded_group_learning_rates", []), ("loaded_group_learning_rates", [True]),
    ("loaded_group_learning_rates", [1.e-5, 1.e-4]), ("loaded_group_learning_rates", [0.]),
    ("optimizer_group_learning_rates_unchanged", 1), ("adaptive_schedule_preserved", False),
    ("multi_gpu", 0), ("touched_field", "optimizer.param_groups"), ("expected_loaded_lr", None),
    ("scope", "fixed_schedule"), ("status", "not_restored"), ("extra", "hidden_change")])
def test_malformed_or_extra_lr_audit_rejected(field, value):
    audit = lr_audit(); audit[field] = value
    with pytest.raises(ValueError):
        publish.validate_lr_audit(audit)


@pytest.mark.parametrize("field", sorted(publish.LR_AUDIT_KEYS))
def test_missing_lr_audit_field_rejected(field):
    audit = lr_audit(); del audit[field]
    with pytest.raises(ValueError):
        publish.validate_lr_audit(audit)


@pytest.mark.parametrize("field,value", [("num_envs", True), ("num_envs", 1024), ("iterations_requested", 599),
    ("seed", 43), ("smoke_steps_passed", 7), ("resume_sha256", "a" * 64), ("task_profile_sha256", "b" * 64)])
def test_actual_training_factor_mismatch_rejected(context, field, value):
    context["records"]["manifest"][field] = value
    with pytest.raises(ValueError): validate(context)


@pytest.mark.parametrize("field,value", [("status", "training"), ("last_iteration", 4594),
    ("training_environment_steps", 29_491_199), ("num_envs", 2048.), ("iterations_requested", 600.)])
def test_incomplete_or_untyped_training_rejected(context, field, value):
    context["records"]["run_status"][field] = value
    with pytest.raises(ValueError): validate(context)


@pytest.mark.parametrize("field", ["controller_gain_diagnostic", "physics_clock_refinement", "reward_sampling_audit", "ground_load_reserve_module_sha256"])
def test_simultaneous_feature_or_orphan_rejected(context, field):
    context["records"]["manifest"][field] = {}
    with pytest.raises(ValueError): validate(context)


def test_startup_audit_missing_or_differs_rejected(context):
    context["records"]["manifest"]["resume_learning_rate_audit"] = {}
    with pytest.raises(ValueError): validate(context)


@pytest.mark.parametrize("field,value", [("verified_before_runner_load", 1), ("rsl_rl_version", "3.1.0"),
    ("actual_source_sha256", {}), ("expected_source_sha256", {})])
def test_installed_source_audit_rejected(context, field, value):
    context["records"]["manifest"]["resume_source_audit"][field] = value
    with pytest.raises(ValueError): validate(context)


def test_source_pin_or_local_source_mutation_rejected(context):
    context["expected_source_sha256"]["training/tron1_resume_lr.py"] = "a" * 64
    with pytest.raises(ValueError): validate(context)


def test_orphan_source_pin_rejected(context):
    context["records"]["manifest"]["source_sha256"]["extra.py"] = "a" * 64
    with pytest.raises(ValueError): validate(context)


@pytest.mark.parametrize("category,field,value", [
    ("final_checkpoint_health", "every_model_optimizer_tensor_finite", 1),
    ("final_checkpoint_health", "adam_parameter_steps", [92059] * 17),
    ("final_checkpoint_health", "model_tensor_count", 25.),
    ("parent_checkpoint_health", "normalizer_counts", {"actor_obs_normalizer.count": 0, "critic_obs_normalizer.count": 0})])
def test_final_cpu_health_or_budget_rejected(context, category, field, value):
    context["completion"][category][field] = value
    with pytest.raises(ValueError): validate(context)


def test_final_proof_input_tampering_rejected(context):
    context["proof"]["input_sha256"]["training/model_final.pt"] = "b" * 64
    with pytest.raises(ValueError): validate(context)


@pytest.mark.parametrize("category,field,value", [
    ("actor_export", "includes_normalizer", 1), ("actor_export", "tensors_compared_bit_exact", 11),
    ("actor_export", "recorded_pre_action_rows_checked", 287), ("actor_export", "baseline_actor_equivalence_required", True),
    ("mujoco_actor_replay", "rows", 287), ("isaac_actor_replay", "max_abs_raw_error", 1.e-4),
    ("mujoco_actor_replay", "predeclared_absolute_tolerance", 1.e-4)])
def test_actor_proof_replay_mismatch_rejected(context, category, field, value):
    context["proof"][category][field] = value
    with pytest.raises(ValueError): validate(context)


def test_proof_booleans_are_not_numbers(context):
    context["proof"]["checks"]["all_recorded_actor_actions_replay"] = 1
    with pytest.raises(ValueError): validate(context)


def test_old_publisher_kept_byte_exact_and_new_code_has_no_apps():
    assert publish.base.sha256(ROOT / "scripts/publish_car_drift.py") == publish.LEGACY_PUBLISHER_SHA256
    tree = ast.parse((ROOT / "scripts/publish_car_drift_lr_sync.py").read_text())
    names = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not names & {"mj_step", "mj_forward", "load", "learn", "step", "simulate"}
    imports = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any("torch" in n or "isaaclab" in n or "mujoco" in n for n in imports)


def test_prepare_wraps_base_without_modifying_three_payloads_or_media(context, monkeypatch):
    c = context
    payloads = {name: b"GENERATED base payload" for name in ("assessment.json", "reference_audit.json", "render_report.json")}
    media = {name: {"path": c["files"]["actor"], "sha256": c["hashes"]["actor"]}
             for name in ("motion_pipeline.mp4", "overview.png", "final.png", "contact_overview.png")}
    summary = {"verdict": "fail", "behavior_pass": False, "input_sha256": c["hashes"], "failed_checks": ["synthetic_failure"]}
    before = copy.deepcopy(payloads)
    monkeypatch.setattr(publish.base, "prepare_publication", lambda *a: (summary, payloads, media, dict(c["files"]), dict(c["hashes"])))
    result, data, returned_media, _, _ = publish.prepare_publication(c["trial"], c["repo_root"] / "reference", c["trial"] / "render",
        protocol_path=c["protocol_path"], expected_protocol_sha256=publish.base.sha256(c["protocol_path"]),
        expected_source_sha256=c["expected_source_sha256"], repo_root=c["repo_root"])
    assert data == before and returned_media == media
    assert result["behavior_pass"] is False and result["failed_checks"] == ["synthetic_failure"]
    assert "learning_rate_resume_context" in result
    text = publish.base.encode(result).decode()
    assert "/data/private-account" not in text and "private-account" not in text


def test_wrong_protocol_pin_fails_before_base(context, monkeypatch):
    monkeypatch.setattr(publish.base, "prepare_publication", lambda *a: pytest.fail("Base must not run on unsealed protocol"))
    with pytest.raises(ValueError):
        publish.prepare_publication(context["trial"], "unused", "unused", protocol_path=context["protocol_path"],
            expected_protocol_sha256="a" * 64, expected_source_sha256=context["expected_source_sha256"])


def test_existing_public_output_rejected_before_prepare(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "prepare_publication", lambda *a, **k: pytest.fail("No preparation before overwrite refusal"))
    with pytest.raises(ValueError):
        publish.publish_trial("unused", "unused", tmp_path, "unused")


@pytest.mark.parametrize("field", ["training", "learning_rate_resume", "unchanged", "evaluation"])
@pytest.mark.parametrize("value", [None, [], True])
def test_malformed_protocol_sections_reject(context, field, value):
    context["protocol"][field] = value
    with pytest.raises(ValueError): validate(context)


@pytest.mark.parametrize("field,value", [("updates", 599), ("num_envs", True), ("saved_iteration", 4594),
    ("additional_environment_steps", 29_491_200.)])
def test_cpu_completion_budget_is_exact_and_typed(context, field, value):
    context["completion"]["training_budget"][field] = value
    with pytest.raises(ValueError): validate(context)


@pytest.mark.parametrize("relative", ["training/manifest.json", "training/run_status.json", "training/runner_config.json",
    "training/model_final.pt", "training/resume_learning_rate_audit.json", "repository/config/car_drift_lr_sync_trial.json",
    "repository/training/tron1_resume_lr.py"])
def test_missing_or_wrong_completion_seals_rejected(context, relative):
    context["completion"]["input_sha256"].pop(relative)
    with pytest.raises(ValueError): validate(context)


@pytest.mark.parametrize("field", ["training_budget", "parent_checkpoint_health", "final_checkpoint_health"])
def test_malformed_completion_object_rejected(context, field):
    context["completion"][field] = None
    with pytest.raises(ValueError): validate(context)


@pytest.mark.parametrize("field", ["checkpoint_tensor_health", "actor_export", "isaac_actor_replay", "mujoco_actor_replay", "input_sha256"])
def test_malformed_actor_proof_object_rejected(context, field):
    context["proof"][field] = None
    with pytest.raises(ValueError): validate(context)


@pytest.mark.parametrize("field,value", [("motion_sha256", "a" * 64), ("asset_sha256", "b" * 64)])
def test_original_reference_asset_not_silently_changed(context, field, value):
    context["records"]["manifest"][field] = value
    with pytest.raises(ValueError): validate(context)


@pytest.mark.parametrize("record", ["contract", "isaac_report", "mujoco_report"])
def test_orphan_actor_lr_metadata_rejected(context, record):
    context["records"][record]["resume_learning_rate_audit"] = lr_audit()
    with pytest.raises(ValueError): validate(context)


def test_final_not_historical_or_intermediate_checkpoint(context):
    context["records"]["run_status"]["checkpoint"] = "/data/private/run/outputs/trial011/training/model_final.pt"
    with pytest.raises(ValueError): validate(context)


def test_legacy_route_is_not_misrepresented_as_lr_aware():
    out = publish.base.training_budget({"resume_learning_rate_audit": lr_audit(), "resume_source_audit": {"verified": True}})
    assert "resume_learning_rate_audit" not in out and "resume_source_audit" not in out


def test_production_source9_defaults_are_reviewed_actual_files():
    assert len(publish.BASE_SOURCE_PINS) + len(publish.NEW_SOURCE_PINS) == 9
    for relative, expected in {**publish.BASE_SOURCE_PINS, **publish.NEW_SOURCE_PINS}.items():
        assert publish.base.sha256(ROOT / relative) == expected
