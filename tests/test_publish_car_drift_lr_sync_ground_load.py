"""Pure offline LR-and-reserve fixtures; none is a training/physics result."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import publish_car_drift_lr_sync_ground_load as publish


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



def full_training_contract(profile):
    """Execute only the pure contract function/constants, never simulator imports."""
    tree = ast.parse((ROOT / "training/tron1_tracking.py").read_text())
    constant_names = {"LEG_JOINT_NAMES", "WHEEL_JOINT_NAMES", "ACTION_JOINT_NAMES", "BODY_NAMES",
                      "NON_WHEEL_BODY_NAMES", "WHEEL_BODY_NAMES", "POLICY_FPS", "LEG_ACTION_SCALE"}
    selected = [node for node in tree.body if
        (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constant_names for t in node.targets))
        or (isinstance(node, ast.FunctionDef) and node.name == "observation_action_contract")]
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<pure actual contract>", "exec"), namespace)
    result = namespace["observation_action_contract"]()
    from training.tron1_domain_randomization import default_profile
    result.update(task_profile=copy.deepcopy(profile), domain_randomization=default_profile(42))
    return result


@pytest.fixture
def context(tmp_path, monkeypatch):
    """Generated complete metadata, tokens only: no policy/checkpoint/physics calls."""
    from test_publish_car_drift import reserve_audit
    from training.tron1_domain_randomization import default_profile
    root = tmp_path / "repository"
    trial = root / "outputs/drift-sim2sim-20261012/trial013"
    sources = {**publish.BASE_SOURCE_PINS, **publish.NEW_SOURCE_PINS}
    for relative in sources:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(ROOT / relative)
    for filename in ("car_drift_contact_task.json", "car_drift_ground_load_task.json"):
        path = root / "config" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / "config" / filename).read_bytes())
    original_profile = publish.base.read_json(root / "config/car_drift_contact_task.json")
    profile = publish.base.read_json(root / "config/car_drift_ground_load_task.json")
    baseline = {"task_profile": original_profile, "contract": full_training_contract(original_profile),
                "domain_randomization_profile": default_profile(42)}
    baseline_path = write(root / "outputs/drift-sim2sim-20261011/trial004/training/manifest.json", baseline)
    monkeypatch.setattr(publish, "PARENT_MANIFEST_SHA256", publish.base.sha256(baseline_path))
    training_contract = copy.deepcopy(baseline["contract"]); training_contract["task_profile"] = profile
    native = training_contract["body_names"][:3] + ["limx_imu"] + training_contract["body_names"][3:]
    runner_path = write(trial / "training/runner_config.json", {"num_steps_per_env": 24})
    monkeypatch.setattr(publish, "RUNNER_SHA256", publish.base.sha256(runner_path))
    protocol = publish.base.read_json(ROOT / "config/car_drift_lr_sync_ground_load_trial.json")
    protocol.update(trial_id="trial013", experiment_type=publish.EXPERIMENT_KIND,
        task_profile="config/car_drift_ground_load_task.json", task_profile_sha256=publish.PROFILE_SHA256,
        publication_entrypoint="scripts/publish_car_drift_lr_sync_ground_load.py")
    protocol["training"]["parent_manifest_sha256"] = publish.PARENT_MANIFEST_SHA256
    protocol["training"]["runner_config_sha256"] = publish.RUNNER_SHA256
    protocol_path = write(root / "config/car_drift_lr_sync_ground_load_trial.json", protocol)
    checkpoint = trial / "training/model_final.pt"; checkpoint.write_bytes(b"GENERATED final checkpoint token")
    actor = trial / "isaac/actor_normalized.pt"; actor.parent.mkdir(parents=True, exist_ok=True)
    actor.write_bytes(b"GENERATED actor token, NEVER EXECUTED")
    cp_sha, actor_sha = publish.base.sha256(checkpoint), publish.base.sha256(actor)
    old_export = copy.deepcopy(baseline["contract"])
    old_export.update(domain_randomization="disabled during evaluation; fixed actuator parameters",
        actor_file="actor_normalized.pt", actor_includes_observation_normalizer=True,
        actor_output_is_unclipped=True, actor_observation_groups=["policy"],
        export_validation_max_abs_error=0., actual_joint_order=old_export["observed_joint_velocity_order"],
        actual_body_order=native, reference_file_sha256=publish.REFERENCE_PINS["motion_sha256"],
        checkpoint_sha256=publish.PARENT_CP_SHA256, default_joint_position_rad=[0.] * 8,
        training_domain_randomization=default_profile(42))
    old_export_path = write(root / "outputs/drift-sim2sim-20261011/trial004/isaac/policy_contract.json", old_export)
    monkeypatch.setattr(publish, "PARENT_EXPORT_SHA256", publish.base.sha256(old_export_path))
    contract = copy.deepcopy(old_export); contract.update(task_profile=profile, checkpoint_sha256=cp_sha)
    audit = lr_audit()
    def material(n, tracks):
        return {"leg_pd_gain_readback": {"kp_range": [500., 500.], "kd_range": [10., 10.]},
            "ground_colliders": [{"collider": "/World/ground/terrain/GroundPlane/CollisionPlane"}],
            "ground_load_reserve": reserve_audit(n, tracks)}
    manifest = {"source_sha256": sources, "versions": {"rsl-rl-lib": "3.0.1"},
        "resume_learning_rate_audit": audit,
        "resume_source_audit": {"schema_version": 1, "rsl_rl_version": "3.0.1",
            "source_root": "/data/private-account/conda/rsl_rl", "expected_source_sha256": publish.RSL_SOURCE_PINS,
            "actual_source_sha256": publish.RSL_SOURCE_PINS, "verified_before_runner_load": True},
        "contract": training_contract, "domain_randomization_profile": baseline["domain_randomization_profile"],
        "task_profile_sha256": publish.PROFILE_SHA256, "task_profile": profile, "num_envs": 2048,
        "iterations_requested": 600, "seed": 42, "smoke_steps_passed": 8, "body_names": native,
        "resume_sha256": publish.PARENT_CP_SHA256, "contact_material_audit": material(2048, False),
        "motion_sha256": publish.REFERENCE_PINS["motion_sha256"], "asset_sha256": publish.REFERENCE_PINS["robot_asset_sha256"]}
    records = {"manifest": manifest, "contract": contract,
        "isaac_report": {"task_profile": profile, "checkpoint_sha256": cp_sha, "contact_material_audit": material(1, True),
            "asset_root_file_sha256": publish.REFERENCE_PINS["robot_asset_sha256"], "policy_export": contract,
            "domain_parameters": None, "evaluator_source_sha256": publish.base.GROUND_LOAD_RESERVE_SOURCE_PINS["scripts/eval_tracking.py"],
            "drift_module_sha256": sources["training/tron1_drift.py"],
            "contact_slip_helper_sha256": sources["training/tron1_contact_slip.py"],
            "ground_load_reserve_module_sha256": publish.RESERVE_SHA256, "physics_recording": {"samples": 1153}},
        "mujoco_report": {"contract": contract, "model_sha256": publish.REFERENCE_PINS["mujoco_model_sha256"],
            "domain_parameters": {"is_nominal": True, "joint_names": contract["observed_joint_velocity_order"],
                "torque_scale": [1.] * 8, "velocity_scale": [1.] * 8, "wheel_friction_nm": [0., 0.],
                "friction_smoothing_rad_s": .5},
            "contact_material_audit": {"contact_margin_and_gap_changed": False, "explicit_contact_solref": [.04, 2.],
                "compiled_solref_ranges": [[.04, .04], [2., 2.]]},
            "physics_steps": 1152, "policy_steps": 288, "root_state_writes": 1, "hidden_resets": 0,
            "physics_stepped": True, "policy_sha256": actor_sha},
        "runner": {"num_steps_per_env": 24},
        "run_status": {"status": "training_completed", "num_envs": 2048, "iterations_requested": 600,
            "training_environment_steps": 29_491_200, "last_iteration": 4595,
            "checkpoint": "/data/private-account/run/outputs/trial013/training/model_final.pt"},
        "assessment": {"provenance_checks": {"same_physics_contact_sampling_clock": {"passed": True}}},
        "adaptation": {"model_sha256": publish.base.GROUND_LOAD_RESERVE_ASSET_PINS["reference_model"]},
        "export_report": {"model_sha256": publish.base.GROUND_LOAD_RESERVE_ASSET_PINS["reference_model"]}}
    names = {"manifest": "training/manifest.json", "contract": "isaac/policy_contract.json",
        "isaac_report": "isaac/report.json", "mujoco_report": "mujoco/report.json", "run_status": "training/run_status.json",
        "assessment": "assessment.json", "adaptation": "reference/adaptation.json", "export_report": "reference/motion.json"}
    files = {key: write(trial / relative, records[key]) for key, relative in names.items()}
    files.update(actor=actor, final_checkpoint=checkpoint, runner=runner_path)
    for relative in (*publish.FINAL_INPUTS, "training/resume_learning_rate_audit.json"):
        path = trial / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("GENERATED archive token: " + relative).encode())
    files.update(isaac_physics_trajectory=trial / "isaac/physics_trajectory.npz", mujoco_trajectory=trial / "mujoco/rollout.npz")
    hashes = {key: publish.base.sha256(path) for key, path in files.items()}
    hashes.update(motion=publish.REFERENCE_PINS["motion_sha256"], robot_reference=publish.REFERENCE_PINS["robot_reference_sha256"])
    completion = {"schema_version": 1, "status": "training_completion_verified", "final_checkpoint_sha256": cp_sha,
        "original_ground_load_profile_and_explicit_ten_training_sources_verified": True,
        "all_input_hashes_unchanged": True, "new_actor_inference_calls": 0, "new_optimizer_updates": 0,
        "simulation_steps": 0, "cuda_allocations": 0, "behavior_verified": False, "hardware_ready": False,
        "actual_adam_step_increment": 12000, "actual_actor_and_critic_normalizer_count_increment": 29_491_200,
        "resume_learning_rate_audit": audit, "resume_source_audit": manifest["resume_source_audit"],
        "training_budget": {"updates": 600, "num_envs": 2048, "rollout_steps_per_update": 24,
            "seed": 42, "additional_environment_steps": 29_491_200, "saved_iteration": 4595}}
    for name, iteration, step, count in (("parent_checkpoint_health", 3996, 80060, 145072128),
                                        ("final_checkpoint_health", 4595, 92060, 174563328)):
        completion[name] = {"saved_iteration": iteration, "model_tensor_count": 25, "optimizer_tensor_count": 51,
            "adam_parameter_state_count": 17, "every_model_optimizer_tensor_finite": True,
            "adam_parameter_steps": [step] * 17,
            "normalizer_counts": {"actor_obs_normalizer.count": count, "critic_obs_normalizer.count": count}}
    checks = {k: True for k in publish.PAIRED_CHECK_NAMES}
    proof = {"schema_version": 1, "status": "deployment_provenance_pass", "checks": checks,
        "new_rollouts": 0, "simulation_steps": 0, "behavior_success_claimed": False,
        "resume_learning_rate_audit": audit, "resume_source_audit": manifest["resume_source_audit"],
        "input_sha256": {relative: publish.base.sha256(trial / relative) for relative in publish.FINAL_INPUTS},
        "checkpoint_tensor_health": {"model_state_dict": {"tensor_count": 25, "all_finite": True},
                                    "optimizer_state_dict": {"tensor_count": 51, "all_finite": True}},
        "actor_export": {"actor_sha256": actor_sha, "includes_normalizer": True, "tensors_compared_bit_exact": 12,
            "recorded_pre_action_rows_checked": 288, "baseline_actor_equivalence_required": False},
        "physics_provenance": {"isaac": {"file": "physics_trajectory.npz", "kind": "physics_rate", "sample_dt_s": .005,
            "sha256": hashes["isaac_physics_trajectory"]},
            "mujoco": {"file": "rollout.npz", "sample_dt_s": .005, "samples": 1153, "sha256": hashes["mujoco_trajectory"]}}}
    for engine in ("isaac", "mujoco"):
        proof[engine + "_actor_replay"] = {"rows": 288, "predeclared_absolute_tolerance": 1.e-5,
                                          "max_abs_raw_error": 3.e-7, "max_abs_clipped_error": 3.e-7}
    dedicated = write(trial / "training/resume_learning_rate_audit.json", audit)
    completion["input_sha256"] = {"training/" + name: publish.base.sha256(trial / "training" / name)
        for name in ("manifest.json", "run_status.json", "runner_config.json", "model_final.pt", "resume_learning_rate_audit.json")}
    completion["input_sha256"].update({"repository/" + key: value for key, value in sources.items()})
    completion["input_sha256"]["repository/config/car_drift_lr_sync_ground_load_trial.json"] = publish.base.sha256(protocol_path)
    files["lr_sync_protocol"], hashes["lr_sync_protocol"] = protocol_path, publish.base.sha256(protocol_path)
    complete_path = write(trial / "diagnostics/training_completion_verification.json", completion)
    proof["training_completion_proof_sha256"] = publish.base.sha256(complete_path)
    proof["input_sha256"].update({"training/resume_learning_rate_audit.json": publish.base.sha256(dedicated),
        "diagnostics/training_completion_verification.json": publish.base.sha256(complete_path),
        "repository/config/car_drift_lr_sync_ground_load_trial.json": publish.base.sha256(protocol_path)})
    proof["input_sha256"].update({"repository/" + key: value for key, value in sources.items()})
    proof_path = write(trial / "diagnostics/deployment_provenance_audit.json", proof)
    return {"records": records, "files": files, "hashes": hashes, "protocol": protocol, "lr_audit": audit,
        "completion": completion, "proof": proof, "repo_root": root, "expected_source_sha256": sources,
        "trial": trial, "protocol_path": protocol_path, "proof_path": proof_path}

def validate(c):
    keys = ("records", "files", "hashes", "protocol", "lr_audit", "completion", "proof", "repo_root", "expected_source_sha256")
    return publish.validate_context(**{k: c[k] for k in keys})


@pytest.fixture
def prepared_context(context, monkeypatch):
    """Stub only existing rendering/archive preflight, not new context guards."""
    c = context
    payloads = {name: b"GENERATED base payload" for name in
                ("assessment.json", "reference_audit.json", "render_report.json")}
    media = {name: {"path": c["files"]["actor"], "sha256": c["hashes"]["actor"]}
             for name in ("motion_pipeline.mp4", "overview.png", "final.png", "contact_overview.png")}

    def base_candidate(*args):
        summary = {"status": "published_evidence", "verdict": "fail", "behavior_pass": False,
                   "failed_checks": ["GENERATED failing behavior"], "input_sha256": dict(c["hashes"])}
        return summary, dict(payloads), dict(media), dict(c["files"]), dict(c["hashes"])

    monkeypatch.setattr(publish.base, "prepare_publication", base_candidate)
    return c


def prepare(c, **overrides):
    args = {"protocol_path": c["protocol_path"],
            "expected_protocol_sha256": publish.base.sha256(c["protocol_path"]),
            "expected_source_sha256": c["expected_source_sha256"], "repo_root": c["repo_root"]}
    args.update(overrides)
    return publish.prepare_publication(c["trial"], c["repo_root"] / "reference", c["trial"] / "render", **args)


def repin_modified_records(c, *names):
    """Retain honest file seals so a semantic guard, not stale hashes, rejects."""
    for name in names:
        write(c["files"][name], c["records"][name])
        c["hashes"][name] = publish.base.sha256(c["files"][name])
    c["completion"]["input_sha256"]["training/manifest.json"] = c["hashes"]["manifest"]
    write(c["trial"] / "diagnostics/training_completion_verification.json", c["completion"])
    c["proof"]["training_completion_proof_sha256"] = publish.base.sha256(
        c["trial"] / "diagnostics/training_completion_verification.json")
    c["proof"]["input_sha256"].update({relative: publish.base.sha256(c["trial"] / relative)
        for relative in (*publish.FINAL_INPUTS, "training/resume_learning_rate_audit.json",
                         "diagnostics/training_completion_verification.json")})
    write(c["proof_path"], c["proof"])


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
    assert publish.base.sha256(ROOT / "scripts/publish_car_drift_lr_sync.py") == publish.LR_PUBLISHER_SHA256
    tree = ast.parse((ROOT / "scripts/publish_car_drift_lr_sync_ground_load.py").read_text())
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
    "training/model_final.pt", "training/resume_learning_rate_audit.json", "repository/config/car_drift_lr_sync_ground_load_trial.json",
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


def test_production_source10_defaults_are_reviewed_actual_files():
    assert len(publish.BASE_SOURCE_PINS) + len(publish.NEW_SOURCE_PINS) == 10
    for relative, expected in {**publish.BASE_SOURCE_PINS, **publish.NEW_SOURCE_PINS}.items():
        assert publish.base.sha256(ROOT / relative) == expected


def reserve(context, engine="manifest"):
    return context["records"][engine]["contact_material_audit"]["ground_load_reserve"]


@pytest.mark.parametrize("engine", ["manifest", "isaac_report"])
@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("module_sha256", "a" * 64),
    ("settings", {"weight": -1., "scale_fraction": .21}),
    ("settings", {"weight": -1., "scale_fraction": True}),
    ("settings", {"weight": -1., "scale_fraction": .2, "hidden": 0}),
    ("sampling", "all_substeps_mean"), ("formula", "exp(-force/threshold)"),
    ("force_evidence", "thresholded contact bearing patches"),
    ("physics_dt_s", .0025), ("policy_dt_s", .01), ("decimation", 4.),
    ("sensor_names", ["wheel_L_ground", "wheel_L_ground"]),
    ("wheel_body_names", ["wheel_R_Link", "wheel_L_Link"]),
    ("sensor_matrix_shape", [1, 2, 1, 3]), ("sensor_matrix_shape", None),
    ("sensor_matrix_shape", [2048., 1, 1, 3]),
    ("sensor_track_contact_points", [None, None]),
    ("native_mass_source", "cached default masses"), ("mass_reduction_dtype", "float32"),
    ("cached_mg_dtype", "float64"), ("cuda_device", "cpu"), ("cuda_device", "cuda:7"),
    ("num_envs", True), ("cost_bounds", [0., 1.]), ("cost_bounds", [False, 2.]),
    ("startup_force_finite_and_nonnegative_z", 1), ("startup_loaded_contact_required", True),
    ("raw_normal_forces_thresholded", True), ("scale_is_minimum_force", True),
    ("continuous_contact_certified", True), ("host_synchronization_free", True),
    ("reset_or_history_state_added", True), ("soft_balanced_load_preference", False),
    ("loaded_normal_vector_and_raw_aggregate_runtime_smoke_required", False),
    ("tracking_tradeoff_and_no_hardware_safety_claim", False), ("unsupported_claim", True),
])
def test_malformed_reserve_startup_schema_is_rejected(context, engine, field, value):
    reserve(context, engine)[field] = value
    with pytest.raises(ValueError):
        validate(context)


@pytest.mark.parametrize("engine", ["manifest", "isaac_report"])
@pytest.mark.parametrize("field,value", [
    ("mass_kg_range", [0., 0.]), ("mass_kg_range", [22.273, float("nan")]),
    ("mass_kg_range", [22.273, float("inf")]), ("mass_kg_range", [True, 22.273]),
    ("mass_kg_range", [24., 24.]), ("first_environment_link_mass_kg", [1.] * 9),
    ("first_environment_link_mass_kg", [1.] * 10),
    ("first_environment_link_mass_kg", [-1.] + [1.] * 9),
    ("mg_N_range", [218., 218.]), ("mg_N_range", [-218., -218.]),
    ("mg_N_range", [218., float("nan")]), ("configured_gravity_m_s2", [0., 0., 9.81]),
    ("native_gravity_m_s2", [0., 0., 0.]), ("native_gravity_m_s2", [1., 0., -9.81]),
    ("native_gravity_m_s2", [0., 0., -9.82]),
])
def test_reserve_live_mass_gravity_and_cached_weight_must_agree(context, engine, field, value):
    reserve(context, engine)[field] = value
    with pytest.raises(ValueError):
        validate(context)


@pytest.mark.parametrize("engine", ["manifest", "isaac_report"])
@pytest.mark.parametrize("field,value", [
    ("path", "/World/robot/wheel"), ("usd_type", "Cylinder"),
    ("collision_api_verified", 1), ("native_cooked_geometry_readback_performed", True),
    ("unsupported", True),
])
def test_reserve_filter_must_match_original_authored_ground(context, engine, field, value):
    reserve(context, engine)["ground_collider"][field] = value
    with pytest.raises(ValueError):
        validate(context)


@pytest.mark.parametrize("change", ["remove", "duplicate", "mass", "mode", "function", "class_instance", "extra"])
@pytest.mark.parametrize("engine", ["manifest", "isaac_report"])
def test_only_original_three_training_and_two_evaluation_events_are_supported(context, change, engine):
    events = reserve(context, engine)["active_event_identity_audit"]
    if change == "remove":
        events.pop()
    elif change == "duplicate":
        events[-1] = copy.deepcopy(events[0])
    elif change == "mass":
        events[-1]["name"] = "randomize_rigid_body_mass"
    elif change == "mode":
        events[0]["mode"] = "reset"
    elif change == "function":
        events[0]["function"] = "my_module.alternative_material"
    elif change == "class_instance":
        events[0]["initialized_exact_class_instance"] = 1
    else:
        events[0]["hidden"] = True
    with pytest.raises(ValueError):
        validate(context)


@pytest.mark.parametrize("engine,tracking", [("manifest", True), ("isaac_report", False)])
def test_training_force_only_and_evaluation_contact_point_modes_do_not_swap(context, engine, tracking):
    reserve(context, engine)["sensor_track_contact_points"] = [tracking, tracking]
    with pytest.raises(ValueError):
        validate(context)


def test_actual_native_ten_link_layout_not_nine_tracked_links(context):
    assert len(reserve(context)["first_environment_link_mass_kg"]) == 10
    assert len(context["records"]["contract"]["body_names"]) == 9
    assert len(context["records"]["contract"]["actual_body_order"]) == 10
    assert validate(context)["ground_load_reserve"]["startup_audits"]["training"]["num_envs"] == 2048


@pytest.mark.parametrize("usd_type", ["Mesh", "Plane"])
def test_audited_authored_ground_mesh_or_plane_is_allowed_without_cooked_equivalence_claim(context, usd_type):
    for engine in ("manifest", "isaac_report"):
        reserve(context, engine)["ground_collider"]["usd_type"] = usd_type
    result = validate(context)["ground_load_reserve"]
    assert result["native_cooked_geometry_equivalence_claimed"] is False
    assert result["full_training_force_validity_independently_proved"] is False


@pytest.mark.parametrize("field,value", [("friction_smoothing_rad_s", .1), ("friction_smoothing_rad_s", True),
    ("friction_smoothing_rad_s", float("nan")), ("resampled", True), ("is_nominal", 1),
    ("torque_scale", [True] * 8), ("wheel_friction_nm", [0., 1.])])
def test_actual_nominal_motors_cannot_change_or_hide_extra_draw_fields(context, field, value):
    context["records"]["mujoco_report"]["domain_parameters"][field] = value
    with pytest.raises(ValueError):
        validate(context)


@pytest.mark.parametrize("field", ["friction_smoothing_rad_s", "wheel_friction_nm"])
def test_nominal_motor_draw_requires_complete_schema(context, field):
    del context["records"]["mujoco_report"]["domain_parameters"][field]
    with pytest.raises(ValueError):
        validate(context)


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("original_ground_load_profile_and_explicit_ten_training_sources_verified", 1),
    ("original_ground_load_profile_and_explicit_ten_training_sources_verified", False),
    ("all_input_hashes_unchanged", 1), ("all_input_hashes_unchanged", False),
    ("new_actor_inference_calls", 1), ("new_actor_inference_calls", False),
    ("new_optimizer_updates", 1), ("simulation_steps", 1), ("cuda_allocations", 1),
    ("cuda_allocations", 0.), ("behavior_verified", True), ("hardware_ready", True),
    ("actual_adam_step_increment", 11999), ("actual_actor_and_critic_normalizer_count_increment", 29_491_199),
])
def test_completion_is_cpu_only_full_budget_source10_not_behavioral_proof(context, field, value):
    context["completion"][field] = value
    with pytest.raises(ValueError):
        validate(context)


@pytest.mark.parametrize("field", ["original_ground_load_profile_and_explicit_ten_training_sources_verified",
    "all_input_hashes_unchanged", "new_actor_inference_calls", "new_optimizer_updates", "simulation_steps", "cuda_allocations"])
def test_completion_requires_explicit_cpu_and_source10_flags(context, field):
    del context["completion"][field]
    with pytest.raises(ValueError):
        validate(context)


@pytest.mark.parametrize("change", ["unknown", "old_trial_name", "missing", "extra", "numeric_true"])
def test_fifteen_pair_checks_must_be_exact_producer_identities(context, change):
    checks = context["proof"]["checks"]
    key = "fresh013_final_CP_and_matching51_120_8_500_10_50_200_contracts_no_overrides"
    if change == "unknown":
        checks["unspecified_check"] = checks.pop(key)
    elif change == "old_trial_name":
        checks[key.replace("013", "011")] = checks.pop(key)
    elif change == "missing":
        del checks[key]
    elif change == "extra":
        checks["undocumented_extra_gate"] = True
    else:
        checks[key] = 1
    with pytest.raises(ValueError):
        validate(context)


@pytest.mark.parametrize("field,value", [("schema_version", True), ("new_rollouts", 1),
    ("new_rollouts", False), ("simulation_steps", 1), ("behavior_success_claimed", True)])
def test_recorded_actor_proof_cannot_add_rollouts_or_claim_success(context, field, value):
    context["proof"][field] = value
    with pytest.raises(ValueError):
        validate(context)


@pytest.mark.parametrize("relative", ["training/resume_learning_rate_audit.json", "diagnostics/training_completion_verification.json"])
def test_both_additional_actual_inputs_are_required_for_fourteen_input_proof(context, relative):
    del context["proof"]["input_sha256"][relative]
    with pytest.raises(ValueError):
        validate(context)


@pytest.mark.parametrize("engine,field,value", [
    ("isaac", "sample_dt_s", .02), ("isaac", "sample_dt_s", True), ("isaac", "kind", "legacy_policy_rate"),
    ("isaac", "file", "trajectory.npz"), ("isaac", "sha256", "a" * 64),
    ("mujoco", "sample_dt_s", .0025), ("mujoco", "samples", 1152), ("mujoco", "samples", 1153.),
    ("mujoco", "sha256", "a" * 64),
])
def test_actual_physics_archives_are_full_original_clock_hash_linked_evidence(context, engine, field, value):
    context["proof"]["physics_provenance"][engine][field] = value
    with pytest.raises(ValueError):
        validate(context)


@pytest.mark.parametrize("field,value", [("physics_steps", 1151), ("policy_steps", 287), ("root_state_writes", 2),
    ("hidden_resets", 1), ("physics_stepped", 1), ("policy_sha256", "b" * 64)])
def test_mujoco_rollout_has_no_hidden_resets_or_actor_substitution(context, field, value):
    context["records"]["mujoco_report"][field] = value
    with pytest.raises(ValueError):
        validate(context)


@pytest.mark.parametrize("field,value", [("trial_id", "trial012"), ("baseline_trial", "trial010"),
    ("parent_trial", "trial012"), ("reserve_helper_sha256", "a" * 64), ("expected_training_source_sha256", {}),
    ("reward", None), ("experiment_type", "nominal_unchanged_objective_continuation")])
def test_protocol_cannot_substitute_old_trial_wrong_source_or_missing_reserve(context, field, value):
    context["protocol"][field] = value
    with pytest.raises(ValueError):
        validate(context)


@pytest.mark.parametrize("field,value", [("weight", -.5), ("weight", True), ("scale_fraction_r", .1),
    ("cost", "minimum_normal_force"), ("sampling", "all_substeps_mean"), ("range", [False, 2.])])
def test_protocol_reserve_cost_scale_weight_and_last_sampling_are_unchanged(context, field, value):
    context["protocol"]["reward"][field] = value
    with pytest.raises(ValueError):
        validate(context)


@pytest.mark.parametrize("path_kind", ["absolute", "relative", "dot_segments", "symlink_inside"])
def test_protocol_resolves_relative_or_absolute_before_proof_seals(prepared_context, monkeypatch, path_kind):
    c = prepared_context
    monkeypatch.chdir(c["repo_root"])
    if path_kind == "absolute":
        protocol = c["protocol_path"]
        monkeypatch.chdir(c["repo_root"].parent)
    elif path_kind == "relative":
        protocol = c["protocol_path"].relative_to(c["repo_root"])
    elif path_kind == "dot_segments":
        protocol = Path("config/../config/car_drift_lr_sync_ground_load_trial.json")
    else:
        alias = c["repo_root"] / "protocol_alias.json"
        alias.symlink_to(c["protocol_path"])
        protocol = Path("protocol_alias.json")
    result, _, _, files, hashes = prepare(c, protocol_path=protocol, repo_root=Path(".") if path_kind != "absolute" else c["repo_root"])
    assert files["lr_sync_protocol"] == c["protocol_path"].resolve()
    assert hashes["lr_sync_protocol"] == publish.base.sha256(c["protocol_path"])
    assert result["experiment_kind"] == publish.EXPERIMENT_KIND
    assert result["verdict"] == "fail"


@pytest.mark.parametrize("path_kind", ["outside_absolute", "outside_symlink", "outside_dot_segments"])
def test_protocol_path_escaping_declared_repository_rejected_before_base(context, monkeypatch, path_kind):
    c = context
    outside = c["repo_root"].parent / "outside_protocol.json"
    outside.write_bytes(c["protocol_path"].read_bytes())
    monkeypatch.chdir(c["repo_root"])
    if path_kind == "outside_symlink":
        alias = c["repo_root"] / "outside_alias.json"
        alias.symlink_to(outside)
        protocol = Path("outside_alias.json")
    elif path_kind == "outside_dot_segments":
        protocol = Path("../outside_protocol.json")
    else:
        protocol = outside
    monkeypatch.setattr(publish.base, "prepare_publication", lambda *a: pytest.fail("Escaped protocol must fail before base"))
    with pytest.raises(ValueError, match="resolve inside"):
        prepare(c, protocol_path=protocol)


@pytest.mark.parametrize("failure", ["force_audit", "completion", "pair_checks", "source", "budget"])
def test_semantically_forged_but_resealed_metadata_fails_before_output_creation(prepared_context, failure):
    c = prepared_context
    if failure == "force_audit":
        reserve(c)["mg_N_range"] = [1., 1.]
    elif failure == "completion":
        c["completion"]["cuda_allocations"] = 1
    elif failure == "pair_checks":
        c["proof"]["checks"]["hidden_gate"] = c["proof"]["checks"].pop("all_input_hashes_unchanged_during_audit")
    elif failure == "source":
        c["records"]["manifest"]["source_sha256"].pop(publish.RESERVE_HELPER)
    else:
        c["records"]["run_status"]["training_environment_steps"] = 29_491_199
    repin_modified_records(c, "manifest", "run_status")
    destination = c["repo_root"] / "public" / "trial013"
    with pytest.raises(ValueError):
        publish.publish_trial(c["trial"], c["repo_root"] / "reference", destination, c["trial"] / "render",
            protocol_path=c["protocol_path"], expected_protocol_sha256=publish.base.sha256(c["protocol_path"]),
            expected_source_sha256=c["expected_source_sha256"], repo_root=c["repo_root"])
    assert not destination.exists() and not destination.parent.exists()


def test_generated_publication_is_exactly_eight_sanitized_files_not_raw_data(prepared_context):
    c = prepared_context
    destination = c["repo_root"] / "public" / "trial013"
    result = publish.publish_trial(c["trial"], c["repo_root"] / "reference", destination, c["trial"] / "render",
        protocol_path=c["protocol_path"], expected_protocol_sha256=publish.base.sha256(c["protocol_path"]),
        expected_source_sha256=c["expected_source_sha256"], repo_root=c["repo_root"])
    assert {path.name for path in destination.iterdir()} == {"summary.json", "assessment.json", "reference_audit.json",
        "render_report.json", "motion_pipeline.mp4", "overview.png", "final.png", "contact_overview.png"}
    assert result["verdict"] == "fail" and result["failed_checks"] == ["GENERATED failing behavior"]
    assert result["additional_training_updates"] == 600
    assert result["lr_synchronized_ground_load_reserve_context"]["comparison_trial"] == "trial012"
    assert result["ground_load_reserve_context"]["settings"] == publish.RESERVE_SETTINGS
    public_text = (destination / "summary.json").read_text()
    assert "/data/private-account" not in public_text and "private-account" not in public_text
    assert not any(path.suffix in {".npz", ".pt"} for path in destination.iterdir())


def test_standalone_cli_help_imports_without_repo_cwd(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/publish_car_drift_lr_sync_ground_load.py"), "--help"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--protocol-sha256" in result.stdout and "--render-dir" in result.stdout

