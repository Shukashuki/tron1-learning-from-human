"""Pure offline LR-and-substep fixtures; none is a training/physics result."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import publish_car_drift_lr_sync_substeps as publish


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



def sampling_audit(num_envs, steps, *, native_offset=2):
    """Generated scalar audit, not captured014 tensor or simulator evidence."""
    dt = struct.unpack("f", struct.pack("f", .005))[0]
    start = 4 * steps - 4 + native_offset
    identity = {"schema_version": 1, "mode": "all_substeps_mean", "wheel_body_ids": (8, 9),
                "decimation": 4, "physics_dt_s": .005, "policy_dt_s": .02, "num_envs": num_envs}
    return {"schema_version": 1, "mode": "all_substeps_mean",
        "fingerprint_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        "installed": True, "inside_step": False, "sampling_open": False, "num_envs": num_envs,
        "wheel_body_ids": [8, 9], "device": "cuda:5", "dtype": "torch.float32", "decimation": 4,
        "physics_dt_s": .005, "policy_dt_s": .02, "steps_started": steps, "steps_completed": steps,
        "steps_failed": 0, "samples_last_control_step": 4, "samples_total": 4 * steps,
        "last_start_counter": 4 * steps - 4, "last_sample_counter": 4 * steps,
        "last_step_completed": True, "native_clock_required": True,
        "native_start_step_index": start, "native_last_step_index": start + 4,
        "native_start_time_s": start * dt, "native_last_time_s": (start + 4) * dt,
        "native_time_float32_dt_rounding_bound_s": abs(dt - .005),
        "native_time_double_rounding_allowance_ulps": 8, "cost_reads": steps}


def native_device_audit():
    """Canonical public stdlib validator only, never a live CUDA call."""
    from training.tron1_runtime_device import validate_device_audit
    return validate_device_audit({"requested_device": "cuda:5", "environment_device": "cuda:5",
        "physics_sim_view_frontend_device": "cuda:5", "robot_native_transform_tensor_device": "cuda:5",
        "native_physics_backend_device_ordinal": 5, "torch_current_device": 5, "physics_cudaDevice_setting": 5,
        "physics_sceneMultiGPUMode_setting": 0, "cuda_visible_devices": None})


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
    from training.tron1_domain_randomization import default_profile
    root = tmp_path / "repository"
    trial = root / "outputs/drift-sim2sim-20261012/trial014"
    sources = {**publish.BASE_SOURCE_PINS, **publish.NEW_SOURCE_PINS}
    for relative in sources:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(ROOT / relative)
    for filename in ("car_drift_contact_task.json", "car_drift_substep_task.json"):
        path = root / "config" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / "config" / filename).read_bytes())
    original_profile = publish.base.read_json(root / "config/car_drift_contact_task.json")
    profile = publish.base.read_json(root / "config/car_drift_substep_task.json")
    baseline = {"task_profile": original_profile, "contract": full_training_contract(original_profile),
                "domain_randomization_profile": default_profile(42)}
    baseline_path = write(root / "outputs/drift-sim2sim-20261011/trial004/training/manifest.json", baseline)
    monkeypatch.setattr(publish, "PARENT_MANIFEST_SHA256", publish.base.sha256(baseline_path))
    training_contract = copy.deepcopy(baseline["contract"]); training_contract["task_profile"] = profile
    native = training_contract["body_names"][:3] + ["limx_imu"] + training_contract["body_names"][3:]
    runner_path = write(trial / "training/runner_config.json", {"num_steps_per_env": 24})
    monkeypatch.setattr(publish, "RUNNER_SHA256", publish.base.sha256(runner_path))
    protocol = publish.base.read_json(ROOT / "config/car_drift_lr_sync_substeps_trial.json")
    protocol.update(trial_id="trial014", experiment_type=publish.EXPERIMENT_KIND,
        task_profile="config/car_drift_substep_task.json", task_profile_sha256=publish.PROFILE_SHA256,
        publication_entrypoint="scripts/publish_car_drift_lr_sync_substeps.py")
    protocol["training"]["parent_manifest_sha256"] = publish.PARENT_MANIFEST_SHA256
    protocol["training"]["runner_config_sha256"] = publish.RUNNER_SHA256
    protocol_path = write(root / "config/car_drift_lr_sync_substeps_trial.json", protocol)
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
        return {"expected_friction": .15, "robot_runtime_friction_range": [.15, .15],
            "leg_pd_gain_readback": {"kp_range": [500., 500.], "kd_range": [10., 10.]},
            "ground_colliders": [{"collider": "/World/ground/terrain/GroundPlane/CollisionPlane",
                                 "static_friction": .15, "dynamic_friction": .15}]}
    manifest = {"source_sha256": sources, "versions": {"rsl-rl-lib": "3.0.1"},
        "resume_learning_rate_audit": audit,
        "resume_source_audit": {"schema_version": 1, "rsl_rl_version": "3.0.1",
            "source_root": "/data/private-account/conda/rsl_rl", "expected_source_sha256": publish.RSL_SOURCE_PINS,
            "actual_source_sha256": publish.RSL_SOURCE_PINS, "verified_before_runner_load": True},
        "contract": training_contract, "domain_randomization_profile": baseline["domain_randomization_profile"],
        "task_profile_sha256": publish.PROFILE_SHA256, "task_profile": profile, "num_envs": 2048,
        "iterations_requested": 600, "seed": 42, "smoke_steps_passed": 8, "body_names": native,
        "resume_sha256": publish.PARENT_CP_SHA256, "contact_material_audit": material(2048, False),
        "motion_sha256": publish.REFERENCE_PINS["motion_sha256"], "asset_sha256": publish.REFERENCE_PINS["robot_asset_sha256"],
        "reward_sampling_audit": sampling_audit(2048, 8), "runtime_device_audit": native_device_audit()}
    records = {"manifest": manifest, "contract": contract,
        "isaac_report": {"task_profile": profile, "checkpoint_sha256": cp_sha, "contact_material_audit": material(1, True),
            "asset_root_file_sha256": publish.REFERENCE_PINS["robot_asset_sha256"], "policy_export": contract,
            "domain_parameters": None, "evaluator_source_sha256": publish.base.GROUND_LOAD_RESERVE_SOURCE_PINS["scripts/eval_tracking.py"],
            "drift_module_sha256": sources["training/tron1_drift.py"],
            "contact_slip_helper_sha256": sources["training/tron1_contact_slip.py"],
            "substep_reward_module_sha256": publish.SAMPLING_SHA256, "physics_recording": {"samples": 1153},
            "reward_sampling_audit": sampling_audit(1, 288), "runtime_device_audit": native_device_audit(),
            "runtime_device_module_sha256": publish.BASE_SOURCE_PINS["training/tron1_runtime_device.py"]},
        "mujoco_report": {"contract": contract, "model_sha256": publish.REFERENCE_PINS["mujoco_model_sha256"],
            "domain_parameters": {"is_nominal": True, "joint_names": contract["observed_joint_velocity_order"],
                "torque_scale": [1.] * 8, "velocity_scale": [1.] * 8, "wheel_friction_nm": [0., 0.],
                "friction_smoothing_rad_s": .5},
            "contact_material_audit": {"expected_friction": .15, "compiled_friction_range": [.15, .15], "colliding_geom_count": 9,
                "contact_margin_and_gap_changed": False, "explicit_contact_solref": [.04, 2.],
                "compiled_solref_ranges": [[.04, .04], [2., 2.]]},
            "physics_steps": 1152, "policy_steps": 288, "root_state_writes": 1, "hidden_resets": 0,
            "physics_stepped": True, "policy_sha256": actor_sha},
        "runner": {"num_steps_per_env": 24},
        "run_status": {"status": "training_completed", "num_envs": 2048, "iterations_requested": 600,
            "training_environment_steps": 29_491_200, "last_iteration": 4595,
            "checkpoint": "/data/private-account/run/outputs/trial014/training/model_final.pt"},
        "assessment": {"provenance_checks": {"same_physics_contact_sampling_clock": {"passed": True}}},
        "adaptation": {"model_sha256": publish.base.GROUND_LOAD_RESERVE_ASSET_PINS["reference_model"]},
        "export_report": {"model_sha256": publish.base.GROUND_LOAD_RESERVE_ASSET_PINS["reference_model"]}}
    final_sampling = sampling_audit(2048, 14408)
    write(trial / "training/reward_sampling_final_audit.json", final_sampling)
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
        "archive_sha256": "b" * 64,
        "original_substep_profile_and_explicit_ten_training_sources_verified": True,
        "all_input_hashes_unchanged": True, "new_actor_inference_calls": 0, "new_optimizer_updates": 0,
        "simulation_steps": 0, "cuda_allocations": 0, "behavior_verified": False, "hardware_ready": False,
        "reward_sampling_initial_audit": manifest["reward_sampling_audit"], "reward_sampling_final_audit": final_sampling,
        "substep_reward_module_sha256": publish.SAMPLING_SHA256,
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
        "reward_sampling_initial_audit": manifest["reward_sampling_audit"], "reward_sampling_final_audit": final_sampling,
        "reward_sampling_evaluation_audit": records["isaac_report"]["reward_sampling_audit"],
        "substep_reward_module_sha256": publish.SAMPLING_SHA256,
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
        for name in ("manifest.json", "run_status.json", "runner_config.json", "model_final.pt", "resume_learning_rate_audit.json",
                     "reward_sampling_final_audit.json")}
    completion["input_sha256"].update({"repository/" + key: value for key, value in sources.items()})
    completion["input_sha256"]["repository/config/car_drift_lr_sync_substeps_trial.json"] = publish.base.sha256(protocol_path)
    files["lr_sync_protocol"], hashes["lr_sync_protocol"] = protocol_path, publish.base.sha256(protocol_path)
    smoke_inputs = {**sources, **publish.SMOKE_SOURCE_PINS,
        **{name: "c" * 64 for name in ("smoke/launch_manifest.json", "smoke/science_report.json",
            "smoke/window_evidence.npz", "smoke/teardown_report.json", "smoke_launch_receipt.json",
            "smoke_started.json", "smoke_exit_report.json", "smoke.log")},
        "deployment.tar": completion["archive_sha256"], "config/car_drift_substep_task.json": publish.PROFILE_SHA256,
        "config/car_drift_contact_task.json": publish.ORIGINAL_PROFILE_SHA256,
        "config/car_drift_lr_sync_substeps_trial.json": hashes["lr_sync_protocol"]}
    completion["current_stack_smoke_prerequisite"] = {"status": "current_stack_smoke_prerequisite_verified",
        "smoke_verification_sha256": "d" * 64, "input_sha256": smoke_inputs}
    proof["current_stack_smoke_prerequisite"] = copy.deepcopy(completion["current_stack_smoke_prerequisite"])
    complete_path = write(trial / "diagnostics/training_completion_verification.json", completion)
    proof["training_completion_proof_sha256"] = publish.base.sha256(complete_path)
    proof["input_sha256"].update({"training/resume_learning_rate_audit.json": publish.base.sha256(dedicated),
        "diagnostics/training_completion_verification.json": publish.base.sha256(complete_path),
        "repository/config/car_drift_lr_sync_substeps_trial.json": publish.base.sha256(protocol_path)})
    proof["input_sha256"].update({"repository/" + key: value for key, value in sources.items()})
    proof_path = write(trial / "diagnostics/deployment_provenance_audit.json", proof)
    return {"records": records, "files": files, "hashes": hashes, "protocol": protocol, "lr_audit": audit,
        "completion": completion, "proof": proof, "sampling_final": final_sampling, "repo_root": root, "expected_source_sha256": sources,
        "trial": trial, "protocol_path": protocol_path, "proof_path": proof_path}

def validate(c):
    keys = ("records", "files", "hashes", "protocol", "lr_audit", "completion", "proof", "sampling_final", "repo_root", "expected_source_sha256")
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


@pytest.mark.parametrize("field", ["controller_gain_diagnostic", "physics_clock_refinement", "ground_load_reserve_module_sha256"])
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
    assert publish.base.sha256(ROOT / "scripts/publish_car_drift_lr_sync_ground_load.py") == publish.GROUND_LOAD_PUBLISHER_SHA256
    tree = ast.parse((ROOT / "scripts/publish_car_drift_lr_sync_substeps.py").read_text())
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
    "training/model_final.pt", "training/resume_learning_rate_audit.json", "repository/config/car_drift_lr_sync_substeps_trial.json",
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
        protocol = Path("config/../config/car_drift_lr_sync_substeps_trial.json")
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


def test_standalone_cli_help_imports_without_repo_cwd(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/publish_car_drift_lr_sync_substeps.py"), "--help"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--protocol-sha256" in result.stdout and "--render-dir" in result.stdout


@pytest.mark.parametrize("num_envs,steps", [(2048, 8), (2048, 14408), (1, 288), (8, 16)])
@pytest.mark.parametrize("native_offset", [0, 2, 20])
def test_exact_native_sampler_stage_accepts_arbitrary_native_epoch_not_counter_gaps(num_envs, steps, native_offset):
    audit = sampling_audit(num_envs, steps, native_offset=native_offset)
    checked = publish.validate_sampling_audit(audit, num_envs=num_envs, expected_steps=steps, body_names=publish.NATIVE_BODY_NAMES)
    assert checked == audit and checked is not audit
    checked["wheel_body_ids"][0] = 0
    assert audit["wheel_body_ids"] == [8, 9]


def test_computed_fingerprints_match_frozen_aeec_configuration():
    assert sampling_audit(2048, 8)["fingerprint_sha256"] == "69a3f7bc62c94d9aa6ad9d56d7afc484c8faa52adabcabf4dd4a0a35c5885cd4"
    assert sampling_audit(1, 288)["fingerprint_sha256"] == "f0a453323f800b028f8100860ccf931775173b0d2eb01309476d90c8f96bf5b6"


@pytest.mark.parametrize("field", sorted(publish.SAMPLING_AUDIT_KEYS))
def test_every_frozen_sampler_audit_field_is_required(field):
    audit = sampling_audit(2048, 8); del audit[field]
    with pytest.raises(ValueError):
        publish.validate_sampling_audit(audit, num_envs=2048, expected_steps=8, body_names=publish.NATIVE_BODY_NAMES)


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("mode", "last_substep"), ("fingerprint_sha256", "a" * 64),
    ("num_envs", 2048.), ("num_envs", True), ("wheel_body_ids", [8., 9]),
    ("wheel_body_ids", [9, 8]), ("wheel_body_ids", [8, 8]), ("wheel_body_ids", [True, 9]),
    ("wheel_body_ids", [8, 9, 9]), ("device", "cuda:7"), ("device", "cpu"),
    ("dtype", "torch.float64"), ("decimation", 4.), ("decimation", 8),
    ("physics_dt_s", .0025), ("policy_dt_s", .01), ("physics_dt_s", float("nan")),
    ("steps_started", 9), ("steps_completed", 7), ("steps_failed", False), ("steps_failed", 1),
    ("samples_last_control_step", 3), ("samples_total", 33), ("cost_reads", 7),
    ("last_start_counter", 27), ("last_sample_counter", 33),
    ("native_start_step_index", -1), ("native_start_step_index", 30.),
    ("native_last_step_index", 35), ("native_last_step_index", 34.),
    ("native_start_time_s", True), ("native_start_time_s", -1.),
    ("native_last_time_s", float("inf")), ("native_last_time_s", float("nan")),
    ("native_last_time_s", .17 + 1e-7), ("native_last_time_s", .15),
    ("native_time_float32_dt_rounding_bound_s", 1e-8),
    ("native_time_double_rounding_allowance_ulps", 8.),
    ("native_time_double_rounding_allowance_ulps", 9), ("unknown_claim", True),
])
def test_malformed_sampler_counter_clock_dtype_device_and_unknown_fields_rejected(field, value):
    audit = sampling_audit(2048, 8); audit[field] = value
    with pytest.raises(ValueError):
        publish.validate_sampling_audit(audit, num_envs=2048, expected_steps=8, body_names=publish.NATIVE_BODY_NAMES)


@pytest.mark.parametrize("field,expected", [("installed", True), ("inside_step", False), ("sampling_open", False),
    ("last_step_completed", True), ("native_clock_required", True)])
@pytest.mark.parametrize("replacement", [None, 0, 1])
def test_sampler_boolean_lifecycle_flags_are_not_numeric(field, expected, replacement):
    audit = sampling_audit(2048, 8); audit[field] = replacement
    with pytest.raises(ValueError):
        publish.validate_sampling_audit(audit, num_envs=2048, expected_steps=8, body_names=publish.NATIVE_BODY_NAMES)


@pytest.mark.parametrize("num_envs,steps", [(2048., 8), (True, 8), (1, 287), (2048, 14400), (8, 17), (1024, 8)])
def test_no_alternate_sampler_stage_or_float_argument(num_envs, steps):
    with pytest.raises(ValueError):
        publish.validate_sampling_audit(sampling_audit(2048, 8), num_envs=num_envs, expected_steps=steps,
                                        body_names=publish.NATIVE_BODY_NAMES)


def test_sampler_ids_use_actual_native_ten_links_not_nine_motion_links():
    wrong = publish.NATIVE_BODY_NAMES[:3] + publish.NATIVE_BODY_NAMES[4:]
    with pytest.raises(ValueError):
        publish.validate_sampling_audit(sampling_audit(2048, 8), num_envs=2048, expected_steps=8, body_names=wrong)


def test_native_time_float32_quantization_allowed_but_clock_refinement_not_allowed():
    audit = sampling_audit(2048, 8)
    checked = publish.validate_sampling_audit(audit, num_envs=2048, expected_steps=8, body_names=publish.NATIVE_BODY_NAMES)
    assert checked["native_last_time_s"] - checked["native_start_time_s"] != .02
    audit["native_last_time_s"] += 2e-9
    with pytest.raises(ValueError):
        publish.validate_sampling_audit(audit, num_envs=2048, expected_steps=8, body_names=publish.NATIVE_BODY_NAMES)


@pytest.mark.parametrize("container", ["startup", "final", "evaluation"])
def test_all_three_actual_sampler_stages_are_required(context, container):
    if container == "startup": context["records"]["manifest"]["reward_sampling_audit"] = None
    elif container == "evaluation": context["records"]["isaac_report"]["reward_sampling_audit"] = None
    else: context["sampling_final"] = None
    with pytest.raises(ValueError): validate(context)


@pytest.mark.parametrize("container", ["startup", "final", "evaluation"])
def test_stage_counts_cannot_be_reused_for_another_phase(context, container):
    if container == "startup": context["records"]["manifest"]["reward_sampling_audit"] = sampling_audit(2048, 14408)
    elif container == "evaluation": context["records"]["isaac_report"]["reward_sampling_audit"] = sampling_audit(2048, 8)
    else: context["sampling_final"] = sampling_audit(2048, 8)
    with pytest.raises(ValueError): validate(context)


@pytest.mark.parametrize("field,value", [("original_substep_profile_and_explicit_ten_training_sources_verified", 1),
    ("original_substep_profile_and_explicit_ten_training_sources_verified", False),
    ("original_ground_load_profile_and_explicit_ten_training_sources_verified", True),
    ("original_all_substep_profile_and_explicit_ten_training_sources_verified", True),
    ("original_nominal_profile_and_explicit_nine_training_sources_verified", True),
    ("all_input_hashes_unchanged", 1), ("new_actor_inference_calls", 1), ("new_optimizer_updates", 1),
    ("simulation_steps", 1), ("cuda_allocations", 1), ("cuda_allocations", False),
    ("behavior_verified", True), ("hardware_ready", True), ("substep_reward_module_sha256", "a" * 64),
    ("reward_sampling_initial_audit", {}), ("reward_sampling_final_audit", {})])
def test_cpu_only_verification_cannot_be_old_reserve_nominal_or_incomplete_sampler(context, field, value):
    context["completion"][field] = value
    with pytest.raises(ValueError): validate(context)


@pytest.mark.parametrize("relative", ["training/reward_sampling_final_audit.json", "training/resume_learning_rate_audit.json",
    "diagnostics/training_completion_verification.json"])
def test_all_fifteen_actual_input_seals_required(context, relative):
    del context["proof"]["input_sha256"][relative]
    with pytest.raises(ValueError): validate(context)


def test_cpu_six_training_file_seals_include_the_actual_final_sampling_json(context):
    del context["completion"]["input_sha256"]["training/reward_sampling_final_audit.json"]
    with pytest.raises(ValueError): validate(context)


@pytest.mark.parametrize("field", ["reward_sampling_initial_audit", "reward_sampling_final_audit",
    "reward_sampling_evaluation_audit", "substep_reward_module_sha256"])
def test_paired_proof_cannot_omit_actual_sampler_aliases(context, field):
    del context["proof"][field]
    with pytest.raises(ValueError): validate(context)


@pytest.mark.parametrize("change", ["missing", "status", "sha", "extra_key", "input_missing", "input_extra",
    "old005_protocol", "reserve_helper", "archive", "private_source", "paired_mismatch"])
def test_sealed_new_current_stack_smoke_prerequisite_not_old_or_orphan(context, change):
    data = context["completion"]["current_stack_smoke_prerequisite"]
    if change == "missing": context["completion"].pop("current_stack_smoke_prerequisite")
    elif change == "status": data["status"] = "smoke_verification_pass"
    elif change == "sha": data["smoke_verification_sha256"] = True
    elif change == "extra_key": data["hardware_ready"] = True
    elif change == "input_missing": data["input_sha256"].pop("smoke/window_evidence.npz")
    elif change == "input_extra": data["input_sha256"]["unapproved_future_record.npz"] = "a" * 64
    elif change == "old005_protocol": data["input_sha256"]["config/car_drift_lr_sync_substeps_trial.json"] = "a" * 64
    elif change == "reserve_helper": data["input_sha256"]["training/tron1_ground_load_reserve.py"] = "a" * 64
    elif change == "archive": data["input_sha256"]["deployment.tar"] = "a" * 64
    elif change == "private_source": data["input_sha256"]["verify_smoke.py"] = "a" * 64
    else: context["proof"]["current_stack_smoke_prerequisite"] = {}
    with pytest.raises(ValueError): validate(context)


@pytest.mark.parametrize("alias", ["training/unapproved_future.json", "isaac/second_episode.npz", "future_debug.npz"])
def test_undeclared_runtime_file_alias_rejected_even_if_hash_shaped(context, alias):
    context["proof"]["input_sha256"][alias] = "a" * 64
    with pytest.raises(ValueError): validate(context)


@pytest.mark.parametrize("change", ["unknown", "old013", "numeric_true", "missing"])
def test_exact_fifteen_producer_check_names_required(context, change):
    checks = context["proof"]["checks"]
    key = "fresh014_final_CP_and_matching51_120_8_500_10_50_200_contracts_no_overrides"
    if change == "unknown": checks["unverified_new_check"] = checks.pop(key)
    elif change == "old013": checks[key.replace("014", "013")] = checks.pop(key)
    elif change == "numeric_true": checks[key] = 1
    else: checks.pop(key)
    with pytest.raises(ValueError): validate(context)


@pytest.mark.parametrize("target", ["manifest", "isaac_report", "mujoco_report"])
def test_orphan_reserve_reward_cache_metadata_cannot_coexist(context, target):
    context["records"][target]["contact_material_audit"]["ground_load_reserve"] = {}
    with pytest.raises(ValueError): validate(context)


def test_reserve_helper_cannot_replace_sampler_even_with_exactly_ten_source_entries(context):
    value = context["records"]["manifest"]["source_sha256"].pop(publish.SAMPLING_HELPER)
    context["records"]["manifest"]["source_sha256"]["training/tron1_ground_load_reserve.py"] = value
    assert len(context["records"]["manifest"]["source_sha256"]) == 10
    with pytest.raises(ValueError): validate(context)


def test_historical005_context_without_lr_synchronization_is_not014(context):
    context["records"]["manifest"].pop("resume_learning_rate_audit")
    with pytest.raises(ValueError): validate(context)


@pytest.mark.parametrize("field,value", [("mode", "last_substep"), ("weight", -4.), ("std_m_s", .03),
    ("weight", True), ("substeps_per_control_window", 4.), ("physics_dt_s", .0025),
    ("initial_or_same_counter_reset_refresh_sampled", True), ("extra_key", True)])
def test_protocol_rejects_changed_sampling_or_unknown_schema(context, field, value):
    context["protocol"]["sampling"][field] = value
    with pytest.raises(ValueError): validate(context)


@pytest.mark.parametrize("field,value", [("wheel_vertical_velocity_weight", -4.),
    ("wheel_vertical_velocity_std_m_s", .04), ("wheel_vertical_velocity_sampling", "last_substep"),
    ("wheel_ground_load_reserve", {"weight": -1., "scale_fraction": .2})])
def test_exact_profile_cannot_mix_reward_factor_or_silent_scale_change(context, field, value):
    context["records"]["manifest"]["task_profile"][field] = value
    with pytest.raises(ValueError): validate(context)


@pytest.mark.parametrize("field,value", [("schema_version", True), ("requested_device", "cuda:7"),
    ("native_physics_backend_device_ordinal", 7), ("torch_current_device", True),
    ("physics_sceneMultiGPUMode_setting", None), ("cuda_visible_devices", "5"),
    ("physical_mapping_verified", True), ("exclusivity_verified", 0), ("unknown_runtime_claim", True)])
@pytest.mark.parametrize("record", ["manifest", "isaac_report"])
def test_native_device_audits_are_actual_unmasked_ordinal_only(context, record, field, value):
    context["records"][record]["runtime_device_audit"][field] = value
    with pytest.raises(ValueError): validate(context)


def test_import_remains_inert_without_torch_isaac_or_mujoco(tmp_path):
    code = f"import sys;sys.path.insert(0,{str(ROOT / 'scripts')!r});import publish_car_drift_lr_sync_substeps;assert not any(x in sys.modules for x in ('torch','isaaclab','mujoco','carb','omni'))"
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_forged_resealed_sampler_fails_before_output_directory_creation(prepared_context):
    c = prepared_context
    c["records"]["manifest"]["reward_sampling_audit"]["cost_reads"] = 7
    repin_modified_records(c, "manifest")
    destination = c["repo_root"] / "public" / "trial014"
    with pytest.raises(ValueError):
        publish.publish_trial(c["trial"], c["repo_root"] / "reference", destination, c["trial"] / "render",
            protocol_path=c["protocol_path"], expected_protocol_sha256=publish.base.sha256(c["protocol_path"]),
            repo_root=c["repo_root"])
    assert not destination.exists() and not destination.parent.exists()


def test_generated_publication_retains_three_base_payloads_four_media_and_honest_limits(prepared_context):
    c = prepared_context; destination = c["repo_root"] / "public" / "trial014"
    result = publish.publish_trial(c["trial"], c["repo_root"] / "reference", destination, c["trial"] / "render",
        protocol_path=c["protocol_path"], expected_protocol_sha256=publish.base.sha256(c["protocol_path"]), repo_root=c["repo_root"])
    assert {x.name for x in destination.iterdir()} == {"summary.json", "assessment.json", "reference_audit.json",
        "render_report.json", "motion_pipeline.mp4", "overview.png", "final.png", "contact_overview.png"}
    for name in ("assessment.json", "reference_audit.json", "render_report.json"):
        assert (destination / name).read_bytes() == b"GENERATED base payload"
    assert result["verdict"] == "fail" and result["experiment_kind"] == publish.EXPERIMENT_KIND
    ctx = result["substep_velocity_sampling_context"]
    assert ctx["settings"] == {"mode": "all_substeps_mean", "weight": -5., "std_m_s": .02}
    assert ctx["full_training_sample_values_independently_archived"] is False
    assert ctx["full_training_live_reward_values_independently_proved"] is False
    assert ctx["continuous_contact_certified"] is False
    assert result["learning_rate_resume_context"]["budget"]["updates"] == 600
    assert result["lr_synchronized_substep_velocity_context"]["comparison_trial"] == "trial012"
    assert "ground_load_reserve_context" not in result
    text = (destination / "summary.json").read_text()
    assert "/data/private-account" not in text and "private-account" not in text
    assert not any(x.suffix in {".npz", ".pt"} for x in destination.iterdir())
