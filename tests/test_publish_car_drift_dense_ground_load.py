"""Trial016 publication tests use synthetic metadata only; no actor or physics."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import publish_car_drift_dense_ground_load as p


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def completion():
    # Existing inert NumPy-only synthetic fixture, never a real future result.
    fixture = imported("synthetic016_completion_fixture", ROOT /
        "outputs/drift-trial016-deployment-20261012/test_verify_training_completion_trial016.py")
    value = fixture.completion_document()
    for alias, filename in value["input_files"].items():
        if filename in p.SOURCE17:
            value["input_sha256"][alias] = p.SOURCE17[filename]
    value["input_sha256"].update({"runtime/protocol": p.PROTOCOL_SHA256,
        "runtime/actual8/protocol": p.PROTOCOL_SHA256, "runtime/profile": p.PROFILE_SHA256,
        "runtime/actual8/profile": p.PROFILE_SHA256, "runtime/deployment_archive": p.ARCHIVE_SHA256,
        "runtime/actual8/deployment_archive": p.ARCHIVE_SHA256})
    filenames = {"launch_smoke_trial016.py": "58b75a6bf6fd8facdd9753818c71744d1e109c79d11a14b63a1560df70714b88",
        "verify_smoke_trial016.py": "017d723f7c7310511627d9fa4df7d771c7b91bd064bd86b3f5d15d811dd32ef1",
        "launch_runtime_trial016.py": "c9e2605b201a5d6140c51e797a6eac881c2747e5079747a5dd5c6d007e517106",
        "verify_runtime_trial016.py": "f9f08c6250de76926fa1631a69248fdf321713899cb9516520ca936ab3f8ea00",
        "config/car_drift_ground_load_task.json": p.strict.BASELINE013_PROFILE_SHA256,
        "config/car_drift_contact_task.json": p.strict.PARENT004_PROFILE_SHA256}
    for alias, name in value["input_files"].items():
        if name in filenames:
            value["input_sha256"][alias] = filenames[name]
    for key, prefix in (("current_stack_runtime_prerequisite", "runtime/"),
                        ("current_stack_smoke_prerequisite", "runtime/actual8/")):
        value[key]["input_sha256"] = {name[len(prefix):]: sha for name, sha in
            value["input_sha256"].items() if name.startswith(prefix)}
    return value


def test_frozen_source_profiles_and_explicit_protocol():
    assert p.sha256(ROOT / p.PROTOCOL_FILE) == p.PROTOCOL_SHA256
    protocol = p.read_json(ROOT / p.PROTOCOL_FILE)
    assert p.validate_protocol(protocol) is protocol
    assert len(p.SOURCE17) == 17 and len(p.TRAINING_SOURCE14) == 14
    assert protocol["expected_source_sha256"] == p.SOURCE17
    assert p.sha256(ROOT / p.PROFILE_FILE) == p.PROFILE_SHA256
    profile = p.read_json(ROOT / p.PROFILE_FILE)
    p.strict.validate_dense_load_profile(profile, p.read_json(ROOT / p.strict.BASELINE013_PROFILE_FILE))
    for filename, expected in p.SOURCE17.items():
        assert p.sha256(ROOT / filename) == expected
    assert p.sha256(p.base.__file__) == p.LEGACY_PUBLISHER_SHA256
    assert p.sha256(p.lr.__file__) == p.LR_PUBLISHER_SHA256


@pytest.mark.parametrize("path,value", [
    (("trial_id",), "trial013"), (("baseline_trial",), "trial015"),
    (("training", "iterations"), 599), (("training", "seed"), 43),
    (("reward", "sampling"), "all_substeps_mean_force"),
    (("reward", "weight"), -2), (("unchanged", "leg_kp_kd"), [500, 12]),
    (("expected_source_sha256", "training/tron1_ground_load_reserve.py"), "0" * 64),
])
def test_protocol_changes_rejected(path, value):
    protocol = p.read_json(ROOT / p.PROTOCOL_FILE)
    target = protocol
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        p.validate_protocol(protocol)


@pytest.mark.parametrize("stage", ["dense_ground_load_sampling_initial_audit", "dense_ground_load_sampling_final_audit"])
def test_dense_sampler_accepts_actual_schema_synthetic(completion, stage):
    value = completion[stage]
    steps = 8 if "initial" in stage else 14408
    before = copy.deepcopy(value)
    assert p.validate_sampler(value, num_envs=2048, steps=steps, device="cuda:6") is value
    assert value == before
    evaluation = copy.deepcopy(value)
    life = evaluation["lifecycle"]
    life.update(num_envs=1, steps_started=288, steps_completed=288, samples_total=1152,
        last_start_counter=1148, last_sample_counter=1152)
    life["fingerprint_sha256"] = p.fingerprint({"schema_version": 1, "mode": "all_substeps_mean",
        "wheel_body_ids": [8, 9], "decimation": 4, "physics_dt_s": .005, "policy_dt_s": .02, "num_envs": 1})
    evaluation.update(force_samples_total=1152, force_cost_reads=288)
    evaluation["fingerprint_sha256"] = p.fingerprint({"schema_version": 1, "mode": "all_substeps_mean_cost",
        "lifecycle_fingerprint_sha256": life["fingerprint_sha256"], "settings": evaluation["settings"],
        "sensor_names": ["wheel_L_ground", "wheel_R_ground"],
        "ground_collider": "/World/ground/terrain/GroundPlane/CollisionPlane",
        "reserve_dependency_sha256": evaluation["reserve_dependency_sha256"],
        "lifecycle_dependency_sha256": evaluation["lifecycle_dependency_sha256"]})
    p.validate_sampler(evaluation, num_envs=1, steps=288, device="cuda:6")


@pytest.mark.parametrize("path,value", [
    (("mode",), "all_substeps_mean"), (("force_cost_reads",), 7),
    (("force_samples_total",), 31), (("inherited_velocity_cost_reads",), False),
    (("inherited_velocity_cost_reads",), 8), (("wheel_vertical_velocity_reward_changed",), True),
    (("settings", "scale_fraction"), .1), (("lifecycle", "samples_total"), False),
    (("lifecycle", "steps_failed"), False), (("lifecycle", "cost_reads"), 8),
    (("lifecycle", "native_last_step_index"), 100), (("lifecycle", "native_last_time_s"), .04),
    (("lifecycle", "device"), "cuda:5"),
])
def test_dense_sampler_rejects_phase_count_type_and_velocity_changes(completion, path, value):
    audit = copy.deepcopy(completion["dense_ground_load_sampling_initial_audit"])
    target = audit
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        p.validate_sampler(audit, num_envs=2048, steps=8, device="cuda:6")


def test_cpu_completion_distinct_from_actor_and_behavior(completion):
    cp = completion["final_checkpoint_sha256"]
    assert p.validate_completion(completion, cp) is completion
    assert len(p.completion_input_files()) == 100
    assert completion["behavior_verified"] is False
    assert completion["dense_ground_load_sampling_final_audit"]["inherited_velocity_cost_reads"] == 0


@pytest.mark.parametrize("path,value", [
    (("trial_id",), "trial014"), (("verification_script_sha256",), "0" * 64),
    (("git_commit",), "0" * 40), (("training_source_sha256", "training/tron1_ground_load_substeps.py"), "0" * 64),
    (("original_dense_ground_load_profile_and_explicit_fourteen_training_sources_verified",), False),
    (("new_actor_inference_calls",), False), (("cuda_allocations",), 1),
    (("behavior_verified",), True), (("production_force_cache_values_independently_archived",), True),
    (("owned_exit_returncode",), 1), (("training_budget", "updates"), 599),
    (("input_sha256", "runtime/source/training/tron1_ground_load_substeps.py"), "0" * 64),
    (("current_stack_runtime_prerequisite", "status"), "dense_ground_load_smoke_verified"),
    (("current_stack_smoke_prerequisite", "proof_sha256"), "0" * 64),
    (("external_input_sha256", "parent004/model_final.pt"), "0" * 64),
    (("external_input_files",), {"forged": "/wrong/path"}),
    (("final_checkpoint_health", "adam_parameter_steps"), [80060] * 17),
    (("final_checkpoint_health", "normalizer_counts", "actor_obs_normalizer.count"), 145072128),
])
def test_cpu_proof_mutations_fail(completion, path, value):
    cp = completion["final_checkpoint_sha256"]
    target = completion
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        p.validate_completion(completion, cp)


def test_actual_cache_fields_directly_validated_no_projection():
    # Past completed013 metadata is copied ONLY to form a synthetic016 cache fixture.
    trial = ROOT / "outputs/drift-sim2sim-20261012/trial013"
    profile = p.read_json(ROOT / p.PROFILE_FILE)
    records = {"manifest": p.read_json(trial / "training/manifest.json"),
        "isaac_report": p.read_json(trial / "isaac/report.json"),
        "mujoco_report": p.read_json(trial / "mujoco/report.json"),
        "contract": p.read_json(trial / "isaac/policy_contract.json"),
        "assessment": p.read_json(trial / "assessment.json"),
        "adaptation": p.read_json(ROOT / "outputs/drift-reference-20261011/adaptation.json"),
        "export_report": p.read_json(ROOT / "outputs/drift-reference-20261011/motion.json")}
    for container in (records["manifest"], records["manifest"]["contract"], records["contract"],
                      records["isaac_report"], records["isaac_report"]["policy_export"],
                      records["mujoco_report"]["contract"]):
        container["task_profile"] = copy.deepcopy(profile)
    records["manifest"]["source_sha256"] = copy.deepcopy(p.TRAINING_SOURCE14)
    records["isaac_report"]["evaluator_source_sha256"] = p.SOURCE17["scripts/eval_tracking_dense_ground_load.py"]
    records["isaac_report"]["baseline_evaluator_source_sha256"] = p.SOURCE17["scripts/eval_tracking.py"]
    originals = copy.deepcopy(records)
    hashes = {key: p.base.GROUND_LOAD_RESERVE_ASSET_PINS[key] for key in ("motion", "robot_reference")}
    context = p.validate_instantaneous_cache_context(records, {}, hashes)
    assert records == originals
    assert context["actual_objective_sampling"] == "all_substeps_mean_cost"
    assert context["startup_audits"]["training"]["sampling"] == "last_physics_substep"
    records["isaac_report"]["evaluator_source_sha256"] = p.SOURCE17["scripts/eval_tracking.py"]
    with pytest.raises(ValueError):
        p.validate_instantaneous_cache_context(records, {}, hashes)


def test_builder_retains_assessment_provenance_and_media_calls():
    tree = ast.parse(Path(p.__file__).read_text())
    builder = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "_prepare_recorded_candidate")
    calls = {node.func.id for node in ast.walk(builder) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)}
    assert {"assess_runs", "validate_provenance", "validate_render", "validate_instantaneous_cache_context"} <= calls
    all_calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)}
    assert not {"mj_step", "load", "forward", "learn", "step", "cuda", "monkeypatch"} & all_calls


def test_no_future_actual_metadata_or_import_work():
    command = [sys.executable, "-c", "import sys; sys.path.insert(0,'scripts'); "
        "import publish_car_drift_dense_ground_load; "
        "assert 'torch' not in sys.modules; assert 'isaaclab' not in sys.modules; assert 'mujoco' not in sys.modules"]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert len(p.RUNTIME_INPUTS) == 15 and len(p.PAIRED_CHECK_NAMES) == 15


def test_publisher_overwrite_and_unsealed_producer_fail_closed(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(ValueError, match="NEW"):
        p.publish_trial(tmp_path / "trial016", tmp_path, output, tmp_path)
    with pytest.raises(ValueError):
        p.digest("UNSEALED", "Unsealed")


def test_exact_current_paired_producer_pin_and15_original_checks():
    filename = ROOT / "outputs/drift-trial016-deployment-20261012/audit_deployment_provenance_trial016.py"
    assert p.sha256(filename) == p.PAIRED_PRODUCER_SHA256
    tree = ast.parse(filename.read_text())
    selected = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"CHECK_NAMES", "FILES"}}
    assert set(selected["CHECK_NAMES"]) == p.PAIRED_CHECK_NAMES
    assert tuple(selected["FILES"]) == p.RUNTIME_INPUTS


@pytest.fixture
def full_context(tmp_path, monkeypatch, completion):
    """Complete GENERATED016 metadata+byte tokens; never executed as a policy."""
    baseline = ROOT / "outputs/drift-sim2sim-20261012/trial013"
    profile = p.read_json(ROOT / p.PROFILE_FILE)
    records = {name: p.read_json(baseline / relative) for name, relative in {
        "manifest": "training/manifest.json", "run_status": "training/run_status.json",
        "runner": "training/runner_config.json", "contract": "isaac/policy_contract.json",
        "isaac_report": "isaac/report.json", "mujoco_report": "mujoco/report.json"}.items()}
    fixture = imported("synthetic016_completion_evaluation", ROOT /
        "outputs/drift-trial016-deployment-20261012/test_verify_training_completion_trial016.py")
    evaluation = fixture.dense_sampler(288)
    evaluation["lifecycle"]["num_envs"] = 1
    evaluation["lifecycle"]["fingerprint_sha256"] = p.fingerprint({"schema_version": 1,
        "mode": "all_substeps_mean", "wheel_body_ids": [8, 9], "decimation": 4,
        "physics_dt_s": .005, "policy_dt_s": .02, "num_envs": 1})
    evaluation["fingerprint_sha256"] = p.fingerprint({"schema_version": 1,
        "mode": "all_substeps_mean_cost",
        "lifecycle_fingerprint_sha256": evaluation["lifecycle"]["fingerprint_sha256"],
        "settings": evaluation["settings"], "sensor_names": ["wheel_L_ground", "wheel_R_ground"],
        "ground_collider": "/World/ground/terrain/GroundPlane/CollisionPlane",
        "reserve_dependency_sha256": evaluation["reserve_dependency_sha256"],
        "lifecycle_dependency_sha256": evaluation["lifecycle_dependency_sha256"]})
    trial = tmp_path / "trial016"
    for relative in p.RUNTIME_INPUTS:
        path = trial / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("SYNTHETIC NEVER EXECUTED: " + relative).encode())
    cp_sha = p.sha256(trial / "training/model_final.pt")
    actor_sha = p.sha256(trial / "isaac/actor_normalized.pt")
    for container in (records["manifest"], records["manifest"]["contract"], records["contract"],
                      records["isaac_report"], records["isaac_report"]["policy_export"],
                      records["mujoco_report"]["contract"]):
        container["task_profile"] = copy.deepcopy(profile)
        if "checkpoint_sha256" in container:
            container["checkpoint_sha256"] = cp_sha
    records["manifest"].update(source_sha256=copy.deepcopy(p.TRAINING_SOURCE14), task_profile_sha256=p.PROFILE_SHA256,
        reward_sampling_audit=completion["dense_ground_load_sampling_initial_audit"])
    records["isaac_report"].update(reward_sampling_audit=evaluation,
        evaluator_source_sha256=p.SOURCE17["scripts/eval_tracking_dense_ground_load.py"],
        baseline_evaluator_source_sha256=p.SOURCE17["scripts/eval_tracking.py"],
        dense_ground_load_module_sha256=p.SOURCE17["training/tron1_ground_load_substeps.py"],
        dense_ground_load_trial_module_sha256=p.SOURCE17["training/tron1_ground_load_substeps_trial.py"],
        substep_reward_module_sha256=p.SOURCE17["training/tron1_substep_rewards.py"],
        dense_ground_load_sampling="all_substeps_mean_cost", checkpoint_sha256=cp_sha)
    records["mujoco_report"]["policy_sha256"] = actor_sha
    records["run_status"]["checkpoint"] = "/data/SYNTHETIC/outputs/trial016/training/model_final.pt"
    records.update(lr_audit=completion["resume_learning_rate_audit"],
                   final_sampling=completion["dense_ground_load_sampling_final_audit"])
    paths = {"manifest": "training/manifest.json", "run_status": "training/run_status.json",
        "runner": "training/runner_config.json", "contract": "isaac/policy_contract.json",
        "isaac_report": "isaac/report.json", "mujoco_report": "mujoco/report.json",
        "lr_audit": "training/resume_learning_rate_audit.json",
        "final_sampling": "training/reward_sampling_final_audit.json"}
    for key, name in paths.items():
        (trial / name).write_bytes(p.encode(records[key]))
    (trial / "training/runner_config.json").write_bytes((baseline / "training/runner_config.json").read_bytes())
    files = {key: trial / name for key, name in paths.items()}
    files.update(actor=trial / "isaac/actor_normalized.pt", final_checkpoint=trial / "training/model_final.pt",
        isaac_physics_trajectory=trial / "isaac/physics_trajectory.npz",
        mujoco_trajectory=trial / "mujoco/rollout.npz")
    hashes = {key: p.sha256(path) for key, path in files.items()}
    completion["final_checkpoint_sha256"] = cp_sha
    for name in ("manifest.json", "run_status.json", "runner_config.json", "model_final.pt",
                 "resume_learning_rate_audit.json", "reward_sampling_final_audit.json"):
        completion["input_sha256"]["training/" + name] = p.sha256(trial / "training" / name)
    complete_path = trial / "diagnostics/training_completion_verification.json"
    complete_path.write_bytes(p.encode(completion))
    files["training_completion"], hashes["training_completion"] = complete_path, p.sha256(complete_path)
    proof = p.read_json(baseline / "diagnostics/deployment_provenance_audit.json")
    producer = "b" * 64
    monkeypatch.setattr(p, "PAIRED_PRODUCER_SHA256", producer)
    proof.update(checks={key: True for key in p.PAIRED_CHECK_NAMES}, audit_script_sha256=producer,
        dense_ground_load_sampling_initial_audit=records["manifest"]["reward_sampling_audit"],
        dense_ground_load_sampling_final_audit=records["final_sampling"],
        dense_ground_load_sampling_evaluation_audit=evaluation,
        resume_learning_rate_audit=records["lr_audit"],
        resume_source_audit=records["manifest"]["resume_source_audit"],
        training_completion_proof_sha256=hashes["training_completion"],
        training_completion_input_sha256=copy.deepcopy(completion["input_sha256"]),
        training_completion_external_input_sha256=copy.deepcopy(completion["external_input_sha256"]),
        completion_local_input_sha256=copy.deepcopy(completion["input_sha256"]),
        completion_local_external_input_sha256=copy.deepcopy(completion["external_input_sha256"]),
        all100_relative_and9_external_local_byte_copies_rehashed=True)
    proof["actor_export"]["actor_sha256"] = actor_sha
    proof["input_sha256"] = {name: p.sha256(trial / name) for name in p.RUNTIME_INPUTS}
    proof["input_sha256"].update({"repository/" + name: value for name, value in p.SOURCE17.items()})
    proof["input_sha256"].update({"repository/" + p.PROTOCOL_FILE: p.PROTOCOL_SHA256,
                                 "repository/" + p.PROFILE_FILE: p.PROFILE_SHA256})
    for engine, key in (("isaac", "isaac_physics_trajectory"), ("mujoco", "mujoco_trajectory")):
        proof["physics_provenance"][engine]["sha256"] = hashes[key]
    files["paired_proof"] = trial / "diagnostics/deployment_provenance_audit.json"
    files["paired_proof"].write_bytes(p.encode(proof))
    hashes["paired_proof"] = p.sha256(files["paired_proof"])
    return records, files, hashes, p.read_json(ROOT / p.PROTOCOL_FILE), completion, proof


def test_complete_synthetic_context_and_sanitized_scope(full_context):
    context = p.validate_context(*full_context)
    assert context["primary_comparison_trial"] == "trial013"
    assert context["only_factor_relative_to_primary"] == "ground_load_reserve_last_to_mean_of_four_instantaneous_costs"
    assert context["held_wheel_vertical_velocity_sampling"] == "last_physics_substep"
    assert context["behavior_gates_changed"] is False
    assert context["production_force_cache_values_independently_archived"] is False
    public = p.sanitize(context, p.private_identities(context))
    encoded = p.encode(public)
    assert b"/data/" not in encoded and b"/home/" not in encoded
    assert b"model_final.pt" not in encoded


@pytest.mark.parametrize("target,path,value", [
    ("records", ("manifest", "task_profile", "wheel_vertical_velocity_sampling"), "all_substeps_mean"),
    ("records", ("manifest", "source_sha256", "training/tron1_ground_load_substeps.py"), "0" * 64),
    ("records", ("isaac_report", "evaluator_source_sha256"), "0" * 64),
    ("records", ("run_status", "checkpoint"), "/data/wrong/outputs/trial013/training/model_final.pt"),
    ("records", ("contract", "leg_kd"), 12.),
    ("proof", ("checks", "all_recorded_actor_actions_replay"), False),
    ("proof", ("isaac_actor_replay", "rows"), 287),
    ("proof", ("actor_export", "tensors_compared_bit_exact"), 11),
    ("proof", ("dense_ground_load_sampling_final_audit", "inherited_velocity_cost_reads"), 14408),
    ("proof", ("input_sha256", "training/reward_sampling_final_audit.json"), "0" * 64),
    ("proof", ("all100_relative_and9_external_local_byte_copies_rehashed",), False),
    ("proof", ("completion_local_input_sha256", "training/model_final.pt"), "0" * 64),
])
def test_complete_context_mutations_fail(full_context, target, path, value):
    records, files, hashes, protocol, completion, proof = full_context
    node = records if target == "records" else proof
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError):
        p.validate_context(records, files, hashes, protocol, completion, proof)


@pytest.mark.parametrize("alias", ["runtime/baseline013_profile", "runtime/parent004_profile",
    "runtime/launcher", "runtime/verifier", "runtime/actual8/launcher", "runtime/actual8/verifier",
    "runtime/actual8/baseline013_profile", "runtime/actual8/parent004_profile"])
def test_consistently_forged_prerequisites_cannot_change_frozen_producer_pins(completion, alias):
    completion["input_sha256"][alias] = "0" * 64
    for key, prefix in (("current_stack_runtime_prerequisite", "runtime/"),
                        ("current_stack_smoke_prerequisite", "runtime/actual8/")):
        completion[key]["input_sha256"] = {name[len(prefix):]: sha for name, sha in
            completion["input_sha256"].items() if name.startswith(prefix)}
    with pytest.raises(ValueError):
        p.validate_completion(completion, completion["final_checkpoint_sha256"])


