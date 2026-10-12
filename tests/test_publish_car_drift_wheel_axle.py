"""Synthetic-only trial017 publication tests; no actor, checkpoint or physics."""
from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import publish_car_drift_wheel_axle as p


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def completion():
    helper = imported("synthetic017_cpu_metadata_only", ROOT /
        "outputs/drift-trial017-deployment-20261012/test_verify_training_completion_trial017.py")
    proof, context = helper.completion_document()
    fixture = p.read_json(ROOT / "outputs/drift-trial017-deployment-20261012/prelaunch_seal.json")
    external_files, external_hashes = p.external_input_maps()
    training = {"schema_version": 1, "status": "trial017_training_stage_sealed", "trial_id": "trial017",
        "git_commit": p.SOURCE_COMMIT, "protocol_sha256": p.PROTOCOL_SHA256,
        "profile_sha256": p.PROFILE_SHA256, "archive_sha256": p.ARCHIVE_SHA256,
        "fixture_prelaunch_seal_sha256": p.FIXTURE_SEAL_SHA256,
        "repository_source_sha256": copy.deepcopy(p.SOURCE27),
        "deployment_source_sha256": copy.deepcopy(p.CPU_TRAINING_PINS),
        "parent_input_sha256": {name.split("/", 1)[1]: value for name, value in external_hashes.items()
                                if name.startswith("parent004/")},
        "rsl_input_sha256": copy.deepcopy(p.lr.RSL_SOURCE_PINS)}
    proof.update(archive_sha256=p.ARCHIVE_SHA256, prelaunch_seal_sha256=p.FIXTURE_SEAL_SHA256,
                 training_prelaunch_seal_sha256="d" * 64)
    known = {**p.SOURCE27, **p.FIXTURE_PRIVATE_PINS, **p.CPU_TRAINING_PINS,
        p.PROTOCOL_FILE: p.PROTOCOL_SHA256, p.PROFILE_FILE: p.PROFILE_SHA256,
        "config/car_drift_dense_ground_load_task.json": p.BASELINE_PROFILE_SHA256,
        "config/car_drift_ground_load_task.json": p.dense.strict.BASELINE013_PROFILE_SHA256,
        "config/car_drift_contact_task.json": p.dense.strict.PARENT004_PROFILE_SHA256,
        "deployment.tar": p.ARCHIVE_SHA256, "prelaunch_seal.json": p.FIXTURE_SEAL_SHA256,
        "training_prelaunch_seal.json": "d" * 64}
    for alias, filename in proof["input_files"].items():
        if filename in known:
            proof["input_sha256"][alias] = known[filename]
    refresh_prerequisites(proof)
    return proof, fixture, training


def refresh_prerequisites(proof):
    for key, prefix, mappings in (
        ("current_stack_smoke_prerequisite", "runtime/actual8/", p.fixture_input_files()),
        ("current_stack_runtime_prerequisite", "runtime/", p.runtime_input_files())):
        proof[key]["input_sha256"] = {name: proof["input_sha256"][prefix + name] for name in mappings}


def validate_completion(value):
    proof, fixture, training = value
    return p.validate_completion(proof, proof["final_checkpoint_sha256"], fixture_seal=fixture,
        training_seal=training, fixture_seal_sha256=proof["prelaunch_seal_sha256"],
        training_seal_sha256=proof["training_prelaunch_seal_sha256"])


def test_actual_posttraining_producer_sources_are_pinned():
    producers = p.validate_publication_producer_sources(ROOT)
    assert len(producers) == 2
    for name, digest in producers.items():
        assert p.sha256(ROOT / name) == digest


@pytest.mark.parametrize("field", ["PAIRED_PRODUCER_SHA256", "EVALUATION_LAUNCHER_SHA256"])
def test_old_or_changed_posttraining_producer_source_rejected(field, monkeypatch):
    monkeypatch.setattr(p, field, "0" * 64)
    with pytest.raises(ValueError, match="Changed post-training producer"):
        p.validate_publication_producer_sources(ROOT)


def test_posttraining_source_binding_precedes_recorded_candidate_work():
    tree = ast.parse(Path(p.__file__).read_text())
    preflight = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "prepare_publication")
    calls = {node.func.id: node.lineno for node in ast.walk(preflight)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert calls["validate_publication_producer_sources"] < calls["_prepare_recorded_candidate"]


def test_frozen_profile_protocol_source_and_old_dependencies():
    protocol = p.read_json(ROOT / p.PROTOCOL_FILE)
    assert p.validate_protocol(protocol) is protocol
    assert len(p.SOURCE27) == 27 and len(p.TRAINING_SOURCE17) == 17
    for name, sha in p.SOURCE27.items():
        assert p.sha256(ROOT / name) == sha
    assert p.sha256(p.dense.__file__) == p.DENSE_PUBLISHER_SHA256
    assert p.sha256(p.base.__file__) == p.LEGACY_PUBLISHER_SHA256
    assert p.sha256(p.lr.__file__) == p.LR_PUBLISHER_SHA256
    profile = p.read_json(ROOT / p.PROFILE_FILE)
    assert p.strict.validate_wheel_axle_profile(profile) == profile
    assert p.sha256(ROOT / p.PROFILE_FILE) == p.PROFILE_SHA256


@pytest.mark.parametrize("path,value", [
    (("trial_id",), "trial016"), (("baseline_trial",), "trial015"),
    (("training", "iterations"), 599), (("training", "seed"), True),
    (("reward", "weight"), 2), (("reward", "target"), "raw_global_body_quat_w"),
    (("unchanged", "contact_geometry_margin_or_gap_changed"), True),
    (("expected_new_source_sha256", "training/tron1_wheel_axle_tracking.py"), "0" * 64),
])
def test_protocol_mutations_rejected(path, value):
    protocol = p.read_json(ROOT / p.PROTOCOL_FILE)
    target = protocol
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    with pytest.raises(ValueError): p.validate_protocol(protocol)


def test_cpu_completion_positive140_and9_and_no_mutation(completion):
    before = copy.deepcopy(completion)
    assert validate_completion(completion) is completion[0]
    assert completion == before
    assert len(p.fixture_input_files()) == 47 and len(p.runtime_input_files()) == 95
    assert len(p.completion_input_files()) == 140 and len(p.external_input_maps()[0]) == 9


@pytest.mark.parametrize("path,value", [
    (("trial_id",), "trial016"), (("verification_script_sha256",), "0" * 64),
    (("original_wheel_axle_profile_and_explicit_seventeen_training_sources_verified",), False),
    (("new_actor_inference_calls",), False), (("simulation_steps",), 1), (("cuda_allocations",), True),
    (("full_training_live_axle_reward_ledger_archived",), True),
    (("wheel_axle_orientation_tracking", "std"), .2),
    (("wheel_axle_target_semantics",), "raw unanchored teacher"),
    (("training_source_sha256", "scripts/train_tracking_wheel_axle.py"), "0" * 64),
    (("sealed_source_sha256", "tests/test_smoke_wheel_axle_tracking.py"), "0" * 64),
    (("training_budget", "updates"), 600.),
    (("input_sha256", "runtime/actual8/private/verify_smoke_trial017.py"), "0" * 64),
    (("input_sha256", "training_prelaunch_seal"), "0" * 64),
    (("external_input_sha256", "rsl/algorithms/ppo.py"), "0" * 64),
    (("resume_learning_rate_audit", "loaded_group_learning_rates"), [1.e-5, 1.e-5]),
    (("resume_source_audit", "actual_source_sha256", "algorithms/ppo.py"), "0" * 64),
    (("final_checkpoint_health", "normalizer_counts", "actor_obs_normalizer.count"), 145072128),
    (("dense_ground_load_sampling_final_audit", "inherited_velocity_cost_reads"), 14408),
])
def test_cpu_mutations_rejected(completion, path, value):
    target = completion[0]
    for name in path[:-1]: target = target[name]
    target[path[-1]] = value
    with pytest.raises(ValueError): validate_completion(completion)


@pytest.mark.parametrize("field", ["input_files", "input_sha256", "external_input_files", "external_input_sha256"])
def test_unknown_cpu_aliases_rejected(completion, field):
    completion[0][field]["unexpected"] = "0" * 64
    with pytest.raises(ValueError): validate_completion(completion)


@pytest.mark.parametrize("target,key,value", [
    (1, "profile_sha256", "0" * 64), (1, "parent_profile_sha256", "0" * 64),
    (2, "status", "trial016_training_stage_sealed"), (2, "unexpected", False),
])
def test_stage_seal_mutations_rejected(completion, target, key, value):
    completion[target][key] = value
    with pytest.raises(ValueError): validate_completion(completion)


@pytest.fixture
def full_context(tmp_path, monkeypatch, completion):
    """Explicit synthetic JSON/tokens, never a learned017 result or policy."""
    proof_cpu, fixture_seal, training_seal = completion
    baseline = ROOT / "outputs/drift-sim2sim-20261012/trial013"
    profile = p.read_json(ROOT / p.PROFILE_FILE)
    records = {name: p.read_json(baseline / relative) for name, relative in {
        "manifest": "training/manifest.json", "run_status": "training/run_status.json",
        "runner": "training/runner_config.json", "contract": "isaac/policy_contract.json",
        "isaac_report": "isaac/report.json", "mujoco_report": "mujoco/report.json"}.items()}
    metadata = imported("synthetic017_sampler_metadata", ROOT /
        "outputs/drift-trial017-deployment-20261012/test_verify_smoke_trial017.py")
    evaluation = metadata.sampler(288, "cuda:2", 1)
    trial = tmp_path / "trial017"
    for relative in p.RUNTIME_INPUTS:
        path = trial / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("SYNTHETIC NEVER EXECUTED: " + relative).encode())
    cp_sha, actor_sha = [p.sha256(trial / name) for name in (
        "training/model_final.pt", "isaac/actor_normalized.pt")]
    for container in (records["manifest"], records["manifest"]["contract"], records["contract"],
                      records["isaac_report"], records["isaac_report"]["policy_export"],
                      records["mujoco_report"]["contract"]):
        container["task_profile"] = copy.deepcopy(profile)
        if "checkpoint_sha256" in container: container["checkpoint_sha256"] = cp_sha
    records["manifest"].update(source_sha256=copy.deepcopy(p.TRAINING_SOURCE17),
        task_profile_sha256=p.PROFILE_SHA256, reward_sampling_audit=proof_cpu["dense_ground_load_sampling_initial_audit"],
        wheel_axle_orientation_tracking={"weight": 1.0, "std": .4}, wheel_axle_target_semantics=p.AXLE_TARGET_SEMANTICS,
        dense_ground_load_sampling="all_substeps_mean_cost",
        baseline_entry_source_sha256=p.SOURCE27["scripts/train_tracking_lr_sync.py"])
    records["isaac_report"].update(reward_sampling_audit=evaluation,
        evaluator_source_sha256=p.SOURCE27["scripts/eval_tracking_wheel_axle.py"],
        baseline_evaluator_source_sha256=p.SOURCE27["scripts/eval_tracking.py"],
        baseline_dense_evaluator_source_sha256=p.SOURCE27["scripts/eval_tracking_dense_ground_load.py"],
        dense_ground_load_module_sha256=p.SOURCE27["training/tron1_ground_load_substeps.py"],
        dense_ground_load_trial_module_sha256=p.SOURCE27["training/tron1_ground_load_substeps_trial.py"],
        substep_reward_module_sha256=p.SOURCE27["training/tron1_substep_rewards.py"],
        wheel_axle_tracking_source_sha256=p.SOURCE27["training/tron1_wheel_axle_tracking.py"],
        wheel_axle_trial_source_sha256=p.SOURCE27["training/tron1_wheel_axle_trial.py"],
        dense_ground_load_sampling="all_substeps_mean_cost", checkpoint_sha256=cp_sha)
    records["mujoco_report"]["policy_sha256"] = actor_sha
    records["run_status"]["checkpoint"] = "/data/SYNTHETIC/outputs/trial017/training/model_final.pt"
    for holder in (records["manifest"], records["isaac_report"]):
        audit = holder["runtime_device_audit"]
        for key in ("requested_device", "environment_device", "physics_sim_view_frontend_device",
                    "robot_native_transform_tensor_device"):
            audit[key] = "cuda:2"
        for key in ("native_physics_backend_device_ordinal", "torch_current_device",
                    "physics_cudaDevice_setting", "device_index"):
            audit[key] = 2
    records.update(lr_audit=proof_cpu["resume_learning_rate_audit"],
                   final_sampling=proof_cpu["dense_ground_load_sampling_final_audit"],
                   fixture_seal=fixture_seal, training_seal=training_seal)
    paths = {"manifest": "training/manifest.json", "run_status": "training/run_status.json",
        "runner": "training/runner_config.json", "contract": "isaac/policy_contract.json",
        "isaac_report": "isaac/report.json", "mujoco_report": "mujoco/report.json",
        "lr_audit": "training/resume_learning_rate_audit.json",
        "final_sampling": "training/reward_sampling_final_audit.json",
        "fixture_seal": "diagnostics/prelaunch_seal.json",
        "training_seal": "diagnostics/training_prelaunch_seal.json"}
    for key, relative in paths.items(): (trial / relative).write_bytes(p.encode(records[key]))
    # Preserve real fixture seal bytes, not a JSON reserialization hash claim.
    (trial / paths["fixture_seal"]).write_bytes((ROOT /
        "outputs/drift-trial017-deployment-20261012/prelaunch_seal.json").read_bytes())
    (trial / "training/runner_config.json").write_bytes((baseline / "training/runner_config.json").read_bytes())
    files = {key: trial / name for key, name in paths.items()}
    files.update(actor=trial / "isaac/actor_normalized.pt", final_checkpoint=trial / "training/model_final.pt",
        isaac_physics_trajectory=trial / "isaac/physics_trajectory.npz", mujoco_trajectory=trial / "mujoco/rollout.npz")
    hashes = {key: p.sha256(path) for key, path in files.items()}
    proof_cpu.update(final_checkpoint_sha256=cp_sha,
        training_prelaunch_seal_sha256=hashes["training_seal"])
    for alias, filename in proof_cpu["input_files"].items():
        if filename == "training_prelaunch_seal.json": proof_cpu["input_sha256"][alias] = hashes["training_seal"]
    for name in ("manifest.json", "run_status.json", "runner_config.json", "model_final.pt",
                 "resume_learning_rate_audit.json", "reward_sampling_final_audit.json"):
        proof_cpu["input_sha256"]["training/" + name] = p.sha256(trial / "training" / name)
    complete_path = trial / "diagnostics/training_completion_verification.json"
    complete_path.write_bytes(p.encode(proof_cpu))
    files["training_completion"], hashes["training_completion"] = complete_path, p.sha256(complete_path)
    monkeypatch.setattr(p, "PAIRED_PRODUCER_SHA256", "b" * 64)
    monkeypatch.setattr(p, "EVALUATION_LAUNCHER_SHA256", "c" * 64)
    evaluation_seal = {"schema_version": 1, "status": "trial017_evaluation_stage_sealed", "trial_id": "trial017",
        "git_commit": p.SOURCE_COMMIT, "protocol_sha256": p.PROTOCOL_SHA256, "profile_sha256": p.PROFILE_SHA256,
        "archive_sha256": p.ARCHIVE_SHA256, "fixture_prelaunch_seal_sha256": hashes["fixture_seal"],
        "training_prelaunch_seal_sha256": hashes["training_seal"],
        "training_completion_proof_sha256": hashes["training_completion"], "final_checkpoint_sha256": cp_sha,
        "deployment_source_sha256": {"launch_eval_trial017.py": "c" * 64}}
    records["evaluation_seal"] = evaluation_seal
    files["evaluation_seal"] = trial / "diagnostics/evaluation_prelaunch_seal.json"
    files["evaluation_seal"].write_bytes(p.encode(evaluation_seal))
    hashes["evaluation_seal"] = p.sha256(files["evaluation_seal"])
    proof = p.read_json(baseline / "diagnostics/deployment_provenance_audit.json")
    proof.update(checks={name: True for name in p.PAIRED_CHECK_NAMES}, audit_script_sha256="b" * 64,
        dense_ground_load_sampling_initial_audit=records["manifest"]["reward_sampling_audit"],
        dense_ground_load_sampling_final_audit=records["final_sampling"],
        dense_ground_load_sampling_evaluation_audit=evaluation,
        resume_learning_rate_audit=records["lr_audit"], resume_source_audit=records["manifest"]["resume_source_audit"],
        training_completion_proof_sha256=hashes["training_completion"],
        training_completion_input_sha256=copy.deepcopy(proof_cpu["input_sha256"]),
        training_completion_external_input_sha256=copy.deepcopy(proof_cpu["external_input_sha256"]),
        completion_local_input_sha256=copy.deepcopy(proof_cpu["input_sha256"]),
        completion_local_external_input_sha256=copy.deepcopy(proof_cpu["external_input_sha256"]),
        all140_relative_and9_external_local_byte_copies_rehashed=True,
        wheel_axle_orientation_tracking={"weight": 1.0, "std": .4}, wheel_axle_target_semantics=p.AXLE_TARGET_SEMANTICS,
        wheel_axle_tracking_source_sha256=p.SOURCE27["training/tron1_wheel_axle_tracking.py"],
        wheel_axle_trial_source_sha256=p.SOURCE27["training/tron1_wheel_axle_trial.py"],
        held_last_wheel_axle_sampling=True, full_training_live_axle_reward_ledger_archived=False,
        prelaunch_seal_sha256=hashes["fixture_seal"], training_prelaunch_seal_sha256=hashes["training_seal"],
        fixture_prelaunch_seal_sha256=hashes["fixture_seal"],
        evaluation_prelaunch_seal_sha256=hashes["evaluation_seal"],
        training_source_sha256=copy.deepcopy(p.TRAINING_SOURCE17), sealed_source_sha256=copy.deepcopy(p.SOURCE27),
        source_backed_wheel_axle_configuration_only=True,
        fixture_raw_arithmetic_freshly_replayed_locally=False,
        stage_seal_input_sha256={"prelaunch_seal.json": hashes["fixture_seal"],
            "training_prelaunch_seal.json": hashes["training_seal"],
            "evaluation_prelaunch_seal.json": hashes["evaluation_seal"]})
    proof["actor_export"]["actor_sha256"] = actor_sha
    proof["input_sha256"] = {name: p.sha256(trial / name) for name in p.RUNTIME_INPUTS}
    proof["input_sha256"].update({"repository/" + name: sha for name, sha in p.SOURCE27.items()})
    proof["input_sha256"].update({"repository/" + p.PROTOCOL_FILE: p.PROTOCOL_SHA256,
                                 "repository/" + p.PROFILE_FILE: p.PROFILE_SHA256,
        "repository/outputs/drift-trial017-deployment-20261012/launch_eval_trial017.py": "c" * 64})
    proof["evaluation_input_sha256"] = {}
    for name in ("eval_launch_receipt.json", "eval_started.json", "eval_exit_report.json",
                 "eval_live_placement.jsonl", "eval.log"):
        path = trial / "diagnostics" / name
        path.write_bytes(("SYNTHETIC NEVER EXECUTED: " + name).encode())
        files["evaluation_stage/" + name] = path
        hashes["evaluation_stage/" + name] = proof["evaluation_input_sha256"][name] = p.sha256(path)
    for engine, key in (("isaac", "isaac_physics_trajectory"), ("mujoco", "mujoco_trajectory")):
        proof["physics_provenance"][engine]["sha256"] = hashes[key]
    files["paired_proof"] = trial / "diagnostics/deployment_provenance_audit.json"
    files["paired_proof"].write_bytes(p.encode(proof))
    hashes["paired_proof"] = p.sha256(files["paired_proof"])
    return records, files, hashes, p.read_json(ROOT / p.PROTOCOL_FILE), proof_cpu, proof


def test_complete_synthetic_context_primary016_and_no_new_teacher(full_context):
    before = copy.deepcopy(full_context[0])
    context = p.validate_context(*full_context)
    assert full_context[0] == before
    assert context["primary_comparison_trial"] == "trial016"
    assert context["descriptive_comparison_trials"] == ["trial013", "trial014", "trial015"]
    assert context["fixed_source_windows_s"] == [[.78, 1.44], [1.82, 2.32]]
    assert context["behavior_gates_changed"] is False
    assert context["full_training_live_axle_reward_ledger_archived"] is False
    assert context["held_wheel_vertical_velocity_sampling"] == "last_physics_substep"
    encoded = p.encode(p.sanitize(context, p.private_identities(context)))
    assert b"/data/" not in encoded and b"/home/" not in encoded and b"model_final.pt" not in encoded


@pytest.mark.parametrize("target,path,value", [
    ("records", ("manifest", "wheel_axle_target_semantics"), "global teacher"),
    ("records", ("manifest", "task_profile", "wheel_axle_orientation_tracking", "weight"), 2.),
    ("records", ("manifest", "source_sha256", "training/tron1_wheel_axle_tracking.py"), "0" * 64),
    ("records", ("isaac_report", "baseline_dense_evaluator_source_sha256"), "0" * 64),
    ("records", ("run_status", "checkpoint"), "/data/wrong/outputs/trial016/training/model_final.pt"),
    ("records", ("contract", "leg_kd"), 12.),
    ("records", ("evaluation_seal", "final_checkpoint_sha256"), "0" * 64),
    ("proof", ("checks", "all_recorded_actor_actions_replay"), False),
    ("proof", ("isaac_actor_replay", "rows"), 287),
    ("proof", ("actor_export", "tensors_compared_bit_exact"), 11),
    ("proof", ("full_training_live_axle_reward_ledger_archived",), True),
    ("proof", ("held_last_wheel_axle_sampling",), False),
    ("proof", ("fixture_raw_arithmetic_freshly_replayed_locally",), True),
    ("proof", ("wheel_axle_tracking_source_sha256",), "0" * 64),
    ("proof", ("evaluation_input_sha256", "eval.log"), "0" * 64),
    ("proof", ("stage_seal_input_sha256", "prelaunch_seal.json"), "0" * 64),
    ("records", ("isaac_report", "runtime_device_audit", "physical_mapping_verified"), True),
    ("proof", ("dense_ground_load_sampling_final_audit", "inherited_velocity_cost_reads"), 14408),
    ("proof", ("input_sha256", "training/reward_sampling_final_audit.json"), "0" * 64),
    ("proof", ("all140_relative_and9_external_local_byte_copies_rehashed",), False),
    ("proof", ("completion_local_input_sha256", "training/model_final.pt"), "0" * 64),
])
def test_complete_context_mutations_rejected(full_context, target, path, value):
    records, files, hashes, protocol, completion, proof = full_context
    node = records if target == "records" else proof
    for key in path[:-1]: node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError): p.validate_context(records, files, hashes, protocol, completion, proof)


def test_current_cache_validated_directly_without_profile_projection(full_context):
    records = {**full_context[0],
        "assessment": p.read_json(ROOT / "outputs/drift-sim2sim-20261012/trial013/assessment.json"),
        "adaptation": p.read_json(ROOT / "outputs/drift-reference-20261011/adaptation.json"),
        "export_report": p.read_json(ROOT / "outputs/drift-reference-20261011/motion.json")}
    originals = copy.deepcopy(records)
    hashes = {key: p.base.GROUND_LOAD_RESERVE_ASSET_PINS[key] for key in ("motion", "robot_reference")}
    result = p.validate_instantaneous_cache_context(records, {}, hashes)
    assert records == originals and result["actual_objective_sampling"] == "all_substeps_mean_cost"
    assert result["full_training_force_validity_independently_proved"] is False


def test_inert_import_and_original_evidence_builder_calls():
    tree = ast.parse(Path(p.__file__).read_text())
    builder = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "_prepare_recorded_candidate")
    calls = {node.func.id for node in ast.walk(builder) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert {"assess_runs", "validate_provenance", "validate_render", "validate_instantaneous_cache_context"} <= calls
    attrs = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not {"mj_step", "load", "forward", "learn", "step", "cuda", "monkeypatch"} & attrs
    result = subprocess.run([sys.executable, "-B", "-c", "import sys; sys.path.insert(0,'scripts'); "
        "import publish_car_drift_wheel_axle; assert not any(k.startswith(('torch','isaaclab','mujoco')) for k in sys.modules)"],
        cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_no_overwrite_or_unsealed_publication(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="NEW"):
        p.publish_trial(tmp_path / "trial017", tmp_path, existing, tmp_path)
    with pytest.raises(ValueError): p.digest("UNSEALED", "Producer")


def test_standalone_cli_from_outside_repository(tmp_path):
    result = subprocess.run([sys.executable, "-B", str(Path(p.__file__)), "--help"],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "--fixture-seal-sha256" in result.stdout and "--evaluation-seal-sha256" in result.stdout


def test_absolute_protocol_preflight_and_failure_creates_nothing(tmp_path):
    output = tmp_path / "trial017"
    kwargs = {"protocol_path": (ROOT / p.PROTOCOL_FILE).resolve(),
        "expected_protocol_sha256": p.PROTOCOL_SHA256,
        "fixture_seal_path": tmp_path / "absent", "fixture_seal_sha256": p.FIXTURE_SEAL_SHA256,
        "training_seal_path": tmp_path / "absent", "training_seal_sha256": "d" * 64,
        "evaluation_seal_path": tmp_path / "absent", "evaluation_seal_sha256": "e" * 64}
    with pytest.raises(ValueError, match="External actual stage seal"):
        p.publish_trial(tmp_path / "trial017", tmp_path, output, tmp_path, **kwargs)
    assert not output.exists()
    kwargs["protocol_path"] = ROOT / "config/car_drift_dense_ground_load_trial.json"
    with pytest.raises(ValueError, match="protocol path"):
        p.publish_trial(tmp_path / "trial017", tmp_path, output, tmp_path, **kwargs)
    assert not output.exists()


def test_publication_keeps_exact8_artifacts_and_failure(tmp_path, monkeypatch):
    """Publisher filesystem-copy unit test, NOT an actual assessed policy."""
    source = tmp_path / "synthetic_media"
    source.mkdir()
    payloads = {name: ("SYNTHETIC sanitized fail: " + name).encode() for name in (
        "assessment.json", "reference_audit.json", "render_report.json")}
    media = {}
    for name in ("motion_pipeline.mp4", "overview.png", "final.png", "contact_overview.png"):
        path = source / name
        path.write_bytes(("SYNTHETIC media token " + name).encode())
        media[name] = {"path": path, "sha256": p.sha256(path)}
    summary = {"status": "assessed_fail", "verdict": "fail", "behavior_pass": False}
    monkeypatch.setattr(p, "prepare_publication", lambda *args, **kwargs: (summary, payloads, media, {}, {}))
    target = tmp_path / "public" / "trial017"
    p.publish_trial(tmp_path / "trial017", tmp_path, target, tmp_path)
    assert set(path.name for path in target.iterdir()) == set(payloads) | set(media)
    assert len(tuple(target.iterdir())) == 8
    for name, value in payloads.items(): assert (target / name).read_bytes() == value
    for name, item in media.items(): assert (target / name).read_bytes() == item["path"].read_bytes()
    assert p.read_json(target / "summary.json")["verdict"] == "fail"
    with pytest.raises(ValueError): p.publish_trial(tmp_path / "trial017", tmp_path, target, tmp_path)
