"""Explicit trial013 publication: held-fixed LR synchronization plus old reserve.

Only metadata/context is added to the frozen car-drift publication candidate.
No actor inference, checkpoint deserialization, physics or new behavior gate.
The caller must supply the independently sealed protocol SHA (avoids a
protocol/publisher self-hash cycle). Missing prospective pins fail closed.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import re
import shutil

import publish_car_drift_lr_sync as lr

base = lr.base

ROOT = base.ROOT
LEGACY_PUBLISHER_SHA256 = "598cd0cf6a78918da8a5e09ca0f20624c0a7379c7311813923366fd2c9b4293d"
BASE_SOURCE_PINS = {
    "training/tron1_tracking.py": "0a29fa27c1d65c0b9add0886221491e549553e3529d6970f1ec5ba35e2532817",
    "training/tron1_domain_randomization.py": "3f3a0b62c85a0af496ae9e9e0d75dda85c3ec30da587e4a47969843f5b1b1dca",
    "training/tron1_terrain.py": "dec819a3c3c526fce706fb6a5a6491388298884b32bfe73a5c331b166f976ebf",
    "training/tron1_drift.py": "6d5f7974602adec9313a5e4feceb8e3caab392ca99351ac4a5f717e57fe1cf8a",
    "training/tron1_contact_slip.py": "52ef38de9cece19a8ee02c120cb85d472ffd831e4587e5f0d1f340974ab699f4",
    "training/tron1_runtime_device.py": "2a1460d1b48e884fc5608e0365edc43b815d1fc74c8439945d4e2cc18027f775",
    "scripts/train_tracking.py": "ea9e4cd14f25ff3ba5202b9ba43ac0f08320418e2311cbae205c8449f3a1da9c",
}
NEW_SOURCE_PINS = {
    "scripts/train_tracking_lr_sync.py": "9bd544b82e211633bb37c3fd8a658448d6d0e9b90f380f01a78dd9e8de02b8cb",
    "training/tron1_resume_lr.py": "596f1e2a455a4bba5644965985791aec9d4a934b5079932e6c6c4a208a1a3064",
    "training/tron1_ground_load_reserve.py": "761eccede3c0d90d744a4d9c45c9671b318426ab9bc6f9762ee7cacb3c14496d",
}
RSL_SOURCE_PINS = {
    "runners/on_policy_runner.py": "e51e4dd49664ccc8743a16b71d2971d392f40a6988136f10fe8c5f03ab6a83b5",
    "algorithms/ppo.py": "deafc8c947eba4df3e91b393869426cdab8d7b71e05974c3734125d2331d7d1c",
    "networks/normalization.py": "7db82f8021d62a660b4229782b52ac47e32d781206dba709d6bc436bf13f6672",
}
PROFILE_SHA256 = "174a0543dfbf3c65bdd54867ab8e9b591047c8a572b0212eaaddbcd0573f2064"
ORIGINAL_PROFILE_SHA256 = "325cf279c699769694fb4934852ce85753b459e09670376b8c2d62d94f7d4e3c"
LR_PUBLISHER_SHA256 = "497ded75605813f916380d01eee387017e444c9e75bb496c5440da96015f71cd"
PARENT_EXPORT_SHA256 = "63bd2ad72bbf5ca390552c7437eedf3a745c8eef1140cfb1540ee4b3e54ccc53"
RESERVE_HELPER = "training/tron1_ground_load_reserve.py"
RESERVE_SHA256 = "761eccede3c0d90d744a4d9c45c9671b318426ab9bc6f9762ee7cacb3c14496d"
RESERVE_SETTINGS = {"weight": -1.0, "scale_fraction": .20}
EXPERIMENT_KIND = "lr_synchronized_ground_load_reserve_continuation"
PARENT_CP_SHA256 = "93bebee533bd59b8a7d707a1cd15b83bdff6c02663910afe59f69df91cd71780"
PARENT_MANIFEST_SHA256 = "b5b4b980425f7058a15a0f86ebbc0f534c216afe6a31d93ac7305bdb964a2a2f"
RUNNER_SHA256 = "7c309e49bed37045a29f337e9ed3f418bfaae437483a27d0c947b9685e170738"
REFERENCE_PINS = {
    "motion_sha256": "8680ac0b331b8c2353701d95dcbce2ea8b0c97342eeb12244867eee33c84fb04",
    "robot_reference_sha256": "ea0b79c1ae84b8deb79961505555ca38decf22f93e44839d8b4cdc8d398deb81",
    "robot_asset_sha256": "327ca5080e753bbc95d8c3c797e9cd49a0f134d400cfe80494af0ba840e23a8e",
    "mujoco_model_sha256": "bd92264b72b4d09084732888dbd81fd94a64a1678978e58a2ab0dd191887f513",
}
FINAL_INPUTS = ("training/manifest.json", "training/run_status.json", "training/runner_config.json",
    "training/model_final.pt", "isaac/report.json", "isaac/policy_contract.json", "isaac/actor_normalized.pt",
    "isaac/trajectory.npz", "isaac/physics_trajectory.npz", "mujoco/report.json", "mujoco/rollout.npz", "assessment.json")
LR_AUDIT_KEYS = {"schema_version", "scope", "status", "scalar_before", "loaded_group_learning_rates",
    "scalar_after", "touched_field", "optimizer_group_learning_rates_unchanged", "adaptive_schedule_preserved",
    "expected_scalar_before", "expected_loaded_lr", "multi_gpu"}
PAIRED_CHECK_NAMES = frozenset({
    "completed600_2048_24_seed42_original004_resume_and_all_manifest_source_profile_contract_fields",
    "actual25_model_51_optimizer_tensors_finite_and_iteration_status_consistent",
    "all_model_optimizer_tensors_finite",
    "fresh013_final_CP_and_matching51_120_8_500_10_50_200_contracts_no_overrides",
    "same_contract_profile_both_engines",
    "all12_exported_actor_normalizer_tensor_bytes_exact_fresh013_final_CP_NOT_baseline004",
    "exported_actor_normalizer_bit_exact_checkpoint",
    "same_actual_I_MJ_actorbytes_and_canonical51D_observation_reconstruction",
    "all288_valid_applied_Isaac_pre_action_observations_CPU_replay_within1e_minus5",
    "all288_MuJoCo_CPU_actor_replay_and_no_normalizer_mutation",
    "all_recorded_actor_actions_replay",
    "genuine200Hz_both_engines_initial_once_and_firstepisode_reset_excluded",
    "reference_asset_model_evaluator_helper_nominalmotor_livegain_and_contactprovenance",
    "all_assessor_provenance_checks_pass",
    "all_input_hashes_unchanged_during_audit",
})


def require(condition, message):
    base.require(condition, "Trial013 LR-and-reserve publication: " + message)


def number(value, expected, label):
    require(type(value) in (int, float), label)
    try:
        finite = math.isfinite(value)
    except (OverflowError, ValueError):
        finite = False
    require(finite and value == expected, label)


def integer(value, expected, label):
    require(type(value) is int and value == expected, label)


def digest(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), label)
    return value


def mapping_field(value, key, label):
    result = value.get(key)
    require(isinstance(result, dict), label)
    return result


validate_lr_audit = lr.validate_lr_audit


def validate_protocol(protocol):
    require(isinstance(protocol, dict), "Protocol object required")
    integer(protocol.get("schema_version"), 1, "Protocol version")
    require(protocol.get("trial_id") == "trial013" and protocol.get("task") == "captured_car_drift"
            and protocol.get("experiment_type") == "lr_synchronized_ground_load_reserve_continuation", "Explicit prospective013 protocol required")
    for key, value in (("baseline_trial", "trial012"), ("parent_trial", "trial004"),
                       ("original_training_task_profile_sha256", ORIGINAL_PROFILE_SHA256),
                       ("task_profile", "config/car_drift_ground_load_task.json"),
                       ("task_profile_sha256", PROFILE_SHA256), ("training_entrypoint", "scripts/train_tracking_lr_sync.py"),
                       ("training_helper", "training/tron1_resume_lr.py"), ("frozen_base_entrypoint", "scripts/train_tracking.py"),
                       ("frozen_base_entrypoint_sha256", BASE_SOURCE_PINS["scripts/train_tracking.py"]),
                       ("publication_entrypoint", "scripts/publish_car_drift_lr_sync_ground_load.py")):
        require(protocol.get(key) == value, "Protocol changed: " + key)
    require(protocol.get("expected_training_source_sha256") == {**BASE_SOURCE_PINS, **NEW_SOURCE_PINS},
            "Protocol must seal exact unchanged source10")
    require(protocol.get("reserve_helper") == RESERVE_HELPER
            and protocol.get("reserve_helper_sha256") == RESERVE_SHA256, "Original reserve helper pin required")
    reserve = mapping_field(protocol, "reward", "Original reserve reward declaration required")
    number(reserve.get("weight"), -1., "Original reserve reward weight")
    number(reserve.get("scale_fraction_r"), .20, "Original reserve exponential scale")
    require(reserve.get("cost") == "sum_i exp(-max(F_ground_i_world_z,0)/(r*mg))"
            and reserve.get("sampling") == "last_physics_substep", "Original last-substep reserve formula required")
    require(type(reserve.get("range")) is list and len(reserve["range"]) == 2, "Original bounded reserve range required")
    for value, expected in zip(reserve["range"], (0., 2.)):
        number(value, expected, "Original bounded reserve range")
    training = mapping_field(protocol, "training", "Protocol training object required")
    for key, value in (("iterations", 600), ("num_envs", 2048), ("rollout_steps_per_update", 24),
                       ("seed", 42), ("smoke_steps", 8), ("additional_environment_steps", 29_491_200),
                       ("parent_saved_iteration", 3996), ("expected_final_saved_iteration", 4595)):
        integer(training.get(key), value, "Protocol training: " + key)
    require(training.get("resume_checkpoint_sha256") == PARENT_CP_SHA256
            and training.get("parent_manifest_sha256") == PARENT_MANIFEST_SHA256
            and training.get("runner_config_sha256") == RUNNER_SHA256, "Original004 resume/config pins required")
    lr = mapping_field(protocol, "learning_rate_resume", "Protocol LR object required")
    require(lr.get("rsl_rl_version") == "3.0.1" and lr.get("installed_source_sha256") == RSL_SOURCE_PINS,
            "Installed RSL3.0.1 sources must match")
    for key, value in (("configured_ppo_learning_rate", .001), ("expected_scalar_before", .001),
                       ("expected_loaded_optimizer_learning_rate", 1.e-5), ("expected_scalar_after", 1.e-5)):
        number(lr.get(key), value, "Protocol LR: " + key)
    require(lr.get("exactly_uniform_optimizer_groups_required") is True
            and lr.get("only_mutated_field") == "algorithm.learning_rate"
            and lr.get("optimizer_group_learning_rates_or_moments_mutated") is False
            and lr.get("normalizers_or_counters_mutated") is False, "Scalar-only protocol required")
    unchanged = mapping_field(protocol, "unchanged", "Protocol unchanged object required")
    require(all(unchanged.get(key) == value for key, value in REFERENCE_PINS.items()),
            "Original motion/reference/robot/physics-model pins required")
    for key, value in (("policy_hz", 50), ("physics_hz_both_engines", 200)):
        integer(unchanged.get(key), value, "Unchanged protocol clock")
    require(unchanged.get("actor_critic_action_dimensions") == [51, 120, 8]
            and unchanged.get("leg_kp_kd") == [500., 10.]
            and unchanged.get("leg_and_wheel_torque_limits_nm") == [80., 12.]
            and unchanged.get("mujoco_contact_solref") == [.04, 2.], "Original dimensions/gains/contact response required")
    number(unchanged.get("contact_friction"), .15, "Original friction")
    for key in ("observations_or_contract_changed", "contact_geometry_margin_or_gap_changed", "surrogate_contact_parameters_adopted"):
        require(unchanged.get(key) is False, "Unchanged protocol: " + key)
    evaluation = mapping_field(protocol, "evaluation", "Protocol evaluation object required")
    for key, value in (("episodes_per_engine", 1), ("num_envs", 1), ("seed", 42)):
        integer(evaluation.get(key), value, "Original evaluation: " + key)
    require(evaluation.get("assessor_source_sha256") == "a2214c47d0f8b262fb8f5b70a709908ef1d7487fd2b1393becb8ad8dfc672fa5",
            "Original assessor required")
    return protocol


def validate_completion(completion, checkpoint_sha):
    """Retain the original budget checks and require the actual013 CPU proof."""
    lr.validate_completion(completion, checkpoint_sha)
    integer(completion.get("schema_version"), 1, "CPU completion schema required")
    require(completion.get("original_ground_load_profile_and_explicit_ten_training_sources_verified") is True,
            "CPU completion must independently verify the original reserve profile and explicit source10")
    require(completion.get("all_input_hashes_unchanged") is True,
            "CPU completion input hashes must remain unchanged")
    for field in ("new_actor_inference_calls", "new_optimizer_updates", "simulation_steps", "cuda_allocations"):
        integer(completion.get(field), 0, "CPU-only completion: " + field)
    require(completion.get("behavior_verified") is False and completion.get("hardware_ready") is False,
            "CPU completion cannot establish behavior or hardware readiness")
    return completion


proof_seals = lr.proof_seals


def validate_actor_proof(proof, trial_dir, files, hashes):
    """Require the exact013 producer checks, retaining every original bound."""
    lr.validate_actor_proof(proof, trial_dir, files, hashes)
    integer(proof.get("schema_version"), 1, "Paired actor-proof schema required")
    require(set(proof["checks"]) == PAIRED_CHECK_NAMES,
            "All exact fifteen013 paired-provenance checks are required")
    integer(proof.get("new_rollouts"), 0, "Recorded-only actor proof cannot create rollouts")
    integer(proof.get("simulation_steps"), 0, "Recorded-only actor proof cannot simulate")
    require(proof.get("behavior_success_claimed") is False,
            "Paired actor provenance alone cannot claim behavioral success")
    return proof


def validate_context(records, files, hashes, protocol, lr_audit, completion, proof, *,
                     repo_root, expected_source_sha256):
    """Pure metadata validation except bounded SHA reads; no actor/checkpoint load."""
    validate_protocol(protocol)
    require(isinstance(records, dict) and isinstance(files, dict) and isinstance(hashes, dict), "Publication record/file/hash objects required")
    require(isinstance(expected_source_sha256, dict), "Approved source pins object required")
    expected = dict(expected_source_sha256)
    require(set(expected) == set(BASE_SOURCE_PINS) | set(NEW_SOURCE_PINS), "Exactly ten prospective training source pins required")
    require(expected == {**BASE_SOURCE_PINS, **NEW_SOURCE_PINS}, "Frozen source10 implementations changed")
    for relative, expected_sha in expected.items():
        digest(expected_sha, "Unsealed source pin: " + relative)
        require(base.sha256(repo_root / relative) == expected_sha, "Local training source differs: " + relative)
    manifest, contract, isaac, mujoco = [mapping_field(records, k, "Publication metadata object required: " + k)
                                       for k in ("manifest", "contract", "isaac_report", "mujoco_report")]
    require(manifest.get("source_sha256") == expected, "Actual manifest must seal exactly approved source10")
    require(manifest.get("versions", {}).get("rsl-rl-lib") == "3.0.1", "Actual installed RSL version required")
    validate_lr_audit(lr_audit)
    require(manifest.get("resume_learning_rate_audit") == lr_audit, "Dedicated LR audit differs from actual startup manifest")
    source_audit = manifest.get("resume_source_audit", {})
    require(isinstance(source_audit, dict) and set(source_audit) == {"schema_version", "rsl_rl_version", "source_root",
                "expected_source_sha256", "actual_source_sha256", "verified_before_runner_load"}
            and isinstance(source_audit.get("source_root"), str) and bool(source_audit["source_root"])
            and type(source_audit.get("schema_version")) is int and source_audit["schema_version"] == 1
            and source_audit.get("rsl_rl_version") == "3.0.1"
            and source_audit.get("expected_source_sha256") == source_audit.get("actual_source_sha256") == RSL_SOURCE_PINS
            and source_audit.get("verified_before_runner_load") is True, "Installed-source startup proof required")
    baseline_path = repo_root / "outputs/drift-sim2sim-20261011/trial004/training/manifest.json"
    require(base.sha256(baseline_path) == PARENT_MANIFEST_SHA256, "Original004 manifest changed")
    baseline = base.read_json(baseline_path)
    profile_path = repo_root / protocol["task_profile"]
    require(base.sha256(profile_path) == PROFILE_SHA256, "Original174a reserve profile changed")
    profile = base.read_json(profile_path)
    original_profile = baseline.get("task_profile")
    original_path = repo_root / "config/car_drift_contact_task.json"
    require(base.sha256(original_path) == ORIGINAL_PROFILE_SHA256
            and base.read_json(original_path) == original_profile, "Original004 profile/source changed")
    require(profile.get("wheel_ground_load_reserve") == RESERVE_SETTINGS
            and {k:v for k,v in profile.items() if k not in ("description", "wheel_ground_load_reserve")}
                == {k:v for k,v in original_profile.items() if k != "description"},
            "Only the original fixed reserve objective may differ")
    original_contract = mapping_field(baseline, "contract", "Original004 training contract required")
    expected_contract = copy.deepcopy(original_contract)
    expected_contract["task_profile"] = profile
    require(manifest.get("contract") == expected_contract
            and manifest.get("domain_randomization_profile") == baseline.get("domain_randomization_profile"),
            "Reserve profile alone may change the original full training contract")
    old_export_path = repo_root / "outputs/drift-sim2sim-20261011/trial004/isaac/policy_contract.json"
    require(base.sha256(old_export_path) == PARENT_EXPORT_SHA256, "Original004 export contract changed")
    old_export = base.read_json(old_export_path)
    require(set(contract) == set(old_export), "Unknown deployment contract fields")
    for key, value in old_export.items():
        if key not in ("task_profile", "checkpoint_sha256", "export_validation_max_abs_error"):
            require(contract[key] == value, "Original full nominal export changed: " + key)
    require(contract == isaac.get("policy_export") == mujoco.get("contract"),
            "Paired actual deployment contracts differ")
    require(manifest.get("task_profile_sha256") == PROFILE_SHA256
            and manifest.get("task_profile") == contract.get("task_profile") == isaac.get("task_profile")
            == mujoco.get("contract", {}).get("task_profile") == profile, "Fixed reserve profile must remain exact in training and both engines")
    require(manifest.get("motion_sha256") == contract.get("reference_file_sha256")
            == hashes["motion"] == REFERENCE_PINS["motion_sha256"]
            and hashes["robot_reference"] == REFERENCE_PINS["robot_reference_sha256"]
            and manifest.get("asset_sha256") == isaac.get("asset_root_file_sha256") == REFERENCE_PINS["robot_asset_sha256"]
            and mujoco.get("model_sha256") == REFERENCE_PINS["mujoco_model_sha256"], "Original reference/asset/physical-model chain required")
    for record in (contract, isaac, mujoco):
        require("resume_learning_rate_audit" not in record and "resume_source_audit" not in record,
                "LR metadata belongs to training, not a changed actor/evaluator contract")
    for record in (manifest, contract, isaac, mujoco):
        require(not any(key in record for key in ("controller_gain_diagnostic", "physics_clock_refinement", "reward_sampling_audit")),
                "No simultaneous gain/clock/all-substep factor")
    require(isaac.get("ground_load_reserve_module_sha256") == RESERVE_SHA256, "Actual reserve evaluator source required")
    reserve_context = base.validate_ground_load_reserve_context(records, files, hashes)
    require(isinstance(reserve_context, dict) and reserve_context.get("settings") == RESERVE_SETTINGS,
            "Original source-backed reserve startup context required")
    require(all(audit.get("cuda_device") == "cuda:5" for audit in reserve_context["startup_audits"].values()),
            "Both startup reserve caches must retain the declared CUDA5 ordinal")
    domain = mapping_field(mujoco, "domain_parameters", "Actual nominal MuJoCo motor draw required")
    require(set(domain) == {"is_nominal", "joint_names", "torque_scale", "velocity_scale",
                            "wheel_friction_nm", "friction_smoothing_rad_s"},
            "Only the original nominal motor-draw schema is supported")
    number(domain.get("friction_smoothing_rad_s"), .5, "Original nominal wheel-friction smoothing required")
    for key, value in (("num_envs", 2048), ("iterations_requested", 600), ("seed", 42), ("smoke_steps_passed", 8)):
        integer(manifest.get(key), value, "Actual training: " + key)
    require(manifest.get("resume_sha256") == PARENT_CP_SHA256, "Original004 parent checkpoint required")
    status, runner = [mapping_field(records, k, "Completion metadata object required: " + k) for k in ("run_status", "runner")]
    require(status.get("status") == "training_completed", "Final completed training required")
    for key, value in (("num_envs", 2048), ("iterations_requested", 600), ("training_environment_steps", 29_491_200), ("last_iteration", 4595)):
        integer(status.get(key), value, "Actual completed budget: " + key)
    integer(runner.get("num_steps_per_env"), 24, "Actual PPO rollout length")
    require(base.sha256(files["runner"]) == RUNNER_SHA256, "Original complete PPO configuration required")
    completed_path = Path(status.get("checkpoint", ""))
    require(completed_path.name == "model_final.pt"
            and completed_path.parent.parts[-3:] == ("outputs", "trial013", "training"),
            "Completed status must name final013, not an intermediate or historical checkpoint")
    checkpoint_sha = hashes["final_checkpoint"]
    require(checkpoint_sha == contract.get("checkpoint_sha256") == isaac.get("checkpoint_sha256"), "Actual final013 checkpoint/export chain required")
    validate_completion(completion, checkpoint_sha)
    integer(completion.get("actual_adam_step_increment"), 12000, "Actual complete600-update Adam increment")
    integer(completion.get("actual_actor_and_critic_normalizer_count_increment"), 29_491_200,
            "Actual complete600-update normalization increment")
    require(completion.get("resume_learning_rate_audit") == lr_audit
            and completion.get("resume_source_audit") == source_audit,
            "Completion must retain exact startup LR/source metadata")
    completion_inputs = completion.get("input_sha256", {})
    require(isinstance(completion_inputs, dict), "CPU completion input seals required")
    training_dir = files["manifest"].parent
    for name in ("manifest.json", "run_status.json", "runner_config.json", "model_final.pt", "resume_learning_rate_audit.json"):
        require(completion_inputs.get("training/" + name) == base.sha256(training_dir / name),
                "CPU completion must seal actual013 training/" + name)
    require(completion_inputs.get("repository/config/car_drift_lr_sync_ground_load_trial.json") == hashes["lr_sync_protocol"],
            "CPU completion must seal prospective protocol")
    for relative, expected_sha in expected.items():
        require(completion_inputs.get("repository/" + relative) == expected_sha,
                "CPU completion must seal training source: " + relative)
    validate_actor_proof(proof, files["manifest"].parent.parent, files, hashes)
    for relative in ("training/resume_learning_rate_audit.json", "diagnostics/training_completion_verification.json"):
        proof_seals(proof, relative, files["manifest"].parent.parent / relative)
    require(proof.get("resume_learning_rate_audit") == lr_audit
            and proof.get("resume_source_audit") == source_audit
            and proof.get("training_completion_proof_sha256") == base.sha256(
                files["manifest"].parent.parent / "diagnostics/training_completion_verification.json"),
            "Actor proof must retain exact sealed LR and completed-training evidence")
    integer(isaac.get("physics_recording", {}).get("samples"), 1153, "All1153 actual Isaac physics states required")
    for key, value in (("physics_steps", 1152), ("policy_steps", 288), ("root_state_writes", 1), ("hidden_resets", 0)):
        integer(mujoco.get(key), value, "Full physics-only MuJoCo: " + key)
    require(mujoco.get("physics_stepped") is True and mujoco.get("policy_sha256") == hashes["actor"],
            "Same actual exported actor must run in MuJoCo physics")
    physics = mapping_field(proof, "physics_provenance", "Actual200Hz proof required")
    for engine, key in (("isaac", "isaac_physics_trajectory"), ("mujoco", "mujoco_trajectory")):
        evidence = mapping_field(physics, engine, "Actual physics engine proof required")
        number(evidence.get("sample_dt_s"), .005, "Original paired200Hz behavior clock")
        require(evidence.get("sha256") == hashes[key], "Actual physics archive proof SHA changed")
        if engine == "isaac":
            require(evidence.get("kind") == "physics_rate" and evidence.get("file") == "physics_trajectory.npz",
                    "Isaac must use actual physics-rate recorder")
        else:
            integer(evidence.get("samples"), 1153, "All1153 actual MuJoCo states required")
    return {
        "schema_version": 1, "experiment_kind": "lr_synchronized_ground_load_reserve_continuation",
        "evidence_scope": "Source-pinned actual startup scalar audit, full CPU budget/tensor proof and paired final-actor provenance; not independent full-minibatch LR telemetry.",
        "restoration": lr_audit, "installed_rsl_sources": source_audit,
        "comparison_trial": "trial012",
        "only_added_factor_relative_to_lr_synchronized_control": "wheel_ground_load_reserve",
        "ground_load_reserve": reserve_context,
        "ground_load_reserve_profile_sha256": PROFILE_SHA256,
        "original_training_task_profile": original_profile,
        "effective_training_and_deployment_task_profile": profile,
        "training_source_sha256": expected, "parent_checkpoint_sha256": PARENT_CP_SHA256,
        "actual_final_checkpoint_sha256": checkpoint_sha, "actual_actor_sha256": hashes["actor"],
        "budget": {"updates": 600, "num_envs": 2048, "rollout_steps_per_update": 24, "seed": 42,
                   "additional_environment_steps": 29_491_200, "saved_iteration": 4595},
        "behavior_gates_changed": False, "learned_actor_equals_baseline_required": False,
        "legacy_publication_route_is_lr_aware": False,
        "limitations": ["The frozen legacy/012 routes are not the explicit combined trial013 publication route.",
            "After initial scalar synchronization, the original adaptive schedule may change LR normally.",
            "Reserve audits are source-backed startup metadata, not independent full-training force/cache or live reward proof.",
            "The symmetric force cost softly favors balanced/high loads and can trade against motion tracking; its scale is not a minimum force.",
            "Last-substep50Hz reward sampling does not establish uninterrupted200Hz bearing-contact patches.",
            "Startup audit/source proof does not independently measure every mini-batch learning rate or establish a causal explanation.",
            "One seed and one paired episode do not establish statistical robustness, contact-law equivalence or hardware readiness."]}


def prepare_publication(trial_dir, reference_dir, render_dir, *, protocol_path, expected_protocol_sha256,
                        expected_source_sha256=None, repo_root=ROOT):
    require(base.sha256(Path(base.__file__)) == LEGACY_PUBLISHER_SHA256, "Frozen legacy publisher source changed")
    require(base.sha256(Path(lr.__file__)) == LR_PUBLISHER_SHA256, "Frozen012 pure validation source changed")
    protocol_path, trial_dir, repo_root = Path(protocol_path).resolve(), Path(trial_dir).resolve(), Path(repo_root).resolve()
    require(protocol_path.is_relative_to(repo_root), "Protocol must resolve inside the declared repository")
    digest(expected_protocol_sha256, "Independent sealed protocol SHA required")
    require(base.sha256(protocol_path) == expected_protocol_sha256, "Prospective protocol SHA differs")
    protocol = validate_protocol(base.read_json(protocol_path))
    require(trial_dir.name == "trial013", "Explicit trial013 only")
    require(render_dir is not None, "Full four-panel render required, including FAIL")
    summary, payloads, media, files, hashes = base.prepare_publication(trial_dir, reference_dir, render_dir)
    additional = {
        "lr_sync_protocol": protocol_path, "resume_learning_rate_audit": trial_dir / "training/resume_learning_rate_audit.json",
        "training_completion_verification": trial_dir / "diagnostics/training_completion_verification.json",
        "deployment_provenance_audit": trial_dir / "diagnostics/deployment_provenance_audit.json",
        "final_checkpoint": trial_dir / "training/model_final.pt", "lr_sync_publisher": Path(__file__),
    }
    for key, path in additional.items():
        require(path.is_file() and path.stat().st_size > 0, "Missing evidence: " + key)
        files[key], hashes[key] = path, base.sha256(path)
    for key in ("run_status", "runner"):
        require(key in files, "Missing completion input: " + key)
    source_pins = {**BASE_SOURCE_PINS, **NEW_SOURCE_PINS} if expected_source_sha256 is None else expected_source_sha256
    records = {key: base.read_json(files[key]) for key in ("manifest", "contract", "isaac_report", "mujoco_report", "run_status", "runner", "assessment", "adaptation", "export_report")}
    lr_audit = base.read_json(files["resume_learning_rate_audit"])
    completion = base.read_json(files["training_completion_verification"])
    proof = base.read_json(files["deployment_provenance_audit"])
    context = validate_context(records, files, hashes, protocol, lr_audit, completion, proof,
                               repo_root=Path(repo_root), expected_source_sha256=source_pins)
    for relative in ("training/resume_learning_rate_audit.json", "diagnostics/training_completion_verification.json"):
        proof_seals(proof, relative, trial_dir / relative)
    for key, path in (("lr_sync_protocol", protocol_path),):
        require(proof.get("input_sha256", {}).get("repository/" + str(path.relative_to(repo_root))) == hashes[key],
                "Paired proof does not seal protocol")
    for relative in source_pins:
        source = Path(repo_root) / relative
        proof_seals(proof, "repository/" + relative, source)
        files["lr_sync_source/" + relative], hashes["lr_sync_source/" + relative] = source, base.sha256(source)
    context["input_sha256"] = {key: hashes[key] for key in additional}
    lr_context = {k:v for k,v in context.items() if k != "ground_load_reserve"}
    summary["learning_rate_resume_context"] = base.sanitize(lr_context, base.private_identities({**records, "source": context}))
    summary["lr_synchronized_ground_load_reserve_context"] = {
        "schema_version": 1, "experiment_kind": EXPERIMENT_KIND, "comparison_trial": "trial012",
        "only_added_factor_relative_to_lr_synchronized_control": "wheel_ground_load_reserve",
        "lr_context_key": "learning_rate_resume_context", "reserve_context_key": "ground_load_reserve_context",
        "additional_training_updates": 600, "behavior_gates_changed": False,
    }
    summary["ground_load_reserve_context"] = base.sanitize(context["ground_load_reserve"],
        base.private_identities({**records, "source": context}))
    summary["experiment_kind"] = EXPERIMENT_KIND
    summary["additional_training_updates"] = 600
    summary["input_sha256"] = hashes
    # No base assessment/reference/render payload or media bytes are changed.
    require(len(payloads) == 3 and len(media) == 4, "Complete original four-panel payload required")
    require(all(base.sha256(path) == hashes[key] for key, path in files.items()), "Input changed during LR-context preflight")
    return summary, payloads, media, files, hashes


def publish_trial(trial_dir, reference_dir, output_dir, render_dir, **kwargs):
    output_dir = Path(output_dir)
    require(not output_dir.exists(), "Output must be NEW; refusing overwrite")
    summary, payloads, media, files, hashes = prepare_publication(trial_dir, reference_dir, render_dir, **kwargs)
    payloads["summary.json"] = base.encode(summary)
    require(all(base.sha256(path) == hashes[key] for key, path in files.items()), "Input changed before publication")
    require(all(base.sha256(item["path"]) == item["sha256"] for item in media.values()), "Media changed before publication")
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        with (output_dir / name).open("xb") as stream:
            stream.write(data)
    for name, item in media.items():
        with item["path"].open("rb") as source, (output_dir / name).open("xb") as target:
            shutil.copyfileobj(source, target)
        require(base.sha256(output_dir / name) == item["sha256"], "Media copy changed")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--protocol", dest="protocol_path", type=Path, default=ROOT / "config/car_drift_lr_sync_ground_load_trial.json")
    parser.add_argument("--protocol-sha256", dest="expected_protocol_sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = vars(parser.parse_args())
    summary = publish_trial(**args)
    print(json.dumps({"status": summary["status"], "verdict": summary["verdict"],
                      "output_dir": str(args["output_dir"])}))


if __name__ == "__main__":
    main()

