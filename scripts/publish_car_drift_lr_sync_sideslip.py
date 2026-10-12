"""Explicit trial015 publication: held-fixed LR/substeps plus sideslip weight.

Adds only validated context to the unchanged recorded car-drift publication.
No actor inference, checkpoint deserialization, physics or new behavior gate.
An independently sealed protocol SHA is required to avoid a self-hash cycle.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import hashlib
import struct
from pathlib import Path
import re
import shutil
import sys

import publish_car_drift_lr_sync as lr

base = lr.base

ROOT = base.ROOT
sys.path.insert(0, str(ROOT))
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
    "training/tron1_substep_rewards.py": "aeec31437226984a326a263109aaadebca004a6ef333f25a1d3377688aa276a5",
}
RSL_SOURCE_PINS = {
    "runners/on_policy_runner.py": "e51e4dd49664ccc8743a16b71d2971d392f40a6988136f10fe8c5f03ab6a83b5",
    "algorithms/ppo.py": "deafc8c947eba4df3e91b393869426cdab8d7b71e05974c3734125d2331d7d1c",
    "networks/normalization.py": "7db82f8021d62a660b4229782b52ac47e32d781206dba709d6bc436bf13f6672",
}
PROFILE_SHA256 = "a067a61a6b817639e26e41a8c7e20b0f3cff503ffda4f459cef43fd605bbd2ff"
BASELINE014_PROFILE_SHA256 = "6e35d93b3a92523d8eed40198f5b48a261cc6497565c953da4961fa106794a05"
SIDESLIP_GUARD_SHA256 = "e03e521377a6df50129cd13200b3232ba8b6905dabc86ff2174268d0207176e7"
FROZEN014_PUBLISHER_SHA256 = "56c2a7c20cd5783d26efe7d6e491c982898468e68674f799fb4659f35d7c5af6"
ORIGINAL_PROFILE_SHA256 = "325cf279c699769694fb4934852ce85753b459e09670376b8c2d62d94f7d4e3c"
LR_PUBLISHER_SHA256 = "497ded75605813f916380d01eee387017e444c9e75bb496c5440da96015f71cd"
PARENT_EXPORT_SHA256 = "63bd2ad72bbf5ca390552c7437eedf3a745c8eef1140cfb1540ee4b3e54ccc53"
SAMPLING_HELPER = "training/tron1_substep_rewards.py"
SAMPLING_SHA256 = "aeec31437226984a326a263109aaadebca004a6ef333f25a1d3377688aa276a5"
SAMPLING_WEIGHT = -5.0
SAMPLING_STD_M_S = .02
NATIVE_BODY_NAMES = ["base_Link", "abad_L_Link", "abad_R_Link", "limx_imu", "hip_L_Link",
                     "hip_R_Link", "knee_L_Link", "knee_R_Link", "wheel_L_Link", "wheel_R_Link"]
GROUND_LOAD_PUBLISHER_SHA256 = "2ebb2130532f5d6d28e4b905872e93c8351aac1a0b483fa216bb3b7cb46c6f11"
SMOKE_SOURCE_PINS = {
    "verify_smoke.py": "229476e99f4ec87fd306fdf1c31c8e9799e35bbd738860f17a2ea4e1c33f1bca",
    "smoke_substeps.py": "21c6c2472cdf1913952780425cb83bba2640d7cad8d1e883e84352a989c90c6b",
    "launch_smoke_trial015.py": "2d07a9965f13dff5aa51ed7d82aaa8130693dffc81a293a9f4a6d9e1984f35ea",
    "smoke_protocol.json": "46fffc079c52909471fcdd53ebaab571bc2c4967dafa8068d18ef29e6c870936",
}
EXPERIMENT_KIND = "lr_synchronized_sideslip_weight_continuation"
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
    "training/model_final.pt", "training/reward_sampling_final_audit.json", "isaac/report.json", "isaac/policy_contract.json", "isaac/actor_normalized.pt",
    "isaac/trajectory.npz", "isaac/physics_trajectory.npz", "mujoco/report.json", "mujoco/rollout.npz", "assessment.json")
LR_AUDIT_KEYS = {"schema_version", "scope", "status", "scalar_before", "loaded_group_learning_rates",
    "scalar_after", "touched_field", "optimizer_group_learning_rates_unchanged", "adaptive_schedule_preserved",
    "expected_scalar_before", "expected_loaded_lr", "multi_gpu"}
PAIRED_CHECK_NAMES = frozenset({
    "completed600_2048_24_seed42_original004_resume_and_all_manifest_source_profile_contract_fields",
    "actual25_model_51_optimizer_tensors_finite_and_iteration_status_consistent",
    "all_model_optimizer_tensors_finite",
    "fresh015_final_CP_and_matching51_120_8_500_10_50_200_contracts_no_overrides",
    "same_contract_profile_both_engines",
    "all12_exported_actor_normalizer_tensor_bytes_exact_fresh015_final_CP_NOT_baseline004",
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
    base.require(condition, "Trial015 LR-and-sideslip publication: " + message)


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
    require(isinstance(value, dict), label)
    result = value.get(key)
    require(isinstance(result, dict), label)
    return result


validate_lr_audit = lr.validate_lr_audit


def validate_protocol(protocol):
    require(isinstance(protocol, dict), "Protocol object required")
    integer(protocol.get("schema_version"), 1, "Protocol version")
    require(protocol.get("trial_id") == "trial015" and protocol.get("task") == "captured_car_drift"
            and protocol.get("experiment_type") == EXPERIMENT_KIND, "Explicit prospective015 protocol required")
    for key, value in (("baseline_trial", "trial014"), ("parent_trial", "trial004"),
                       ("original_training_task_profile_sha256", ORIGINAL_PROFILE_SHA256),
                       ("task_profile", "config/car_drift_lr_sync_sideslip_task.json"),
                       ("task_profile_sha256", PROFILE_SHA256), ("training_entrypoint", "scripts/train_tracking_lr_sync.py"),
                       ("training_helper", "training/tron1_resume_lr.py"), ("frozen_base_entrypoint", "scripts/train_tracking.py"),
                       ("frozen_base_entrypoint_sha256", BASE_SOURCE_PINS["scripts/train_tracking.py"]),
                       ("publication_entrypoint", "scripts/publish_car_drift_lr_sync_sideslip.py"),
                       ("sampling_helper", SAMPLING_HELPER), ("sampling_helper_sha256", SAMPLING_SHA256)):
        require(protocol.get(key) == value, "Protocol changed: " + key)
    require(protocol.get("expected_training_source_sha256") == {**BASE_SOURCE_PINS, **NEW_SOURCE_PINS},
            "Protocol must seal exact source10: original012 nine plus frozen aeec sampler, never reserve761")
    sampling = mapping_field(protocol, "sampling", "Original substep sampling declaration required")
    require(set(sampling) == {"mode", "signal", "formula", "quadrature", "weight", "std_m_s",
            "substeps_per_control_window", "physics_dt_s", "policy_dt_s",
            "initial_or_same_counter_reset_refresh_sampled", "startup_sampler_counts",
            "completed_training_sampler_counts", "evaluation_sampler_counts",
            "additional_final_training_file", "extra_observer_only_in_runtime_smoke"},
            "Exact original sampling declaration schema required")
    require(sampling.get("mode") == "all_substeps_mean"
            and sampling.get("signal") == "world-frame wheel LINK-ORIGIN vertical velocity, not COM velocity or contact-point separation"
            and sampling.get("formula") == "mean_s(sum_w((v_link_z_w_s / std_m_s)^2)) over exactly four completed native physical substeps",
            "Original square-before-mean wheel LINK-origin signal required")
    number(sampling.get("weight"), SAMPLING_WEIGHT, "Original substep weight")
    number(sampling.get("std_m_s"), SAMPLING_STD_M_S, "Original substep normalization")
    integer(sampling.get("substeps_per_control_window"), 4, "Four physical substeps required")
    number(sampling.get("physics_dt_s"), .005, "Original sampling physics clock")
    number(sampling.get("policy_dt_s"), .02, "Original sampling policy clock")
    require(sampling.get("initial_or_same_counter_reset_refresh_sampled") is False
            and sampling.get("additional_final_training_file") == "training/reward_sampling_final_audit.json",
            "No initialization/reset samples; actual final sampler file required")
    for field, steps in (("startup_sampler_counts", 8), ("completed_training_sampler_counts", 14408),
                         ("evaluation_sampler_counts", 288)):
        counts = mapping_field(sampling, field, "Explicit complete sampler counts required")
        require(set(counts) == {"control_steps", "substep_samples", "cost_reads", "failed_steps"}, "Exact sampler count schema")
        for key, value in (("control_steps", steps), ("substep_samples", 4 * steps), ("cost_reads", steps), ("failed_steps", 0)):
            integer(counts.get(key), value, "Protocol sampler: " + field + "/" + key)
    training = mapping_field(protocol, "training", "Protocol training object required")
    for key, value in (("iterations", 600), ("num_envs", 2048), ("rollout_steps_per_update", 24),
                       ("seed", 42), ("smoke_steps", 8), ("additional_environment_steps", 29_491_200),
                       ("parent_saved_iteration", 3996), ("expected_final_saved_iteration", 4595)):
        integer(training.get(key), value, "Protocol training: " + key)
    require(training.get("resume_checkpoint_sha256") == PARENT_CP_SHA256
            and training.get("parent_manifest_sha256") == PARENT_MANIFEST_SHA256
            and training.get("runner_config_sha256") == RUNNER_SHA256, "Original004 resume/config pins required")
    settings = mapping_field(protocol, "learning_rate_resume", "Protocol LR object required")
    require(settings.get("rsl_rl_version") == "3.0.1" and settings.get("installed_source_sha256") == RSL_SOURCE_PINS,
            "Installed RSL3.0.1 sources must match")
    for key, value in (("configured_ppo_learning_rate", .001), ("expected_scalar_before", .001),
                       ("expected_loaded_optimizer_learning_rate", 1.e-5), ("expected_scalar_after", 1.e-5)):
        number(settings.get(key), value, "Protocol LR: " + key)
    require(settings.get("exactly_uniform_optimizer_groups_required") is True
            and settings.get("only_mutated_field") == "algorithm.learning_rate"
            and settings.get("optimizer_group_learning_rates_or_moments_mutated") is False
            and settings.get("normalizers_or_counters_mutated") is False, "Scalar-only LR protocol required")
    unchanged = mapping_field(protocol, "unchanged", "Protocol unchanged object required")
    require(all(unchanged.get(key) == value for key, value in REFERENCE_PINS.items()),
            "Original motion/reference/robot/physics-model pins required")
    for key, value in (("policy_hz", 50), ("physics_hz_both_engines", 200)):
        integer(unchanged.get(key), value, "Unchanged protocol clock")
    for key, values in (("actor_critic_action_dimensions", [51, 120, 8]),):
        actual = unchanged.get(key)
        require(isinstance(actual, list) and len(actual) == len(values), "Original dimensions shape")
        for value, expected in zip(actual, values):
            integer(value, expected, "Original integer dimensions")
    for key, values in (("leg_kp_kd", [500., 10.]), ("leg_and_wheel_torque_limits_nm", [80., 12.]),
                        ("mujoco_contact_solref", [.04, 2.])):
        actual = unchanged.get(key)
        require(isinstance(actual, list) and len(actual) == len(values), "Original numeric control/contact shape")
        for value, expected in zip(actual, values):
            number(value, expected, "Original control/contact: " + key)
    number(unchanged.get("contact_friction"), .15, "Original friction")
    for key in ("observations_or_contract_changed", "contact_geometry_margin_or_gap_changed", "surrogate_contact_parameters_adopted"):
        require(unchanged.get(key) is False, "Unchanged protocol: " + key)
    evaluation = mapping_field(protocol, "evaluation", "Protocol evaluation object required")
    for key, value in (("episodes_per_engine", 1), ("num_envs", 1), ("seed", 42)):
        integer(evaluation.get(key), value, "Original evaluation: " + key)
    require(evaluation.get("assessor_source_sha256") == "a2214c47d0f8b262fb8f5b70a709908ef1d7487fd2b1393becb8ad8dfc672fa5",
            "Original assessor required")
    factor = mapping_field(protocol, "single_factor", "Declared single reward-weight factor required")
    for key, expected in (("before", 2.0), ("after", 4.0), ("error_scale_rad", .12),
                          ("reference_speed_mask_m_s", .3), ("actual_speed_mask_m_s", .25)):
        number(factor.get(key), expected, "Typed single factor: " + key)
    require(factor.get("signed_reference_error") is True
            and factor.get("contact_or_slip_or_duration_constraint_added") is False
            and factor.get("new_wheel_teacher") is False, "Typed unchanged single-factor flags")
    require(factor == {
        "baseline_profile": "config/car_drift_substep_task.json",
        "baseline_profile_sha256": BASELINE014_PROFILE_SHA256,
        "changed_path": "reward_weights.sideslip", "before": 2.0, "after": 4.0,
        "metadata_only_exception": "description",
        "formula": "exp(-(wrap(beta-beta_ref)/0.12)^2) * (reference_speed > 0.3) * (actual_speed > 0.25)",
        "angle_units": "radians", "signed_reference_error": True, "error_scale_rad": .12,
        "reference_speed_mask_m_s": .3, "actual_speed_mask_m_s": .25,
        "unchanged_frame_convention": "Reward uses full inverse root quaternion; original assessor uses horizontal yaw-only beta. Preserve this existing difference.",
        "contact_or_slip_or_duration_constraint_added": False, "new_wheel_teacher": False,
    }, "Only predeclared signed sideslip weight2->4; no error-scale/frame/mask/teacher change")
    return protocol


def validate_completion(completion, checkpoint_sha):
    """Retain the original budget checks and require the actual015 CPU proof."""
    lr.validate_completion(completion, checkpoint_sha)
    integer(completion.get("schema_version"), 1, "CPU completion schema required")
    require(completion.get("original_sideslip_profile_and_explicit_ten_training_sources_verified") is True,
            "CPU completion must independently verify the original all-substep profile and explicit source10")
    require(not any(key in completion for key in (
        "original_substep_profile_and_explicit_ten_training_sources_verified",
        "original_ground_load_profile_and_explicit_ten_training_sources_verified",
        "original_all_substep_profile_and_explicit_ten_training_sources_verified",
        "original_nominal_profile_and_explicit_nine_training_sources_verified")),
        "No stale or mixed experiment-verification flags")
    require(completion.get("all_input_hashes_unchanged") is True,
            "CPU completion input hashes must remain unchanged")
    cpu_inputs = mapping_field(completion, "input_sha256", "CPU source/input seals required")
    require(cpu_inputs.get("repository/training/tron1_drift_sideslip_trial.py") == SIDESLIP_GUARD_SHA256
            and cpu_inputs.get("repository/config/car_drift_substep_task.json") == BASELINE014_PROFILE_SHA256,
            "CPU proof must seal the new pure helper and unchanged014 control profile")
    for field in ("new_actor_inference_calls", "new_optimizer_updates", "simulation_steps", "cuda_allocations"):
        integer(completion.get(field), 0, "CPU-only completion: " + field)
    require(completion.get("behavior_verified") is False and completion.get("hardware_ready") is False,
            "CPU completion cannot establish behavior or hardware readiness")
    return completion


proof_seals = lr.proof_seals


def validate_actor_proof(proof, trial_dir, files, hashes):
    """Require the exact015 producer checks, retaining every original bound."""
    lr.validate_actor_proof(proof, trial_dir, files, hashes)
    integer(proof.get("schema_version"), 1, "Paired actor-proof schema required")
    for relative in FINAL_INPUTS:
        proof_seals(proof, relative, Path(trial_dir) / relative)
    require(all(isinstance(key, str) for key in proof["input_sha256"]), "String proof-input aliases required")
    actual_aliases = {key for key in proof["input_sha256"] if not key.startswith("repository/")}
    require(actual_aliases == set(FINAL_INPUTS) | {"training/resume_learning_rate_audit.json",
            "diagnostics/training_completion_verification.json"}, "Exactly fifteen actual runtime-input aliases required")
    require(set(proof["checks"]) == PAIRED_CHECK_NAMES,
            "All exact fifteen015 paired-provenance checks are required")
    integer(proof.get("new_rollouts"), 0, "Recorded-only actor proof cannot create rollouts")
    integer(proof.get("simulation_steps"), 0, "Recorded-only actor proof cannot simulate")
    require(proof.get("behavior_success_claimed") is False,
            "Paired actor provenance alone cannot claim behavioral success")
    return proof


SAMPLING_AUDIT_KEYS = frozenset({
    "schema_version", "mode", "fingerprint_sha256", "installed", "inside_step", "sampling_open",
    "num_envs", "wheel_body_ids", "device", "dtype", "decimation", "physics_dt_s", "policy_dt_s",
    "steps_started", "steps_completed", "steps_failed", "samples_last_control_step", "samples_total",
    "last_start_counter", "last_sample_counter", "last_step_completed", "native_clock_required",
    "native_start_step_index", "native_last_step_index", "native_start_time_s", "native_last_time_s",
    "native_time_float32_dt_rounding_bound_s", "native_time_double_rounding_allowance_ulps", "cost_reads",
})


def finite(value, label, *, minimum=0.):
    require(type(value) in (int, float), label)
    try:
        valid = math.isfinite(value)
    except (OverflowError, ValueError):
        valid = False
    require(valid and value >= minimum, label)
    return float(value)


def validate_sampling_audit(audit, *, num_envs, expected_steps, body_names):
    """Validate aeec scalar metadata; never infer actual per-sample tensors.

    Supported full-stage counts are predeclared. Native epoch/index offsets
    need not equal the environment epoch; only actual consecutive windows and
    the original float32 callback-dt plus eight-double-ULP allowance are used.
    """
    require(isinstance(audit, dict) and set(audit) == SAMPLING_AUDIT_KEYS,
            "Exact frozen aeec sampling-audit schema required")
    require(type(num_envs) is int and type(expected_steps) is int
            and (num_envs, expected_steps) in {(2048, 8), (2048, 14408), (1, 288), (8, 16)},
            "Only declared sampler stages supported")
    require(body_names == NATIVE_BODY_NAMES, "Actual native ten-body order required")
    for field, expected in (("schema_version", 1), ("num_envs", num_envs), ("decimation", 4),
                           ("steps_started", expected_steps), ("steps_completed", expected_steps),
                           ("steps_failed", 0), ("samples_last_control_step", 4),
                           ("samples_total", 4 * expected_steps), ("cost_reads", expected_steps),
                           ("last_start_counter", 4 * expected_steps - 4),
                           ("last_sample_counter", 4 * expected_steps),
                           ("native_time_double_rounding_allowance_ulps", 8)):
        integer(audit.get(field), expected, "Sampler exact counter: " + field)
    require(audit.get("mode") == "all_substeps_mean" and audit.get("device") == "cuda:5"
            and audit.get("dtype") == "torch.float32", "Original all-substep CUDA5/float32 signal required")
    ids = audit.get("wheel_body_ids")
    require(isinstance(ids, list) and len(ids) == 2, "Exactly two native wheel IDs required")
    for value, expected in zip(ids, (8, 9)):
        integer(value, expected, "Named left/right native wheel ID")
    require([body_names[i] for i in ids] == ["wheel_L_Link", "wheel_R_Link"],
            "Sampler IDs must identify the actual wheel links")
    for field, expected in (("installed", True), ("inside_step", False), ("sampling_open", False),
                           ("last_step_completed", True), ("native_clock_required", True)):
        require(audit.get(field) is expected, "Sampler lifecycle flag: " + field)
    number(audit.get("physics_dt_s"), .005, "Actual native sampler 200Hz clock")
    number(audit.get("policy_dt_s"), .02, "Actual native sampler 50Hz clock")
    fingerprint = {"schema_version": 1, "mode": "all_substeps_mean", "wheel_body_ids": (8, 9),
                   "decimation": 4, "physics_dt_s": .005, "policy_dt_s": .02, "num_envs": num_envs}
    require(audit.get("fingerprint_sha256") == hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest(),
            "Actual sampler fingerprint differs from frozen aeec configuration")
    start, end = audit.get("native_start_step_index"), audit.get("native_last_step_index")
    require(type(start) is int and type(end) is int and start >= 0 and end == start + 4,
            "Native physics must advance exactly four actual steps")
    start_time = finite(audit.get("native_start_time_s"), "Native start time must be finite nonnegative")
    end_time = finite(audit.get("native_last_time_s"), "Native last time must be finite nonnegative")
    rounding = abs(struct.unpack("f", struct.pack("f", .005))[0] - .005)
    number(audit.get("native_time_float32_dt_rounding_bound_s"), rounding, "Exact float32 native-dt bound required")
    tolerance = 4 * rounding + 8 * math.ulp(max(abs(start_time), abs(end_time), .005))
    require(tolerance < .005 and end_time > start_time
            and abs((end_time - start_time) - .02) <= tolerance,
            "Native elapsed window must equal four5ms steps within only f32-dt/eight-double-ULP bounds")
    return copy.deepcopy(audit)


def validate_native_device_audit(audit, label):
    """Reuse public stdlib-only validator; reject extra or invented readbacks."""
    from training.tron1_runtime_device import validate_device_audit
    require(isinstance(audit, dict), "Actual native device audit required: " + label)
    checked = validate_device_audit(audit, required_unmasked=True)
    require(set(audit) == set(checked) and audit == checked, "Exact native device audit schema: " + label)
    integer(audit.get("schema_version"), 1, "Native device audit version")
    require(audit.get("requested_device") == "cuda:5", "Declared native CUDA5 target required: " + label)
    for field, value in (("required_unmasked", True), ("native_and_tensor_device_consistency_verified", True),
                         ("physical_mapping_verified", False), ("exclusivity_verified", False)):
        require(audit.get(field) is value, "Native device evidence scope: " + field)
    return checked


def validate_smoke_prerequisite(completion, proof, source_pins, protocol_sha256):
    """Validate sealed prerequisite metadata; never import the private verifier.

    The prior independent producer verifies raw fixture values and owned exit.
    This publication route preserves its source/input seals, not a second GPU
    fixture execution or an independent all2048-training cost measurement.
    """
    prerequisite = mapping_field(completion, "current_stack_smoke_prerequisite", "Current-stack smoke prerequisite required")
    require(set(prerequisite) == {"smoke_verification_sha256", "status", "input_sha256", "sideslip_reward_configuration"}
            and prerequisite.get("status") == "current_stack_smoke_prerequisite_verified",
            "Exact current-stack smoke prerequisite schema/status")
    from training.tron1_drift_sideslip_trial import validate_live_reward_audit
    validate_live_reward_audit(prerequisite["sideslip_reward_configuration"], wheel_vz_capture_expected=True)
    digest(prerequisite.get("smoke_verification_sha256"), "Actual smoke-verification SHA required")
    inputs = mapping_field(prerequisite, "input_sha256", "Current-stack smoke input seals required")
    names = set(source_pins) | set(SMOKE_SOURCE_PINS) | {
        "smoke/launch_manifest.json", "smoke/science_report.json", "smoke/window_evidence.npz",
        "smoke/teardown_report.json", "smoke_launch_receipt.json", "smoke_started.json",
        "smoke_exit_report.json", "smoke.log", "deployment.tar", "config/car_drift_lr_sync_sideslip_task.json",
        "config/car_drift_contact_task.json", "config/car_drift_lr_sync_sideslip_trial.json",
        "config/car_drift_substep_task.json", "training/tron1_drift_sideslip_trial.py",
        "smoke/live_reward_configuration.json",
    }
    require(set(inputs) == names, "Exact current-source smoke input aliases; old005/reserve smoke cannot substitute")
    for value in inputs.values():
        digest(value, "Every current-stack smoke input needs a SHA256 seal")
    archive = digest(completion.get("archive_sha256"), "CPU completion archive SHA required")
    expected = {**source_pins, **SMOKE_SOURCE_PINS, "deployment.tar": archive,
                "config/car_drift_lr_sync_sideslip_task.json": PROFILE_SHA256,
                "config/car_drift_contact_task.json": ORIGINAL_PROFILE_SHA256,
                "config/car_drift_lr_sync_sideslip_trial.json": protocol_sha256,
                "config/car_drift_substep_task.json": BASELINE014_PROFILE_SHA256,
                "training/tron1_drift_sideslip_trial.py": SIDESLIP_GUARD_SHA256}
    require(all(inputs.get(key) == value for key, value in expected.items()),
            "Current-source smoke profile/helper/protocol/archive seals differ")
    require(proof.get("current_stack_smoke_prerequisite") == prerequisite,
            "Paired actor proof must preserve the actual CPU-verified smoke prerequisite")
    return copy.deepcopy(prerequisite)


def validate_sampling_context(records, sampling_final, profile):
    manifest, isaac, mujoco, contract = [mapping_field(records, key, "Sampler metadata object required") for key in
                                        ("manifest", "isaac_report", "mujoco_report", "contract")]
    from training.tron1_drift import validate_task_profile
    validate_task_profile(profile)
    for record in (manifest, isaac, mujoco, contract, manifest["contract"], mujoco["contract"]):
        require(not any(key in record for key in ("ground_load_reserve", "ground_load_reserve_module_sha256",
                                                  "ground_load_reserve_context", "wheel_ground_load_reserve")),
                "No simultaneous or orphan reserve reward/helper/cache context")
        material = record.get("contact_material_audit", {})
        require(isinstance(material, dict) and "ground_load_reserve" not in material,
                "Substep velocity task cannot include a reserve cache/material audit")
    require("wheel_ground_load_reserve" not in profile, "No reserve reward in original sampling-only profile")
    require(isaac.get("substep_reward_module_sha256") == SAMPLING_SHA256,
            "Actual eval sampler helper pin required")
    require("substep_reward_module_sha256" not in manifest and "substep_reward_module_sha256" not in contract
            and "reward_sampling_audit" not in contract and "reward_sampling_audit" not in mujoco
            and "substep_reward_module_sha256" not in mujoco, "Sampler metadata belongs to actual Isaac execution only")
    require(isaac.get("evaluator_source_sha256") == "fd90089810d3d14bfd96210723afdd3a36113a19f667ec60b2df617dd032f992"
            and isaac.get("runtime_device_module_sha256") == BASE_SOURCE_PINS["training/tron1_runtime_device.py"],
            "Actual frozen evaluator/native-device sources required")
    require(manifest.get("body_names") == contract.get("actual_body_order") == NATIVE_BODY_NAMES,
            "Actual training/export native body order changed")
    audits = {
        "training_startup": validate_sampling_audit(manifest.get("reward_sampling_audit"),
            num_envs=2048, expected_steps=8, body_names=manifest["body_names"]),
        "training_final": validate_sampling_audit(sampling_final,
            num_envs=2048, expected_steps=14408, body_names=manifest["body_names"]),
        "isaac_evaluation": validate_sampling_audit(isaac.get("reward_sampling_audit"),
            num_envs=1, expected_steps=288, body_names=contract["actual_body_order"]),
    }
    devices = {engine: validate_native_device_audit(parent.get("runtime_device_audit"), engine)
               for engine, parent in (("training", manifest), ("isaac_evaluation", isaac))}
    for label, parent in (("Training", manifest), ("Isaac", isaac), ("MuJoCo", mujoco)):
        base.check_material_audit(parent.get("contact_material_audit"), .15, label)
    for label, parent in (("Training", manifest), ("Isaac", isaac)):
        base.check_leg_gain_audit(parent["contact_material_audit"], 500., 10., label)
    contact = mujoco["contact_material_audit"]
    require(contact.get("contact_margin_and_gap_changed") is False
            and contact.get("explicit_contact_solref") == [.04, 2.]
            and contact.get("compiled_solref_ranges") == [[.04, .04], [2., 2.]],
            "Original MuJoCo contact response/margin/gap required")
    return {
        "schema_version": 1, "validation_kind": "source_backed_native_sampler_metadata",
        "supported_helper_sha256": SAMPLING_SHA256,
        "settings": {"mode": "all_substeps_mean", "weight": SAMPLING_WEIGHT, "std_m_s": SAMPLING_STD_M_S},
        "signal": "world-frame wheel LINK-origin vertical velocity, not COM velocity or contact-point separation",
        "formula": "mean_s(sum_w((v_link_z_w_s / std_m_s)^2)); square each sample before arithmetic averaging",
        "unchanged_reward_manager_integration_dt_s": .02,
        "sampler_audits": audits, "native_device_audits": devices,
        "full_training_sample_values_independently_archived": False,
        "full_training_live_reward_values_independently_proved": False,
        "continuous_contact_certified": False, "behavior_gates_changed": False,
        "normal_force_envelope_equivalence_claimed": False, "hardware_ready": False,
        "limitations": [
            "Source-backed scalar configuration and sampling-counter metadata, not independent full2048 training tensor/reward archives.",
            "LINK-origin vertical velocity is not COM velocity or actual contact-point separation; correct quadrature does not guarantee load-bearing contact.",
            "Square-before-mean is the original formula; the mean cost can be below the last-substep cost, not necessarily stronger.",
            "The native clock tolerates only declared float32 callback-dt rounding and eight double ULPs; no counter gaps or reset samples are bridged.",
            "Unmasked native/tensor CUDA ordinal readback is not physical UUID mapping, exclusive resource reservation or hardware readiness.",
        ],
    }


def validate_context(records, files, hashes, protocol, lr_audit, completion, proof, sampling_final, *,
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
    versions = mapping_field(manifest, "versions", "Actual installed version map required")
    require(versions.get("rsl-rl-lib") == "3.0.1", "Actual installed RSL version required")
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
    require(base.sha256(profile_path) == PROFILE_SHA256, "Declared trial015 sideslip-weight profile changed")
    profile = base.read_json(profile_path)
    guard_path = repo_root / "training/tron1_drift_sideslip_trial.py"
    require(base.sha256(guard_path) == SIDESLIP_GUARD_SHA256, "New pure single-factor guard source changed")
    from training.tron1_drift_sideslip_trial import validate_sideslip_profile, validate_original_parent_profile
    baseline014_path = repo_root / "config/car_drift_substep_task.json"
    require(base.sha256(baseline014_path) == BASELINE014_PROFILE_SHA256, "Immutable014 profile changed")
    baseline014_profile = base.read_json(baseline014_path)
    validate_sideslip_profile(profile, baseline014_profile)
    original_profile = baseline.get("task_profile")
    original_path = repo_root / "config/car_drift_contact_task.json"
    require(base.sha256(original_path) == ORIGINAL_PROFILE_SHA256
            and base.read_json(original_path) == original_profile, "Original004 profile/source changed")
    validate_original_parent_profile(profile, original_profile)
    number(profile.get("wheel_vertical_velocity_weight"), SAMPLING_WEIGHT, "Unchanged original wheel-vz weight")
    number(profile.get("wheel_vertical_velocity_std_m_s"), SAMPLING_STD_M_S, "Unchanged original wheel-vz standard deviation")
    original_contract = mapping_field(baseline, "contract", "Original004 training contract required")
    expected_contract = copy.deepcopy(original_contract)
    expected_contract["task_profile"] = profile
    require(manifest.get("contract") == expected_contract
            and manifest.get("domain_randomization_profile") == baseline.get("domain_randomization_profile"),
            "Only declared sideslip profile may change the original full training contract")
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
            == mujoco.get("contract", {}).get("task_profile") == profile, "Fixed substep profile must remain exact in training and both engines")
    require(manifest.get("motion_sha256") == contract.get("reference_file_sha256")
            == hashes["motion"] == REFERENCE_PINS["motion_sha256"]
            and hashes["robot_reference"] == REFERENCE_PINS["robot_reference_sha256"]
            and manifest.get("asset_sha256") == isaac.get("asset_root_file_sha256") == REFERENCE_PINS["robot_asset_sha256"]
            and mujoco.get("model_sha256") == REFERENCE_PINS["mujoco_model_sha256"], "Original reference/asset/physical-model chain required")
    for record in (contract, isaac, mujoco):
        require("resume_learning_rate_audit" not in record and "resume_source_audit" not in record,
                "LR metadata belongs to training, not a changed actor/evaluator contract")
    for record in (manifest, contract, isaac, mujoco):
        require(not any(key in record for key in ("controller_gain_diagnostic", "physics_clock_refinement")),
                "No simultaneous gain or clock factor")
    sampling_context = validate_sampling_context(records, sampling_final, profile)
    domain = mapping_field(mujoco, "domain_parameters", "Actual nominal MuJoCo motor draw required")
    require(set(domain) == {"is_nominal", "joint_names", "torque_scale", "velocity_scale",
                            "wheel_friction_nm", "friction_smoothing_rad_s"},
            "Only the original nominal motor-draw schema is supported")
    require(domain.get("is_nominal") is True and domain.get("joint_names") == contract["observed_joint_velocity_order"],
            "Named nominal MuJoCo motor draw required")
    for field, count, expected_value in (("torque_scale", 8, 1.), ("velocity_scale", 8, 1.), ("wheel_friction_nm", 2, 0.)):
        values = domain.get(field)
        require(isinstance(values, list) and len(values) == count, "Nominal motor draw shape: " + field)
        for value in values:
            number(value, expected_value, "Nominal motor draw: " + field)
    number(domain.get("friction_smoothing_rad_s"), .5, "Original nominal wheel-friction smoothing required")
    require(isaac.get("domain_parameters") is None, "Nominal Isaac motors required")
    for key, value in (("num_envs", 2048), ("iterations_requested", 600), ("seed", 42), ("smoke_steps_passed", 8)):
        integer(manifest.get(key), value, "Actual training: " + key)
    require(manifest.get("resume_sha256") == PARENT_CP_SHA256, "Original004 parent checkpoint required")
    status, runner = [mapping_field(records, k, "Completion metadata object required: " + k) for k in ("run_status", "runner")]
    require(status.get("status") == "training_completed", "Final completed training required")
    for key, value in (("num_envs", 2048), ("iterations_requested", 600), ("training_environment_steps", 29_491_200), ("last_iteration", 4595)):
        integer(status.get(key), value, "Actual completed budget: " + key)
    integer(runner.get("num_steps_per_env"), 24, "Actual PPO rollout length")
    require(base.sha256(files["runner"]) == RUNNER_SHA256, "Original complete PPO configuration required")
    require(isinstance(status.get("checkpoint"), str) and bool(status["checkpoint"]), "Completed checkpoint path required")
    completed_path = Path(status["checkpoint"])
    require(completed_path.name == "model_final.pt"
            and completed_path.parent.parts[-3:] == ("outputs", "trial015", "training"),
            "Completed status must name final015, not an intermediate or historical checkpoint")
    checkpoint_sha = hashes["final_checkpoint"]
    require(checkpoint_sha == contract.get("checkpoint_sha256") == isaac.get("checkpoint_sha256"), "Actual final015 checkpoint/export chain required")
    validate_completion(completion, checkpoint_sha)
    integer(completion.get("actual_adam_step_increment"), 12000, "Actual complete600-update Adam increment")
    integer(completion.get("actual_actor_and_critic_normalizer_count_increment"), 29_491_200,
            "Actual complete600-update normalization increment")
    require(completion.get("reward_sampling_initial_audit") == manifest["reward_sampling_audit"]
            and completion.get("reward_sampling_final_audit") == sampling_final
            and completion.get("substep_reward_module_sha256") == SAMPLING_SHA256,
            "CPU completion must retain exact actual startup/final sampler metadata and helper source")
    require(completion.get("resume_learning_rate_audit") == lr_audit
            and completion.get("resume_source_audit") == source_audit,
            "Completion must retain exact startup LR/source metadata")
    completion_inputs = completion.get("input_sha256", {})
    require(isinstance(completion_inputs, dict), "CPU completion input seals required")
    training_dir = files["manifest"].parent
    for name in ("manifest.json", "run_status.json", "runner_config.json", "model_final.pt", "resume_learning_rate_audit.json",
                 "reward_sampling_final_audit.json"):
        require(completion_inputs.get("training/" + name) == base.sha256(training_dir / name),
                "CPU completion must seal actual015 training/" + name)
    require(completion_inputs.get("repository/config/car_drift_lr_sync_sideslip_trial.json") == hashes["lr_sync_protocol"],
            "CPU completion must seal prospective protocol")
    for relative, expected_sha in expected.items():
        require(completion_inputs.get("repository/" + relative) == expected_sha,
                "CPU completion must seal training source: " + relative)
    validate_actor_proof(proof, files["manifest"].parent.parent, files, hashes)
    require(proof.get("reward_sampling_initial_audit") == manifest["reward_sampling_audit"]
            and proof.get("reward_sampling_final_audit") == sampling_final
            and proof.get("reward_sampling_evaluation_audit") == isaac["reward_sampling_audit"]
            and proof.get("substep_reward_module_sha256") == SAMPLING_SHA256,
            "Paired proof must preserve exact actual startup/final/evaluation sampler metadata")
    smoke_prerequisite = validate_smoke_prerequisite(completion, proof, expected, hashes["lr_sync_protocol"])
    for relative in ("training/resume_learning_rate_audit.json", "diagnostics/training_completion_verification.json",
                     "training/reward_sampling_final_audit.json"):
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
        "schema_version": 1, "experiment_kind": "lr_synchronized_sideslip_weight_continuation",
        "evidence_scope": "Source-pinned actual startup scalar audit, full CPU budget/tensor proof and paired final-actor provenance; not independent full-minibatch LR telemetry.",
        "restoration": lr_audit, "installed_rsl_sources": source_audit,
        "comparison_trial": "trial014",
        "only_added_factor_relative_to_lr_synchronized_control": "sideslip_reward_weight",
        "substep_velocity_sampling": sampling_context,
        "current_stack_smoke_prerequisite": smoke_prerequisite,
        "sampling_profile_sha256": PROFILE_SHA256,
        "sampling_profile_provenance": "Immutable014 all-substep profile is the primary control; only sideslip weight2->4 and description metadata differ. Initial LR synchronization and all-four sampling are held fixed.",
        "single_factor": copy.deepcopy(protocol["single_factor"]),
        "single_factor_guard_source_sha256": SIDESLIP_GUARD_SHA256,
        "original_training_task_profile": original_profile,
        "effective_training_and_deployment_task_profile": profile,
        "training_source_sha256": expected, "parent_checkpoint_sha256": PARENT_CP_SHA256,
        "actual_final_checkpoint_sha256": checkpoint_sha, "actual_actor_sha256": hashes["actor"],
        "budget": {"updates": 600, "num_envs": 2048, "rollout_steps_per_update": 24, "seed": 42,
                   "additional_environment_steps": 29_491_200, "saved_iteration": 4595},
        "behavior_gates_changed": False, "learned_actor_equals_baseline_required": False,
        "legacy_publication_route_is_lr_aware": False,
        "limitations": ["The frozen legacy/012/013 routes are not the explicit combined trial015 publication route.",
            "After initial scalar synchronization, the original adaptive schedule may change LR normally.",
            "Counts and fingerprints are source-backed sampler metadata, not an independent archive of every training tensor or live reward value.",
            "The sealed current-stack eight-environment smoke remains limited fixture evidence, not a full-training or contact-physics equivalence test.",
            "Wheel LINK-origin vertical velocity is not COM velocity or contact-point separation; substep coverage does not establish continuous bearing contact.",
            "The mean squares each native sample before averaging; it is not necessarily larger than the last-sample cost.",
            "Startup audit/source proof does not independently measure every mini-batch learning rate or establish a causal explanation.",
            "One seed and one paired episode do not establish statistical robustness, contact-law equivalence or hardware readiness."]}


def prepare_publication(trial_dir, reference_dir, render_dir, *, protocol_path, expected_protocol_sha256,
                        expected_source_sha256=None, repo_root=ROOT):
    require(base.sha256(Path(base.__file__)) == LEGACY_PUBLISHER_SHA256, "Frozen legacy publisher source changed")
    require(base.sha256(Path(lr.__file__)) == LR_PUBLISHER_SHA256, "Frozen012 pure validation source changed")
    require(base.sha256(Path(__file__).with_name("publish_car_drift_lr_sync_ground_load.py")) == GROUND_LOAD_PUBLISHER_SHA256,
            "Frozen013 publisher source changed")
    require(base.sha256(Path(__file__).with_name("publish_car_drift_lr_sync_substeps.py")) == FROZEN014_PUBLISHER_SHA256,
            "Frozen014 publisher source changed")
    protocol_path, trial_dir, repo_root = Path(protocol_path).resolve(), Path(trial_dir).resolve(), Path(repo_root).resolve()
    require(protocol_path.is_relative_to(repo_root), "Protocol must resolve inside the declared repository")
    digest(expected_protocol_sha256, "Independent sealed protocol SHA required")
    require(base.sha256(protocol_path) == expected_protocol_sha256, "Prospective protocol SHA differs")
    protocol = validate_protocol(base.read_json(protocol_path))
    require(trial_dir.name == "trial015", "Explicit trial015 only")
    require(render_dir is not None, "Full four-panel render required, including FAIL")
    summary, payloads, media, files, hashes = base.prepare_publication(trial_dir, reference_dir, render_dir)
    additional = {
        "lr_sync_protocol": protocol_path, "resume_learning_rate_audit": trial_dir / "training/resume_learning_rate_audit.json",
        "training_completion_verification": trial_dir / "diagnostics/training_completion_verification.json",
        "deployment_provenance_audit": trial_dir / "diagnostics/deployment_provenance_audit.json",
        "final_checkpoint": trial_dir / "training/model_final.pt", "lr_sync_publisher": Path(__file__),
        "reward_sampling_final_audit": trial_dir / "training/reward_sampling_final_audit.json",
        "sideslip_factor_guard": repo_root / "training/tron1_drift_sideslip_trial.py",
        "baseline014_profile": repo_root / "config/car_drift_substep_task.json",
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
    sampling_final = base.read_json(files["reward_sampling_final_audit"])
    context = validate_context(records, files, hashes, protocol, lr_audit, completion, proof, sampling_final,
                               repo_root=Path(repo_root), expected_source_sha256=source_pins)
    for relative in ("training/resume_learning_rate_audit.json", "diagnostics/training_completion_verification.json"):
        proof_seals(proof, relative, trial_dir / relative)
    for key, path in (("lr_sync_protocol", protocol_path),):
        require(proof.get("input_sha256", {}).get("repository/" + str(path.relative_to(repo_root))) == hashes[key],
                "Paired proof does not seal protocol")
    for relative in ("training/tron1_drift_sideslip_trial.py", "config/car_drift_substep_task.json"):
        proof_seals(proof, "repository/" + relative, repo_root / relative)
    for relative in source_pins:
        source = Path(repo_root) / relative
        proof_seals(proof, "repository/" + relative, source)
        files["lr_sync_source/" + relative], hashes["lr_sync_source/" + relative] = source, base.sha256(source)
    context["input_sha256"] = {key: hashes[key] for key in additional}
    lr_context = {k:v for k,v in context.items() if k != "substep_velocity_sampling"}
    summary["learning_rate_resume_context"] = base.sanitize(lr_context, base.private_identities({**records, "source": context}))
    summary["lr_synchronized_sideslip_weight_context"] = {
        "schema_version": 1, "experiment_kind": EXPERIMENT_KIND, "comparison_trial": "trial014",
        "only_added_factor_relative_to_lr_synchronized_control": "sideslip_reward_weight",
        "lr_context_key": "learning_rate_resume_context", "sampling_context_key": "substep_velocity_sampling_context",
        "additional_training_updates": 600, "behavior_gates_changed": False,
    }
    summary["substep_velocity_sampling_context"] = base.sanitize(context["substep_velocity_sampling"],
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
    parser.add_argument("--protocol", dest="protocol_path", type=Path, default=ROOT / "config/car_drift_lr_sync_sideslip_trial.json")
    parser.add_argument("--protocol-sha256", dest="expected_protocol_sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = vars(parser.parse_args())
    summary = publish_trial(**args)
    print(json.dumps({"status": summary["status"], "verdict": summary["verdict"],
                      "output_dir": str(args["output_dir"])}))


if __name__ == "__main__":
    main()
