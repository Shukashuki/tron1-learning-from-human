"""Explicit trial017 wheel-axle publication; read-only preflight before NEW output.

Directly validates actual017 profiles, source maps, two completion seals and
paired evidence. Frozen016 cache/builder logic is copied with explicit017 hooks;
there is no record projection, monkeypatch, checkpoint or actor execution.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct

import publish_car_drift_lr_sync as lr

base = lr.base
ROOT = base.ROOT
from training import tron1_wheel_axle_trial as strict
import publish_car_drift_dense_ground_load as dense

require_base = base.require
read_json, sha256, encode = base.read_json, base.sha256, base.encode
sanitize, private_identities, pick = base.sanitize, base.private_identities, base.pick
assess_runs, validate_provenance = base.assess_runs, base.validate_provenance
check_material_audit, check_leg_gain_audit = base.check_material_audit, base.check_leg_gain_audit
validate_controller_gain_context = base.validate_controller_gain_context
validate_render, training_budget = base.validate_render, base.training_budget
GROUND_LOAD_RESERVE_SOURCE_PINS = base.GROUND_LOAD_RESERVE_SOURCE_PINS
GROUND_LOAD_RESERVE_BASELINE_SHA = base.GROUND_LOAD_RESERVE_BASELINE_SHA
GROUND_LOAD_RESERVE_ASSET_PINS = base.GROUND_LOAD_RESERVE_ASSET_PINS

PROTOCOL_FILE = "config/car_drift_wheel_axle_trial.json"
PROFILE_FILE = "config/car_drift_wheel_axle_task.json"
PROTOCOL_SHA256 = "250983571ebe2a757cf800ce6e3769763b719600c3312890e12fac049c995b3f"
PROFILE_SHA256 = "1db17ea1bb8820396e37bdf7043b6852151fda2b500c69f5d9a7716283932f78"
SOURCE_COMMIT = "a50cd96989c70cb7cc1f3fb4863959e7eeb4326a"
ARCHIVE_SHA256 = "5b1cbab5e680aff6ec8cc9174b5292e83790f9c14f00dff2f416c68e7a177fbd"
CPU_PRODUCER_SHA256 = "fc348af116fd002639f923fd05cb4fbc21373297b5ae601321ce5155ab27c5e7"
CPU_LAUNCHER_SHA256 = "78858401033ac69c9f7dec790b1133deab03d296ddaa0599f5ef432fc0f8b875"
PAIRED_PRODUCER_SHA256 = "07e3d2b44c892691ce1e6fb9a008d86a01f9d96c34820cd708e362698b54d9fe"
EVALUATION_LAUNCHER_SHA256 = "8e065bae62c7f5dc0b4a80831f830a6e5e54cce76311c93479507317d95976b9"
SHARED_ENTRYPOINT_SHA256 = "78886a27d22a9305b08f4e37d771d9f128685ee754b4e6d047dce722d6f3ac25"
SHARED_AUTHORIZATION_SHA256 = "3b17c49aaf09422247f4dd3a3ad7c1c815f4165de3cd6ec2b34f3e203ba671bf"
DENSE_PUBLISHER_SHA256 = "cb3897f3185974f02cc29636aff0472784ffb952ff4055defd2c59a68e493f06"
LEGACY_PUBLISHER_SHA256 = "598cd0cf6a78918da8a5e09ca0f20624c0a7379c7311813923366fd2c9b4293d"
LR_PUBLISHER_SHA256 = "497ded75605813f916380d01eee387017e444c9e75bb496c5440da96015f71cd"
EXPERIMENT_KIND = "lr_synchronized_wheel_axle_priority_continuation"
SOURCE27 = {
    "training/tron1_tracking.py": "0a29fa27c1d65c0b9add0886221491e549553e3529d6970f1ec5ba35e2532817",
    "training/tron1_drift.py": "6d5f7974602adec9313a5e4feceb8e3caab392ca99351ac4a5f717e57fe1cf8a",
    "training/tron1_domain_randomization.py": "3f3a0b62c85a0af496ae9e9e0d75dda85c3ec30da587e4a47969843f5b1b1dca",
    "training/tron1_contact_slip.py": "52ef38de9cece19a8ee02c120cb85d472ffd831e4587e5f0d1f340974ab699f4",
    "training/tron1_runtime_device.py": "2a1460d1b48e884fc5608e0365edc43b815d1fc74c8439945d4e2cc18027f775",
    "training/tron1_terrain.py": "dec819a3c3c526fce706fb6a5a6491388298884b32bfe73a5c331b166f976ebf",
    "scripts/train_tracking.py": "ea9e4cd14f25ff3ba5202b9ba43ac0f08320418e2311cbae205c8449f3a1da9c",
    "scripts/train_tracking_lr_sync.py": "9bd544b82e211633bb37c3fd8a658448d6d0e9b90f380f01a78dd9e8de02b8cb",
    "training/tron1_resume_lr.py": "596f1e2a455a4bba5644965985791aec9d4a934b5079932e6c6c4a208a1a3064",
    "training/tron1_ground_load_reserve.py": "761eccede3c0d90d744a4d9c45c9671b318426ab9bc6f9762ee7cacb3c14496d",
    "training/tron1_substep_rewards.py": "aeec31437226984a326a263109aaadebca004a6ef333f25a1d3377688aa276a5",
    "scripts/eval_tracking.py": "fd90089810d3d14bfd96210723afdd3a36113a19f667ec60b2df617dd032f992",
    "training/tron1_ground_load_substeps.py": "4df069ea80e37cf7575346201687be934927434db45ea143f5ce797d4979ecd2",
    "training/tron1_ground_load_substeps_trial.py": "1917e3147783cddc47d8f8aeae366a46f224dd1b4091cc695a7327eded0b5b3f",
    "scripts/train_tracking_dense_ground_load.py": "361c76f4a2ef9bc3ce4d95a3fbd878ddc7728af2da80c44cf70cc0566fcc4fcd",
    "scripts/eval_tracking_dense_ground_load.py": "ba44bc4215f2913e59d6fa8c616d2d057c2d9c7fc1bc98b7fc9df68470d14185",
    "scripts/smoke_dense_ground_load.py": "620c8f2906114cf4cf7aa375413d43034b416acab091ca144a792801436c3d08",
    "training/tron1_wheel_axle_tracking.py": "aa3dac2bdae03f029d2c4bb174e79007f9b7927b5eaa04d35e50eefcff7de256",
    "training/tron1_wheel_axle_trial.py": "92301d90d844c3b3db0aa82e336abf7823855cde2ec21b01b639924be36e8421",
    "scripts/train_tracking_wheel_axle.py": "654acc37ccd8ff41f7b10d0b71abe9c85051cafe58be7f3b5d893755906c7bc3",
    "scripts/eval_tracking_wheel_axle.py": "c865d498c94ceafee1bf20f72ded62ac7ee417ec91e370c475ecbaa029fc1663",
    "tests/test_tron1_wheel_axle_tracking.py": "f1e8ee6bc6c8a1914565b0ac50aa647146c104fd79af9d693ff17e5fd3357fac",
    "tests/test_wheel_axle_trial.py": "f00fd99ef17a7b1a3ada824a852c3efbe747c43282a651b9153e6c190123268e",
    "tests/test_wheel_axle_entrypoints.py": "8498abcb9406983b35b98b101357ed766454c660ce3bb1783d1b0a34a4bc5cbc",
    "config/car_drift_wheel_axle_task.json": "1db17ea1bb8820396e37bdf7043b6852151fda2b500c69f5d9a7716283932f78",
    "scripts/smoke_wheel_axle_tracking.py": "8ba91d74a5af5eded0c3cdc81e7dc9318459b0b2a5111f55edf6ff9a0457b561",
    "tests/test_smoke_wheel_axle_tracking.py": "daa01174c0e00a14bae67570534c663f5aa0105b27bd5c6d04940aa9233a80ca"
}
TRAINING_SOURCE17 = {name: SOURCE27[name] for name in ["training/tron1_tracking.py","training/tron1_domain_randomization.py","training/tron1_terrain.py","training/tron1_drift.py","training/tron1_contact_slip.py","scripts/train_tracking_wheel_axle.py","training/tron1_substep_rewards.py","training/tron1_ground_load_substeps.py","training/tron1_ground_load_substeps_trial.py","scripts/train_tracking_lr_sync.py","scripts/train_tracking_dense_ground_load.py","training/tron1_wheel_axle_trial.py","training/tron1_wheel_axle_tracking.py","training/tron1_ground_load_reserve.py","training/tron1_runtime_device.py","scripts/train_tracking.py","training/tron1_resume_lr.py"]}
RUNTIME_INPUTS = (*lr.FINAL_INPUTS, "training/resume_learning_rate_audit.json",
    "diagnostics/training_completion_verification.json", "training/reward_sampling_final_audit.json")
PAIRED_CHECK_NAMES = frozenset((
    "completed600_2048_24_seed42_original004_resume_and_all_manifest_source_profile_contract_fields",
    "actual25_model_51_optimizer_tensors_finite_and_iteration_status_consistent",
    "all_model_optimizer_tensors_finite",
    "fresh017_final_CP_and_matching51_120_8_500_10_50_200_contracts_no_overrides",
    "same_contract_profile_both_engines",
    "all12_exported_actor_normalizer_tensor_bytes_exact_fresh017_final_CP_NOT_baseline004",
    "exported_actor_normalizer_bit_exact_checkpoint",
    "same_actual_I_MJ_actorbytes_and_canonical51D_observation_reconstruction",
    "all288_valid_applied_Isaac_pre_action_observations_CPU_replay_within1e_minus5",
    "all288_MuJoCo_CPU_actor_replay_and_no_normalizer_mutation",
    "all_recorded_actor_actions_replay",
    "genuine200Hz_both_engines_initial_once_and_firstepisode_reset_excluded",
    "reference_asset_model_evaluator_helper_nominalmotor_livegain_and_contactprovenance",
    "all_assessor_provenance_checks_pass",
    "all_input_hashes_unchanged_during_audit"))


def require(condition, message):
    require_base(condition, "Trial017 wheel-axle publication: " + message)


def typed_equal(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return set(a) == set(b) and all(typed_equal(a[k], b[k]) for k in a)
    if type(a) is list:
        return len(a) == len(b) and all(typed_equal(x, y) for x, y in zip(a, b))
    return a == b


def fields(value, expected, label):
    require(type(value) is dict and all(key in value and typed_equal(value[key], item)
            for key, item in expected.items()), label)


def digest(value, label):
    require(type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None, label)
    return value


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), label)
    return value


def validate_instantaneous_cache_context(records, files, hashes, *, repo_root=ROOT):
    """Validate actual017 records' original761 instantaneous-cache metadata.

    Adapted from the frozen source-backed validator, not a virtual013 view.
    Actual017 profile/new-evaluator fields are checked directly; the cache's
    unchanged sampling label is separate from authoritative mean-cost audits.
    """
    key, helper = "wheel_ground_load_reserve", "training/tron1_ground_load_reserve.py"
    manifest, isaac, mujoco, contract = [records[k] for k in (
        "manifest", "isaac_report", "mujoco_report", "contract")]
    containers = (manifest, isaac, contract, manifest.get("contract", {}), mujoco.get("contract", {}))
    profiles = [c.get("task_profile") if isinstance(c, dict) else None for c in containers]
    active = any(isinstance(p, dict) and key in p for p in profiles)
    audits = [c.get("contact_material_audit", {}) if isinstance(c, dict) else {} for c in (manifest, isaac, mujoco)]
    source = manifest.get("source_sha256")
    orphan = (isinstance(source, dict) and helper in source
              or any(isinstance(c, dict) and "ground_load_reserve_module_sha256" in c for c in (*containers, mujoco))
              or any(isinstance(a, dict) and "ground_load_reserve" in a for a in audits))
    if not active:
        require(not orphan, "Orphan ground-load reserve evidence without an explicit task-profile key")
        return None
    require(all(isinstance(p, dict) and key in p for p in profiles),
            "Ground-load reserve must be explicit in training and both deployment contracts")
    require(not any("controller_gain_diagnostic" in c or "physics_clock_refinement" in c
                    for c in containers if isinstance(c, dict)),
            "Ground-load reserve context cannot combine a controller or clock diagnostic")
    require(all(isinstance(a, dict) for a in audits)
            and "ground_load_reserve" not in audits[2],
            "Reserve startup audits belong to training/evaluation Isaac, not MuJoCo")
    require(all("ground_load_reserve_module_sha256" not in c for c in
                (manifest, mujoco, contract, manifest.get("contract", {}), mujoco.get("contract", {}))),
            "Reserve module evidence must use the recorded training source map and Isaac report fields")

    def number(value, label):
        require(type(value) in (int, float), f"Reserve {label} must be a finite non-boolean number")
        try:
            finite = math.isfinite(value)
        except (OverflowError, ValueError):
            finite = False
        require(finite, f"Reserve {label} must be finite")
        return value

    def numeric_list(value, length, label):
        require(isinstance(value, list) and len(value) == length, f"Invalid reserve {label} shape")
        return [number(v, label) for v in value]

    source = manifest.get("source_sha256")
    require(isinstance(source, dict), "Missing reserve training source pins")
    for relative, expected in GROUND_LOAD_RESERVE_SOURCE_PINS.items():
        path = repo_root / relative
        require(path.is_file() and sha256(path) == expected,
                f"Unsupported reserve source implementation: {relative}")
        if relative != "scripts/eval_tracking.py":
            require(source.get(relative) == expected, f"Missing or stale reserve training source pin: {relative}")
        input_key = "ground_load_reserve_source_" + path.stem
        files[input_key], hashes[input_key] = path, expected
    require(isaac.get("evaluator_source_sha256") == SOURCE27["scripts/eval_tracking_wheel_axle.py"]
            and isaac.get("baseline_evaluator_source_sha256") == SOURCE27["scripts/eval_tracking.py"]
            and isaac.get("drift_module_sha256") == GROUND_LOAD_RESERVE_SOURCE_PINS["training/tron1_drift.py"]
            and isaac.get("contact_slip_helper_sha256") == GROUND_LOAD_RESERVE_SOURCE_PINS["training/tron1_contact_slip.py"]
            and isaac.get("ground_load_reserve_module_sha256") == GROUND_LOAD_RESERVE_SOURCE_PINS[helper],
            "Missing or stale reserve evaluation source pins")
    baseline_path = repo_root / "config/car_drift_contact_task.json"
    require(baseline_path.is_file() and sha256(baseline_path) == GROUND_LOAD_RESERVE_BASELINE_SHA,
            "Original reserve baseline profile changed")
    files["ground_load_reserve_baseline_profile"] = baseline_path
    hashes["ground_load_reserve_baseline_profile"] = GROUND_LOAD_RESERVE_BASELINE_SHA
    baseline = read_json(baseline_path)
    profile = profiles[0]
    baseline013 = read_json(repo_root / "config/car_drift_ground_load_task.json")
    for candidate in profiles:
        strict.validate_wheel_axle_profile(candidate)
    require(all(p == profile for p in profiles), "Reserve training/deployment profiles differ")
    require({k: v for k, v in profile.items() if k not in (key, "description", "wheel_ground_load_reserve_sampling", strict.AXLE_KEY)}
            == {k: v for k, v in baseline.items() if k != "description"},
            "Only held reserve and mean-cost quadrature may differ from original004")
    from training.tron1_ground_load_reserve import validate_load_reserve_settings
    settings = validate_load_reserve_settings(profile[key])
    for target in ("motion", "robot_reference"):
        require(hashes.get(target) == GROUND_LOAD_RESERVE_ASSET_PINS[target],
                f"Reserve must preserve the original {target}")
    require(manifest.get("asset_sha256") == isaac.get("asset_root_file_sha256")
            == GROUND_LOAD_RESERVE_ASSET_PINS["asset"], "Reserve robot asset differs from baseline")
    require(mujoco.get("model_sha256") == GROUND_LOAD_RESERVE_ASSET_PINS["model"],
            "Reserve MuJoCo physics model differs from baseline")
    require(records["adaptation"].get("model_sha256") == records["export_report"].get("model_sha256")
            == GROUND_LOAD_RESERVE_ASSET_PINS["reference_model"],
            "Reserve reference-generation model differs from baseline")
    require(isaac.get("domain_parameters") is None, "Reserve evaluation must use nominal Isaac motors")
    from training.tron1_domain_randomization import default_profile
    require(manifest.get("domain_randomization_profile") == default_profile(42),
            "Reserve must retain the original motor-v1 randomization")
    domain = mujoco.get("domain_parameters", {})
    require(isinstance(domain, dict) and domain.get("is_nominal") is True
            and domain.get("joint_names") == contract.get("observed_joint_velocity_order"),
            "Reserve evaluation requires named nominal MuJoCo motors")
    for field, count, expected in (("torque_scale", 8, 1.), ("velocity_scale", 8, 1.),
                                   ("wheel_friction_nm", 2, 0.)):
        require(all(v == expected for v in numeric_list(domain.get(field), count, field)),
                f"Reserve nominal motor {field} changed")
    contact = audits[2]
    require(contact.get("contact_margin_and_gap_changed") is False
            and contact.get("explicit_contact_solref") == [.04, 2.]
            and contact.get("compiled_solref_ranges") == [[.04, .04], [2., 2.]],
            "Reserve MuJoCo contact response/margin/gap changed")
    require(records["assessment"].get("provenance_checks", {}).get(
        "same_physics_contact_sampling_clock", {}).get("passed") is True,
        "Reserve requires paired actual 200Hz contact evidence")
    for c in (manifest.get("contract", {}), contract, mujoco.get("contract", {})):
        for field, expected in (("leg_kp", 500.), ("leg_kd", 10.), ("physics_dt", .005), ("policy_fps", 50.)):
            require(number(c.get(field), field) == expected, f"Reserve {field} changed")
        require(all(type(c.get(field)) is int and c[field] == expected for field, expected in (
            ("decimation", 4), ("actor_dim", 51), ("critic_dim", 120), ("action_dim", 8))),
            "Reserve decimation/observation/action dimensions changed")
    require(type(manifest.get("num_envs")) is int and manifest["num_envs"] == 2048,
            "Reserve training must declare 2048 environments")
    require(type(manifest.get("iterations_requested")) is int and manifest["iterations_requested"] == 600
            and type(manifest.get("seed")) is int and manifest["seed"] == 42
            and manifest.get("resume_sha256") == "93bebee533bd59b8a7d707a1cd15b83bdff6c02663910afe59f69df91cd71780",
            "Reserve must retain the predeclared 600-update/seed42/original004-parent training")
    tracked = ["base_Link", "abad_L_Link", "abad_R_Link", "hip_L_Link", "hip_R_Link",
               "knee_L_Link", "knee_R_Link", "wheel_L_Link", "wheel_R_Link"]
    native = tracked[:3] + ["limx_imu"] + tracked[3:]
    require(contract.get("body_names") == tracked
            and contract.get("actual_body_order") == native == manifest.get("body_names"),
            "Reserve tracked/native body identities differ from the original robot")
    body_count = len(native)
    checked = {}
    for engine, parent, n, tracks in (("training", audits[0], 2048, False), ("isaac_evaluation", audits[1], 1, True)):
        check_leg_gain_audit(parent, 500., 10., engine)
        a = parent.get("ground_load_reserve")
        require(isinstance(a, dict), f"Missing {engine} reserve startup audit")
        require(type(a.get("schema_version")) is int and a["schema_version"] == 1
                and a.get("module_sha256") == GROUND_LOAD_RESERVE_SOURCE_PINS[helper]
                and a.get("settings") == settings,
                f"Invalid {engine} reserve schema/source/settings")
        validate_load_reserve_settings(a.get("settings"))
        for field, expected in (("physics_dt_s", .005), ("policy_dt_s", .02)):
            require(number(a.get(field), field) == expected, f"Reserve {engine} {field} changed")
        require(type(a.get("num_envs")) is int and a["num_envs"] == n
                and type(a.get("decimation")) is int and a["decimation"] == 4
                and a.get("sensor_matrix_shape") == [n, 1, 1, 3]
                and all(type(v) is int for v in a["sensor_matrix_shape"]),
                f"Invalid {engine} reserve native layout/clock")
        require(a.get("sensor_names") == ["wheel_L_ground", "wheel_R_ground"]
                and a.get("wheel_body_names") == ["wheel_L_Link", "wheel_R_Link"]
                and a.get("sensor_track_contact_points") == [tracks, tracks]
                and all(type(v) is bool for v in a["sensor_track_contact_points"]),
                f"Invalid {engine} reserve wheel sensor/telemetry mode")
        expected_strings = {
            "sampling": "last_physics_substep",
            "formula": "sum(exp(-clamp(ground_normal_force_world_z_N,min=0)/(scale_fraction*mg_N)))",
            "force_evidence": "filtered ContactSensor.data.force_matrix_w; normal forces only",
            "native_mass_source": "ArticulationView.get_masses() CPU (N,bodies)",
            "mass_reduction_dtype": "float64", "cached_mg_dtype": "float32",
        }
        require(all(a.get(k) == v for k, v in expected_strings.items()), f"Invalid {engine} reserve equation/evidence/dtypes")
        flags = {"training_contact_point_tracking_required": False,
                 "startup_force_finite_and_nonnegative_z": True, "startup_loaded_contact_required": False,
                 "loaded_normal_vector_and_raw_aggregate_runtime_smoke_required": True,
                 "raw_normal_forces_thresholded": False, "scale_is_minimum_force": False,
                 "continuous_contact_certified": False, "host_synchronization_free": False,
                 "reset_or_history_state_added": False, "soft_balanced_load_preference": True,
                 "tracking_tradeoff_and_no_hardware_safety_claim": True}
        schema = {"schema_version", "module_sha256", "settings", "physics_dt_s", "policy_dt_s", "decimation",
                  "sensor_names", "wheel_body_names", "sensor_matrix_shape", "ground_collider",
                  "sensor_track_contact_points", "mass_kg_range", "first_environment_link_mass_kg",
                  "mg_N_range", "configured_gravity_m_s2", "native_gravity_m_s2", "cuda_device",
                  "num_envs", "active_event_identity_audit", "cost_bounds"} | set(expected_strings) | set(flags)
        require(set(a) == schema, f"Invalid {engine} reserve startup audit schema (missing/unsupported claims)")
        require(all(a.get(k) is v for k, v in flags.items()), f"Invalid {engine} reserve semantic flags")
        require(numeric_list(a.get("cost_bounds"), 2, "cost bounds") == [0., 2.], "Invalid reserve cost bounds")
        ground = a.get("ground_collider")
        require(isinstance(ground, dict)
                and set(ground) == {"path", "usd_type", "collision_api_verified", "native_cooked_geometry_readback_performed"}
                and ground.get("path") == "/World/ground/terrain/GroundPlane/CollisionPlane"
                and ground.get("usd_type") in ("Mesh", "Plane")
                and ground.get("collision_api_verified") is True
                and ground.get("native_cooked_geometry_readback_performed") is False
                and ground["path"] in {g.get("collider") for g in parent.get("ground_colliders", [])},
                f"Invalid {engine} reserve ground filter/schema audit")
        require(isinstance(a.get("cuda_device"), str) and re.fullmatch(r"cuda:(0|[1-9][0-9]*)", a["cuda_device"]),
                "Reserve cached weight must have an explicit CUDA ordinal")
        mass = numeric_list(a.get("mass_kg_range"), 2, "live mass range")
        mg = numeric_list(a.get("mg_N_range"), 2, "live weight range")
        require(0 < mass[0] <= mass[1] and 0 < mg[0] <= mg[1]
                and all(abs(v - 22.273000011220574) <= 1e-5 for v in mass),
                "Reserve live mass/weight range differs from unchanged asset")
        links = numeric_list(a.get("first_environment_link_mass_kg"), body_count, "link masses")
        require(all(v >= 0 for v in links) and mass[0] - 1e-6 <= sum(links) <= mass[1] + 1e-6,
                "Reserve link mass sum differs from live total")
        cfg_g = numeric_list(a.get("configured_gravity_m_s2"), 3, "configured gravity")
        native_g = numeric_list(a.get("native_gravity_m_s2"), 3, "native gravity")
        require(all(g[:2] == [0., 0.] and g[2] < 0 and abs(g[2] + 9.81) <= 1e-5 for g in (cfg_g, native_g))
                and all(math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-7) for x, y in zip(cfg_g, native_g)),
                "Reserve native/configured downward gravity mismatch")
        require(all(math.isclose(w, m * abs(native_g[2]), rel_tol=1e-9, abs_tol=1e-6)
                    for w, m in zip(mg, mass)), "Reserve CPU-double mass/gravity weight is inconsistent")
        events = a.get("active_event_identity_audit")
        require(isinstance(events, list) and len(events) == (3 if tracks is False else 2),
                "Reserve unexpected event count (mass/gravity/random push events are not supported)")
        expected_events = {
            "physics_material": ("startup", "isaaclab.envs.mdp.events.randomize_rigid_body_material", True),
            "legacy_joint_friction": ("startup", "training.tron1_tracking.clear_legacy_joint_friction", False),
        }
        if not tracks:
            expected_events["domain_randomization"] = ("reset", "training.tron1_domain_randomization.reset_domain_randomization", False)
        seen = set()
        for event in events:
            require(isinstance(event, dict) and isinstance(event.get("name"), str)
                    and set(event) == {"name", "mode", "function", "initialized_exact_class_instance"}
                    and event["name"] in expected_events and event["name"] not in seen,
                    "Reserve unsafe/duplicate event identity")
            name = event["name"]
            mode, function, instance = expected_events[name]
            require(event.get("mode") == mode and event.get("function") == function
                    and event.get("initialized_exact_class_instance") is instance,
                    "Reserve event function/class instance or mode differs from inspected implementation")
            seen.add(name)
        checked[engine] = a
    return {
        "schema_version": 1, "validation_kind": "source_backed_validated_startup_context",
        "supported_helper_sha256": GROUND_LOAD_RESERVE_SOURCE_PINS[helper],
        "source_sha256": dict(GROUND_LOAD_RESERVE_SOURCE_PINS), "baseline_profile_sha256": GROUND_LOAD_RESERVE_BASELINE_SHA,
        "actual_objective_sampling": "all_substeps_mean_cost",
        "settings": settings, "sampling": "instantaneous_original761_cache_metadata_not_objective_quadrature",
        "training_sensor_telemetry": "two force-only single-wheel ground-filtered matrices; contact-point tracking disabled",
        "evaluation_sensor_telemetry": "same filtered matrices plus original wheel-point/nonwheel-contact recording; telemetry-only difference",
        "startup_audits": checked, "behavior_gates_changed": False,
        "live_reward_manager_configuration_independently_recorded": False,
        "all_cached_gpu_weight_values_independently_recorded": False,
        "full_training_force_validity_independently_proved": False,
        "normal_force_envelope_equivalence_claimed": False, "native_cooked_geometry_equivalence_claimed": False,
        "continuous_contact_certified": False, "hardware_ready": False,
        "limitations": [
            "Source-backed validated startup context, not independently recorded live RewardManager configuration, all cached GPU weight values or full-run force proof.",
            "Approved helper guards establish the declared native layout, 5 ms impulse conversion and inspected event identities; omitted runtime readbacks are not invented.",
            "Any separate 8-env/16-window/64-sample GPU smoke establishes configuration/equation validity only in that fixture, not 2048-env training, policy force continuity or simulator force-envelope equivalence.",
            "The original761 cache audit retains sampling=last_physics_substep as instantaneous helper metadata. The authoritative016 objective separately averages four instantaneous costs; this cache label is not the reward quadrature.",
            "The e-fold scale is not a minimum-force gate or hard 50:50 load target; dense soft costs do not certify uninterrupted bearing-contact patches.",
        ],
    }


def _prepare_recorded_candidate(trial_dir, reference_dir, render_dir=None, *, repo_root=ROOT):
    trial_dir, reference_dir = Path(trial_dir), Path(reference_dir)
    files = {
        "assessment": trial_dir / "assessment.json", "manifest": trial_dir / "training/manifest.json",
        "isaac_report": trial_dir / "isaac/report.json", "contract": trial_dir / "isaac/policy_contract.json",
        "actor": trial_dir / "isaac/actor_normalized.pt", "isaac_trajectory": trial_dir / "isaac/trajectory.npz",
        "mujoco_report": trial_dir / "mujoco/report.json", "mujoco_trajectory": trial_dir / "mujoco/rollout.npz",
        "adaptation": reference_dir / "adaptation.json", "motion": reference_dir / "motion.npz",
        "robot_reference": reference_dir / "robot_reference.npz", "export_report": reference_dir / "motion.json",
    }
    for key, path in files.items():
        require(path.is_file() and path.stat().st_size > 0, f"Missing or empty {key} evidence")
    hashes = {key: sha256(path) for key, path in files.items()}
    records = {key: read_json(path) for key, path in files.items() if path.suffix == ".json"}
    isaac_report = records["isaac_report"]
    if "physics_trajectory_file" in isaac_report:
        require(isaac_report["physics_trajectory_file"] == "physics_trajectory.npz",
                "Isaac physics trajectory must use the local physics_trajectory.npz evidence file")
        from assess_drift import load_isaac_behavior_arrays
        _, physics_path, physics_info = load_isaac_behavior_arrays(trial_dir / "isaac", isaac_report)
        require(physics_info.get("kind") == "physics_rate",
                "Isaac physics trajectory was not validated as physics-rate evidence")
        files["isaac_physics_trajectory"] = Path(physics_path)
        hashes["isaac_physics_trajectory"] = sha256(physics_path)
        require(physics_info.get("sha256") == hashes["isaac_physics_trajectory"],
                "Validated Isaac physics trajectory SHA differs from the actual input")
    stored = records["assessment"]
    fresh = assess_runs(trial_dir / "isaac", trial_dir / "mujoco", files["motion"])
    require(encode(stored) == encode(fresh), "Supplied assessment differs from freshly assessed current evidence")
    require(stored.get("task") == "captured_car_drift" and stored.get("verdict") in ("pass", "fail"),
            "Assessment must identify captured_car_drift with an explicit verdict")
    isaac, mujoco, contract, manifest, adaptation, export = [records[key] for key in (
        "isaac_report", "mujoco_report", "contract", "manifest", "adaptation", "export_report")]
    provenance = validate_provenance(trial_dir / "isaac", trial_dir / "mujoco", isaac, mujoco)
    require(isaac.get("policy_export") == contract, "Saved Isaac policy contract differs from evaluation report")
    require(provenance == stored.get("provenance"), "Assessment same-actor provenance differs from current input")
    for record, key, target in ((manifest, "motion_sha256", "motion"),
                               (contract, "reference_file_sha256", "motion"),
                               (adaptation, "motion_sha256", "motion"),
                               (adaptation, "robot_reference_sha256", "robot_reference"),
                               (export, "source_sha256", "robot_reference"),
                               (export, "output_sha256", "motion")):
        require(record.get(key) == hashes[target], f"{key} is not linked to the actual {target}")
    require(export.get("upstream_task_adaptation") == adaptation.get("calibration_and_adaptation"),
            "Reference adaptation and exported recipe differ")
    require(export.get("source_car_provenance") == adaptation.get("provenance"),
            "Car source provenance differs between reference and export")
    if "model_sha256" in adaptation or "model_sha256" in export:
        require(adaptation.get("model_sha256") == export.get("model_sha256"),
                "Reference generation and export use different robot model hashes")
    original_task_profile = strict.validate_wheel_axle_profile(manifest.get("task_profile"))
    diagnostic = validate_controller_gain_context(records, files, hashes)
    require(diagnostic is None, "Trial017 cannot combine gain diagnostics")
    task_profile = original_task_profile
    require(task_profile == contract.get("task_profile") == mujoco.get("contract", {}).get("task_profile")
            == isaac.get("task_profile"), "Training and both deployments must use the same drift task/material profile")
    mu = task_profile["contact_friction"]
    material = {"training": check_material_audit(manifest.get("contact_material_audit"), mu, "Training Isaac"),
                "isaac": check_material_audit(isaac.get("contact_material_audit"), mu, "Isaac"),
                "mujoco": check_material_audit(mujoco.get("contact_material_audit"), mu, "MuJoCo")}
    for engine in ("isaac", "mujoco"):
        require(stored.get(engine, {}).get("checks", {}).get("physics_only_no_hidden_reset_or_root_teleport", {}).get("passed") is True,
                f"{engine} trial does not establish a physics-only recorded rollout")
    optional = {}
    for name, relative in (("run_status", "training/run_status.json"), ("runner", "training/runner_config.json")):
        path = trial_dir / relative
        if path.exists():
            require(path.is_file(), f"Invalid optional {name} input")
            files[name], hashes[name], optional[name] = path, sha256(path), read_json(path)
    if diagnostic is not None:
        require(training_budget(manifest, optional.get("run_status"), optional.get("runner"))[
            "requested_budget_completion_verified"] and "runner" in optional,
            "Diagnostic must retain the completed original training budget evidence")
    load_reserve_context = validate_instantaneous_cache_context(records, files, hashes, repo_root=repo_root)
    if load_reserve_context is not None:
        require("run_status" in optional and "runner" in optional
                and training_budget(manifest, optional["run_status"], optional["runner"])[
                    "requested_budget_completion_verified"]
                and type(optional["run_status"].get("num_envs")) is int
                and type(optional["run_status"].get("iterations_requested")) is int
                and type(optional["run_status"].get("training_environment_steps")) is int
                and optional["run_status"]["training_environment_steps"] == 600 * 2048 * 24
                and type(optional["runner"].get("num_steps_per_env")) is int
                and optional["runner"]["num_steps_per_env"] == 24,
                "Reserve publication requires completed predeclared training and 24 PPO steps per environment")
    render, media = None, {}
    if render_dir is not None:
        render, media, report_path = validate_render(render_dir, hashes, adaptation)
        files["render_report"] = report_path
        hashes["render_report"] = sha256(report_path)
    identities = private_identities({**records, **optional, "render": render})
    source = adaptation["provenance"]
    require(isinstance(source, dict) and all(isinstance(source.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", source[key])
            for key in ("source_csv_sha256", "source_npz_sha256", "source_audit_sha256", "source_configuration_sha256")),
            "Incomplete pinned car-source hashes")
    summary = {
        "schema_version": 1, "task": "captured_car_drift", "status": "assessed_" + stored["verdict"],
        "verdict": stored["verdict"], "behavior_pass": stored["verdict"] == "pass",
        "failed_checks": stored["failed_checks"], "same_actor_reference_verified": True,
        "actual_physics_rollouts_verified": True,
        "trial_snapshot_identity_sha256": hashlib.sha256(encode({key: hashes[key] for key in ("manifest", "assessment", "actor")})).hexdigest(),
        "input_sha256": hashes,
        "car_source": source,
        "reference": {"adaptation": adaptation["calibration_and_adaptation"], "diagnostics": adaptation["diagnostics"],
                      "frames": export.get("frames"), "fps": export.get("fps"), "sample_span_s": export.get("sample_span_s"),
                      "positions": export.get("positions"), "linear_velocity": export.get("linear_velocity"),
                      "source_time_mapping": export.get("source_time_mapping")},
        "training": training_budget(manifest, optional.get("run_status"), optional.get("runner")),
        "task_profile": task_profile, "contact_material_audits": material,
        "same_actor_provenance": provenance,
        "behavior_evidence": stored.get("behavior_evidence"),
        "provenance_checks": stored.get("provenance_checks"),
        "engines": {engine: pick(stored[engine], ("verdict", "failed_checks", "metrics", "checks",
            "contact_patch_slip_evidence_available", "wheel_ground_contact_evidence_available", "nonwheel_contact_evidence_available",
            "complete_contact_evidence")) for engine in ("isaac", "mujoco")},
        "evaluation": {
            "isaac": pick(isaac, ("terminal_states_captured_before_auto_reset", "reference_state_initialization_during_evaluation",
                                  "wheel_contact_slip_evidence", "contact_slip_module_sha256", "evaluator_source_sha256", "task_source_sha256",
                                  "physics_trajectory_file", "physics_trajectory_sha256", "physics_dt_s", "physics_recording",
                                  "contact_slip_measurement", "wheel_lateral_contact_slip_evidence_kind",
                                  "contact_slip_helper_sha256", "drift_module_sha256")),
            "mujoco": pick(mujoco, ("physics_stepped", "root_state_writes", "hidden_resets", "contact_slip_evidence_kind",
                                    "contact_slip_module_sha256", "evaluator_source_sha256", "task_source_sha256", "model_sha256",
                                    "physics_dt_s", "policy_dt_s", "physics_recording", "contact_slip_measurement",
                                    "wheel_lateral_contact_slip_evidence_kind", "contact_slip_helper_sha256",
                                    "nonwheel_ground_contact_recorded")),
        },
        "common_window": stored.get("common_window"), "complete_contact_evidence": stored.get("complete_contact_evidence", False),
        "rendered_comparison_included": render is not None,
        "safety_verified": False, "hardware_ready": False, "statistical_robustness_benchmark": False,
        "limitations": stored.get("limitations", []),
        "artifacts": {"assessment": "assessment.json", "reference_audit": "reference_audit.json"},
    }
    if diagnostic is not None:
        summary.update({"experiment_kind": "zero_shot_controller_gain_diagnostic",
                        "additional_training_updates": 0,
                        "original_training_task_profile": original_task_profile,
                        "effective_deployment_task_profile": task_profile,
                        "controller_gain_diagnostic": diagnostic})
        summary["training"]["budget_scope"] = "original checkpoint training; no diagnostic training"
        if diagnostic["scope"] == "zero_shot_common_leg_kd":
            summary["experiment_kind"] = "zero_shot_controller_damping_diagnostic"
            summary["controller_gain_change"] = {
                "parameter": "leg_kd", "trained_leg_kd": diagnostic["trained_leg_kd"],
                "deployment_leg_kd": diagnostic["deployment_leg_kd"],
                "unchanged_leg_kp": diagnostic["unchanged_leg_kp"],
            }
    if load_reserve_context is not None:
        summary["ground_load_reserve_context"] = load_reserve_context
    payloads = {"assessment.json": encode(sanitize(stored, identities)),
                "reference_audit.json": encode(sanitize(adaptation, identities))}
    if render is not None:
        payloads["render_report.json"] = encode(sanitize(render, identities))
        summary["artifacts"].update(render_report="render_report.json", video="motion_pipeline.mp4", overview="overview.png")
    summary["published_artifact_sha256"] = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
    summary["published_artifact_sha256"].update({name: item["sha256"] for name, item in media.items()})
    return sanitize(summary, identities), payloads, media, files, hashes


def validate_protocol(protocol, *, repo_root=ROOT):
    """The already prospective whole protocol is pinned, not reconstructed."""
    path = repo_root / PROTOCOL_FILE
    require(sha256(path) == PROTOCOL_SHA256, "Frozen prospective017 protocol changed")
    require(typed_equal(protocol, read_json(path)), "Explicit017 protocol must match every sealed field")
    fields(protocol, {"trial_id": "trial017", "baseline_trial": "trial016",
        "parent_trial": "trial004", "experiment_type": EXPERIMENT_KIND,
        "task_profile_sha256": PROFILE_SHA256}, "Protocol context")
    require(typed_equal({**protocol["expected_historical_source_sha256"],
                        **protocol["expected_new_source_sha256"]}, SOURCE27), "Exact27 sealed repository sources")
    return protocol


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_sampler(audit, *, num_envs, steps, device):
    """Exact frozen4df force audit; last-substep-vz remains a separate reward."""
    require(type(num_envs) is int and type(steps) is int and (num_envs, steps) in {
        (2048, 8), (2048, 14408), (1, 288)}, "Predesignated sampler stage")
    require(type(device) is str and re.fullmatch(r"cuda:[0-7]", device), "Explicit native CUDA ordinal")
    dt = .005
    rounding = abs(struct.unpack("f", struct.pack("f", dt))[0] - dt)
    life_fp = fingerprint({"schema_version": 1, "mode": "all_substeps_mean",
        "wheel_body_ids": [8, 9], "decimation": 4, "physics_dt_s": dt,
        "policy_dt_s": .02, "num_envs": num_envs})
    settings = {"weight": -1., "scale_fraction": .20}
    force_fp = fingerprint({"schema_version": 1, "mode": "all_substeps_mean_cost",
        "lifecycle_fingerprint_sha256": life_fp, "settings": settings,
        "sensor_names": ["wheel_L_ground", "wheel_R_ground"],
        "ground_collider": "/World/ground/terrain/GroundPlane/CollisionPlane",
        "reserve_dependency_sha256": SOURCE27["training/tron1_ground_load_reserve.py"],
        "lifecycle_dependency_sha256": SOURCE27["training/tron1_substep_rewards.py"]})
    expected = {"schema_version": 1, "mode": "all_substeps_mean_cost",
        "fingerprint_sha256": force_fp, "settings": settings,
        "formula": "mean_over_4_physics_substeps(sum_wheels(exp(-clamp(ground_normal_force_world_z_N,min=0)/(.20*mg_N))))",
        "cache_bound": True, "force_samples_last_control_step": 4,
        "force_samples_total": steps * 4, "force_cost_reads": steps,
        "force_cost_read_in_last_window": True, "inherited_velocity_collection_unused": True,
        "inherited_velocity_cost_reads": 0, "wheel_vertical_velocity_reward_changed": False,
        "extra_physics_actor_reward_manager_reset_rng_calls": 0, "host_synchronization_free": False,
        "continuous_contact_certified": False,
        "reserve_dependency_sha256": SOURCE27["training/tron1_ground_load_reserve.py"],
        "lifecycle_dependency_sha256": SOURCE27["training/tron1_substep_rewards.py"],
        "module_sha256": SOURCE27["training/tron1_ground_load_substeps.py"]}
    fields(audit, expected, "Actual mean-of-instantaneous-cost audit")
    require(set(audit) == set(expected) | {"lifecycle"}, "Exact force sampler schema")
    life = audit["lifecycle"]
    expected_life = {"schema_version": 1, "mode": "all_substeps_mean",
        "fingerprint_sha256": life_fp, "installed": True, "inside_step": False, "sampling_open": False,
        "num_envs": num_envs, "wheel_body_ids": [8, 9], "device": device, "dtype": "torch.float32",
        "decimation": 4, "physics_dt_s": dt, "policy_dt_s": .02,
        "steps_started": steps, "steps_completed": steps, "steps_failed": 0,
        "samples_last_control_step": 4, "samples_total": steps * 4,
        "last_start_counter": steps * 4 - 4, "last_sample_counter": steps * 4,
        "last_step_completed": True, "native_clock_required": True,
        "native_time_float32_dt_rounding_bound_s": rounding,
        "native_time_double_rounding_allowance_ulps": 8, "cost_reads": 0}
    fields(life, expected_life, "Actual inherited lifecycle audit; no velocity-cost consumption")
    clocks = {"native_start_step_index", "native_last_step_index", "native_start_time_s", "native_last_time_s"}
    require(set(life) == set(expected_life) | clocks, "Exact lifecycle schema")
    start, end = life["native_start_step_index"], life["native_last_step_index"]
    require(type(start) is int and start >= 0 and type(end) is int and end == start + 4,
            "Exactly four actual native physical steps")
    t0, t1 = finite(life["native_start_time_s"], "Native t0"), finite(life["native_last_time_s"], "Native t1")
    require(t0 >= 0 and t1 > t0 and abs(t1 - (t0 + .02)) <=
            4 * rounding + 8 * max(math.ulp(t0), math.ulp(t1), math.ulp(.02)),
            "Native20ms including explicit float32 callback quantization")
    return audit


FIXTURE_PRIVATE_PINS = {
    "launch_smoke_trial017.py": "4964ad34a991697aec5870d3418bd172b1aabea150d5bd427a6613eb39f67ac4",
    "launch_runtime_trial017.py": "6444d7dec31db918d665d8df0b86943bb1291e32b64805718a83f195a9388257",
    "verify_smoke_trial017.py": "60ac8b3ac9ccd616e8437ef30c51ef40f49670f099a43f5b39a3d620300a0e01",
    "verify_runtime_trial017.py": "a8ad012df7f2e083f4211264df236e4ba9074d1b1f9b09615602ee7ffa3685a1",
}
FIXTURE_SEAL_SHA256 = "2b41e39374755fa9ad26c9317ae0fafbfc4f9da0ba4fd54dd016e3c686b78e3a"
BASELINE_PROFILE_SHA256 = "989215d55b548e245d6add928411d9e6fa16852a21f6863a777179014e3230b5"
AXLE_TARGET_SEMANTICS = (
    "Existing yaw-reanchored robot reference local+Y axle directions; "
    "LAST physical substep; no wheel spin/quaternion phase or new car teacher.")
CPU_TRAINING_PINS = {"launch_training_trial017.py": CPU_LAUNCHER_SHA256,
                     "verify_training_completion_trial017.py": CPU_PRODUCER_SHA256}


def fixture_input_files(*, runtime=False):
    """Exact47 current fixture aliases; no import of ignored private modules."""
    prefix = "runtime_smoke" if runtime else "smoke"
    files = {"source/" + name: name for name in SOURCE27}
    files.update({"private/" + name: name for name in FIXTURE_PRIVATE_PINS})
    files.update({"protocol": PROTOCOL_FILE, "profile": PROFILE_FILE,
        "baseline016_profile": "config/car_drift_dense_ground_load_task.json",
        "baseline013_profile": "config/car_drift_ground_load_task.json",
        "parent004_profile": "config/car_drift_contact_task.json",
        "deployment_archive": "deployment.tar", "prelaunch_seal": "prelaunch_seal.json",
        "manifest": prefix + "/launch_manifest.json", "science": prefix + "/science_report.json",
        "teardown": prefix + "/teardown_report.json", "evidence": prefix + "/window_evidence.npz",
        "launch_receipt": prefix + "_launch_receipt.json", "started": prefix + "_started.json",
        "exit": prefix + "_exit_report.json", "live_placement": prefix + "_live_placement.jsonl",
        "log": prefix + ".log"})
    require(len(files) == 47, "Current fixture47 aliases")
    return files


def runtime_input_files():
    files = fixture_input_files(runtime=True)
    files["actual8_proof"] = "smoke_verification.json"
    files.update({"actual8/" + key: name for key, name in fixture_input_files().items()})
    require(len(files) == 95, "Current runtime95 aliases")
    return files


def completion_input_files():
    files = {"runtime/" + key: name for key, name in runtime_input_files().items()}
    files.update({"source/" + name: name for name in SOURCE27})
    files.update({"training/" + name: "outputs/trial017/training/" + name for name in (
        "manifest.json", "run_status.json", "runner_config.json", "model_final.pt",
        "resume_learning_rate_audit.json", "reward_sampling_final_audit.json")})
    files.update({"training_launcher": "launch_training_trial017.py",
        "completion_verifier": "verify_training_completion_trial017.py",
        "training_prelaunch_seal": "training_prelaunch_seal.json",
        "fixture_prelaunch_seal": "prelaunch_seal.json",
        "training_launch_receipt": "training_launch_receipt.json", "training_started": "training_started.json",
        "training_exit": "training_exit_report.json", "training_live_placement": "training_live_placement.jsonl",
        "training_log": "training.log", "training_metadata_completion": "training_metadata_completion.json",
        "smoke_proof": "smoke_verification.json", "runtime_proof": "runtime_smoke_verification.json"})
    require(len(files) == 140, "Current completion140 aliases")
    return files


def external_input_maps():
    parent_root = "/data/zengxiong/workspaces/tron1-drift-20261011/outputs/trial004/training/"
    rsl_root = "/data/zengxiong/.conda/envs/regrind/lib/python3.11/site-packages/rsl_rl/"
    hashes = {"motion": base.GROUND_LOAD_RESERVE_ASSET_PINS["motion"],
        "asset": base.GROUND_LOAD_RESERVE_ASSET_PINS["asset"],
        "parent004/model_final.pt": lr.PARENT_CP_SHA256,
        "parent004/manifest.json": lr.PARENT_MANIFEST_SHA256,
        "parent004/runner_config.json": lr.RUNNER_SHA256,
        "parent004/run_status.json": "8356fcd8d399d65a206d6f2a64072ad9a9475eb01b559cd003b7ae04bdcb41fb",
        **{"rsl/" + name: value for name, value in lr.RSL_SOURCE_PINS.items()}}
    files = {"motion": "/data/zengxiong/workspaces/tron1-drift-20261011/outputs/reference/motion.npz",
        "asset": "/data/zengxiong/workspaces/tron1-beyondmimic-20261002/outputs/training-asset/WF_TRON1A.usda",
        **{"parent004/" + name: parent_root + name for name in (
            "model_final.pt", "manifest.json", "runner_config.json", "run_status.json")},
        **{"rsl/" + name: rsl_root + name for name in lr.RSL_SOURCE_PINS}}
    return files, hashes


def validate_stage_seals(fixture, training, *, fixture_sha, training_sha):
    require(fixture_sha == FIXTURE_SEAL_SHA256, "Externally pinned immutable fixture seal")
    digest(training_sha, "External training seal")
    fields(fixture, {"schema_version": 1, "trial_id": "trial017", "git_commit": SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256, "archive_sha256": ARCHIVE_SHA256,
        "profile_sha256": PROFILE_SHA256, "baseline_profile_sha256": BASELINE_PROFILE_SHA256,
        "baseline_original_profile_sha256": dense.strict.BASELINE013_PROFILE_SHA256,
        "parent_profile_sha256": dense.strict.PARENT004_PROFILE_SHA256,
        "repository_source_sha256": SOURCE27, "deployment_source_sha256": FIXTURE_PRIVATE_PINS},
        "Current fixture external seal")
    external_files, external_hashes = external_input_maps()
    require(typed_equal(fixture.get("external_input_sha256"),
        {external_files[k]: external_hashes[k] for k in ("motion", "asset")}),
        "Fixture absolute external seals")
    expected = {"schema_version": 1, "status": "trial017_training_stage_sealed", "trial_id": "trial017",
        "git_commit": SOURCE_COMMIT, "protocol_sha256": PROTOCOL_SHA256, "profile_sha256": PROFILE_SHA256,
        "archive_sha256": ARCHIVE_SHA256, "fixture_prelaunch_seal_sha256": fixture_sha,
        "repository_source_sha256": SOURCE27, "deployment_source_sha256": CPU_TRAINING_PINS,
        "parent_input_sha256": {name.split("/", 1)[1]: value for name, value in external_hashes.items()
                               if name.startswith("parent004/")},
        "rsl_input_sha256": lr.RSL_SOURCE_PINS}
    require(typed_equal(training, expected), "Exact12-field training external seal")


def validate_completion(proof, checkpoint_sha, *, fixture_seal, training_seal,
                        fixture_seal_sha256, training_seal_sha256):
    validate_stage_seals(fixture_seal, training_seal, fixture_sha=fixture_seal_sha256,
                         training_sha=training_seal_sha256)
    lr.validate_completion(proof, checkpoint_sha)
    fields(proof, {"schema_version": 1, "trial_id": "trial017", "git_commit": SOURCE_COMMIT,
        "archive_sha256": ARCHIVE_SHA256, "protocol_sha256": PROTOCOL_SHA256,
        "profile_sha256": PROFILE_SHA256, "verification_script_sha256": CPU_PRODUCER_SHA256,
        "training_launcher_source_sha256": CPU_LAUNCHER_SHA256,
        "training_source_sha256": TRAINING_SOURCE17, "sealed_source_sha256": SOURCE27,
        "ground_load_substep_module_sha256": SOURCE27["training/tron1_ground_load_substeps.py"],
        "lifecycle_module_sha256": SOURCE27["training/tron1_substep_rewards.py"],
        "checkpoint_helpers_reference_sha256": "a87b1a5ff34a859ac030a98a60f5292306a157aca83ba9c120d3b8aa52735b09",
        "original_wheel_axle_profile_and_explicit_seventeen_training_sources_verified": True,
        "original_model_and_optimizer_layout_verified": True, "all_input_hashes_unchanged": True,
        "actual_adam_step_increment": 12000, "actual_actor_and_critic_normalizer_count_increment": 29491200,
        "new_actor_inference_calls": 0, "new_optimizer_updates": 0, "simulation_steps": 0, "cuda_allocations": 0,
        "owned_exit_returncode": 0, "behavior_verified": False, "hardware_ready": False,
        "physical_uuid_native_mapping_claimed": False, "physical_GPU_isolation_claimed": False,
        "native_app_close_return_claimed": False, "production_force_cache_values_independently_archived": False,
        "prelaunch_seal_sha256": fixture_seal_sha256, "training_prelaunch_seal_sha256": training_seal_sha256,
        "wheel_axle_orientation_tracking": {"weight": 1.0, "std": .4},
        "wheel_axle_target_semantics": AXLE_TARGET_SEMANTICS,
        "full_training_live_axle_reward_ledger_archived": False},
        "Actual source-sealed017 CPU completion")
    require(not any(key.startswith("original_") and key.endswith("_sources_verified")
            and key != "original_wheel_axle_profile_and_explicit_seventeen_training_sources_verified"
            for key in proof), "No historical completion labels")
    files, seals = completion_input_files(), proof.get("input_sha256")
    require(typed_equal(proof.get("input_files"), files) and type(seals) is dict
            and set(seals) == set(files), "Exact140 CPU input aliases")
    for value in seals.values(): digest(value, "CPU input SHA")
    known = {**SOURCE27, **FIXTURE_PRIVATE_PINS, **CPU_TRAINING_PINS,
        PROTOCOL_FILE: PROTOCOL_SHA256, PROFILE_FILE: PROFILE_SHA256,
        "config/car_drift_dense_ground_load_task.json": BASELINE_PROFILE_SHA256,
        "config/car_drift_ground_load_task.json": dense.strict.BASELINE013_PROFILE_SHA256,
        "config/car_drift_contact_task.json": dense.strict.PARENT004_PROFILE_SHA256,
        "deployment.tar": ARCHIVE_SHA256, "prelaunch_seal.json": fixture_seal_sha256,
        "training_prelaunch_seal.json": training_seal_sha256,
        "outputs/trial017/training/runner_config.json": lr.RUNNER_SHA256,
        "outputs/trial017/training/model_final.pt": checkpoint_sha}
    duplicate = {}
    for alias, name in files.items():
        require(name not in duplicate or duplicate[name] == seals[alias], "Duplicated CPU aliases disagree")
        duplicate[name] = seals[alias]
        if name in known: require(seals[alias] == known[name], "Frozen CPU source/seal " + alias)
    for key, status, file_key, prefix, mappings in (
        ("current_stack_smoke_prerequisite", "wheel_axle_smoke_verified", "smoke_proof",
         "runtime/actual8/", fixture_input_files()),
        ("current_stack_runtime_prerequisite", "wheel_axle_runtime_smoke_verified", "runtime_proof",
         "runtime/", runtime_input_files())):
        item = proof.get(key)
        require(type(item) is dict and set(item) == {"status", "proof_sha256", "input_sha256"},
                "Distinct source-backed current fixture prerequisite")
        fields(item, {"status": status, "proof_sha256": seals[file_key],
            "input_sha256": {name: seals[prefix + name] for name in mappings}}, "Fixture prerequisite seals")
    external_files, external_hashes = external_input_maps()
    fields(proof, {"external_input_files": external_files, "external_input_sha256": external_hashes},
           "Exact9 external byte copies")
    require(set(proof["external_input_files"]) == set(external_files)
            and set(proof["external_input_sha256"]) == set(external_hashes), "Exact9 external aliases")
    require(typed_equal(proof.get("resume_learning_rate_audit"), {
        "schema_version": 1, "scope": "adaptive_ppo_learning_rate_scalar_from_loaded_optimizer",
        "status": "restored_from_loaded_optimizer", "scalar_before": .001,
        "loaded_group_learning_rates": [1.e-5], "scalar_after": 1.e-5,
        "touched_field": "algorithm.learning_rate", "optimizer_group_learning_rates_unchanged": True,
        "adaptive_schedule_preserved": True, "expected_scalar_before": .001,
        "expected_loaded_lr": 1.e-5, "multi_gpu": False}), "Exact restored scalar-only LR audit")
    fields(proof, {"resume_source_audit": {"schema_version": 1, "rsl_rl_version": "3.0.1",
        "source_root": "/data/zengxiong/.conda/envs/regrind/lib/python3.11/site-packages/rsl_rl",
        "expected_source_sha256": lr.RSL_SOURCE_PINS, "actual_source_sha256": lr.RSL_SOURCE_PINS,
        "verified_before_runner_load": True}}, "Exact installed RSL source audit")
    for health in ("parent_checkpoint_health", "final_checkpoint_health"):
        for names, count in (("model_tensor_names", 25), ("optimizer_tensor_names", 51)):
            value = proof[health].get(names)
            require(type(value) is list and len(value) == len(set(value)) == count
                    and all(type(v) is str for v in value), "Distinct tensor names")
        if health == "final_checkpoint_health":
            for key in ("model_tensor_names", "optimizer_tensor_names"):
                require(typed_equal(proof[health][key], proof["parent_checkpoint_health"][key]),
                        "Original checkpoint tensor layout")
    device = proof.get("device")
    validate_sampler(proof.get("dense_ground_load_sampling_initial_audit"), num_envs=2048, steps=8, device=device)
    validate_sampler(proof.get("dense_ground_load_sampling_final_audit"), num_envs=2048, steps=14408, device=device)
    require(type(proof.get("limitations")) is list and bool(proof["limitations"]), "CPU proof limitations")
    return proof


def validate_actor_proof(proof, trial_dir, files, hashes):
    """Separate recorded actor replay from CPU completion and behavioral verdict."""
    lr.validate_actor_proof(proof, trial_dir, files, hashes)
    fields(proof, {"schema_version": 1, "new_rollouts": 0, "simulation_steps": 0,
                  "behavior_success_claimed": False}, "Recorded-only paired proof")
    require(set(proof["checks"]) == PAIRED_CHECK_NAMES, "Exact original15 trial017 check names")
    fields(proof, {"audit_script_sha256": digest(PAIRED_PRODUCER_SHA256, "Unsealed paired producer")},
           "Actual reviewed017 paired producer")
    for relative in RUNTIME_INPUTS:
        lr.proof_seals(proof, relative, trial_dir / relative)
    return proof


def validate_evaluation_seal(seal, hashes):
    expected = {"schema_version": 1, "status": "trial017_evaluation_stage_sealed", "trial_id": "trial017",
        "git_commit": SOURCE_COMMIT, "protocol_sha256": PROTOCOL_SHA256, "profile_sha256": PROFILE_SHA256,
        "archive_sha256": ARCHIVE_SHA256, "fixture_prelaunch_seal_sha256": hashes["fixture_seal"],
        "training_prelaunch_seal_sha256": hashes["training_seal"],
        "training_completion_proof_sha256": hashes["training_completion"],
        "final_checkpoint_sha256": hashes["final_checkpoint"],
        "deployment_source_sha256": {"launch_eval_trial017.py":
                                    digest(EVALUATION_LAUNCHER_SHA256, "Unsealed new evaluation guard")}}
    require(typed_equal(seal, expected), "Exact12-field final-checkpoint evaluation seal")


def validate_native_audit(audit, device):
    require(type(device) is str and re.fullmatch(r"cuda:[0-7]", device), "Explicit selected native ordinal")
    index = int(device.split(":")[1])
    fields(audit, {"schema_version": 1, "status": "verified", "requested_device": device,
        "environment_device": device, "physics_sim_view_frontend_device": device,
        "robot_native_transform_tensor_device": device, "native_physics_backend_device_ordinal": index,
        "torch_current_device": index, "physics_cudaDevice_setting": index,
        "physics_sceneMultiGPUMode_setting": 0, "cuda_visible_devices": None,
        "required_unmasked": True, "device_index": index, "index_scope": "unmasked_cuda_ordinal",
        "native_and_tensor_device_consistency_verified": True,
        "physical_mapping_verified": False, "exclusivity_verified": False}, "Actual native/tensor device audit")


def validate_context(records, files, hashes, protocol, completion, proof, *, repo_root=ROOT):
    validate_protocol(protocol, repo_root=repo_root)
    for name, expected in SOURCE27.items():
        require(sha256(repo_root / name) == expected, "Frozen science source changed: " + name)
    manifest, contract, isaac, mujoco = [records[name] for name in (
        "manifest", "contract", "isaac_report", "mujoco_report")]
    require(typed_equal(manifest.get("source_sha256"), TRAINING_SOURCE17), "Exactly17 actual training sources")
    profile = read_json(repo_root / PROFILE_FILE)
    require(sha256(repo_root / PROFILE_FILE) == PROFILE_SHA256, "New profile byte pin")
    strict.validate_wheel_axle_profile(profile)
    parent_path = repo_root / "outputs/drift-sim2sim-20261011/trial004/training/manifest.json"
    require(sha256(parent_path) == lr.PARENT_MANIFEST_SHA256, "Original004 manifest")
    parent = read_json(parent_path)
    strict.validate_wheel_axle_contract(manifest["contract"], parent["contract"], profile)
    fields(manifest, {"wheel_axle_orientation_tracking": {"weight": 1.0, "std": .4},
        "wheel_axle_target_semantics": AXLE_TARGET_SEMANTICS,
        "dense_ground_load_sampling": "all_substeps_mean_cost",
        "baseline_entry_source_sha256": SOURCE27["scripts/train_tracking_lr_sync.py"]},
        "Actual source-backed LAST axle training objective")
    require(typed_equal(manifest.get("task_profile"), profile)
        and typed_equal(manifest.get("domain_randomization_profile"), parent.get("domain_randomization_profile"))
        and manifest.get("resume_sha256") == lr.PARENT_CP_SHA256
        and manifest.get("task_profile_sha256") == PROFILE_SHA256, "Held parent/profile/domain")
    old_export_path = repo_root / "outputs/drift-sim2sim-20261011/trial004/isaac/policy_contract.json"
    require(sha256(old_export_path) == "63bd2ad72bbf5ca390552c7437eedf3a745c8eef1140cfb1540ee4b3e54ccc53",
            "Original004 nominal export")
    old_export = read_json(old_export_path)
    require(set(contract) == set(old_export), "No new policy contract fields")
    for key, value in old_export.items():
        if key not in {"task_profile", "checkpoint_sha256", "export_validation_max_abs_error"}:
            require(typed_equal(contract[key], value), "Held nominal export field: " + key)
    require(typed_equal(contract["task_profile"], profile) and typed_equal(contract, isaac["policy_export"])
            and typed_equal(contract, mujoco["contract"]), "Actual profiles/contracts match")
    for record in (manifest, contract, isaac, mujoco):
        require("controller_gain_diagnostic" not in record and "physics_clock_refinement" not in record,
                "No other controller/clock factor")
    fields(isaac, {"evaluator_source_sha256": SOURCE27["scripts/eval_tracking_wheel_axle.py"],
        "baseline_evaluator_source_sha256": SOURCE27["scripts/eval_tracking.py"],
        "dense_ground_load_module_sha256": SOURCE27["training/tron1_ground_load_substeps.py"],
        "dense_ground_load_trial_module_sha256": SOURCE27["training/tron1_ground_load_substeps_trial.py"],
        "substep_reward_module_sha256": SOURCE27["training/tron1_substep_rewards.py"],
        "ground_load_reserve_module_sha256": SOURCE27["training/tron1_ground_load_reserve.py"],
        "baseline_dense_evaluator_source_sha256": SOURCE27["scripts/eval_tracking_dense_ground_load.py"],
        "wheel_axle_tracking_source_sha256": SOURCE27["training/tron1_wheel_axle_tracking.py"],
        "wheel_axle_trial_source_sha256": SOURCE27["training/tron1_wheel_axle_trial.py"],
        "dense_ground_load_sampling": "all_substeps_mean_cost"}, "Actual dense evaluator dependencies")
    for record in (contract, mujoco):
        require("reward_sampling_audit" not in record, "No imaginary MuJoCo reward-manager sampler")
    for name in ("num_envs", "iterations_requested", "seed", "smoke_steps_passed"):
        fields(manifest, {name: {"num_envs": 2048, "iterations_requested": 600, "seed": 42,
                                "smoke_steps_passed": 8}[name]}, "Training budget")
    fields(records["run_status"], {"status": "training_completed", "num_envs": 2048,
        "iterations_requested": 600, "training_environment_steps": 29491200, "last_iteration": 4595},
        "Completed final600 status")
    final_path = Path(records["run_status"].get("checkpoint", ""))
    require(final_path.name == "model_final.pt" and final_path.parent.parts[-3:] ==
            ("outputs", "trial017", "training"), "Predesignated final017 only")
    require(sha256(files["runner"]) == lr.RUNNER_SHA256, "Original PPO runner")
    require(hashes["final_checkpoint"] == contract["checkpoint_sha256"] == isaac["checkpoint_sha256"],
            "Actual final checkpoint/export chain")
    validate_completion(completion, hashes["final_checkpoint"],
        fixture_seal=records["fixture_seal"], training_seal=records["training_seal"],
        fixture_seal_sha256=hashes["fixture_seal"], training_seal_sha256=hashes["training_seal"])
    validate_native_audit(manifest.get("runtime_device_audit"), completion["device"])
    require(typed_equal(completion.get("resume_learning_rate_audit"), records["lr_audit"])
        and typed_equal(manifest.get("resume_learning_rate_audit"), records["lr_audit"])
        and typed_equal(completion.get("resume_source_audit"), manifest.get("resume_source_audit")),
        "Exact dedicated and startup LR/source audits")
    source_audit = manifest.get("resume_source_audit")
    fields(source_audit, {"schema_version": 1, "rsl_rl_version": "3.0.1",
        "expected_source_sha256": lr.RSL_SOURCE_PINS, "actual_source_sha256": lr.RSL_SOURCE_PINS,
        "verified_before_runner_load": True,
        "source_root": "/data/zengxiong/.conda/envs/regrind/lib/python3.11/site-packages/rsl_rl"},
        "Actual installed RSL source audit")
    require(set(source_audit) == {"schema_version", "rsl_rl_version", "source_root",
        "expected_source_sha256", "actual_source_sha256", "verified_before_runner_load"},
        "Exact installed RSL source audit fields")
    require(manifest.get("versions", {}).get("rsl-rl-lib") == "3.0.1", "RSL version")
    for name in ("manifest.json", "run_status.json", "runner_config.json", "model_final.pt",
                 "resume_learning_rate_audit.json", "reward_sampling_final_audit.json"):
        require(completion["input_sha256"]["training/" + name] == sha256(files["manifest"].parent / name),
                "CPU proof must seal actual training file " + name)
    initial = manifest.get("reward_sampling_audit")
    final = records["final_sampling"]
    evaluation = isaac.get("reward_sampling_audit")
    require(typed_equal(initial, completion["dense_ground_load_sampling_initial_audit"])
            and typed_equal(final, completion["dense_ground_load_sampling_final_audit"]), "Actual CPU sampler audits")
    validate_sampler(evaluation, num_envs=1, steps=288, device=evaluation.get("lifecycle", {}).get("device"))
    validate_native_audit(isaac.get("runtime_device_audit"), evaluation["lifecycle"]["device"])
    validate_actor_proof(proof, files["manifest"].parent.parent, files, hashes)
    for key, value in (("dense_ground_load_sampling_initial_audit", initial),
                       ("dense_ground_load_sampling_final_audit", final),
                       ("dense_ground_load_sampling_evaluation_audit", evaluation),
                       ("resume_learning_rate_audit", records["lr_audit"]),
                       ("resume_source_audit", source_audit)):
        require(typed_equal(proof.get(key), value), "Paired proof current context: " + key)
    require(proof.get("training_completion_proof_sha256") == hashes["training_completion"],
            "Paired proof ties actual CPU completion")
    validate_evaluation_seal(records["evaluation_seal"], hashes)
    fields(proof, {"wheel_axle_orientation_tracking": {"weight": 1.0, "std": .4},
        "wheel_axle_target_semantics": AXLE_TARGET_SEMANTICS,
        "wheel_axle_tracking_source_sha256": SOURCE27["training/tron1_wheel_axle_tracking.py"],
        "wheel_axle_trial_source_sha256": SOURCE27["training/tron1_wheel_axle_trial.py"],
        "held_last_wheel_axle_sampling": True,
        "full_training_live_axle_reward_ledger_archived": False,
        "prelaunch_seal_sha256": hashes["fixture_seal"],
        "fixture_prelaunch_seal_sha256": hashes["fixture_seal"],
        "training_prelaunch_seal_sha256": hashes["training_seal"],
        "evaluation_prelaunch_seal_sha256": hashes["evaluation_seal"],
        "training_source_sha256": TRAINING_SOURCE17, "sealed_source_sha256": SOURCE27,
        "source_backed_wheel_axle_configuration_only": True,
        "fixture_raw_arithmetic_freshly_replayed_locally": False,
        "stage_seal_input_sha256": {"prelaunch_seal.json": hashes["fixture_seal"],
            "training_prelaunch_seal.json": hashes["training_seal"],
            "evaluation_prelaunch_seal.json": hashes["evaluation_seal"]}},
        "Actual paired axle objective and three externally sealed stages")
    evaluation_inputs = proof.get("evaluation_input_sha256")
    require(type(evaluation_inputs) is dict and set(evaluation_inputs) == {
        "eval_launch_receipt.json", "eval_started.json", "eval_exit_report.json",
        "eval_live_placement.jsonl", "eval.log"}, "Exact five actual owned-evaluation receipt/log seals")
    for value in evaluation_inputs.values(): digest(value, "Actual owned-evaluation input SHA")
    for name, value in evaluation_inputs.items():
        path = files.get("evaluation_stage/" + name)
        require(isinstance(path, Path) and path.is_file() and sha256(path) == value,
                "Paired owned-evaluation receipt/log differs from actual copy: " + name)
    require(proof.get("input_sha256", {}).get(
        "repository/outputs/drift-trial017-deployment-20261012/launch_eval_trial017.py") == EVALUATION_LAUNCHER_SHA256,
        "Paired producer seals actual evaluated guard source")
    for key, expected in (("training_completion_input_sha256", completion["input_sha256"]),
                          ("training_completion_external_input_sha256", completion["external_input_sha256"]),
                          ("completion_local_input_sha256", completion["input_sha256"]),
                          ("completion_local_external_input_sha256", completion["external_input_sha256"])):
        require(typed_equal(proof.get(key), expected), "Paired proof full completion input-copy rehash: " + key)
    require(proof.get("all140_relative_and9_external_local_byte_copies_rehashed") is True,
            "Paired producer must rehash all140 relative and9 external completion inputs")
    for name, value in SOURCE27.items():
        require(proof.get("input_sha256", {}).get("repository/" + name) == value, "Paired source seal: " + name)
    require(proof.get("input_sha256", {}).get("repository/" + PROTOCOL_FILE) == PROTOCOL_SHA256
            and proof.get("input_sha256", {}).get("repository/" + PROFILE_FILE) == PROFILE_SHA256,
            "Paired prospective protocol/profile seals")
    fields(isaac.get("physics_recording"), {"samples": 1153}, "All actual Isaac physics samples")
    fields(mujoco, {"physics_steps": 1152, "policy_steps": 288, "root_state_writes": 1,
                   "hidden_resets": 0, "physics_stepped": True, "policy_sha256": hashes["actor"]},
           "Complete original MuJoCo physics-only first episode")
    physics = proof.get("physics_provenance")
    for engine, key in (("isaac", "isaac_physics_trajectory"), ("mujoco", "mujoco_trajectory")):
        fields(physics.get(engine) if type(physics) is dict else None,
            {"sample_dt_s": .005, "sha256": hashes[key]}, "Actual200Hz engine archive")
    return {"schema_version": 1, "experiment_kind": EXPERIMENT_KIND, "primary_comparison_trial": "trial016",
        "descriptive_comparison_trials": ["trial013", "trial014", "trial015"],
        "only_factor_relative_to_primary": "added_LAST_spin_invariant_localY_wheel_axle_reward_weight1_std0.4",
        "objective": protocol["reward"], "wheel_axle_target_semantics": AXLE_TARGET_SEMANTICS,
        "fixed_source_windows_s": [[.78, 1.44], [1.82, 2.32]],
        "reference_alignment_changed": False, "gap_bridging_or_gate_tolerance_added": False,
        "full_training_live_axle_reward_ledger_archived": False,
        "optional_axle_analysis_requires_same_yaw_reanchored_LAST_target": True, "held_wheel_vertical_velocity_sampling": "last_physics_substep",
        "fixture_raw_arithmetic_freshly_replayed_locally": False,
        "fixture_arithmetic_bit_exact_portability_claimed": False,
        "inherited_velocity_collection_consumed_as_reward": False, "held_sideslip_reward_weight": 2.0,
        "training_source_sha256": TRAINING_SOURCE17, "sealed_scientific_source_sha256": SOURCE27,
        "scientific_source_commit": SOURCE_COMMIT, "deployment_archive_sha256": ARCHIVE_SHA256,
        "restoration": records["lr_audit"], "installed_rsl_sources": source_audit,
        "parent_checkpoint_sha256": lr.PARENT_CP_SHA256,
        "actual_final_checkpoint_sha256": hashes["final_checkpoint"], "actual_actor_sha256": hashes["actor"],
        "budget": completion["training_budget"],
        "sampling_audits": {"training_startup": initial, "training_final": final, "isaac_evaluation": evaluation},
        "current_fixture_prerequisites": {key: completion[key] for key in (
            "current_stack_smoke_prerequisite", "current_stack_runtime_prerequisite")},
        "cpu_completion_proof_sha256": hashes["training_completion"],
        "paired_actor_proof_sha256": hashes["paired_proof"], "behavior_gates_changed": False,
        "fixture_prelaunch_seal_sha256": hashes["fixture_seal"],
        "training_prelaunch_seal_sha256": hashes["training_seal"],
        "evaluation_prelaunch_seal_sha256": hashes["evaluation_seal"],
        "learned_actor_equals_baseline_required": False, "hardware_ready": False,
        "production_force_cache_values_independently_archived": False,
        "normal_force_envelope_equivalence_claimed": False,
        "limitations": ["CPU checkpoint completion is not paired actor provenance or behavioral success.",
            "Actual fixtures validate equation/layout in their16 windows, not full-run production force values.",
            "Actual source-sealed remote fixture proofs and raw hashes are retained; this publisher does not freshly replay their floating-point arithmetic locally or claim bit-exact portability.",
            "Dense force-aggregate soft costs do not certify bearing contacts or absent native manifolds.",
            "Original761 instantaneous cache audit labels do not redefine016 mean-of-cost sampling.",
            "Inherited all-substep velocity statistics are unused; the original LAST wheel-vz reward stays.",
            "Axle tracking reweights the SAME robot reference's yaw-reanchored local+Y directions, not measured car-wheel motion or spin.",
            "The positive LAST/50Hz axle objective does not supervise all 5ms contact states; knee orientation already carries the same cue.",
            "One seed and one nominal episode per engine do not establish causality, robustness or hardware safety.",
            "Native ordinal/owned exit evidence does not establish UUID mapping, GPU isolation or app.close return."]}


def validate_publication_producer_sources(repo_root):
    """Bind actual post-training producer bytes, not only their proof labels."""
    repo_root = Path(repo_root).resolve()
    producers = {
        "outputs/drift-trial017-deployment-20261012/audit_deployment_provenance_trial017.py": PAIRED_PRODUCER_SHA256,
        "outputs/drift-trial017-deployment-20261012/launch_eval_trial017.py": EVALUATION_LAUNCHER_SHA256,
    }
    for name, expected in producers.items():
        path = repo_root / name
        require(path.is_file() and not path.is_symlink() and path.resolve() == path,
                "Missing/remapped post-training producer " + name)
        require(sha256(path) == digest(expected, "Post-training producer source"),
                "Changed post-training producer " + name)
    return producers


def validate_shared_publication_evidence(proof, files, hashes):
    """Do not relabel this explicitly authorized shared run as an idle-GPU run."""
    for key, expected in (("shared_entrypoint", SHARED_ENTRYPOINT_SHA256),
                          ("shared_authorization", SHARED_AUTHORIZATION_SHA256)):
        require(hashes.get(key) == expected and sha256(files[key]) == expected,
                "Changed actual shared-evaluation evidence: " + key)
    receipt = read_json(files["evaluation_stage/eval_launch_receipt.json"])
    fields(receipt, {"status": "sealed_and_authorized_shared_before_launch",
        "resource_mode": "user_authorized_shared", "physical_gpu_index": 5,
        "physical_gpu_uuid_pin": "GPU-f3fe90b7-ba72-b055-11a3-440d7726d34e",
        "entrypoint_sha256": SHARED_ENTRYPOINT_SHA256,
        "shared_authorization_sha256": SHARED_AUTHORIZATION_SHA256}, "Actual shared eval receipt")
    fields(proof.get("evaluation_placement"), {"resource_mode": "user_authorized_shared",
        "selected_gpu_index": 5, "entrypoint_sha256": SHARED_ENTRYPOINT_SHA256,
        "shared_authorization_sha256": SHARED_AUTHORIZATION_SHA256,
        "physical_isolation_claimed": False, "all8_owned_contexts_absent": True}, "Paired shared eval evidence")
    for name, expected in (("launch_eval_trial017_shared.py", SHARED_ENTRYPOINT_SHA256),
                           ("shared_evaluation_authorization.json", SHARED_AUTHORIZATION_SHA256)):
        require(proof.get("input_sha256", {}).get(
            "repository/outputs/drift-trial017-deployment-20261012/" + name) == expected,
            "Paired proof must seal actual shared entrypoint/authority")
    return {"mode": "user_authorized_shared", "selected_gpu_index": 5,
        "physical_isolation_claimed": False, "harmlessness_claimed": False,
        "entrypoint_sha256": SHARED_ENTRYPOINT_SHA256,
        "authorization_sha256": SHARED_AUTHORIZATION_SHA256}


def prepare_publication(trial_dir, reference_dir, render_dir, *, protocol_path,
                        expected_protocol_sha256, fixture_seal_path, fixture_seal_sha256,
                        training_seal_path, training_seal_sha256,
                        evaluation_seal_path, evaluation_seal_sha256, repo_root=ROOT):
    repo_root, trial_dir, protocol_path = Path(repo_root).resolve(), Path(trial_dir).resolve(), Path(protocol_path).resolve()
    require(trial_dir.name == "trial017" and render_dir is not None, "Explicit017 complete four-panel trial only")
    require(protocol_path == repo_root / PROTOCOL_FILE, "Exact prospective017 protocol path")
    require(expected_protocol_sha256 == PROTOCOL_SHA256 and sha256(protocol_path) == PROTOCOL_SHA256,
            "Externally supplied prospective seal")
    require(sha256(base.__file__) == LEGACY_PUBLISHER_SHA256 and sha256(lr.__file__) == LR_PUBLISHER_SHA256,
            "Frozen publication utilities")
    require(sha256(dense.__file__) == DENSE_PUBLISHER_SHA256, "Frozen016 sampler/publication dependency")
    digest(PAIRED_PRODUCER_SHA256, "Paired producer must be sealed before any publication")
    digest(EVALUATION_LAUNCHER_SHA256, "Evaluation guard must be sealed before any publication")
    validate_publication_producer_sources(repo_root)
    for name, value in SOURCE27.items():
        require(sha256(repo_root / name) == value, "Frozen science source " + name)
    protocol = validate_protocol(read_json(protocol_path), repo_root=repo_root)
    supplied = {"fixture_seal": (fixture_seal_path, fixture_seal_sha256),
        "training_seal": (training_seal_path, training_seal_sha256),
        "evaluation_seal": (evaluation_seal_path, evaluation_seal_sha256)}
    for name, (path, expected) in supplied.items():
        require(Path(path).is_file() and sha256(path) == digest(expected, name), "External actual stage seal " + name)
    validate_stage_seals(read_json(fixture_seal_path), read_json(training_seal_path),
        fixture_sha=fixture_seal_sha256, training_sha=training_seal_sha256)
    summary, payloads, media, files, hashes = _prepare_recorded_candidate(
        trial_dir, reference_dir, render_dir, repo_root=repo_root)
    extra = {"protocol017": protocol_path, "final_checkpoint": trial_dir / "training/model_final.pt",
        "lr_audit": trial_dir / "training/resume_learning_rate_audit.json",
        "final_sampling": trial_dir / "training/reward_sampling_final_audit.json",
        "training_completion": trial_dir / "diagnostics/training_completion_verification.json",
        "paired_proof": trial_dir / "diagnostics/deployment_provenance_audit.json", "publisher017": Path(__file__),
        **{name: Path(path) for name, (path, _) in supplied.items()}}
    for name in ("eval_launch_receipt.json", "eval_started.json", "eval_exit_report.json",
                 "eval_live_placement.jsonl", "eval.log"):
        extra["evaluation_stage/" + name] = Path(evaluation_seal_path).resolve().parent / name
    extra["shared_entrypoint"] = Path(evaluation_seal_path).resolve().parent / "launch_eval_trial017_shared.py"
    extra["shared_authorization"] = Path(evaluation_seal_path).resolve().parent / "shared_evaluation_authorization.json"
    for name, path in extra.items():
        require(path.is_file() and path.stat().st_size > 0, "Missing actual017 evidence " + name)
        files[name], hashes[name] = path, sha256(path)
    for name, value in SOURCE27.items():
        files["science_source/" + name], hashes["science_source/" + name] = repo_root / name, value
    records = {key: read_json(files[key]) for key in ("manifest", "contract", "isaac_report", "mujoco_report",
        "runner", "run_status", "lr_audit", "final_sampling", "adaptation", "export_report", "assessment",
        "fixture_seal", "training_seal", "evaluation_seal")}
    context = validate_context(records, files, hashes, protocol, read_json(files["training_completion"]),
        read_json(files["paired_proof"]), repo_root=repo_root)
    resource = validate_shared_publication_evidence(read_json(files["paired_proof"]), files, hashes)
    identities = private_identities({**records, "context": context})
    summary.update({"experiment_kind": EXPERIMENT_KIND, "additional_training_updates": 600,
        "wheel_axle_context": sanitize(context, identities), "evaluation_resource": resource, "input_sha256": hashes})
    require(len(payloads) == 3 and len(media) == 4, "Unchanged full four-panel8-file format")
    require(all(sha256(path) == hashes[key] for key, path in files.items()), "Inputs changed during preflight")
    return sanitize(summary, identities), payloads, media, files, hashes


def publish_trial(trial_dir, reference_dir, output_dir, render_dir, **kwargs):
    output_dir = Path(output_dir)
    require(output_dir.name == "trial017" and not output_dir.exists(),
            "Output must be a NEW trial017 snapshot; refusing overwrite or historical relabeling")
    summary, payloads, media, files, hashes = prepare_publication(trial_dir, reference_dir, render_dir, **kwargs)
    payloads["summary.json"] = encode(summary)
    require(all(sha256(path) == hashes[key] for key, path in files.items()), "Inputs changed before publication")
    require(all(sha256(item["path"]) == item["sha256"] for item in media.values()), "Media changed before publication")
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in payloads.items():
        with (output_dir / name).open("xb") as stream:
            stream.write(value)
    for name, item in media.items():
        with item["path"].open("rb") as source, (output_dir / name).open("xb") as target:
            shutil.copyfileobj(source, target)
        require(sha256(output_dir / name) == item["sha256"], "Media copy changed")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--protocol", dest="protocol_path", type=Path, default=ROOT / PROTOCOL_FILE)
    parser.add_argument("--protocol-sha256", dest="expected_protocol_sha256", required=True)
    for stage in ("fixture", "training", "evaluation"):
        parser.add_argument("--" + stage + "-seal", dest=stage + "_seal_path", type=Path, required=True)
        parser.add_argument("--" + stage + "-seal-sha256", dest=stage + "_seal_sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = vars(parser.parse_args())
    result = publish_trial(**args)
    print(json.dumps({"status": result["status"], "verdict": result["verdict"],
                      "output_dir": str(args["output_dir"])}))


if __name__ == "__main__":
    main()
