"""Publish one sanitized, hash-linked car-drift trial without raw states/models.

Recompute the assessment and complete every provenance/material/media preflight
before creating the new snapshot. Physical behavior failures remain visible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assess_drift import assess_runs
from compare_tracking import validate_provenance
from publish_motion_suite import private_identities, read_json, sanitize, sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pick(value, fields):
    return {key: value[key] for key in fields if key in value}


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def check_material_audit(audit, mu, engine):
    """Read-back material values must match the declared task on both sides."""
    require(isinstance(audit, dict), f"Missing {engine} contact-material audit")
    require(isinstance(audit.get("expected_friction"), (int, float))
            and not isinstance(audit["expected_friction"], bool)
            and math.isfinite(audit["expected_friction"])
            and abs(audit["expected_friction"] - mu) <= 1e-6,
            f"{engine} material audit target differs from task profile")
    if engine == "MuJoCo":
        bounds = audit.get("compiled_friction_range")
        require(isinstance(audit.get("colliding_geom_count"), int)
                and audit["colliding_geom_count"] > 0, "MuJoCo material audit has no colliders")
    else:
        bounds = audit.get("robot_runtime_friction_range")
        ground = audit.get("ground_colliders")
        require(isinstance(ground, list) and len(ground) > 0, f"{engine} material audit has no ground colliders")
        for collider in ground:
            require(isinstance(collider, dict), f"Invalid {engine} ground-material entry")
            values = [collider.get("static_friction"), collider.get("dynamic_friction")]
            require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                        and abs(v - mu) <= 1e-6 for v in values),
                    f"{engine} ground friction differs from task profile")
    require(isinstance(bounds, list) and len(bounds) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    and abs(v - mu) <= 1e-6 for v in bounds),
            f"{engine} robot/contact friction differs from task profile")
    return audit


def training_budget(manifest, status=None, runner=None):
    fields = ("method", "upstream_revision", "num_envs", "iterations_requested", "seed",
              "resume_sha256", "smoke_steps_passed", "versions", "gpu", "asset_sha256",
              "source_sha256", "task_profile_sha256", "domain_randomization_profile")
    out = pick(manifest, fields)
    out["completion_status"] = status.get("status") if status is not None else "not_supplied"
    verified = False
    if status is not None:
        out["reported_run_status"] = pick(status, ("status", "last_iteration", "training_environment_steps",
                                                    "elapsed_seconds", "num_envs", "iterations_requested"))
        verified = (status.get("status") == "training_completed"
                    and status.get("num_envs") == manifest.get("num_envs")
                    and status.get("iterations_requested") == manifest.get("iterations_requested"))
    out["requested_budget_completion_verified"] = verified
    out["completed_requested_iterations"] = manifest.get("iterations_requested") if verified else None
    if runner is not None:
        out["num_steps_per_env"] = runner.get("num_steps_per_env")
    return out


def check_leg_gain_audit(audit, kp, kd, engine):
    """Diagnostic-only gain proof; legacy publication requirements are unchanged."""
    gain = audit.get("leg_pd_gain_readback") if isinstance(audit, dict) else None
    require(isinstance(gain, dict), f"Missing {engine} live leg gain readback")
    for key, expected in (("kp_range", kp), ("kd_range", kd)):
        values = gain.get(key)
        require(isinstance(values, list) and len(values) == 2
                and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) and abs(value - expected) <= 1e-6
                        for value in values),
                f"{engine} live leg gains differ from the diagnostic contract")


def validate_controller_gain_context(records, files, hashes):
    """Allow only a hash-pinned, zero-training single-gain deployment diagnostic.

    This branch is explicit; absent diagnostic metadata, the historical strict
    training/deployment profile equality and public schema remain untouched.
    """
    isaac, mujoco, contract, manifest = [records[key] for key in (
        "isaac_report", "mujoco_report", "contract", "manifest")]
    key = "controller_gain_diagnostic"
    candidates = (isaac, mujoco, contract, manifest, mujoco.get("contract", {}),
                  manifest.get("contract", {}))
    if not any(isinstance(value, dict) and key in value for value in candidates):
        return None
    require(key not in manifest and key not in manifest.get("contract", {}),
            "Diagnostic must preserve the original training manifest and contract")
    require(isinstance(contract.get(key), dict), "Missing explicit controller gain diagnostic metadata")
    require(isaac.get(key) == contract[key], "Isaac diagnostic metadata differs from its saved contract")
    require(mujoco.get("contract") == contract,
            "Diagnostic deployments must use exactly the same saved contract")
    if key in mujoco:
        require(mujoco[key] == contract[key], "MuJoCo diagnostic metadata differs from its saved contract")
    checkpoint = files["manifest"].parent / "model_final.pt"
    require(checkpoint.is_file() and checkpoint.stat().st_size > 0,
            "Diagnostic requires the original training checkpoint")
    files["original_checkpoint"] = checkpoint
    hashes["original_checkpoint"] = sha256(checkpoint)
    scope = contract[key].get("scope")
    if scope == "zero_shot_common_leg_kp":
        helper_path = ROOT / "training/tron1_controller_gain_diagnostic.py"
    elif scope == "zero_shot_common_leg_kd":
        helper_path = ROOT / "training/tron1_controller_damping_diagnostic.py"
        dependency = ROOT / "training/tron1_controller_gain_diagnostic.py"
        require(dependency.is_file(), "Original controller validator dependency is unavailable")
        files["controller_gain_original_validation_dependency"] = dependency
        hashes["controller_gain_original_validation_dependency"] = sha256(dependency)
    else:
        raise ValueError("Unknown single-gain diagnostic scope")
    require(helper_path.is_file(), "Controller diagnostic validator is unavailable")
    files["controller_gain_helper"] = helper_path
    hashes["controller_gain_helper"] = sha256(helper_path)
    if scope == "zero_shot_common_leg_kp":
        from training.tron1_controller_gain_diagnostic import validate_deployment_gain_diagnostic
    else:
        from training.tron1_controller_damping_diagnostic import validate_deployment_gain_diagnostic
    audit = validate_deployment_gain_diagnostic(
        contract, manifest, original_manifest_sha256=hashes["manifest"],
        checkpoint_sha256=hashes["original_checkpoint"])
    require(isaac.get("checkpoint_sha256") == hashes["original_checkpoint"]
            == contract.get("checkpoint_sha256"),
            "Diagnostic must evaluate the original training checkpoint")
    require(records["assessment"].get("provenance_checks", {}).get(
        "same_physics_contact_sampling_clock", {}).get("passed") is True,
        "Diagnostic requires paired actual physics-rate contact evidence")
    trained = manifest.get("contract", {})
    check_leg_gain_audit(manifest.get("contact_material_audit"),
                         trained.get("leg_kp"), trained.get("leg_kd"), "Original training Isaac")
    check_leg_gain_audit(isaac.get("contact_material_audit"),
                         contract["leg_kp"], contract["leg_kd"], "Diagnostic Isaac")
    asset_hash = manifest.get("asset_sha256")
    require(isinstance(asset_hash, str) and re.fullmatch(r"[0-9a-f]{64}", asset_hash)
            and isaac.get("asset_root_file_sha256") == asset_hash,
            "Diagnostic Isaac asset differs from the original training asset")
    require(isaac.get("domain_parameters") is None,
            "Gain diagnostic cannot combine an Isaac actuator-domain override")
    domain = mujoco.get("domain_parameters")
    require(isinstance(domain, dict) and domain.get("is_nominal") is True
            and domain.get("joint_names") == contract["observed_joint_velocity_order"]
            and domain.get("torque_scale") == [1.] * contract["action_dim"]
            and domain.get("velocity_scale") == [1.] * contract["action_dim"]
            and domain.get("wheel_friction_nm") == [0., 0.],
            "Gain diagnostic requires actual nominal MuJoCo actuator parameters")
    contact = mujoco.get("contact_material_audit", {})
    require(contact.get("contact_margin_and_gap_changed") is False,
            "Gain diagnostic cannot change contact margin or gap")
    solref = contract["task_profile"].get("mujoco_contact_solref")
    if solref is not None:
        require(contact.get("explicit_contact_solref") == solref
                and contact.get("compiled_solref_ranges") == [[value, value] for value in solref],
                "Diagnostic MuJoCo contact response differs from the original task profile")
    return audit


GROUND_LOAD_RESERVE_SOURCE_PINS = {
    "training/tron1_tracking.py": "0a29fa27c1d65c0b9add0886221491e549553e3529d6970f1ec5ba35e2532817",
    "training/tron1_drift.py": "6d5f7974602adec9313a5e4feceb8e3caab392ca99351ac4a5f717e57fe1cf8a",
    "training/tron1_domain_randomization.py": "3f3a0b62c85a0af496ae9e9e0d75dda85c3ec30da587e4a47969843f5b1b1dca",
    "training/tron1_contact_slip.py": "52ef38de9cece19a8ee02c120cb85d472ffd831e4587e5f0d1f340974ab699f4",
    "training/tron1_ground_load_reserve.py": "761eccede3c0d90d744a4d9c45c9671b318426ab9bc6f9762ee7cacb3c14496d",
    "scripts/train_tracking.py": "ea9e4cd14f25ff3ba5202b9ba43ac0f08320418e2311cbae205c8449f3a1da9c",
    "scripts/eval_tracking.py": "fd90089810d3d14bfd96210723afdd3a36113a19f667ec60b2df617dd032f992",
}
GROUND_LOAD_RESERVE_BASELINE_SHA = "325cf279c699769694fb4934852ce85753b459e09670376b8c2d62d94f7d4e3c"
GROUND_LOAD_RESERVE_ASSET_PINS = {
    "motion": "8680ac0b331b8c2353701d95dcbce2ea8b0c97342eeb12244867eee33c84fb04",
    "robot_reference": "ea0b79c1ae84b8deb79961505555ca38decf22f93e44839d8b4cdc8d398deb81",
    "asset": "327ca5080e753bbc95d8c3c797e9cd49a0f134d400cfe80494af0ba840e23a8e",
    "model": "bd92264b72b4d09084732888dbd81fd94a64a1678978e58a2ab0dd191887f513",
    "reference_model": "30c8bb94b32b1804aa2f7de2425f7a8dd02be7792881255388097e2e68092e52",
}


def validate_ground_load_reserve_context(records, files, hashes):
    """Validate ORIGINAL761 source-backed startup context, not live force proof.

    There are no imports, additional input hashes or summary fields on the
    historical path. Runtime readbacks absent from that approved helper are
    explicitly not invented: this verifies its pinned guards and saved audit.
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
        path = ROOT / relative
        require(path.is_file() and sha256(path) == expected,
                f"Unsupported reserve source implementation: {relative}")
        if relative != "scripts/eval_tracking.py":
            require(source.get(relative) == expected, f"Missing or stale reserve training source pin: {relative}")
        input_key = "ground_load_reserve_source_" + path.stem
        files[input_key], hashes[input_key] = path, expected
    require(isaac.get("evaluator_source_sha256") == GROUND_LOAD_RESERVE_SOURCE_PINS["scripts/eval_tracking.py"]
            and isaac.get("drift_module_sha256") == GROUND_LOAD_RESERVE_SOURCE_PINS["training/tron1_drift.py"]
            and isaac.get("contact_slip_helper_sha256") == GROUND_LOAD_RESERVE_SOURCE_PINS["training/tron1_contact_slip.py"]
            and isaac.get("ground_load_reserve_module_sha256") == GROUND_LOAD_RESERVE_SOURCE_PINS[helper],
            "Missing or stale reserve evaluation source pins")
    baseline_path = ROOT / "config/car_drift_contact_task.json"
    require(baseline_path.is_file() and sha256(baseline_path) == GROUND_LOAD_RESERVE_BASELINE_SHA,
            "Original reserve baseline profile changed")
    files["ground_load_reserve_baseline_profile"] = baseline_path
    hashes["ground_load_reserve_baseline_profile"] = GROUND_LOAD_RESERVE_BASELINE_SHA
    baseline = read_json(baseline_path)
    profile = profiles[0]
    from training.tron1_drift import validate_task_profile
    for candidate in profiles:
        validate_task_profile(candidate)
    require(all(p == profile for p in profiles), "Reserve training/deployment profiles differ")
    require({k: v for k, v in profile.items() if k not in (key, "description")}
            == {k: v for k, v in baseline.items() if k != "description"},
            "Reserve may change only its new objective, not previous rewards/control/materials")
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
        "settings": settings, "sampling": "last_physics_substep_at_50Hz_reward_clock",
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
            "Initial zero ground forces are legitimate; last-substep sampling cannot supervise all intervening physics steps. The e-fold scale is not a minimum-force gate or hard 50:50 load target.",
        ],
    }


def validate_render(render_dir, input_hashes, reference_audit):
    """Require a freshly hash-linked recorded-rollout render; copy only media."""
    render_dir = Path(render_dir)
    path = render_dir / "render_report.json"
    report = read_json(path)
    require(report.get("status") == "rendered" and report.get("simulation_steps") == 0
            and report.get("full_decode_verified") is True,
            "Render must verify decoded playback of recorded evidence with zero simulation steps")
    expected = {
        "motion": input_hashes["motion"], "robot_reference": input_hashes["robot_reference"],
        "isaac": input_hashes["isaac_trajectory"], "mujoco": input_hashes["mujoco_trajectory"],
        "isaac_report": input_hashes["isaac_report"], "mujoco_report": input_hashes["mujoco_report"],
        "adaptation": input_hashes["adaptation"], "export_report": input_hashes["export_report"],
        "assessment": input_hashes["assessment"],
        "captured_source": reference_audit["provenance"]["source_npz_sha256"],
    }
    source_hashes = report.get("source_hashes", {})
    require(isinstance(source_hashes, dict), "Render source_hashes must be a mapping")
    if "isaac_physics_trajectory" in input_hashes:
        expected["isaac_physics"] = input_hashes["isaac_physics_trajectory"]
    else:
        require("isaac_physics" not in source_hashes,
                "Render selects Isaac physics evidence absent from this trial report")
    for key, expected_hash in expected.items():
        require(source_hashes.get(key) == expected_hash, f"Render {key} source SHA does not match this trial")
    media = {}
    for filename, key, required in (("motion_pipeline.mp4", "video_sha256", True),
                                    ("overview.png", "overview_sha256", True),
                                    ("final.png", "final_sha256", False),
                                    ("contact_overview.png", "contact_overview_sha256", False)):
        artifact = render_dir / filename
        if not required and not artifact.exists():
            continue
        require(artifact.is_file() and artifact.stat().st_size > 0, f"Missing rendered artifact: {filename}")
        actual = sha256(artifact)
        require(report.get(key) == actual, f"Render {filename} SHA missing or inconsistent")
        media[filename] = {"path": artifact, "sha256": actual}
    return report, media, path


def prepare_publication(trial_dir, reference_dir, render_dir=None):
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
    from training.tron1_drift import validate_task_profile
    original_task_profile = validate_task_profile(manifest.get("task_profile"))
    diagnostic = validate_controller_gain_context(records, files, hashes)
    task_profile = original_task_profile if diagnostic is None else validate_task_profile(contract.get("task_profile"))
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
    load_reserve_context = validate_ground_load_reserve_context(records, files, hashes)
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


def publish_trial(trial_dir, reference_dir, output_dir, render_dir=None):
    output_dir = Path(output_dir)
    require(not output_dir.exists(), "Output must be NEW; refusing overwrite or reuse of an existing snapshot")
    summary, payloads, media, files, hashes = prepare_publication(trial_dir, reference_dir, render_dir)
    for key, path in files.items():
        require(sha256(path) == hashes[key], f"Input changed during publication preflight: {key}")
    for name, item in media.items():
        require(sha256(item["path"]) == item["sha256"], f"Media changed during publication preflight: {name}")
    payloads["summary.json"] = encode(summary)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        with (output_dir / name).open("xb") as stream:
            stream.write(payload)
    for name, item in media.items():
        with item["path"].open("rb") as source, (output_dir / name).open("xb") as dest:
            shutil.copyfileobj(source, dest)
        require(sha256(output_dir / name) == item["sha256"], f"Published media copy SHA mismatch: {name}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = publish_trial(args.trial_dir, args.reference_dir, args.output_dir, args.render_dir)
    print(json.dumps({"status": summary["status"], "verdict": summary["verdict"],
                      "output_dir": str(args.output_dir)}, allow_nan=False))


if __name__ == "__main__":
    main()
