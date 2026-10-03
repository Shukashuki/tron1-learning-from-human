"""Publish a six-task, hash-linked evidence snapshot, never models or raw motion.

All inputs are checked before creating any output. Behavioral failures are valid
publication inputs; missing evidence and inconsistent provenance are not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np

from compare_tracking import load_episode, read_npz

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("forward_jump", "turn_jump", "side_jump", "rolling_stop", "crouch", "step_up")
DEFAULT_CONFIG = ROOT / "config/suite/motion_suite.json"
ASSESSMENT_FIELDS = (
    "schema_version", "task", "verdict", "failed_checks", "thresholds", "threshold_policy",
    "provenance", "provenance_checks", "terrain_comparison", "actuator_contract", "common_window",
    "isaac", "mujoco", "complete_contact_evidence", "safety_verified", "hardware_ready",
    "statistical_robustness_benchmark", "limitations",
)
SENSITIVE_KEY = re.compile(
    r"^(?:host(?:name)?|user(?:name)?|ssh.*|password|passwd|.*(?:access_token|api_key|secret|private_key)|"
    r"command|command_line|environment_variables)$", re.I)
PRIVATE_PATH = re.compile(r"(?<![\w:/])/(?:home|data|root|tmp|mnt|opt|usr|Users|workspace|workspaces)(?:/[^\s\"'<>;,\]\)}]*)?")
WINDOWS_PATH = re.compile(r"\b[A-Za-z]:[\\/][^\s\"'<>;,\]\)}]*")
IP_ADDRESS = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
SSH_IDENTITY = re.compile(r"\b[\w.+-]+@(?:[\w.-]+|\[[\da-fA-F:]+\])")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream, parse_constant=invalid)
    # Also catches overflow written as a syntactically valid JSON number.
    json.dumps(data, allow_nan=False)
    return data


def private_identities(value):
    """Collect host/account strings for redaction without publishing their values."""
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"host", "hostname", "user", "username"} and isinstance(item, str):
                found.add(item)
            found.update(private_identities(item))
    elif isinstance(value, list):
        for item in value:
            found.update(private_identities(item))
    elif isinstance(value, str):
        found.update(re.findall(r"/(?:home|data)/([^/\s]+)", value))
    return found


def sanitize(value, identities=(), *, version_context=False):
    """Remove private locations/identities, preserving scientific values and hashes."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if SENSITIVE_KEY.fullmatch(str(key)):
                continue
            clean = sanitize(str(key), identities)
            if clean in result:
                raise ValueError("Sanitization would collapse distinct JSON keys")
            result[clean] = sanitize(item, identities, version_context=(version_context or key in {"version", "versions"} or str(key).endswith("_version")))
        return result
    if isinstance(value, (list, tuple)):
        return [sanitize(item, identities, version_context=version_context) for item in value]
    if isinstance(value, str):
        result = WINDOWS_PATH.sub("<private-path>", PRIVATE_PATH.sub("<private-path>", value))
        result = SSH_IDENTITY.sub("<private-identity>", result)
        # Version fields may legitimately contain four numeric components.
        if not version_context:
            result = IP_ADDRESS.sub("<private-ip>", result)
        for identity in sorted(identities, key=len, reverse=True):
            if identity:
                result = re.sub(r"(?<![\w-])" + re.escape(identity) + r"(?![\w-])", "<private-identity>", result)
        return result
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("Public JSON must contain only finite JSON-compatible values")


def pick(mapping, names):
    return {key: mapping[key] for key in names if key in mapping}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference_summary(path):
    motion = read_npz(path)
    frames = len(motion["joint_pos"])
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    require(frames > 1 and math.isfinite(fps) and fps > 0, "Invalid reference sampling")
    result = {"frames": frames, "fps": fps, "sample_span_s": (frames - 1) / fps,
              "action_duration_s": frames / fps,
              "initialization": "reference frame zero including reference velocities; not a standstill transition"}
    for name, key in (("root_com_linear_velocity_m_s", "body_lin_vel_w"),
                      ("root_angular_velocity_rad_s", "body_ang_vel_w"),
                      ("joint_velocity_rad_s", "joint_vel")):
        require(key in motion, f"Reference lacks initial velocity evidence: {key}")
        initial = np.asarray(motion[key])[0]
        if key.startswith("body_"):
            initial = initial[0]
        require(np.isfinite(initial).all(), "Non-finite reference initial state")
        result[name] = initial.tolist()
    return result


def yaw_diagnostics(path, engine, episode_assessment):
    episode = load_episode(read_npz(path), engine)
    w, x, y, z = episode["quat"].T
    angles = np.rad2deg(np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y*y + z*z))))
    time = episode["time"]
    flights = []
    for interval in episode_assessment["metrics"].get("flight_intervals", []):
        begin = float(interval["takeoff_s"])
        end = begin + float(interval["no_contact_duration_s"])
        require(time[0] - 1e-8 <= begin < end <= time[-1] + 1e-8, "Flight diagnostic outside recorded episode")
        start_yaw, end_yaw = np.interp([begin, end], time, angles)
        flights.append({"takeoff_s": begin, "no_contact_end_s": end,
                        "yaw_before_takeoff_deg": float(start_yaw - angles[0]),
                        "airborne_yaw_change_deg": float(end_yaw - start_yaw)})
    return {"status": "posthoc descriptive only; not used to change the verdict",
            "initial_yaw_deg": float(angles[0]), "total_yaw_change_deg": float(angles[-1] - angles[0]),
            "flights": flights, "does_not_certify_90_degree_airborne_turn": True}


def budget_summary(manifest, status, runner, audit, contract, expected):
    actual = {"num_envs": manifest["num_envs"], "iterations_per_motion": manifest["iterations_requested"],
              "seed": manifest["seed"], "env_steps_per_motion": status["training_environment_steps"],
              "resume_sha256": manifest["resume_sha256"], "last_iteration": status["last_iteration"],
              "physics_hz": 1 / contract["physics_dt"], "policy_hz": contract["policy_fps"],
              "num_steps_per_env": runner["num_steps_per_env"]}
    checks = {key: actual[key] == expected[key] for key in (
        "num_envs", "iterations_per_motion", "seed", "env_steps_per_motion", "resume_sha256", "physics_hz", "policy_hz")}
    checks.update(status_completed=status.get("status") == "training_completed",
                  status_envs_match=status.get("num_envs") == actual["num_envs"],
                  status_iterations_match=status.get("iterations_requested") == actual["iterations_per_motion"],
                  counted_steps_match=actual["env_steps_per_motion"] == actual["num_envs"] * actual["iterations_per_motion"] * actual["num_steps_per_env"],
                  random_audit_enabled=audit.get("enabled") is True and audit.get("mode") == "random",
                  draw_readback_verified=audit.get("all_sampled_parameters_readback_verified") is True,
                  draw_hidden_from_actor=audit.get("parameters_hidden_from_actor") is True,
                  dc_cache_verified=audit.get("dc_overspeed_cache_verified") is True,
                  dr_profile_matches=audit.get("profile") == manifest.get("domain_randomization_profile") == contract.get("training_domain_randomization"),
                  reset_counts_cover_all_envs=len(audit.get("reset_counts", [])) == actual["num_envs"])
    if "resume_iteration" in expected:
        actual["expected_last_iteration"] = expected["resume_iteration"] + expected["iterations_per_motion"] - 1
        checks["final_iteration_matches"] = actual["last_iteration"] == actual["expected_last_iteration"]
    return {"pass": all(checks.values()), "checks": checks, "actual": actual,
            "is_behavior_success": False,
            "final_iteration_offset_verified": "resume_iteration" in expected}


def prepare_task(item, suite_root, expected, project_root):
    task = item["task"]
    trial = suite_root / task / "trial"
    files = {"assessment": trial / "assessment.json", "manifest": trial / "training/manifest.json",
             "status": trial / "training/run_status.json", "runner": trial / "training/runner_config.json",
             "audit": trial / "training/domain_randomization_final_audit.json",
             "checkpoint": trial / "training/model_final.pt", "isaac_report": trial / "isaac/report.json",
             "contract": trial / "isaac/policy_contract.json", "actor": trial / "isaac/actor_normalized.pt",
             "isaac_trajectory": trial / "isaac/trajectory.npz", "mujoco_report": trial / "mujoco/report.json",
             "mujoco_trajectory": trial / "mujoco/rollout.npz", "render": trial / "render/render_report.json",
             "video": trial / "render/tracking_comparison.mp4", "overview": trial / "render/overview.png",
             "reference": project_root / item["reference_path"]}
    for name, path in files.items():
        require(path.is_file() and path.stat().st_size > 0, f"{task}: missing/empty {name}")
    hashes = {name + "_sha256": sha256(path) for name, path in files.items()}
    records = {name: read_json(path) for name, path in files.items() if path.suffix == ".json"}
    assessment, manifest, isaac, mujoco, render, contract = (
        records[key] for key in ("assessment", "manifest", "isaac_report", "mujoco_report", "render", "contract"))
    require(assessment.get("task") == task, f"{task}: assessment task mismatch")
    require(assessment.get("verdict") in {"pass", "fail"}, f"{task}: invalid assessment verdict")
    for engine in ("isaac", "mujoco"):
        require(assessment.get(engine, {}).get("verdict") in {"pass", "fail"}, f"{task}: missing {engine} verdict")
        require((assessment[engine]["verdict"] == "pass") == (not assessment[engine].get("failed_checks")),
                f"{task}: contradictory {engine} verdict/failures")
    require((assessment["verdict"] == "pass") == (not assessment.get("failed_checks")),
            f"{task}: contradictory suite verdict/failures")
    if assessment["verdict"] == "pass":
        require(all(assessment[engine]["verdict"] == "pass" for engine in ("isaac", "mujoco")),
                f"{task}: suite pass contradicts engine failure")
    provenance = assessment["provenance"]
    for key in ("reference", "actor", "isaac_trajectory", "mujoco_trajectory", "isaac_report", "mujoco_report"):
        require(provenance.get(key + "_sha256") == hashes[key + "_sha256"], f"{task}: assessment {key} hash mismatch")
    required_hashes = [(manifest, "motion_sha256", "reference"), (isaac, "motion_file_sha256", "reference"),
                       (mujoco, "motion_sha256", "reference"), (contract, "reference_file_sha256", "reference"),
                       (isaac, "checkpoint_sha256", "checkpoint"), (contract, "checkpoint_sha256", "checkpoint"),
                       (isaac, "runner_config_sha256", "runner"), (mujoco, "policy_sha256", "actor"),
                       (render, "trajectory_sha256", "isaac_trajectory"), (render, "comparison_trajectory_sha256", "mujoco_trajectory"),
                       (render, "motion_sha256", "reference")]
    for record, key, source in required_hashes:
        require(record.get(key) == hashes[source + "_sha256"], f"{task}: {key} does not match actual {source}")
    require(isaac.get("policy_export") == contract, f"{task}: exported contract/report mismatch")
    require(render.get("status") == "rendered" and render.get("simulation_steps") == 0
            and render.get("right_panel_kind") == "recorded_policy" and render.get("full_decode_verified") is True,
            f"{task}: renderer did not verify a recorded-rollout comparison")
    require(records["status"].get("status") == "training_completed", f"{task}: training incomplete")
    require(all(key in manifest.get("source_sha256", {}) for key in (
        "training/tron1_tracking.py", "training/tron1_domain_randomization.py", "scripts/train_tracking.py")),
        f"{task}: missing training source hashes")
    identities = private_identities(records)
    public_assessment = sanitize(pick(assessment, ASSESSMENT_FIELDS), identities)
    budget = budget_summary(manifest, records["status"], records["runner"], records["audit"], contract, expected)
    source = dict(item["source"])
    catalog = read_json(project_root / "config/mocap_sources.json")
    require(source["manifest_key"] in catalog, f"{task}: unknown motion source")
    catalog_source = catalog[source["manifest_key"]]
    source["attribution"] = pick(catalog_source, ("source_page", "fps", "frame_count", "usage_terms", "acknowledgment", "terms_summary", "integrity_note"))
    source["files"] = [pick(record, ("url", "sha256", "bytes")) for record in catalog_source.get("files", [])]
    summary = {"task": task, "source": source, "raw_skill_intent": item["raw_skill_intent"],
               "raw_skill_intent_certified": False,
               "raw_skill_intent_note": "Only the explicitly listed behavior gates were assessed; source labels are not additional success claims.",
               "caveats": item["caveats"], "budget": budget,
               "behavior_pass": assessment["verdict"] == "pass", "verdict": assessment["verdict"],
               "failed_checks": assessment["failed_checks"], "common_window": assessment["common_window"],
               "engines": {engine: pick(assessment[engine], ("verdict", "failed_checks", "metrics", "complete_contact_evidence", "contact_force_source", "nonwheel_contact_evidence_available")) for engine in ("isaac", "mujoco")},
               "reference": reference_summary(files["reference"]),
               "reference_recipe": item.get("reference_recipe"), "reference_quality": item.get("reference_quality"),
               "hashes": hashes, "training_source_sha256": manifest["source_sha256"],
               "training": pick(manifest, ("method", "upstream_revision", "versions", "gpu", "smoke_steps_passed", "asset_sha256")),
               "final_domain_randomization_audit": pick(records["audit"], ("enabled", "mode", "profile", "seed", "joint_names", "nominal_environment_count", "actual_parameter_min_max", "dc_overspeed_cache_verified", "all_sampled_parameters_readback_verified", "parameters_hidden_from_actor", "net_effort_guard_semantics")),
               "render": pick(render, ("renderer", "method", "simulation_steps", "primary_label", "comparison_label", "video_frames", "video_fps", "playback_speed", "video_resolution", "full_decode_verified", "first_episode_only", "terminal_pose_included", "comparison_terminal_time_s", "comparison_terminal_pose_frozen_and_labeled")),
               "terrain": {"specification": assessment.get("terrain_comparison"),
                           "training_runtime_audit": manifest.get("terrain_runtime_audit"),
                           "isaac_runtime_audit": isaac.get("terrain_runtime_audit"),
                           "mujoco_runtime_audit": mujoco.get("terrain_runtime_audit")},
               "evaluation_source_sha256": {"isaac": pick(isaac, ("task_source_sha256", "evaluator_source_sha256", "terrain_module_sha256")), "mujoco": pick(mujoco, ("task_source_sha256", "evaluator_source_sha256", "domain_source_sha256", "terrain_module_sha256", "model_sha256", "model_hash_scope"))},
               "complete_contact_evidence": assessment["complete_contact_evidence"],
               "safety_verified": False, "hardware_ready": False,
               "artifacts": {"assessment": f"{task}/assessment.json", "video": f"{task}/tracking_comparison.mp4", "overview": f"{task}/overview.png"}}
    optional = trial / "terrain_implementation_provenance.json"
    if optional.is_file():
        summary["terrain"]["implementation_provenance"] = read_json(optional)
        summary["hashes"]["terrain_implementation_provenance_sha256"] = sha256(optional)
    if task == "turn_jump":
        summary["supplemental_yaw_diagnostics"] = {engine: yaw_diagnostics(files[key], label, assessment[engine])
            for engine, key, label in (("isaac", "isaac_trajectory", "Isaac"), ("mujoco", "mujoco_trajectory", "MuJoCo"))}
    return sanitize(summary, identities), public_assessment, files


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def publish_suite(suite_root, output_dir, config=DEFAULT_CONFIG, *, project_root=ROOT):
    suite_root, output_dir, project_root = Path(suite_root), Path(output_dir), Path(project_root)
    require(not output_dir.exists() or (output_dir.is_dir() and not any(output_dir.iterdir())),
            "Output must be absent or an empty directory; refusing overwrite")
    configuration = read_json(config)
    items = configuration.get("tasks", [])
    require(len(items) == len(TASKS) and {item.get("task") for item in items} == set(TASKS),
            "Configuration must contain each of the six tasks exactly once")
    prepared, errors = [], []
    # Complete all preflight attempts, rather than publishing a partial first task.
    for item in items:
        try:
            prepared.append(prepare_task(item, suite_root, configuration["budget"], project_root))
        except (KeyError, ValueError, OSError, TypeError) as error:
            errors.append(f"{item['task']}: {sanitize(str(error))}")
    require(not errors, "Suite preflight failed; no output written:\n" + "\n".join(errors))
    payloads = []
    for summary, assessment, files in prepared:
        payload = encode(assessment)
        summary["hashes"]["published_assessment_sha256"] = hashlib.sha256(payload).hexdigest()
        payloads.append((summary, payload, files))
    summary = {"schema_version": 1, "date": configuration.get("date"),
               "scope": "Six separately fine-tuned policies; one nominal frame-zero episode per engine per task",
               "config_sha256": sha256(config), "publisher_sha256": sha256(Path(__file__)),
               "expected_budget": sanitize(configuration["budget"]),
               "budget_pass_count": sum(item[0]["budget"]["pass"] for item in payloads),
               "behavior_pass_count": sum(item[0]["behavior_pass"] for item in payloads),
               "task_count": len(TASKS), "statistical_robustness_benchmark": False,
               "safety_verified": False, "hardware_ready": False,
               "limitations": sanitize(configuration.get("limitations", [])),
               "publication": "Source-hash-linked compact snapshot. No raw CMU motion, robot assets, checkpoints, actor weights, full trajectories or logs redistributed. Third-party licenses remain applicable.",
               "tasks": [item[0] for item in payloads]}
    summary_bytes = encode(summary)
    # No destination writes occur before every input and output payload is checked.
    output_dir.mkdir(parents=True, exist_ok=True)
    for item, assessment_bytes, files in payloads:
        task_dir = output_dir / item["task"]
        task_dir.mkdir()
        (task_dir / "assessment.json").write_bytes(assessment_bytes)
        for key, filename in (("video", "tracking_comparison.mp4"), ("overview", "overview.png")):
            destination = task_dir / filename
            shutil.copy2(files[key], destination)
            require(sha256(destination) == item["hashes"][key + "_sha256"], "Copied media hash mismatch")
    (output_dir / "summary.json").write_bytes(summary_bytes)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, default=ROOT / "outputs/motion-suite-20261003")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    try:
        result = publish_suite(args.suite_root, args.output_dir, args.config)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"{sanitize(str(error))}\n")
    print(json.dumps({"tasks": result["task_count"], "budget_pass_count": result["budget_pass_count"],
                      "behavior_pass_count": result["behavior_pass_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
