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
    task_profile = validate_task_profile(manifest.get("task_profile"))
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
