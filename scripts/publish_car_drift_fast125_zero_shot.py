"""Publish the fixed-actor 1.25x drift-reference diagnostic; no new training.

All physics, source identity, actor tensors, original gates and complete video
are checked before creating a new eight-artifact public directory. The raw
rollouts and checkpoint remain under ignored outputs/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from assess_drift import assess_runs
from publish_car_drift import validate_render
from publish_motion_suite import private_identities, sanitize, sha256

REFERENCE = ROOT / "outputs/drift-reference-fast125-20261013"
TRIAL = ROOT / "outputs/drift-fast125-zero-shot-20261013"
BASE = ROOT / "outputs/drift-sim2sim-20261013/trial022"
ARTIFACTS = frozenset(("assessment.json", "reference_audit.json", "render_report.json", "summary.json",
                       "motion_pipeline.mp4", "overview.png", "final.png", "contact_overview.png"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def encode(value, identities):
    data = (json.dumps(sanitize(value, identities), sort_keys=True, indent=2,
                       ensure_ascii=False, allow_nan=False) + "\n").encode()
    require(all(secret not in data for secret in (b"/home/roy", b"/data/zengxiong", b".ssh", b"id_ed25519")),
            "Private paths must not enter publication")
    return data


def decode_video(path, report):
    import imageio_ffmpeg
    reader = imageio_ffmpeg.read_frames(str(path))
    try:
        meta = next(reader)
        require(list(meta["size"]) == report["video_resolution"] == [1920, 1080]
                and meta["codec"] == "h264" and abs(meta["fps"] - report["video_fps"]) <= 1e-6,
                "Original four-panel H264 geometry/fps")
        count = 0
        for frame in reader:
            require(len(frame) == 1920 * 1080 * 3, "Complete RGB frame")
            count += 1
        require(count == report["video_frames"] == 116, "All 116 video frames independently decode")
        return {"frames": count, "resolution": [1920, 1080], "codec": "h264",
                "fps": meta["fps"], "independent_complete_decode": True}
    finally:
        reader.close()


def preflight():
    import torch
    files = {
        "motion": REFERENCE / "motion.npz", "robot_reference": REFERENCE / "robot_reference.npz",
        "adaptation": REFERENCE / "adaptation.json", "export_report": REFERENCE / "motion.json",
        "assessment": TRIAL / "assessment.json", "isaac_report": TRIAL / "isaac/report.json",
        "mujoco_report": TRIAL / "mujoco_paired/report.json",
        "isaac_trajectory": TRIAL / "isaac/trajectory.npz",
        "isaac_physics_trajectory": TRIAL / "isaac/physics_trajectory.npz",
        "mujoco_trajectory": TRIAL / "mujoco_paired/rollout.npz",
        "actor": TRIAL / "isaac/actor_normalized.pt",
        "old_actor": BASE / "isaac/actor_normalized.pt",
        "checkpoint": BASE / "training/model_final.pt",
        "baseline_assessment": BASE / "assessment.json",
        "captured_source": ROOT / "outputs/drift-source-screening-20261011/sa_traj34/captured_source.npz",
    }
    require(all(p.is_file() and p.stat().st_size for p in files.values()), "Complete private evidence required")
    hashes = {key: sha256(path) for key, path in files.items()}
    require(hashes["motion"] == "1baea367b5282396dc7da4cc579145c9c11c3e9fadcfbe5b3cce5e38a1b6642e"
            and hashes["checkpoint"] == "a1b95d85c462e2be314155b895e5089501d021519285d55b7f42afe9152e5231"
            and hashes["old_actor"] == "c4677c07ccffc45bb5f98adab06ddedc97d04dd9e3b14e68f3c132deb01df629"
            and hashes["actor"] == "d93c8d7e0790ddb16832eb2dee2e27b8d26e042fa460153bbf50bf084cf0e4e5",
            "Pinned reference, actual checkpoint and two exports")
    adaptation = read(files["adaptation"])
    original = read(files["baseline_assessment"])
    assessment = read(files["assessment"])
    fresh = assess_runs(TRIAL / "isaac", TRIAL / "mujoco_paired", files["motion"])
    require(assessment == fresh and assessment["verdict"] == "fail"
            and assessment["failed_checks"] == ["isaac.stable_exit_sideslip",
                "mujoco.sustained_grounded_contact_patch_drift", "mujoco.stable_exit_sideslip",
                "mujoco.stable_exit_planar_speed"], "Unchanged original assessor and FAIL verdict")
    require(all(item.get("passed") is True for item in assessment["provenance_checks"].values())
            and all(item["kind"] == "physics_rate" and item["sample_dt_s"] == .005
                    for item in assessment["behavior_evidence"].values()),
            "Same actor/reference/clock provenance and actual 200 Hz contacts")
    isaac, mujoco = read(files["isaac_report"]), read(files["mujoco_report"])
    require(isaac["checkpoint_sha256"] == hashes["checkpoint"]
            and isaac["motion_file_sha256"] == hashes["motion"]
            and isaac["physics_recording"]["samples"] == 921
            and mujoco["policy_sha256"] == hashes["actor"]
            and mujoco["motion_sha256"] == hashes["motion"]
            and mujoco["isaac_observation_validation"]["status"] == "passed",
            "Actual checkpoint, both rollouts and Isaac-observation replay")
    old = torch.jit.load(str(files["old_actor"]), map_location="cpu").eval()
    new = torch.jit.load(str(files["actor"]), map_location="cpu").eval()
    a, b = old.state_dict(), new.state_dict()
    require(len(a) == 12 and list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a),
            "Reexport must contain exactly the same 12 actor and normalizer tensors")
    with np.load(files["isaac_trajectory"], allow_pickle=False) as archive:
        observations = torch.from_numpy(archive["action_observation"][:, 0].astype(np.float32))
    with torch.inference_mode():
        require(len(observations) == 231 and torch.equal(old(observations), new(observations)),
                "Both exports must produce exactly the same fast-reference actor outputs")
    with np.load(files["motion"], allow_pickle=False) as fast, np.load(
            ROOT / "outputs/drift-reference-20261011/motion.npz", allow_pickle=False) as baseline:
        require(len(fast["joint_pos"]) == 230 and len(baseline["joint_pos"]) == 288
                and float(np.asarray(fast["fps"]).reshape(-1)[0]) == 50., "Original and faster task clocks")
        peak = {"baseline_m_s": float(np.linalg.norm(baseline["body_lin_vel_w"][:, 0, :2], axis=1).max()),
                "fast125_m_s": float(np.linalg.norm(fast["body_lin_vel_w"][:, 0, :2], axis=1).max())}
    require(adaptation["motion_sha256"] == hashes["motion"]
            and adaptation["provenance"]["source_npz_sha256"] == hashes["captured_source"]
            and adaptation["calibration_and_adaptation"]["time_scale"] == 1.6
            and adaptation["calibration_and_adaptation"]["recovery_duration_s"] == 1.2
            and adaptation["calibration_and_adaptation"]["stationary_tail_s"] == .4,
            "Same captured source, 1.25x whole-task timing")
    render, media, _ = validate_render(TRIAL / "render", hashes, adaptation)
    require(set(media) == ARTIFACTS - {"assessment.json", "reference_audit.json", "render_report.json", "summary.json"}
            and render["task_verdict"] == "fail" and render["simulation_steps"] == 0,
            "Full four-panel FAIL render")
    decoded = decode_video(media["motion_pipeline.mp4"]["path"], render)
    return files, hashes, adaptation, original, assessment, render, media, peak, decoded


def publish(output):
    require(output.resolve().is_relative_to(ROOT / "results") and not output.exists(),
            "New public result under results/ only; never overwrite")
    files, hashes, adaptation, original, assessment, render, media, peak, decoded = preflight()
    identities = private_identities([adaptation, original, assessment, render,
                                     read(files["isaac_report"]), read(files["mujoco_report"])])
    metrics = lambda report: {engine: {name: report[engine]["metrics"][name] for name in (
        "maximum_contiguous_drift_s", "drift_total_observed_duration_s",
        "full_two_wheel_support_fraction", "tail_max_planar_com_speed_m_s",
        "tail_max_abs_sideslip_deg", "final_root_error_m")} for engine in ("isaac", "mujoco")}
    summary = {
        "schema_version": 1, "kind": "fixed_trial022_actor_fast125_zero_shot_diagnostic",
        "training_on_faster_reference_performed": False, "same_trained_actor_tensors": True,
        "actor_and_normalizer_tensors_bit_identical": 12, "actor_outputs_equal_on_fast_observations": 231,
        "task_speed_factor": 1.25, "reference_time_scale_old_new": [2.0, 1.6],
        "recovery_s_old_new": [1.5, 1.2], "stationary_tail_s_old_new": [0.5, 0.4],
        "reference_duration_s_old_new": [5.76, 4.6], "reference_peak_planar_speed_m_s": peak,
        "original_behavior_gates_and_physics_clocks_unchanged": True,
        "baseline_verdict": original["verdict"], "fast125_verdict": assessment["verdict"],
        "baseline_failed_checks": original["failed_checks"], "fast125_failed_checks": assessment["failed_checks"],
        "baseline_metrics": metrics(original), "fast125_metrics": metrics(assessment),
        "cross_engine_root_rmse_m_old_new": [original["common_window"]["base_position_rmse_m"],
                                           assessment["common_window"]["base_position_rmse_m"]],
        "input_sha256": hashes, "video_decode": decoded,
        "resource": {"mode": "user_authorized_shared_GPU7", "foreign_jobs_stopped": False,
                     "physical_isolation_claimed": False},
        "artifacts": sorted(ARTIFACTS), "safety_verified": False, "hardware_ready": False,
        "limitations": ["Zero-shot speed stress test of the completed trial022 checkpoint; no 1.25x PPO training or new-policy transfer claim.",
                        "The source clock is estimated, the car source is simulated, and the robot stop is synthetic.",
                        "50 Hz resampling omits a small subframe endpoint; XY path length changes slightly, so timing is the intended but not mathematically sole numerical difference.",
                        "One deterministic episode per engine is not a success probability or hardware validation."],
    }
    payloads = {"assessment.json": encode(assessment, identities),
                "reference_audit.json": encode(adaptation, identities),
                "render_report.json": encode(render, identities),
                "summary.json": encode(summary, identities)}
    require(all(sha256(path) == hashes[key] for key, path in files.items())
            and all(sha256(item["path"]) == item["sha256"] for item in media.values()),
            "All inputs and media unchanged during publication")
    output.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        with (output / name).open("xb") as stream:
            stream.write(data)
    for name, item in media.items():
        with item["path"].open("rb") as source, (output / name).open("xb") as target:
            shutil.copyfileobj(source, target)
        require(sha256(output / name) == item["sha256"], "Published media copy differs")
    require({p.name for p in output.iterdir()} == ARTIFACTS, "Exactly eight public artifacts")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = publish(args.output_dir)
    print(json.dumps({"kind": report["kind"], "verdict": report["fast125_verdict"],
                      "artifacts": len(ARTIFACTS)}))


if __name__ == "__main__":
    main()
