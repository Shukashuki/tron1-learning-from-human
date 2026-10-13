"""Publish019 with the source-frozen pre-rollout actual pair/torque audit.

No simulator, training, record projection or old publication context is used.
Failures remain FAIL. Preflight, CPU actor replay and complete video decode all
finish before creating a NEW public destination. Raw models/states stay private.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
PAIR_FILE = "outputs/drift-trial019-preparation-20261012/audit_deployment_provenance_trial019.py"
PAIR_SHA = "f20c192803e9a5c3daf4494ebd515ca0c11ddbd0221acb6c744f1df93aa6f4d4"
BASE_SHA = "598cd0cf6a78918da8a5e09ca0f20624c0a7379c7311813923366fd2c9b4293d"
RENDERER_SHA = "e73dcbb83ad65ee20c0ef3c923b2f90ea46c1f22f9dfa51ae4739e2516aa22e8"
ARTIFACTS = frozenset(("assessment.json", "reference_audit.json", "render_report.json", "summary.json",
                       "motion_pipeline.mp4", "overview.png", "final.png", "contact_overview.png"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dependency(path, digest, name):
    require(path.is_file() and path.resolve() == path and sha(path) == digest, "Changed source: " + name)
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def decode_video(video, report):
    """Read every H264 frame independently; a declared decode flag is insufficient."""
    import imageio_ffmpeg
    reader = imageio_ffmpeg.read_frames(str(video))
    try:
        metadata = next(reader)
        require(list(metadata["size"]) == report["video_resolution"] == [1920, 1080], "Actual1920x1080 video")
        require(metadata["codec"] == "h264" and abs(metadata["fps"] - report["video_fps"]) <= 1e-6, "Actual H264/fps")
        count = 0
        for frame in reader:
            require(len(frame) == 1920 * 1080 * 3, "Complete decoded RGB frame")
            count += 1
        require(count == report["video_frames"] == 145, "All145 rendered samples must decode")
        require(abs(metadata["duration"] - count / metadata["fps"]) <= .1, "Actual video duration")
        return {"frames": count, "resolution": [1920, 1080], "codec": metadata["codec"], "fps": metadata["fps"],
                "duration_seconds": metadata["duration"], "independent_complete_decode": True}
    finally:
        reader.close()


def prepare(trial, reference, render_dir, completion_sha, proof_sha):
    sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
    base = dependency(ROOT / "scripts/publish_car_drift.py", BASE_SHA, "public019_base_helpers")
    pair = dependency(ROOT / PAIR_FILE, PAIR_SHA, "public019_actual_pair")
    renderer = dependency(ROOT / "scripts/render_car_drift_pipeline.py", RENDERER_SHA, "public019_existing_renderer")
    proof_path = trial / "diagnostics/deployment_provenance_audit.json"
    require(sha(proof_path) == proof_sha, "Externally pinned actual paired proof")
    proof = pair.load(proof_path)
    require(proof["status"] == "deployment_provenance_pass" and proof["verification_script_sha256"] == PAIR_SHA
            and proof["completion_proof_sha256"] == completion_sha, "Actual019 producer/completion identity")
    # Replay the full reviewed proof here: a forged PASS JSON cannot publish.
    fresh_proof = pair.audit(trial, completion_sha)
    require(proof == fresh_proof, "Entire recorded-only paired proof must reproduce exactly")
    files = {
        "assessment": trial / "assessment.json", "manifest": trial / "training/manifest.json",
        "run_status": trial / "training/run_status.json", "runner": trial / "training/runner_config.json",
        "isaac_report": trial / "isaac/report.json", "mujoco_report": trial / "mujoco/report.json",
        "isaac_trajectory": trial / "isaac/trajectory.npz", "isaac_physics_trajectory": trial / "isaac/physics_trajectory.npz",
        "mujoco_trajectory": trial / "mujoco/rollout.npz", "actor": trial / "isaac/actor_normalized.pt",
        "motion": reference / "motion.npz", "robot_reference": reference / "robot_reference.npz",
        "adaptation": reference / "adaptation.json", "export_report": reference / "motion.json", "paired_proof": proof_path,
    }
    hashes = {k: sha(p) for k, p in files.items()}
    records = {k: pair.load(p) for k, p in files.items() if p.suffix == ".json"}
    assessment, adaptation = records["assessment"], records["adaptation"]
    require(assessment["verdict"] == proof["behavior_verdict"], "Preserve actual behavior verdict, including FAIL")
    render, media, render_path = base.validate_render(render_dir, hashes, adaptation)
    require(set(media) == {"motion_pipeline.mp4", "overview.png", "final.png", "contact_overview.png"}, "Complete four-panel media set")
    require(render["task_verdict"] == assessment["verdict"] and render["behavior_recorded_samples"] == {"isaac": 1153, "mujoco": 1153}
            and render["metrics_downsampled_for_rendering"] is False and render["simulation_steps"] == 0
            and render["source_hashes"]["renderer"] == RENDERER_SHA, "Rendered actual full-rate pair, not playback as physics")
    generic = pair.pinned("outputs/drift-sim2sim-20261011/trial010/diagnostics/audit_deployment_provenance_v2.py")
    # Renderer reports already redact private path values; compare the same
    # display representation. The full raw assessment is independently checked
    # by pair.audit above, with no changed numerical or behavioral tolerance.
    require(generic.json_agrees(render["assessment"], base.sanitize(assessment), tolerance=1e-12), "Identical original thresholds and metrics in movie report")
    decoded = decode_video(media["motion_pipeline.mp4"]["path"], render)
    identities = base.private_identities([*records.values(), render])
    payloads = {"assessment.json": encode(base.sanitize(assessment, identities)),
                "reference_audit.json": encode(base.sanitize(adaptation, identities)),
                "render_report.json": encode(base.sanitize(render, identities))}
    summary = {
        "schema_version": 1, "trial_id": "trial019", "task": "captured_car_drift", "verdict": assessment["verdict"],
        "publication_source_sha256": sha(__file__),
        "status": "assessed_" + assessment["verdict"], "failed_checks": assessment["failed_checks"],
        "scientific_change": "Only existing LAST directed wheel-axle reward angular std .4 to .2 rad; weight1 and original018 max-load objective, reference, controls, physics and gates held.",
        "scientific_commit": pair.COMMIT, "task_profile": records["manifest"]["task_profile"],
        "training": base.training_budget(records["manifest"], records["run_status"], records["runner"]),
        "provenance": {"status": proof["status"], "producer_source_sha256": PAIR_SHA, "proof_sha256": proof_sha,
            "completion_proof_sha256": completion_sha, "phase_seal_sha256": pair.PHASE_SHA,
            "actual_final_checkpoint_sha256": records["isaac_report"]["checkpoint_sha256"],
            "actual_actor_sha256": proof["actual_actor_sha256"], "actor_normalizer_tensor_count": len(proof["actor_tensor_proofs"]),
            "actor_normalizer_tensor_bytes_exact_final_checkpoint": all(p["byte_identical"] is True for p in proof["actor_tensor_proofs"]),
            "isaac_actor_replay": proof["isaac_actor_replay"], "mujoco_actor_replay": proof["mujoco_actor_replay"],
            "mujoco_controller_torque_replay": proof["mujoco_controller_torque_replay"],
            "verification_method": proof["verification_method"],
            "all_explicit_local_byte_copies_rehashed": proof["all_explicit_local_byte_copies_rehashed"],
            "full_canonical_assessment_replay_equal": proof["full_canonical_assessment_replay_equal"]},
        "engines": {k: assessment[k] for k in ("isaac", "mujoco")}, "common_window": assessment["common_window"],
        "behavior_evidence": assessment["behavior_evidence"], "complete_contact_evidence": assessment["complete_contact_evidence"],
        "resource": {"mode": "user_authorized_shared", "physical_isolation_claimed": False, "foreign_jobs_stopped": False},
        "video_decode": decoded, "input_sha256": hashes,
        "artifacts": {name.removesuffix(".json"): name for name in ARTIFACTS},
        "safety_verified": False, "hardware_ready": False, "statistical_robustness_benchmark": False,
        "limitations": assessment["limitations"] + ["Captured source is a simulated CARLA car with estimated timestamps; stopping/rest are explicitly synthetic.",
            "Runtime fixtures replay actual force arrays, but full-training force/cache values are not independently archived.",
            "This is one nominal first episode per engine, not a success probability or hardware result."],
    }
    summary["published_artifact_sha256"] = {n: hashlib.sha256(b).hexdigest() for n, b in payloads.items()}
    summary["published_artifact_sha256"].update({n: v["sha256"] for n, v in media.items()})
    payloads["summary.json"] = encode(base.sanitize(summary, identities))
    require(set(payloads) | set(media) == ARTIFACTS and len(payloads) == 4 and len(media) == 4, "Exactly eight artifacts")
    require(all(sha(p) == hashes[k] for k, p in files.items()) and sha(proof_path) == proof_sha, "Input bytes changed during publication")
    require(all(sha(v["path"]) == v["sha256"] for v in media.values()), "Media changed during publication")
    return payloads, media


def publish(trial, reference, render_dir, output, completion_sha, proof_sha):
    require(not output.exists(), "Public snapshot must be NEW; never overwrite failures or prior results")
    payloads, media = prepare(trial, reference, render_dir, completion_sha, proof_sha)
    output.mkdir(parents=True, exist_ok=False)
    for name, content in payloads.items():
        with (output / name).open("xb") as stream:
            stream.write(content)
    for name, item in media.items():
        with item["path"].open("rb") as source, (output / name).open("xb") as target:
            shutil.copyfileobj(source, target)
        require(sha(output / name) == item["sha256"], "Published media copy differs")
    require({p.name for p in output.iterdir()} == ARTIFACTS, "Only eight intended public artifacts")
    return json.loads(payloads["summary.json"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trial-dir", "reference-dir", "render-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--completion-sha256", required=True)
    parser.add_argument("--paired-proof-sha256", required=True)
    args = parser.parse_args()
    result = publish(args.trial_dir.resolve(), args.reference_dir.resolve(), args.render_dir.resolve(),
                     args.output_dir.resolve(), args.completion_sha256, args.paired_proof_sha256)
    print(json.dumps({"trial_id": "trial019", "verdict": result["verdict"], "artifacts": len(ARTIFACTS)}))


if __name__ == "__main__": main()
