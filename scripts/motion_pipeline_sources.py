"""Audited source clocks for the wheel-adapted four-panel visualizations."""
import json
from pathlib import Path
import numpy as np
from render_tracking import ROOT, PoseClip, load_rollout, read_npz, sha256
from compare_tracking import validate_provenance
from publish_motion_suite import sanitize


def adapted_source_times(raw, adaptation, reference, robot):
    """Compose export clock -> adapted robot time -> original human timestamp."""
    detail = adaptation["calibration_and_adaptation"]
    scale = float(detail["time_scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Adaptation time scale must be finite and positive")
    frames = raw["frame_numbers"]
    for name, times in (("human", raw["time_s"]), ("robot", robot["time_s"]), ("export", reference["time_s"])):
        if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0)):
            raise ValueError(f"Invalid {name} timestamps")
    if (frames.shape != raw["time_s"].shape or not np.isfinite(frames).all()
            or np.any(np.diff(frames) != 1) or not np.isfinite(float(raw["fps"])) or float(raw["fps"]) <= 0):
        raise ValueError("Invalid source frame numbers or FPS")
    if (reference["source_time_s"].shape != reference["time_s"].shape
            or not np.isfinite(reference["source_time_s"]).all()
            or np.any(np.diff(reference["source_time_s"]) < 0)
            or robot["time_s"][0] != 0 or reference["time_s"][0] != 0):
        raise ValueError("Invalid adapted export clock")
    first, last = detail["source_first_frame_inclusive"], detail["source_last_frame_inclusive"]
    ids = np.flatnonzero((frames >= first) & (frames <= last))
    if len(ids) < 3 or frames[ids[0]] != first or frames[ids[-1]] != last:
        raise ValueError("Adaptation source frame range is missing")
    np.testing.assert_allclose(robot["source_frame_numbers"], first + robot["time_s"] / scale * float(raw["fps"]), atol=1e-8)
    if reference["source_time_s"][-1] > robot["time_s"][-1] + 1e-8:
        raise ValueError("Export clock exceeds adapted robot reference")
    np.testing.assert_allclose(reference["source_time_s"], np.minimum(reference["time_s"], robot["time_s"][-1]), atol=1e-8)
    mapped = raw["time_s"][ids[0]] + reference["source_time_s"] / scale
    if mapped[-1] > raw["time_s"][ids[-1]] + 1e-8:
        raise ValueError("Mapped clock exceeds selected human source")
    return mapped, ids


def with_outcome(prepared, task):
    path = Path(task) / "trial/assessment.json"
    assessment = json.loads(path.read_text())
    if assessment["task"] != Path(task).name or assessment["verdict"] not in ("pass", "fail"):
        raise ValueError("Assessment task/result mismatch")
    for key in ("reference_sha256", "actor_sha256", "isaac_trajectory_sha256", "mujoco_trajectory_sha256"):
        if assessment["provenance"][key] != prepared["provenance"][key]:
            raise ValueError("Assessment does not describe displayed recordings")
    prepared["reports"]["assessment"] = assessment
    prepared["hashes"].update(assessment=sha256(path), source_adapter=sha256(Path(__file__)))
    prepared["task"] = Path(task).name
    prepared.setdefault("adapted", False)
    prepared.setdefault("hold_start_task_s", float(prepared["source_times"][-1]))
    prepared.setdefault("adaptation_note", "Human: one heading alignment + uniform scale. Gold points: pelvis / ankles (not wheel targets).")
    return prepared


def prepare_adapted(task, model_path):
    from render_motion_pipeline import KeypointClip, source_clock
    task = Path(task)
    adaptation = json.loads((task / "adaptation.json").read_text())
    paths = {"human": ROOT / adaptation["source_human_npz"], "robot": task / "robot_reference.npz",
             "reference": task / "motion.npz", "isaac": task / "trial/isaac/trajectory.npz",
             "mujoco": task / "trial/mujoco/rollout.npz"}
    report_paths = {"export": task / "motion.json", "adaptation": task / "adaptation.json",
                    "isaac": task / "trial/isaac/report.json", "mujoco": task / "trial/mujoco/report.json"}
    reports = {k: json.loads(p.read_text()) for k, p in report_paths.items()}
    checks = [("human", adaptation, "source_human_sha256"), ("robot", adaptation, "robot_reference_sha256"),
              ("reference", adaptation, "motion_sha256"), ("robot", reports["export"], "source_sha256"),
              ("reference", reports["export"], "output_sha256")]
    for key, record, field in checks:
        if sha256(paths[key]) != record.get(field):
            raise ValueError(f"Adaptation hash mismatch: {key}")
    if sha256(model_path) != reports["export"]["model_sha256"] or sha256(model_path) != adaptation["model_sha256"]:
        raise ValueError("Adapted reference model mismatch")
    if reports["export"]["upstream_task_adaptation"] != adaptation["calibration_and_adaptation"]:
        raise ValueError("Conflicting adaptation metadata")
    provenance = validate_provenance(report_paths["isaac"].parent, report_paths["mujoco"].parent,
                                     reports["isaac"], reports["mujoco"])
    if provenance["reference_sha256"] != sha256(paths["reference"]):
        raise ValueError("Recorded policy/reference mismatch")
    if reports["isaac"].get("terrain") is not None or reports["mujoco"].get("terrain") is not None:
        raise ValueError("Wheel-adaptation renderer expects original flat-floor tasks")
    arrays = {k: read_npz(p) for k, p in paths.items()}
    raw, ref = arrays["human"], arrays["reference"]
    mapped, ids = adapted_source_times(raw, adaptation, ref, arrays["robot"])
    source_clock(ref["time_s"], mapped, 0.)
    human = KeypointClip(raw["time_s"][ids], raw["joint_positions_world_m"][ids], raw["joint_names"], raw["parent_indices"])
    q = ref["qpos_mujoco"]
    reference = PoseClip(ref["time_s"], q[:, :3], q[:, 3:7], q[:, 7:], reports["export"]["source_joint_names"])
    np.testing.assert_allclose(reference.root, ref["body_pos_w"][:, 0], atol=1e-6)
    detail = adaptation["calibration_and_adaptation"]
    note = (f"Human time stretched x{detail['time_scale']:g}; travel x{detail['translation_scale']:g}, height x{detail['height_scale']:g}; footsteps removed."
            if adaptation["skill"] == "rolling_stop" else
            f"Human frames {detail['source_first_frame_inclusive']}-{detail['source_last_frame_inclusive']}; time stretched x{detail['time_scale']:g}; stationary {detail['depth_m']:.2f} m crouch.")
    metadata = json.loads(raw["metadata_json"].item())
    return with_outcome({"human": human, "reference": reference,
        "isaac": load_rollout(arrays["isaac"]), "mujoco": load_rollout(arrays["mujoco"]),
        "task_times": ref["time_s"], "source_times": mapped, "reports": reports,
        "source_clip": Path(metadata["source"]["amc_path"]).stem, "scale": 1.,
        "human_display_floor_m": 0., "human_heading_alignment": np.eye(3).tolist(),
        "human_seed_root_m": human.positions[0, human.names.index("root")].tolist(),
        "adapted": True, "adaptation_note": note, "hold_start_task_s": float(ref["source_time_s"][-1]),
        "provenance": sanitize(provenance),
        "hashes": {**{k: sha256(p) for k, p in paths.items()}, **{k + "_report": sha256(p) for k, p in report_paths.items()},
                   "model": sha256(model_path), "renderer_script": sha256(ROOT / "scripts/render_motion_pipeline.py")}}, task)
