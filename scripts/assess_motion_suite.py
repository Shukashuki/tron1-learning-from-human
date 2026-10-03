"""Offline, preregistered six-motion acceptance; never steps either simulator.

Behavior gates and same-actor provenance are separate from common-window
engine-to-engine RMSE. A behavioral pass with incomplete contact channels is
NOT safety verification, statistical robustness, or hardware readiness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_tracking import (common_window_rmse, load_episode, read_json,
                              read_npz, sha256, validate_provenance)
from assess_sim2sim import (actuator_contract_comparison, check_provenance,
                           legacy_friction_audit_valid)
from training.tron1_terrain import validate_terrain

TASKS = ("forward_jump", "turn_jump", "side_jump", "rolling_stop", "crouch", "step_up")
THRESHOLDS = {
    "final_root_error_m": .15, "final_height_error_m": .06,
    "final_heading_error_deg": 15., "tail_seconds": .5,
    "tail_max_tilt_deg": 15., "wheel_contact_threshold_n": 5.,
    "tail_min_two_wheel_support_fraction": .9, "tail_mean_planar_speed_m_s": .2,
    "root_rmse_m": .15, "jump_min_flight_s": .08, "jump_min_rise_m": .08,
    "support_before_after_flight_s": .1, "minimum_progress_fraction": .7,
    "nontrivial_translation_m": .03, "nontrivial_yaw_deg": 5.,
    "nontrivial_crouch_depth_m": .03, "nontrivial_step_height_m": .03,
    "nontrivial_rolling_speed_m_s": .03, "wheel_radius_m": .127,
    "ledge_height_tolerance_m": .04, "nonwheel_impact_threshold_n": 1.,
}


def _check(checks, name, passed, detail=None):
    checks[name] = {"passed": bool(passed)}
    if detail is not None:
        checks[name]["detail"] = detail


def _series(arrays, key, engine):
    value = np.asarray(arrays[key])
    if engine == "Isaac":
        mask = np.asarray(arrays["valid_mask"])
        if value.shape[:2] != mask.shape:
            raise ValueError(f"{key} must have time/environment leading dimensions")
        return value[mask[:, 0], 0]
    return value


def yaw(quat):
    w, x, y, z = np.asarray(quat).T
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def duration_fraction(times, values, seconds):
    """Exact duration integral of zero-order-held sample values over the tail."""
    times, values = np.asarray(times), np.asarray(values)
    if values.shape != times.shape or seconds <= 0 or times[-1] - times[0] < seconds - 1e-9:
        return None
    widths = np.maximum(0., times[1:] - np.maximum(times[:-1], times[-1] - seconds))
    return float(np.dot(widths, values[:-1]) / seconds)


def _runs(times, mask):
    mask = np.asarray(mask, bool)
    bounds = np.diff(np.r_[False, mask, False].astype(int))
    return [(float(times[a]), float(times[b] if b < len(times) else times[-1]))
            for a, b in zip(np.flatnonzero(bounds == 1), np.flatnonzero(bounds == -1))]


def flight_intervals(times, force, thresholds=THRESHOLDS):
    """Time-based support/flight runs, including irregular telemetry sampling."""
    contact = np.linalg.norm(force, axis=-1) >= thresholds["wheel_contact_threshold_n"]
    support = [(a, b) for a, b in _runs(times, np.all(contact, axis=1))
               if b - a + 1e-9 >= thresholds["support_before_after_flight_s"]]
    flights = []
    for a, b in _runs(times, ~np.any(contact, axis=1)):
        landing = next((c for c, _ in support if c >= b - 1e-9), None)
        if (b - a + 1e-9 >= thresholds["jump_min_flight_s"]
                and any(d <= a + 1e-9 for _, d in support) and landing is not None):
            flights.append({"takeoff_s": a, "landing_s": landing, "no_contact_duration_s": b - a})
    return flights


def reference_episode(reference):
    fps = float(np.asarray(reference["fps"]).reshape(-1)[0])
    if not np.isclose(fps, 50.):
        raise ValueError("Suite references must explicitly use 50 Hz")
    names = np.asarray(reference["joint_names"]).astype(str).tolist()
    bodies = np.asarray(reference["body_names"]).astype(str).tolist()
    if not bodies or bodies[0] != "base_Link":
        raise ValueError("Reference body zero must be base_Link link frame, not COM")
    root = np.asarray(reference["body_pos_w"], float)[:, 0]
    quat = np.asarray(reference["body_quat_w"], float)[:, 0]
    joints = np.asarray(reference["joint_pos"], float)
    n = len(root)
    if n < 2 or root.shape != (n, 3) or quat.shape != (n, 4) or joints.shape != (n, 8):
        raise ValueError("Malformed reference state shapes")
    if len(names) != 8 or len(set(names)) != 8 or not all(np.isfinite(v).all() for v in (root, quat, joints)):
        raise ValueError("Reference requires finite states and eight named joints")
    if not np.allclose(np.linalg.norm(quat, axis=1), 1., atol=1e-3):
        raise ValueError("Reference quaternions must be normalized")
    if "qpos_mujoco" in reference:
        np.testing.assert_allclose(np.asarray(reference["qpos_mujoco"])[:, :3], root, atol=1e-6)
    # Simulation executes N policy actions: retain the final pose from (N-1)/fps to N/fps.
    return {"time": np.arange(n + 1) / fps, "root": np.vstack((root, root[-1])),
            "quat": np.vstack((quat, quat[-1])), "joints": np.vstack((joints, joints[-1])),
            "names": names, "frames": n, "duration": n / fps}


def terrain_comparison(exported, deployed, task):
    first, second = exported.get("terrain"), deployed.get("terrain")
    try:
        normalized = [validate_terrain(v) if v is not None else None for v in (first, second)]
        matched = normalized[0] == normalized[1]
        if task == "step_up":
            matched = matched and normalized[0] is not None and bool(normalized[0]["boxes"])
        return {"matched": bool(matched), "isaac": normalized[0], "mujoco": normalized[1]}
    except (ValueError, TypeError, KeyError) as exc:
        return {"matched": False, "error": str(exc)}


def assess_episode(data, reference, task, *, engine="MuJoCo", report=None, contract=None):
    """Pure NumPy assessment of env0's recorded first episode; no simulator imports."""
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}")
    report = {} if report is None else report
    contract = contract if contract is not None else report.get(
        "policy_export" if engine == "Isaac" else "contract", {})
    ep, ref = load_episode(data, engine), reference_episode(reference)
    t, p, quat = ep["time"], ep["root"], ep["quat"]
    if engine == "Isaac":
        outcomes = [v for v in report.get("summary", {}).get("episodes", []) if v.get("environment") == 0]
        outcome = outcomes[0] if len(outcomes) == 1 else {}
    else:
        outcome = report
    th, checks = THRESHOLDS, {}
    ids = _series(data, "reference_frame", engine)
    if ids.shape != t.shape or ids.dtype.kind not in "iu":
        raise ValueError("Reference frame IDs must be aligned integer samples")
    _check(checks, "frame_zero_start", abs(t[0]) <= 1e-9 and ids[0] == 0
           and outcome.get("start_reference_frame") == 0)
    _check(checks, "monotonic_reference_without_reset", np.all(np.diff(ids) >= 0)
           and np.all(np.diff(ids) <= 1) and np.all((ids >= 0) & (ids < ref["frames"])))
    _check(checks, "completed_full_reference", outcome.get("completed_full_reference") is True
           and ids[-1] == ref["frames"] - 1 and outcome.get("final_reference_frame") == ref["frames"] - 1
           and report.get("reference_frames") == ref["frames"]
           and abs(t[-1] - ref["duration"]) <= 1e-6)
    terms = outcome.get("termination_terms")
    _check(checks, "no_early_termination", isinstance(terms, list)
           and not any(term != "motion_end" for term in terms)
           and outcome.get("early_terminated", False) is False
           and (outcome.get("end_reason") == "timeout" if engine == "Isaac"
                else outcome.get("termination") == "motion_end"))
    safe = (contract.get("root_prescribed_during_steps") is False
            and contract.get("motion_end_hidden_teleport") is False)
    if engine == "Isaac":
        safe = (safe and report.get("reference_state_initialization_during_evaluation") is False
                and report.get("terminal_states_captured_before_auto_reset") is True)
    else:
        safe = (safe and report.get("physics_stepped") is True
                and report.get("root_state_writes") == 1 and report.get("hidden_resets") == 0)
    _check(checks, "physics_only_no_hidden_reset_or_root_teleport", safe)
    final_delta = p[-1] - ref["root"][-1]
    _check(checks, "final_root_xyz", np.linalg.norm(final_delta) <= th["final_root_error_m"] + 1e-9)
    _check(checks, "final_root_height", abs(final_delta[2]) <= th["final_height_error_m"] + 1e-9)
    actual_yaw, ref_yaw = yaw(quat), yaw(ref["quat"])
    heading_error = abs(np.degrees(np.arctan2(np.sin(actual_yaw[-1] - ref_yaw[-1]),
                                              np.cos(actual_yaw[-1] - ref_yaw[-1]))))
    _check(checks, "final_heading", heading_error <= th["final_heading_error_deg"] + 1e-9)
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * np.sum(quat[:, 1:3] ** 2, axis=1), -1., 1.)))
    bracket = max(0, np.searchsorted(t, t[-1] - th["tail_seconds"], side="right") - 1)
    max_tilt = float(tilt[bracket:].max())
    _check(checks, "stable_tail_tilt", max_tilt <= th["tail_max_tilt_deg"] + 1e-9)
    speeds = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1) / np.diff(t)
    tail_speed = duration_fraction(t, np.r_[speeds, speeds[-1]], th["tail_seconds"])
    _check(checks, "tail_stopped", tail_speed is not None and tail_speed <= th["tail_mean_planar_speed_m_s"] + 1e-9)
    comparison = common_window_rmse(ep, ref, ref["names"][:6])
    root_rmse = comparison["base_position_rmse_m"]
    _check(checks, "full_motion_root_rmse", root_rmse <= th["root_rmse_m"] + 1e-9)
    force_key = ("wheel_ground_contact_force_w_n" if "wheel_ground_contact_force_w_n" in data
                 else "wheel_contact_force_w_n" if engine == "Isaac" and "wheel_contact_force_w_n" in data else None)
    forces = _series(data, force_key, engine) if force_key else np.empty((0, 2, 3))
    force_valid = forces.shape == (len(t), 2, 3) and np.isfinite(forces).all()
    _check(checks, "wheel_contact_evidence_available", force_valid)
    support_fraction, flights = None, []
    if force_valid:
        support = np.all(np.linalg.norm(forces, axis=-1) >= th["wheel_contact_threshold_n"], axis=1)
        support_fraction = duration_fraction(t, support, th["tail_seconds"])
        flights = flight_intervals(t, forces)
    _check(checks, "tail_two_wheel_support", support_fraction is not None
           and support_fraction + 1e-9 >= th["tail_min_two_wheel_support_fraction"])
    nonwheel_available, impact = False, None
    for key in ("nonwheel_ground_contact_count", "nonwheel_ground_contact_force_norm_n"):
        if key in data:
            values = np.asarray(_series(data, key, engine), float)
            nonwheel_available = bool(values.shape == t.shape and np.isfinite(values).all() and np.all(values >= 0))
            if nonwheel_available:
                # Current count telemetry counts individual contacts with force >=1 N.
                impact = bool(np.any(values > 0)) if key.endswith("count") else bool(
                    np.any(values >= th["nonwheel_impact_threshold_n"]))
            break
    if engine == "MuJoCo":
        _check(checks, "no_nonwheel_ground_impact", nonwheel_available and impact is False)
    elif nonwheel_available:
        _check(checks, "no_nonwheel_ground_impact", impact is False)
    gain = float(p[:, 2].max() - p[0, 2])
    metrics = {"final_root_error_m": float(np.linalg.norm(final_delta)),
               "final_height_error_m": float(abs(final_delta[2])), "final_heading_error_deg": float(heading_error),
               "root_rmse_m": root_rmse, "root_rmse_window_s": [comparison["start_s"], comparison["end_s"]],
               "tail_max_tilt_deg": max_tilt, "tail_mean_planar_speed_m_s": tail_speed,
               "tail_two_wheel_support_fraction": support_fraction, "root_rise_m": gain,
               "max_planar_speed_m_s": float(speeds.max()), "flight_intervals": flights}
    if task.endswith("_jump"):
        _check(checks, "sustained_flight_and_landing", bool(flights))
        _check(checks, "minimum_actual_rise", gain + 1e-9 >= th["jump_min_rise_m"])
    if task in ("forward_jump", "side_jump", "rolling_stop"):
        ref_net = ref["root"][-1, :2] - ref["root"][0, :2]
        net = p[-1, :2] - p[0, :2]
        magnitude = np.linalg.norm(ref_net)
        _check(checks, "nontrivial_reference_translation", magnitude > th["nontrivial_translation_m"])
        progress = float(np.dot(net, ref_net) / magnitude**2) if magnitude > 1e-12 else None
        _check(checks, "intended_net_displacement_progress", progress is not None
               and progress + 1e-9 >= th["minimum_progress_fraction"])
        metrics["net_displacement_progress_fraction"] = progress
        if task == "side_jump":
            excursions = ref["root"][:, :2] - ref["root"][0, :2]
            vector = excursions[np.linalg.norm(excursions, axis=1).argmax()]
            length = np.linalg.norm(vector)
            excursion_progress = float(np.max((p[:, :2] - p[0, :2]) @ vector) / length**2) if length > 1e-12 else None
            _check(checks, "nontrivial_reference_side_excursion", length > th["nontrivial_translation_m"])
            _check(checks, "intended_side_excursion_progress", excursion_progress is not None
                   and excursion_progress + 1e-9 >= th["minimum_progress_fraction"])
            metrics["side_excursion_progress_fraction"] = excursion_progress
    if task == "turn_jump":
        expected = ref_yaw[-1] - ref_yaw[0]
        progress = float((actual_yaw[-1] - actual_yaw[0]) / expected) if abs(expected) > 1e-12 else None
        _check(checks, "nontrivial_reference_yaw", abs(np.degrees(expected)) > th["nontrivial_yaw_deg"])
        _check(checks, "intended_yaw_progress", progress is not None and progress + 1e-9 >= th["minimum_progress_fraction"])
        metrics["yaw_progress_fraction"] = progress
    if task == "rolling_stop":
        reference_speed = float((np.linalg.norm(np.diff(ref["root"][:, :2], axis=0), axis=1) / np.diff(ref["time"])).max())
        _check(checks, "nontrivial_reference_rolling_speed", reference_speed > th["nontrivial_rolling_speed_m_s"])
        _check(checks, "rolling_speed_progress", speeds.max() + 1e-9 >= th["minimum_progress_fraction"] * reference_speed)
        metrics["reference_peak_planar_speed_m_s"] = reference_speed
    if task == "crouch":
        depth = float(p[0, 2] - p[:, 2].min())
        expected = float(ref["root"][0, 2] - ref["root"][:, 2].min())
        _check(checks, "nontrivial_reference_crouch", expected > th["nontrivial_crouch_depth_m"])
        _check(checks, "crouch_depth_progress", depth + 1e-9 >= th["minimum_progress_fraction"] * expected)
        _check(checks, "crouch_recovered", abs(p[-1, 2] - p[0, 2]) <= th["final_height_error_m"] + 1e-9)
        metrics.update(crouch_depth_m=depth, reference_crouch_depth_m=expected)
    if task == "step_up":
        try:
            terrain = validate_terrain(contract["terrain"])
            bodies = list(map(str, reference["body_names"]))
            ref_wheels = np.asarray(reference["body_pos_w"])[-1, [bodies.index("wheel_L_Link"), bodies.index("wheel_R_Link")]]
            selected = []
            for box in terrain["boxes"]:
                center, size = np.asarray(box["center"]), np.asarray(box["size"])
                top = center[2] + size[2] / 2
                if (np.all(np.abs(ref_wheels[:, :2] - center[:2]) <= size[:2] / 2 + 1e-9)
                        and np.all(np.abs(ref_wheels[:, 2] - th["wheel_radius_m"] - top) <= th["ledge_height_tolerance_m"])):
                    selected.append(box)
            _check(checks, "reference_identifies_single_target_ledge", len(selected) == 1)
            key = "wheel_position_m" if engine == "Isaac" else "wheel_pos"
            wheels = np.asarray(_series(data, key, engine), float)
            valid_wheels = wheels.shape == (len(t), 2, 3) and np.isfinite(wheels).all()
            _check(checks, "actual_wheel_center_evidence", valid_wheels)
            on_top = False
            if len(selected) == 1 and valid_wheels:
                box = selected[0]; center, size = np.asarray(box["center"]), np.asarray(box["size"])
                top = center[2] + size[2] / 2
                on_top = (np.all(np.abs(wheels[-1, :, :2] - center[:2]) <= size[:2] / 2 + 1e-9)
                          and np.all(np.abs(wheels[-1, :, 2] - th["wheel_radius_m"] - top) <= th["ledge_height_tolerance_m"]))
                metrics["target_ledge"] = box
            _check(checks, "both_final_wheels_on_target_ledge", on_top)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            _check(checks, "valid_step_terrain_and_wheel_evidence", False, str(exc))
        expected = float(ref["root"][-1, 2] - ref["root"][0, 2])
        _check(checks, "nontrivial_reference_step_height", expected > th["nontrivial_step_height_m"])
        _check(checks, "step_height_progress", p[-1, 2] - p[0, 2] + 1e-9 >= th["minimum_progress_fraction"] * expected)
    failed = [name for name, check in checks.items() if not check["passed"]]
    complete = bool(force_key == "wheel_ground_contact_force_w_n" and nonwheel_available)
    return {"verdict": "fail" if failed else "pass", "failed_checks": failed, "checks": checks,
            "metrics": metrics, "complete_contact_evidence": complete,
            "contact_force_source": force_key, "nonwheel_contact_evidence_available": nonwheel_available,
            "safety_verified": False, "hardware_ready": False}


def assess_runs(task, motion_file, isaac_dir, mujoco_dir):
    motion_file, isaac_dir, mujoco_dir = map(Path, (motion_file, isaac_dir, mujoco_dir))
    isaac, mujoco = read_json(isaac_dir / "report.json"), read_json(mujoco_dir / "report.json")
    reference = read_npz(motion_file)
    provenance_checks, provenance = {}, {}
    try:
        provenance = validate_provenance(isaac_dir, mujoco_dir, isaac, mujoco)
        strict = check_provenance(isaac_dir, mujoco_dir, isaac, mujoco)
        provenance_checks.update(strict["checks"])
        _check(provenance_checks, "supplied_motion_hash", sha256(motion_file) == provenance["reference_sha256"])
        provenance["strict_checks"] = strict
    except (ValueError, KeyError, TypeError, OSError) as exc:
        _check(provenance_checks, "valid_same_actor_provenance", False, str(exc))
    exported, deployed = isaac.get("policy_export", {}), mujoco.get("contract", {})
    actuator = actuator_contract_comparison(exported, deployed)
    _check(provenance_checks, "matching_actuator_contract", actuator["matched"], actuator["mismatched_fields"])
    if actuator["dc_motor_involved"]:
        _check(provenance_checks, "legacy_friction_readback_zero", legacy_friction_audit_valid(isaac, exported))
    terrain = terrain_comparison(exported, deployed, task)
    _check(provenance_checks, "matching_terrain_contract", terrain["matched"])
    if task == "step_up":
        for engine, report in (("isaac", isaac), ("mujoco", mujoco)):
            audit = report.get("terrain_runtime_audit")
            _check(provenance_checks, f"{engine}_terrain_runtime_audit_passed",
                   isinstance(audit, dict) and audit.get("status") == "passed")
    arrays = {"isaac": read_npz(isaac_dir / "trajectory.npz"), "mujoco": read_npz(mujoco_dir / "rollout.npz")}
    results = {}
    for key, engine, report in (("isaac", "Isaac", isaac), ("mujoco", "MuJoCo", mujoco)):
        try:
            results[key] = assess_episode(arrays[key], reference, task, engine=engine, report=report)
        except (ValueError, TypeError, KeyError, IndexError, AssertionError) as exc:
            results[key] = {"verdict": "fail", "failed_checks": ["invalid_episode_evidence"],
                            "error": str(exc), "complete_contact_evidence": False, "safety_verified": False}
    try:
        common = common_window_rmse(load_episode(arrays["isaac"], "Isaac"),
                                    load_episode(arrays["mujoco"], "MuJoCo"), exported["leg_joint_names"])
    except (ValueError, TypeError, KeyError) as exc:
        common = {"error": str(exc)}
        _check(provenance_checks, "valid_common_window", False, str(exc))
    failed = ["provenance." + k for k, v in provenance_checks.items() if not v["passed"]]
    failed += [engine + "." + name for engine, result in results.items() for name in result["failed_checks"]]
    return {"schema_version": 1, "task": task, "verdict": "fail" if failed else "pass",
            "failed_checks": failed, "thresholds": dict(THRESHOLDS), "threshold_policy": "Fixed before suite policy outcomes; no task-specific relaxation from observed results.",
            "provenance": provenance, "provenance_checks": provenance_checks, "terrain_comparison": terrain,
            "actuator_contract": actuator, "common_window": common, **results,
            "complete_contact_evidence": all(r["complete_contact_evidence"] for r in results.values()),
            "safety_verified": False, "hardware_ready": False, "statistical_robustness_benchmark": False,
            "limitations": ["One frame-zero episode per engine; deterministic duplicate environments are not independent trials.",
                           "Root positions use base-link origin, not COM. Speeds use base-link finite differences.",
                           "Contact support and flight use time integrals of held force samples, bounded by telemetry rate.",
                           "Isaac net wheel forces may include self-contact; missing non-wheel channels never establish safety.",
                           "Common-window sim2sim RMSE is reported separately; no phase/space alignment or extrapolation.",
                           "A flight/height/progress gate measures actual rollout, never reference playback or RSI."]}


def json_native(value):
    """Normalize NumPy containers/scalars without masking nonfinite evidence."""
    if isinstance(value, dict):
        return {key: json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_native(value.tolist())
    if isinstance(value, np.generic):
        return json_native(value.item())
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--isaac-dir", type=Path, required=True)
    parser.add_argument("--mujoco-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = assess_runs(args.task, args.motion_file, args.isaac_dir, args.mujoco_dir)
    # Validate the ENTIRE payload before creating a file: serialization failure
    # must not leave a partial JSON that appears to be an assessment artifact.
    result = json_native(result)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(payload)
    print(json.dumps({"verdict": result["verdict"], "failed_checks": result["failed_checks"]}))


if __name__ == "__main__":
    main()
