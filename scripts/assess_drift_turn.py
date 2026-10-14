"""Assess a physical WF rollout for one uninterrupted high-speed sliding turn.

This adds a NEW-task gate; it never regrades historical DOA trials.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contiguous_windows(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def physical_episode(archive, engine):
    """Map actual engine measurements to one common first-episode schema."""
    if engine == "mujoco":
        return {key: archive[key] for key in ("time_s", "root_pos", "root_com_lin_vel_w",
                "root_quat_wxyz", "wheel_ground_contact_force_w_n",
                "wheel_ground_contact_count", "wheel_lateral_contact_slip_m_s")}
    if engine != "isaac":
        raise ValueError("Expected mujoco or isaac")
    valid = np.asarray(archive["valid_mask"])
    if valid.ndim != 2 or valid.shape[1] != 1 or valid.dtype.kind != "b":
        raise ValueError("Isaac assessment needs one first-episode validity column")
    selected = np.flatnonzero(valid[:, 0])
    if not len(selected) or np.any(np.diff(selected) != 1) or selected[0] != 0:
        raise ValueError("Isaac first episode must be contiguous from physical frame zero")
    names = {"root_pos": "root_position_m",
             "root_com_lin_vel_w": "root_linear_velocity_w_m_s",
             "root_quat_wxyz": "root_quaternion_wxyz",
             "wheel_ground_contact_force_w_n": "wheel_ground_contact_force_w_n",
             "wheel_ground_contact_count": "wheel_ground_contact_count",
             "wheel_lateral_contact_slip_m_s": "wheel_lateral_contact_slip_m_s"}
    return {"time_s": archive["time_s"][selected],
            **{target: archive[source][selected, 0] for target, source in names.items()}}


def completed_from_report(report, engine):
    if engine == "mujoco":
        value = report.get("completed_full_reference")
    elif engine == "isaac":
        episodes = report.get("summary", {}).get("episodes", [])
        value = episodes[0].get("completed_full_reference") if len(episodes) == 1 else None
    else:
        raise ValueError("Expected mujoco or isaac")
    if type(value) is not bool:
        raise ValueError("Missing exactly one physical episode completion verdict")
    return value


def assess(rollout, reference, *, min_speed=.8, min_beta_deg=8., min_patch_slip=.04,
           min_force=5., min_duration=.2, min_course_turn_deg=45.,
           max_path_rmse=.15, max_path_error=.3):
    """Every gate applies to the SAME native-clock contiguous physical samples."""
    time = np.asarray(rollout["time_s"], dtype=float)
    position = np.asarray(rollout["root_pos"], dtype=float)
    velocity = np.asarray(rollout["root_com_lin_vel_w"], dtype=float)[:, :2]
    quat = np.asarray(rollout["root_quat_wxyz"], dtype=float)
    force = np.asarray(rollout["wheel_ground_contact_force_w_n"], dtype=float)
    count = np.asarray(rollout["wheel_ground_contact_count"])
    patch_slip = np.asarray(rollout["wheel_lateral_contact_slip_m_s"], dtype=float)
    if (time.ndim != 1 or len(time) < 3 or np.any(np.diff(time) <= 0)
            or position.shape != (len(time), 3) or velocity.shape != (len(time), 2)
            or quat.shape != (len(time), 4)
            or force.shape != (len(time), 2, 3) or count.shape != (len(time), 2)
            or patch_slip.shape != (len(time), 2)
            or not all(np.isfinite(x).all() for x in (time, position, velocity, quat, force, patch_slip))):
        raise ValueError("Invalid native-clock physical rollout")
    rt = np.asarray(reference["time_s"], dtype=float)
    rows = np.asarray(reference["source_frame_numbers"], dtype=float)
    phase = np.asarray(reference["source_phase"])
    target_position = np.asarray(reference["root_target_position_m"], dtype=float)
    if not (rt.shape == rows.shape == phase.shape and target_position.shape == (len(rt), 3)
            and len(rt) >= 3 and np.all(np.diff(rt) > 0)
            and np.isfinite(target_position).all()):
        raise ValueError("Invalid robot reference source mapping")
    core = np.flatnonzero((rows >= 1553) & (rows <= 1584) & (phase == 0))
    if len(core) < 3 or np.any(np.diff(core) != 1):
        raise ValueError("Missing pinned drift_drl turn core")
    in_core = (time >= rt[core[0]] - 1e-8) & (time <= rt[core[-1]] + 1e-8)
    speed = np.linalg.norm(velocity, axis=1)
    course = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    rotation = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix()
    yaw = np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
    beta = np.abs((course - yaw + np.pi) % (2 * np.pi) - np.pi) * (180. / np.pi)
    supported = ((count > 0) & (np.linalg.norm(force, axis=2) >= min_force)).all(axis=1)
    sliding = np.mean(np.abs(patch_slip), axis=1) >= min_patch_slip
    qualifying = in_core & (speed >= min_speed) & (beta >= min_beta_deg) & supported & sliding
    windows = []
    for start, end in contiguous_windows(qualifying):
        windows.append({"start_s": float(time[start]), "end_s": float(time[end]),
                        "duration_s": float(time[end] - time[start]),
                        "abs_course_turn_deg": float(abs(np.rad2deg(course[end] - course[start]))),
                        "minimum_speed_m_s": float(speed[start:end + 1].min())})
    passed = any(item["duration_s"] >= min_duration
                 and item["abs_course_turn_deg"] >= min_course_turn_deg for item in windows)
    indices = np.flatnonzero(in_core)
    if not len(indices):
        raise ValueError("No physical rollout samples overlap the source turn core")
    expected_xy = np.c_[np.interp(time[indices], rt, target_position[:, 0]),
                        np.interp(time[indices], rt, target_position[:, 1])]
    path_error = np.linalg.norm(position[indices, :2] - expected_xy, axis=1)
    path_rmse = float(np.sqrt(np.mean(path_error**2)))
    path_max = float(path_error.max())
    passed = passed and path_rmse <= max_path_rmse and path_max <= max_path_error
    return {
        "status": "PASS" if passed else "FAIL", "all_criteria_same_uninterrupted_window": True,
        "criteria": {"minimum_speed_m_s": min_speed, "minimum_abs_body_sideslip_deg": min_beta_deg,
                     "minimum_mean_abs_patch_slip_m_s": min_patch_slip,
                     "minimum_each_wheel_normal_force_n": min_force,
                     "minimum_uninterrupted_duration_s": min_duration,
                     "minimum_abs_travel_course_turn_deg": min_course_turn_deg,
                     "maximum_core_path_rmse_m": max_path_rmse,
                     "maximum_core_path_error_m": max_path_error},
        "reference_core_time_s": [float(rt[core[0]]), float(rt[core[-1]])],
        "physical_core_samples": len(indices),
        "whole_core_course_turn_deg_not_a_pass_gate": float(np.rad2deg(course[indices[-1]] - course[indices[0]])),
        "whole_core_min_speed_m_s": float(speed[indices].min()),
        "core_path_rmse_m": path_rmse, "core_max_path_error_m": path_max,
        "core_fraction_speed_ok": float(np.mean(speed[indices] >= min_speed)),
        "core_fraction_sideslip_ok": float(np.mean(beta[indices] >= min_beta_deg)),
        "core_fraction_two_wheel_support_ok": float(np.mean(supported[indices])),
        "core_fraction_patch_slip_ok": float(np.mean(sliding[indices])),
        "qualifying_windows": windows,
        "longest_qualifying_duration_s": max((item["duration_s"] for item in windows), default=0.),
        "largest_qualifying_course_turn_deg": max((item["abs_course_turn_deg"] for item in windows), default=0.),
        "measurement": "actual native-clock root XY velocity, root yaw, and wheel-ground contact/patch slip; no frame-gap bridging",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--robot-reference", type=Path, required=True)
    parser.add_argument("--engine", choices=("mujoco", "isaac"), default="mujoco")
    parser.add_argument("--engine-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Assessment output already exists")
    with np.load(args.rollout, allow_pickle=False) as archive:
        rollout = physical_episode(archive, args.engine)
    with np.load(args.robot_reference, allow_pickle=False) as archive:
        reference = {key: archive[key] for key in archive.files}
    report = json.loads(args.engine_report.read_text())
    completed = completed_from_report(report, args.engine)
    result = assess(rollout, reference)
    result["engine"] = args.engine
    result["full_reference_completed"] = completed
    result["turn_window_status"] = result["status"]
    result["status"] = "PASS" if result["status"] == "PASS" and completed else "FAIL"
    result["rollout_sha256"] = sha256(args.rollout)
    result["robot_reference_sha256"] = sha256(args.robot_reference)
    result["engine_report_sha256"] = sha256(args.engine_report)
    result["assessor_sha256"] = sha256(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
