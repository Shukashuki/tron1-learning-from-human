"""Fixed, offline acceptance of a same-policy TRON1 drift in Isaac and MuJoCo.

Required new contact telemetry: ``wheel_ground_contact_count`` and
``wheel_lateral_contact_slip_m_s``. Both are [time,2] for MuJoCo and
[time,environment,2] for Isaac, in left/right wheel order. Slip is measured
at actual wheel-ground contact patches relative to the ground; base velocity,
wheel-center velocity and reference poses cannot substitute for that channel.
COM velocity uses ``root_com_lin_vel_w`` (MuJoCo) or
``root_linear_velocity_w_m_s`` (Isaac), in world coordinates. Samples are
assessed on the recorded simulation clock without phase or spatial alignment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_tracking import (common_window_rmse, load_episode, read_json,
                              read_npz, sha256, validate_provenance)

THRESHOLDS = {
    "drift_min_planar_speed_m_s": .4,
    "drift_min_abs_sideslip_deg": 8.,
    "drift_min_mean_abs_contact_slip_m_s": .04,
    "drift_min_contiguous_duration_s": .20,
    "sideslip_defined_min_speed_m_s": .1,
    "initial_max_abs_sideslip_deg": 5.,
    "initial_com_velocity_error_m_s": .0001,
    "tail_seconds": .5,
    "tail_max_abs_sideslip_deg": 5.,
    "tail_max_planar_speed_m_s": .2,
    "tail_max_tilt_deg": 15.,
    "tail_min_two_wheel_support_fraction": .9,
    "full_min_two_wheel_support_fraction": .9,
    "initial_contact_settle_seconds": .1,
    "max_both_wheel_flight_s": .04,
    "root_rmse_m": .15,
    "final_root_error_m": .15,
    "final_height_error_m": .06,
    "final_heading_error_deg": 15.,
    "minimum_reference_progress_fraction": .7,
    "minimum_reference_translation_m": .03,
    "minimum_reference_yaw_deg": 5.,
    "nonwheel_impact_threshold_n": 1.,
}


def _check(checks, name, passed, detail=None):
    checks[name] = {"passed": bool(passed)}
    if detail is not None:
        checks[name]["detail"] = detail


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def yaw(quaternions):
    w, x, y, z = np.asarray(quaternions, float).T
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def sideslip(quaternions, world_com_velocity):
    """Heading-plane sideslip, independent of roll/pitch and world orientation.

    Return speed and signed beta in radians. Callers must mask near-rest samples:
    atan2(0,0) is numerically zero but does not define a physical slip angle.
    """
    velocity = np.asarray(world_com_velocity, float)
    quat = np.asarray(quaternions, float)
    if velocity.shape != (len(quat), 3) or quat.shape != (len(quat), 4):
        raise ValueError("Sideslip needs aligned [time,4] quaternions and [time,3] COM velocity")
    if not np.isfinite(velocity).all() or not np.isfinite(quat).all():
        raise ValueError("Sideslip evidence must be finite")
    if not np.allclose(np.linalg.norm(quat, axis=1), 1., atol=1e-3):
        raise ValueError("Sideslip quaternions must be normalized")
    heading = yaw(quat)
    forward = np.cos(heading) * velocity[:, 0] + np.sin(heading) * velocity[:, 1]
    lateral = -np.sin(heading) * velocity[:, 0] + np.cos(heading) * velocity[:, 1]
    return np.linalg.norm(velocity[:, :2], axis=1), np.arctan2(lateral, forward)


def intervals(times, mask, *, start=None):
    """Consecutive true zero-order-held intervals, clipped to observed time.

    Sample i represents [time_i,time_(i+1)); the terminal sample adds no time.
    An isolated last sample therefore cannot manufacture a sustained maneuver.
    """
    times, mask = np.asarray(times, float), np.asarray(mask)
    if (times.ndim != 1 or len(times) < 2 or mask.shape != times.shape
            or mask.dtype.kind != "b" or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Intervals require strictly increasing times and an aligned boolean mask")
    changes = np.diff(np.r_[False, mask, False].astype(int))
    result = []
    for first, last in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        left = max(float(times[first]), float(start)) if start is not None else float(times[first])
        right = float(times[last] if last < len(times) else times[-1])
        if right > left:
            result.append({"start_s": left, "end_s": right, "duration_s": right - left})
    return result


def duration_mean(times, values, *, start, end):
    """Exact duration mean of held samples; no extrapolation outside telemetry."""
    times, values = np.asarray(times, float), np.asarray(values, float)
    if (values.shape != times.shape or not np.isfinite(values).all()
            or end <= start or start < times[0] - 1e-9 or end > times[-1] + 1e-9):
        return None
    widths = np.maximum(0., np.minimum(times[1:], end) - np.maximum(times[:-1], start))
    return float(widths @ values[:-1] / (end - start))


def _series(arrays, key, engine):
    values = np.asarray(arrays[key])
    if engine == "Isaac":
        valid = np.asarray(arrays["valid_mask"])
        if values.ndim < 2 or values.shape[:2] != valid.shape:
            raise ValueError(f"{key} needs time/environment leading dimensions")
        return values[valid[:, 0], 0]
    return values


def reference_episode(reference):
    fps = float(np.asarray(reference["fps"]).reshape(-1)[0])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Reference fps must be finite and positive")
    names = np.asarray(reference["joint_names"]).astype(str).tolist()
    bodies = np.asarray(reference["body_names"]).astype(str).tolist()
    if not bodies or bodies[0] != "base_Link":
        raise ValueError("Reference root must be the base_Link origin")
    root, quat = np.asarray(reference["body_pos_w"], float)[:, 0], np.asarray(reference["body_quat_w"], float)[:, 0]
    velocity, joints = np.asarray(reference["body_lin_vel_w"], float)[:, 0], np.asarray(reference["joint_pos"], float)
    n = len(root)
    if (n < 2 or root.shape != (n, 3) or quat.shape != (n, 4)
            or velocity.shape != (n, 3) or joints.shape != (n, 8)
            or len(names) != 8 or len(set(names)) != 8
            or not all(np.isfinite(v).all() for v in (root, quat, velocity, joints))):
        raise ValueError("Reference needs finite poses, COM velocities and eight named joints")
    if not np.allclose(np.linalg.norm(quat, axis=1), 1., atol=1e-3):
        raise ValueError("Reference quaternions must be normalized")
    # N actions at 1/fps: hold final reference pose for the final policy interval.
    result = {"time": np.arange(n + 1) / fps, "names": names, "frames": n, "duration": n / fps}
    result.update({key: np.concatenate((value, value[-1:]), axis=0)
                   for key, value in (("root", root), ("quat", quat), ("joints", joints), ("velocity", velocity))})
    return result


def assess_episode(arrays, reference, *, engine="MuJoCo", report=None):
    """Assess recorded env0 behavior, including contact-patch evidence, offline."""
    report = {} if report is None else report
    episode, ref = load_episode(arrays, engine), reference_episode(reference)
    times, root, quat = episode["time"], episode["root"], episode["quat"]
    th, checks = THRESHOLDS, {}
    velocity_key = "root_linear_velocity_w_m_s" if engine == "Isaac" else "root_com_lin_vel_w"
    velocity = np.asarray(_series(arrays, velocity_key, engine), float)
    speed, beta = sideslip(quat, velocity)
    beta_deg = np.degrees(beta)
    ref_speed, ref_beta = sideslip(ref["quat"], ref["velocity"])
    ref_drift = ((ref_speed >= th["drift_min_planar_speed_m_s"])
                 & (np.abs(np.degrees(ref_beta)) >= th["drift_min_abs_sideslip_deg"]))
    ref_runs = intervals(ref["time"], ref_drift)
    _check(checks, "nontrivial_captured_reference_drift", any(
        r["duration_s"] + 1e-9 >= th["drift_min_contiguous_duration_s"] for r in ref_runs))
    _check(checks, "initial_reference_sideslip_not_injected", ref_speed[0] < th["sideslip_defined_min_speed_m_s"]
           or abs(np.degrees(ref_beta[0])) <= th["initial_max_abs_sideslip_deg"] + 1e-9)
    _check(checks, "initial_actual_sideslip_not_injected", speed[0] < th["sideslip_defined_min_speed_m_s"]
           or abs(beta_deg[0]) <= th["initial_max_abs_sideslip_deg"] + 1e-9)
    initial_velocity_error = float(np.linalg.norm(velocity[0] - ref["velocity"][0]))
    _check(checks, "initial_com_velocity_matches_reference", initial_velocity_error
           <= th["initial_com_velocity_error_m_s"] + 1e-9)
    if engine == "Isaac":
        outcomes = [e for e in report.get("summary", {}).get("episodes", []) if e.get("environment") == 0]
        outcome = outcomes[0] if len(outcomes) == 1 else {}
        contract = report.get("policy_export", {})
    else:
        outcome, contract = report, report.get("contract", {})
    ids = np.asarray(_series(arrays, "reference_frame", engine))
    if ids.shape != times.shape or ids.dtype.kind not in "iu":
        raise ValueError("Reference frame IDs must be aligned integer samples")
    _check(checks, "frame_zero_start", abs(times[0]) <= 1e-9 and ids[0] == 0
           and outcome.get("start_reference_frame") == 0)
    _check(checks, "monotonic_reference_without_reset", np.all(np.diff(ids) >= 0)
           and np.all(np.diff(ids) <= 1) and np.all((ids >= 0) & (ids < ref["frames"])))
    _check(checks, "completed_full_reference", outcome.get("completed_full_reference") is True
           and ids[-1] == ref["frames"] - 1 and outcome.get("final_reference_frame") == ref["frames"] - 1
           and report.get("reference_frames") == ref["frames"]
           and abs(times[-1] - ref["duration"]) <= 1e-6)
    terms = outcome.get("termination_terms")
    _check(checks, "no_early_termination", isinstance(terms, list)
           and not any(term != "motion_end" for term in terms)
           and outcome.get("early_terminated", False) is False
           and (outcome.get("end_reason") == "timeout" if engine == "Isaac"
                else outcome.get("termination") == "motion_end"))
    safe = (contract.get("root_prescribed_during_steps") is False
            and contract.get("motion_end_hidden_teleport") is False)
    if engine == "Isaac":
        safe = safe and report.get("reference_state_initialization_during_evaluation") is False and report.get(
            "terminal_states_captured_before_auto_reset") is True
    else:
        safe = safe and report.get("physics_stepped") is True and report.get("root_state_writes") == 1 and report.get("hidden_resets") == 0
    _check(checks, "physics_only_no_hidden_reset_or_root_teleport", safe)

    contact_valid = slip_valid = False
    support = np.zeros(len(times), bool)
    any_supported = np.zeros(len(times), bool)
    abs_slip = np.zeros((len(times), 2))
    for key, shape in (("wheel_ground_contact_count", (len(times), 2)),
                       ("wheel_lateral_contact_slip_m_s", (len(times), 2))):
        try:
            value = np.asarray(_series(arrays, key, engine))
            valid = value.shape == shape and value.dtype.kind in "iuf" and np.isfinite(value).all()
            if key.endswith("count"):
                contact_valid = bool(valid and np.all(value >= 0) and np.all(value == np.floor(value)))
                if contact_valid:
                    support, any_supported = np.all(value > 0, axis=1), np.any(value > 0, axis=1)
            else:
                slip_valid = bool(valid)
                if slip_valid:
                    abs_slip = np.abs(value)
        except (ValueError, TypeError, KeyError):
            pass
    _check(checks, "wheel_ground_contact_evidence_available", contact_valid)
    _check(checks, "contact_patch_lateral_slip_evidence_available", slip_valid)
    mean_slip = abs_slip.mean(axis=1)
    moving_drift = ((speed >= th["drift_min_planar_speed_m_s"])
                    & (np.abs(beta_deg) >= th["drift_min_abs_sideslip_deg"])
                    & (np.cos(beta) > 0.))
    drifting = (moving_drift & support & (mean_slip >= th["drift_min_mean_abs_contact_slip_m_s"]))
    drift_runs = intervals(times, drifting) if contact_valid and slip_valid else []
    accepted_runs = [r for r in drift_runs if r["duration_s"] + 1e-9 >= th["drift_min_contiguous_duration_s"]]
    _check(checks, "sustained_grounded_contact_patch_drift", bool(accepted_runs))
    support_fraction = duration_mean(times, support, start=times[0], end=times[-1]) if contact_valid else None
    _check(checks, "full_episode_two_wheel_support", support_fraction is not None
           and support_fraction + 1e-9 >= th["full_min_two_wheel_support_fraction"])
    flight_runs = intervals(times, ~any_supported, start=times[0] + th["initial_contact_settle_seconds"]) if contact_valid else []
    max_flight = max((r["duration_s"] for r in flight_runs), default=0.) if contact_valid else None
    _check(checks, "no_sustained_both_wheel_flight", max_flight is not None
           and max_flight <= th["max_both_wheel_flight_s"] + 1e-9)

    tail_start = times[-1] - th["tail_seconds"]
    enough_tail = tail_start >= times[0] - 1e-9
    # Include the sample whose held interval straddles tail_start and the terminal state.
    tail_first = max(0, int(np.searchsorted(times, tail_start, side="right")) - 1)
    tail_speed = float(speed[tail_first:].max()) if enough_tail else None
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * np.sum(quat[:, 1:3] ** 2, axis=1), -1., 1.)))
    tail_tilt = float(tilt[tail_first:].max()) if enough_tail else None
    tail_moving = speed[tail_first:] >= th["sideslip_defined_min_speed_m_s"]
    tail_beta = float(np.max(np.abs(beta_deg[tail_first:][tail_moving]))) if enough_tail and tail_moving.any() else (0. if enough_tail else None)
    tail_support = duration_mean(times, support, start=tail_start, end=times[-1]) if contact_valid else None
    _check(checks, "stable_exit_sideslip", tail_beta is not None and tail_beta <= th["tail_max_abs_sideslip_deg"] + 1e-9)
    _check(checks, "stable_exit_planar_speed", tail_speed is not None and tail_speed <= th["tail_max_planar_speed_m_s"] + 1e-9)
    _check(checks, "stable_exit_tilt", tail_tilt is not None and tail_tilt <= th["tail_max_tilt_deg"] + 1e-9)
    _check(checks, "stable_exit_two_wheel_support", tail_support is not None
           and tail_support + 1e-9 >= th["tail_min_two_wheel_support_fraction"])
    nonwheel_available = False
    nonwheel_checks = []
    for key in ("nonwheel_ground_contact_count", "nonwheel_ground_contact_force_norm_n"):
        if key in arrays:
            try:
                value = np.asarray(_series(arrays, key, engine), float)
                valid_nonwheel = bool(value.shape == times.shape and np.isfinite(value).all() and np.all(value >= 0))
                impact = bool(np.any(value > 0)) if key.endswith("count") else bool(np.any(value >= th["nonwheel_impact_threshold_n"]))
                nonwheel_checks.append((valid_nonwheel, not impact))
            except (ValueError, TypeError, KeyError):
                nonwheel_checks.append((False, False))
    if nonwheel_checks:
        nonwheel_available = all(valid for valid, _ in nonwheel_checks)
        _check(checks, "no_nonwheel_ground_impact", nonwheel_available and all(no_impact for _, no_impact in nonwheel_checks))

    leg_names = [name for name in ref["names"] if not name.startswith("wheel_")]
    if len(leg_names) != 6:
        raise ValueError("Reference needs exactly six named leg joints")
    comparison = common_window_rmse(episode, ref, leg_names)
    final_error = root[-1] - ref["root"][-1]
    actual_yaw, ref_yaw = yaw(quat), yaw(ref["quat"])
    final_yaw_error = float(abs(np.degrees(wrap_angle(actual_yaw[-1] - ref_yaw[-1]))))
    _check(checks, "root_motion_rmse", comparison["base_position_rmse_m"] <= th["root_rmse_m"] + 1e-9)
    _check(checks, "final_root_position", np.linalg.norm(final_error) <= th["final_root_error_m"] + 1e-9)
    _check(checks, "final_root_height", abs(final_error[2]) <= th["final_height_error_m"] + 1e-9)
    _check(checks, "final_heading", final_yaw_error <= th["final_heading_error_deg"] + 1e-9)
    expected_yaw = ref_yaw[-1] - ref_yaw[0]
    yaw_progress = float((actual_yaw[-1] - actual_yaw[0]) / expected_yaw) if abs(expected_yaw) > 1e-12 else None
    _check(checks, "nontrivial_reference_turn", abs(np.degrees(expected_yaw)) >= th["minimum_reference_yaw_deg"])
    _check(checks, "captured_reference_yaw_progress", yaw_progress is not None
           and yaw_progress + 1e-9 >= th["minimum_reference_progress_fraction"])
    expected_translation = ref["root"][-1, :2] - ref["root"][0, :2]
    length = np.linalg.norm(expected_translation)
    translation_progress = float((root[-1, :2] - root[0, :2]) @ expected_translation / length**2) if length > 1e-12 else None
    _check(checks, "nontrivial_reference_translation", length >= th["minimum_reference_translation_m"])
    _check(checks, "captured_reference_translation_progress", translation_progress is not None
           and translation_progress + 1e-9 >= th["minimum_reference_progress_fraction"])
    moving = speed >= th["sideslip_defined_min_speed_m_s"]
    drift_widths = np.diff(times) * drifting[:-1]
    drift_observed_duration = float(drift_widths.sum())
    drift_mean_slip = float(drift_widths @ mean_slip[:-1] / drift_observed_duration) if drift_observed_duration else None
    failed = [name for name, value in checks.items() if not value["passed"]]
    return {"verdict": "fail" if failed else "pass", "checks": checks, "failed_checks": failed,
            "metrics": {"drift_intervals": drift_runs, "accepted_drift_intervals": accepted_runs,
                        "maximum_contiguous_drift_s": max((r["duration_s"] for r in drift_runs), default=0.),
                        "reference_drift_intervals": ref_runs,
                        "max_planar_com_speed_m_s": float(speed.max()),
                        "max_moving_abs_sideslip_deg": float(np.max(np.abs(beta_deg[moving]))) if moving.any() else None,
                        "initial_planar_com_speed_m_s": float(speed[0]),
                        "initial_abs_sideslip_deg": float(abs(beta_deg[0])) if moving[0] else None,
                        "initial_com_velocity_error_m_s": initial_velocity_error,
                        "full_two_wheel_support_fraction": support_fraction,
                        "both_wheel_flight_intervals_after_settle": flight_runs,
                        "max_both_wheel_flight_s": max_flight,
                        "tail_max_abs_sideslip_deg": tail_beta, "tail_sideslip_moving_samples": int(tail_moving.sum()),
                        "tail_max_planar_com_speed_m_s": tail_speed, "tail_max_tilt_deg": tail_tilt,
                        "tail_two_wheel_support_fraction": tail_support,
                        "drift_mean_abs_contact_slip_m_s": drift_mean_slip,
                        "drift_total_observed_duration_s": drift_observed_duration,
                        "full_mean_abs_contact_slip_m_s": duration_mean(times, mean_slip,
                            start=times[0], end=times[-1]) if slip_valid else None,
                        "peak_abs_contact_slip_m_s_by_wheel": abs_slip.max(axis=0).tolist() if slip_valid else None,
                        "actual_yaw_change_deg": float(np.degrees(actual_yaw[-1] - actual_yaw[0])),
                        "reference_yaw_change_deg": float(np.degrees(expected_yaw)),
                        "yaw_progress_fraction": yaw_progress, "translation_progress_fraction": translation_progress,
                        "final_heading_error_deg": final_yaw_error, "final_root_error_m": float(np.linalg.norm(final_error)),
                        "reference_tracking": comparison},
            "contact_patch_slip_evidence_available": slip_valid, "wheel_ground_contact_evidence_available": contact_valid,
            "nonwheel_contact_evidence_available": nonwheel_available,
            "complete_contact_evidence": contact_valid and slip_valid and nonwheel_available,
            "safety_verified": False, "hardware_ready": False}


def assess_runs(isaac_dir, mujoco_dir, reference_file):
    isaac_dir, mujoco_dir, reference_file = map(Path, (isaac_dir, mujoco_dir, reference_file))
    isaac, mujoco = read_json(isaac_dir / "report.json"), read_json(mujoco_dir / "report.json")
    reference = read_npz(reference_file)
    checks, provenance = {}, {}
    _check(checks, "isaac_engine", isaac.get("simulator") == "IsaacLab/PhysX")
    _check(checks, "mujoco_engine", mujoco.get("engine") == "MuJoCo")
    try:
        provenance = validate_provenance(isaac_dir, mujoco_dir, isaac, mujoco)
        _check(checks, "same_actor_reference_and_observation_contract", True)
        _check(checks, "supplied_reference_hash", sha256(reference_file) == provenance["reference_sha256"])
    except (ValueError, KeyError, TypeError, OSError) as exc:
        _check(checks, "same_actor_reference_and_observation_contract", False, str(exc))
    exported, deployed = isaac.get("policy_export", {}), mujoco.get("contract", {})
    wheels = exported.get("wheel_joint_names")
    _check(checks, "matching_wheel_action_order", isinstance(wheels, list) and len(wheels) == 2
           and len(set(wheels)) == 2 and wheels == deployed.get("wheel_joint_names"))
    fields = ("leg_kp", "leg_kd", "leg_position_scale", "wheel_torque_scale_nm", "leg_torque_limit_nm",
              "wheel_torque_limit_nm", "physics_dt", "decimation", "policy_fps", "raw_action_clip", "actuator_model")
    mismatches = [key for key in fields if key not in exported or key not in deployed or exported[key] != deployed[key]]
    _check(checks, "matching_actuator_and_clock_contract", not mismatches, mismatches)
    arrays = {"isaac": read_npz(isaac_dir / "trajectory.npz"), "mujoco": read_npz(mujoco_dir / "rollout.npz")}
    results = {}
    for key, engine, report in (("isaac", "Isaac", isaac), ("mujoco", "MuJoCo", mujoco)):
        try:
            results[key] = assess_episode(arrays[key], reference, engine=engine, report=report)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            results[key] = {"verdict": "fail", "failed_checks": ["valid_recorded_episode_evidence"],
                            "error": str(exc), "complete_contact_evidence": False}
    try:
        common = common_window_rmse(load_episode(arrays["isaac"], "Isaac"),
                                    load_episode(arrays["mujoco"], "MuJoCo"), exported["leg_joint_names"])
    except (ValueError, TypeError, KeyError) as exc:
        common = {"error": str(exc)}
        _check(checks, "valid_common_window", False, str(exc))
    failed = ["provenance." + name for name, value in checks.items() if not value["passed"]]
    failed += [key + "." + name for key, result in results.items() for name in result["failed_checks"]]
    return {"schema_version": 1, "task": "captured_car_drift", "verdict": "fail" if failed else "pass",
            "failed_checks": failed, "thresholds": dict(THRESHOLDS),
            "threshold_policy": "Fixed before policy drift evaluation outcomes; no relaxation based on results.",
            "provenance": provenance, "provenance_checks": checks, "common_window": common, **results,
            "assessor_source_sha256": sha256(__file__), "reference_sha256": sha256(reference_file),
            "complete_contact_evidence": all(v["complete_contact_evidence"] for v in results.values()),
            "safety_verified": False, "hardware_ready": False, "statistical_robustness_benchmark": False,
            "limitations": ["One first episode per engine; this is not a measured robustness probability.",
                            "Contact support and contiguous drift duration use held samples on the actual logged clock.",
                            "Near-rest sideslip is undefined and excluded below the fixed speed threshold.",
                            "Contact-patch slip must be recorded by each engine; it is never inferred from base velocity.",
                            "Missing nonwheel contact telemetry leaves complete contact evidence false.",
                            "Common-window RMSE is diagnostic and uses no phase shift, spatial alignment or extrapolation."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-dir", type=Path, required=True)
    parser.add_argument("--mujoco-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Preserving existing drift assessment: {args.output}")
    result = assess_runs(args.isaac_dir, args.mujoco_dir, args.reference)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(json.dumps({"verdict": result["verdict"], "failed_checks": result["failed_checks"]}))


if __name__ == "__main__":
    main()
