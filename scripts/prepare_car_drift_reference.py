"""Adapt captured car motion to a grounded TRON1 kinematic tracking reference.

Source XY and yaw retain their measured profile on the explicitly estimated
source clock. Uniform spatial/time scaling and a synthetic stop are recorded.
Wheel spin cancels only the rolling component of geometric contact velocity;
lateral slip is retained. This module performs no dynamics simulation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "assets/robots/WF_TRON1A/mujoco/robot.xml"
DEFAULT_SOURCE = ROOT / "outputs/drift-source-screening-20261011/sa_traj34/captured_source.npz"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wrap(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def smoothstep(u):
    u = np.clip(u, 0., 1.)
    return u ** 3 * (10. - 15. * u + 6. * u ** 2)


def integral_taper(u):
    """Integral from zero to u of 1 - quintic smoothstep(u)."""
    u = np.clip(u, 0., 1.)
    return u - 2.5 * u ** 4 + 3. * u ** 5 - u ** 6


def load_verified_source(path=DEFAULT_SOURCE):
    """Verify pinned raw CSV, capture audit, and every decoded NPZ array."""
    from capture_car_drift import decode_csv
    path = Path(path)
    source_id = path.parent.name
    source_root = path.parent.parent
    config_path = ROOT / "config/drift_sources.json"
    audit_path = source_root / "source_audit.json"
    config = json.loads(config_path.read_text())
    audit = json.loads(audit_path.read_text())
    if (audit["source_commit"] != config["commit"]
            or audit["configuration_sha256"] != sha256(config_path)):
        raise ValueError("Capture audit is not linked to the pinned source configuration")
    entries = [item for item in config["files"] if item["id"] == source_id]
    if len(entries) != 1:
        raise ValueError("Captured source must identify exactly one pinned CSV")
    entry = entries[0]
    raw_path = source_root / "raw" / Path(entry["path"]).name
    raw_hash = sha256(raw_path)
    if (raw_hash != entry["sha256"]
            or raw_hash != audit["captures"][source_id]["source_sha256"]):
        raise ValueError("Raw CSV SHA does not match capture and source pin")
    decoded = decode_csv(raw_path.read_bytes())
    with np.load(path, allow_pickle=False) as archive:
        source = {name: archive[name].copy() for name in archive.files}
    if set(source) != set(decoded):
        raise ValueError("Captured NPZ fields differ from the pinned CSV decoder")
    for name in source:
        if not np.array_equal(source[name], decoded[name]):
            raise ValueError(f"Captured NPZ differs from pinned CSV: {name}")
    provenance = {
        "source_id": source_id, "source_repository": config["repository"],
        "source_commit": config["commit"], "source_csv_path": entry["path"],
        "source_csv_sha256": raw_hash, "source_npz_sha256": sha256(path),
        "source_audit_sha256": sha256(audit_path), "source_configuration_sha256": sha256(config_path),
        "source_clock_status": "reconstructed_estimate_not_original_timestamps",
        "source_is_physical_car_capture": False,
    }
    return source, provenance


def adapt_source(source, *, spatial_scale=.08, time_scale=2., root_height=.75,
                 recovery_s=1.5, stationary_tail_s=.5, output_fps=50.):
    """Preserve car path/yaw with uniform scaling, then add a disclosed stop."""
    parameters = np.array([spatial_scale, time_scale, root_height, recovery_s,
                           stationary_tail_s, output_fps], dtype=float)
    if (not np.isfinite(parameters).all() or min(parameters[[0, 1, 2, 3, 5]]) <= 0
            or stationary_tail_s < 0):
        raise ValueError("Finite positive scales, height, recovery and FPS required; tail must be nonnegative")
    for duration in (recovery_s, stationary_tail_s):
        if not np.isclose(duration * output_fps, round(duration * output_fps), atol=1e-9):
            raise ValueError("Recovery and stationary tail must be whole output frames")
    t_all = np.asarray(source["time_s_estimated"], dtype=float)
    ids = np.asarray(source["unique_pose_indices"])
    n = len(t_all)
    xy_all = np.asarray(source["position_xy_m"], dtype=float)
    vel_all = np.asarray(source["velocity_xy_m_s"], dtype=float)
    yaw_all = np.asarray(source["yaw_rad"], dtype=float)
    if (t_all.ndim != 1 or n < 3 or xy_all.shape != (n, 2) or vel_all.shape != (n, 2)
            or yaw_all.shape != (n,) or ids.ndim != 1 or ids.dtype.kind not in "iu"
            or len(ids) < 3 or ids[0] != 0 or np.any(np.diff(ids) <= 0)
            or ids[-1] >= n or np.any(np.diff(t_all) < 0)
            or not all(np.isfinite(v).all() for v in (t_all, xy_all, vel_all, yaw_all))):
        raise ValueError("Invalid captured moving source arrays or distinct-pose indices")
    t = t_all[ids] - t_all[ids[0]]
    if t[0] != 0. or np.any(np.diff(t) <= 0.):
        raise ValueError("Distinct poses need a strictly increasing estimated source clock")
    source_count = int(np.floor(t[-1] * time_scale * output_fps + 1e-9)) + 1
    if source_count < 3:
        raise ValueError("Source is too short at requested output rate")
    source_task_t = np.arange(source_count) / output_fps
    query = source_task_t / time_scale
    initial_yaw = yaw_all[ids[0]]
    heading = np.array([[np.cos(initial_yaw), np.sin(initial_yaw)],
                        [-np.sin(initial_yaw), np.cos(initial_yaw)]])
    xy = spatial_scale * (xy_all[ids] - xy_all[ids[0]]) @ heading.T
    vel = spatial_scale / time_scale * vel_all[ids] @ heading.T
    interpolate = lambda values: np.stack([np.interp(query, t, values[:, j])
                                           for j in range(values.shape[1])], axis=1)
    sampled_xy = interpolate(xy)
    sampled_vel = interpolate(vel)
    psi = np.interp(query, t, yaw_all[ids]) - initial_yaw
    root = np.c_[sampled_xy, np.full(source_count, root_height)]
    yaw = psi.copy()
    source_mapping = query + t_all[ids[0]]
    row_mapping = np.interp(query, t, ids.astype(float))
    phase = np.zeros(source_count, dtype=np.int8)
    # The stop begins at the final sampled source pose. Its initial speed and
    # yaw rate come from reference pose derivatives, avoiding a moving hold.
    v0 = (sampled_xy[-1] - sampled_xy[-2]) * output_fps
    speed0 = np.linalg.norm(v0)
    beta0 = float(wrap(np.arctan2(v0[1], v0[0]) - psi[-1]))
    r0 = (psi[-1] - psi[-2]) * output_fps
    recovery_count = int(round(recovery_s * output_fps))
    local_t = np.arange(recovery_count + 1) / output_fps
    u = local_t / recovery_s
    taper = 1. - smoothstep(u)
    recovery_yaw = psi[-1] + r0 * recovery_s * integral_taper(u)
    recovery_beta = beta0 * taper
    course = recovery_yaw + recovery_beta
    recovery_vel = speed0 * taper[:, None] * np.c_[np.cos(course), np.sin(course)]
    recovery_xy = np.zeros_like(recovery_vel)
    recovery_xy[1:] = np.cumsum(.5 * (recovery_vel[:-1] + recovery_vel[1:]) / output_fps, axis=0)
    recovery_xy += sampled_xy[-1]
    root = np.concatenate((root, np.c_[recovery_xy[1:], np.full(recovery_count, root_height)]))
    yaw = np.r_[yaw, recovery_yaw[1:]]
    phase = np.r_[phase, np.ones(recovery_count, dtype=np.int8)]
    source_mapping = np.r_[source_mapping, np.repeat(source_mapping[-1], recovery_count)]
    row_mapping = np.r_[row_mapping, np.repeat(row_mapping[-1], recovery_count)]
    tail_count = int(round(stationary_tail_s * output_fps))
    if tail_count:
        root = np.concatenate((root, np.repeat(root[-1:], tail_count, axis=0)))
        yaw = np.r_[yaw, np.repeat(yaw[-1], tail_count)]
        phase = np.r_[phase, np.full(tail_count, 2, dtype=np.int8)]
        source_mapping = np.r_[source_mapping, np.repeat(source_mapping[-1], tail_count)]
        row_mapping = np.r_[row_mapping, np.repeat(row_mapping[-1], tail_count)]
    time_s = np.arange(len(root)) / output_fps
    qxyzw = Rotation.from_euler("z", yaw[:, None]).as_quat()
    metadata = {
        "spatial_scale": spatial_scale, "time_scale": time_scale,
        "nominal_root_height_m": root_height, "output_fps": output_fps,
        "initial_heading_alignment_rad": float(-initial_yaw), "heading_alignment_matrix_xy": heading.tolist(),
        "source_clock_status": "reconstructed_estimate_not_original_timestamps",
        "source_initial_time_estimated_s": float(t_all[ids[0]]),
        "source_last_time_estimated_s": float(t_all[ids[-1]]),
        "sampled_source_last_time_estimated_s": float(query[-1] + t_all[ids[0]]),
        "omitted_source_subframe_tail_estimated_s": float(t[-1] - query[-1]),
        "source_frames": source_count, "source_task_end_s": float(source_task_t[-1]),
        "source_position_mapping": "uniform scale * initial-heading rotation * (captured XY - initial XY)",
        "source_yaw_mapping": "interpolate captured unwrapped yaw on estimated clock; subtract initial yaw; no amplitude scaling",
        "smoothing_source_s": 0., "height_and_roll_pitch": "Explicit constant lowered root height; zero roll/pitch; source z/pitch discarded",
        "recovery_duration_s": recovery_s, "recovery_initial_speed_m_s": float(speed0),
        "recovery_initial_yaw_rate_rad_s": float(r0), "recovery_initial_sideslip_rad": beta0,
        "recovery_adaptation": "Synthetic quintic taper of speed/yaw rate and sideslip to zero; trapezoidal velocity integration",
        "stationary_tail_s": stationary_tail_s,
        "phase_codes": {"0": "captured_scaled_car_motion", "1": "synthetic_stop_recovery", "2": "stationary_tail"},
        "source_motion": "Starts already moving; retain initial root/wheel velocities on reset",
        "scaled_logged_velocity_vs_pose_difference_max_m_s": float(np.linalg.norm(
            np.gradient(sampled_xy, 1 / output_fps, axis=0) - sampled_vel, axis=1).max()),
        "physics_validated": False,
    }
    return {"time_s": time_s, "root_position": root, "root_quaternion": qxyzw[:, [3, 0, 1, 2]],
            "root_yaw_rad": yaw, "demonstration_time_s_estimated": source_mapping,
            "source_row_numbers": row_mapping, "source_phase": phase, "metadata": metadata}


def differentiate(model, qpos, times):
    import mujoco
    intervals = np.empty((len(times) - 1, model.nv))
    for index, dt in enumerate(np.diff(times)):
        mujoco.mj_differentiatePos(model, intervals[index], dt, qpos[index], qpos[index + 1])
    qvel = np.empty((len(times), model.nv))
    qvel[0], qvel[-1] = intervals[0], intervals[-1]
    qvel[1:-1] = .5 * (intervals[:-1] + intervals[1:])
    return qvel


def solve_reference(intent, model_path=MODEL):
    """Bounded static support pose plus arbitrary-yaw rolling-only wheel IK."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    names = [model.joint(index).name for index in range(1, model.njnt)]
    qids = [int(model.joint(name).qposadr[0]) for name in names]
    vids = [int(model.joint(name).dofadr[0]) for name in names]
    ids = dict(zip(names, qids))
    wheel_names = ("wheel_L_Joint", "wheel_R_Joint")
    wheel_qids = [ids[name] for name in wheel_names]
    wheel_vids = [int(model.joint(name).dofadr[0]) for name in wheel_names]
    wheel_bids = [model.body(name).id for name in ("wheel_L_Link", "wheel_R_Link")]
    gids = [model.geom(name).id for name in ("wheel_L_collision", "wheel_R_collision")]
    base_id = model.body("base_Link").id
    radii = model.geom_size[gids, 0]
    if not np.allclose(radii, .127, rtol=0, atol=1e-12):
        raise ValueError("Expected official cylindrical TRON1 wheels of radius 0.127m")
    pairs = (("hip_L_Joint", "hip_R_Joint"), ("knee_L_Joint", "knee_R_Joint"))
    lower = [max(model.joint(left).range[0], -model.joint(right).range[1]) + 1e-5
             for left, right in pairs]
    upper = [min(model.joint(left).range[1], -model.joint(right).range[0]) - 1e-5
             for left, right in pairs]
    data.qpos[:] = model.qpos0
    data.qpos[:3] = [0., 0., intent["root_position"][0, 2]]
    data.qpos[3:7] = [1., 0., 0., 0.]
    data.qpos[[ids["abad_L_Joint"], ids["abad_R_Joint"]]] = 0.
    def residual(values):
        for (left, right), value in zip(pairs, values):
            data.qpos[[ids[left], ids[right]]] = [value, -value]
        mujoco.mj_forward(model, data)
        midpoint = data.geom_xpos[gids].mean(axis=0)
        return np.array([midpoint[2] - radii.mean(), data.subtree_com[base_id, 0] - midpoint[0]])
    solution = least_squares(residual, [0., .1], bounds=(lower, upper), max_nfev=100,
                             ftol=1e-12, xtol=1e-12, gtol=1e-12)
    error = residual(solution.x)
    if not solution.success or np.max(np.abs(error)) > 1e-6:
        raise ValueError(f"No bounded static wheel support pose at requested height: {error.tolist()}")
    static_pose = data.qpos.copy()
    qpos = np.repeat(static_pose[None], len(intent["time_s"]), axis=0)
    qpos[:, :3], qpos[:, 3:7] = intent["root_position"], intent["root_quaternion"]
    times = intent["time_s"]
    qvel = differentiate(model, qpos, times)
    wheel_rates = np.empty((len(times), 2))
    rotations = Rotation.from_quat(qpos[:, [4, 5, 6, 3]]).as_matrix()
    for frame in range(len(times)):
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        direction = rotations[frame, :, 0]
        for side, (gid, bid, vid, radius) in enumerate(zip(gids, wheel_bids, wheel_vids, radii)):
            jac = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jac, None, data.geom_xpos[gid] + [0., 0., -radius], bid)
            spin_coefficient = direction @ jac[:, vid]
            if abs(spin_coefficient) < .1:
                raise ValueError("Wheel rolling Jacobian does not match the local sagittal axis")
            wheel_rates[frame, side] = -(direction @ (jac @ qvel[frame])) / spin_coefficient
    qpos[1:, wheel_qids] += np.cumsum(.5 * (wheel_rates[:-1] + wheel_rates[1:])
                                      * np.diff(times)[:, None], axis=0)
    qvel = differentiate(model, qpos, times)
    centers, collision_centers, com, contact_vel, support, site_positions = [], [], [], [], [], []
    penetration, self_penetration, nonwheel_penetration = [], [], []
    site_names = ["pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_wheel", "right_wheel"]
    for frame in range(len(times)):
        data.qpos[:], data.qvel[:] = qpos[frame], qvel[frame]
        mujoco.mj_forward(model, data)
        centers.append(data.xpos[wheel_bids].copy())
        collision_centers.append(data.geom_xpos[gids].copy())
        com.append(data.subtree_com[base_id].copy())
        midpoint = data.geom_xpos[gids].mean(axis=0)
        support.append((data.subtree_com[base_id] - midpoint) @ rotations[frame])
        velocities = []
        for gid, bid, radius in zip(gids, wheel_bids, radii):
            jac = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jac, None, data.geom_xpos[gid] + [0., 0., -radius], bid)
            velocities.append((jac @ qvel[frame]) @ rotations[frame])
        contact_vel.append(velocities)
        ground = self_depth = other_depth = 0.
        for contact in data.contact:
            if contact.dist >= 0.:
                continue
            if 0 in model.geom_bodyid[[contact.geom1, contact.geom2]]:
                ground = max(ground, -contact.dist)
                if not any(gid in gids for gid in (contact.geom1, contact.geom2)):
                    other_depth = max(other_depth, -contact.dist)
            else:
                self_depth = max(self_depth, -contact.dist)
        penetration.append(ground); self_penetration.append(self_depth); nonwheel_penetration.append(other_depth)
        site_positions.append([data.site(name).xpos.copy() for name in site_names])
    ranges = np.array([model.joint(name).range for name in names])
    limited = np.array([bool(model.joint(name).limited) for name in names])
    violation = np.maximum(np.maximum(ranges[:, 0] - qpos[:, qids], qpos[:, qids] - ranges[:, 1]), 0.)[:, limited]
    contact_vel = np.asarray(contact_vel)
    support = np.asarray(support)
    report = {
        "all_frames_checked": len(times), "simulation_steps": 0,
        "max_ik_residual_m": float(np.max(np.abs(error))),
        "static_leg_joint_positions_rad": {name: float(static_pose[ids[name]]) for name in names if not name.startswith("wheel_")},
        "max_joint_limit_violation_rad": float(violation.max()),
        "max_ground_penetration_m": float(max(penetration)),
        "max_self_penetration_m": float(max(self_penetration)),
        "max_nonwheel_ground_penetration_m": float(max(nonwheel_penetration)),
        "max_static_com_foreaft_support_error_m": float(np.abs(support[:, 0]).max()),
        "max_contact_rolling_velocity_residual_m_s": float(np.abs(contact_vel[:, :, 0]).max()),
        "max_reference_contact_lateral_slip_m_s": float(np.abs(contact_vel[:, :, 1]).max()),
        "lateral_slip_constraint_imposed": False,
        "max_wheel_speed_rad_s": float(np.abs(qvel[:, wheel_vids]).max()),
        "max_root_speed_m_s": float(np.linalg.norm(qvel[:, :3], axis=1).max()),
        "max_root_acceleration_m_s2": float(np.linalg.norm(np.gradient(qvel[:, :3], times, axis=0), axis=1).max()),
        "final_root_speed_m_s": float(np.linalg.norm(qvel[-1, :3])),
        "final_root_angular_speed_rad_s": float(np.linalg.norm(qvel[-1, 3:6])),
        "final_wheel_speed_rad_s": float(np.abs(qvel[-1, wheel_vids]).max()),
        "contact_validation": "Geometry and contact Jacobian only; static COM support does not establish dynamic or tire-force feasibility",
    }
    if max(report["max_joint_limit_violation_rad"], report["max_ground_penetration_m"],
           report["max_self_penetration_m"], report["max_nonwheel_ground_penetration_m"]) > 1e-4:
        raise ValueError(f"Reference violates geometry or joint bounds: {report}")
    arrays = {
        "fps": np.array(1 / (times[1] - times[0])), "time_s": times,
        "qpos": qpos, "qvel": qvel, "joint_names": np.array(names),
        "joint_positions_rad": qpos[:, qids], "joint_velocities_rad_s": qvel[:, vids],
        "joint_ranges_rad": ranges, "joint_limited": limited,
        "site_names": np.array(site_names), "site_positions_world_m": np.asarray(site_positions),
        "root_target_position_m": intent["root_position"], "root_target_quat_wxyz": intent["root_quaternion"],
        "source_frame_numbers": intent["source_row_numbers"],
        "demonstration_time_s_estimated": intent["demonstration_time_s_estimated"],
        "source_phase": intent["source_phase"], "root_yaw_rad": intent["root_yaw_rad"],
        "wheel_center_positions_world_m": np.asarray(centers),
        "wheel_collision_center_positions_world_m": np.asarray(collision_centers),
        "whole_robot_com_world_m": np.asarray(com),
        "ground_penetration_m": np.asarray(penetration), "self_penetration_m": np.asarray(self_penetration),
        "geometric_contact_velocity_heading_local_m_s": contact_vel,
        "retarget_method": np.array("captured car XY/yaw adaptation + bounded static leg IK + rolling-only wheel Jacobian; lateral slip retained"),
    }
    return arrays, report


def prepare_reference(source_path, output_dir, **options):
    from export_tracking_motion import convert_motion
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Use a new or empty drift-reference output directory")
    source, provenance = load_verified_source(source_path)
    intent = adapt_source(source, **options)
    arrays, diagnostics = solve_reference(intent)
    output_dir.mkdir(parents=True, exist_ok=True)
    robot_path = output_dir / "robot_reference.npz"
    with robot_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    exported, metadata = convert_motion(robot_path, output_fps=50., append_s=0., z_offset=0.)
    for key in ("demonstration_time_s_estimated", "source_phase", "root_yaw_rad"):
        if len(exported["time_s"]) != len(arrays[key]):
            raise ValueError("Exporter changed the declared 50 Hz reference sample count")
        exported[key] = arrays[key]
    metadata.update({
        "wheel_spin_reference": "Geometric rolling-direction contact Jacobian; lateral slip allowed; wheel spin is unobserved and tracking remains masked",
        "upstream_task_adaptation": intent["metadata"], "source_car_provenance": provenance,
        "source_time_mapping": "source_time_s uses robot reference clock; demonstration_time_s_estimated maps separately to captured car clock; synthetic phases are marked",
        "recovery_is_demonstrated": False, "source_controls_are_robot_targets": False,
    })
    metadata["warnings"] = [warning for warning in metadata["warnings"]
                            if not warning.startswith(("No coordinate reset", "Endpoint holds"))]
    metadata["warnings"].append("Upstream explicitly scales car XY/time and appends a synthetic stop. Geometry passes do not establish physical drifting.")
    motion_path = output_dir / "motion.npz"
    with motion_path.open("xb") as stream:
        np.savez_compressed(stream, **exported)
    metadata["output_sha256"] = sha256(motion_path)
    (output_dir / "motion.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    report = {
        "schema_version": 1, "skill": "car_drift", "status": "kinematic_reference_ready_for_physics_training",
        "kind": "car_demonstration_task_adaptation", "source_modified": False,
        "provenance": provenance, "calibration_and_adaptation": intent["metadata"],
        "diagnostics": diagnostics, "model_sha256": sha256(MODEL),
        "generator_sha256": sha256(Path(__file__)), "robot_reference_sha256": sha256(robot_path),
        "motion_sha256": sha256(motion_path), "frames": len(exported["time_s"]),
        "sample_span_s": float(exported["time_s"][-1]),
        "physics_validated": False, "policy_trained": False, "hardware_ready": False,
    }
    (output_dir / "adaptation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spatial-scale", type=float, default=.08)
    parser.add_argument("--time-scale", type=float, default=2.)
    parser.add_argument("--root-height", type=float, default=.75)
    parser.add_argument("--recovery-s", type=float, default=1.5)
    parser.add_argument("--stationary-tail-s", type=float, default=.5)
    args = parser.parse_args()
    report = prepare_reference(args.source, args.output_dir, spatial_scale=args.spatial_scale,
                               time_scale=args.time_scale, root_height=args.root_height,
                               recovery_s=args.recovery_s, stationary_tail_s=args.stationary_tail_s)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
