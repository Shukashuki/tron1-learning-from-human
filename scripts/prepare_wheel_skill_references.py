"""Explicitly adapt human pelvis intent into grounded, wheeled TRON1 references.

This is task adaptation, NOT unchanged human retargeting or dynamic validation.
The symmetric sagittal leg IK preserves ground clearance and a static COM
support condition; wheel rates solve the full contact Jacobian, not x/r alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "assets/robots/WF_TRON1A/mujoco/robot.xml"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibrated_pitch_and_heading(quaternions):
    """Same initial ASF/heading calibration as retarget_mink, without Mink."""
    a = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    human = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).as_matrix() @ a.T
    heading = Rotation.from_euler("z", -Rotation.from_matrix(human[0]).as_euler("ZYX")[0]).as_matrix()
    relative = heading @ human @ human[0].T @ heading.T
    return Rotation.from_matrix(relative).as_euler("ZYX")[:, 1], heading


def load_human(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in archive.files}
    names = result["joint_names"].tolist()
    if len(names) != len(set(names)) or "root" not in names:
        raise ValueError("Human names must be unique and contain root")
    fps = float(result["fps"])
    times = result["time_s"]
    if (not np.isfinite(fps) or fps <= 0 or len(times) < 3
            or not np.isfinite(times).all()
            or not np.allclose(np.diff(times), 1 / fps, atol=1e-9, rtol=0)):
        raise ValueError("Human motion needs finite, contiguous uniform timestamps and positive fps")
    pos, quat = result["joint_positions_world_m"], result["joint_rotations_world_wxyz"]
    if (pos.shape != (len(times), len(names), 3) or quat.shape != (len(times), len(names), 4)
            or not np.isfinite(pos).all() or not np.isfinite(quat).all()
            or not np.allclose(np.linalg.norm(quat, axis=-1), 1, atol=1e-6)):
        raise ValueError("Invalid human position/quaternion arrays")
    return result


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x ** 3 * (10 - 15 * x + 6 * x ** 2)


def adapt_intent(human, skill, *, first_frame=None, last_frame=None,
                 time_scale=2., root_height=.89, translation_scale=.35,
                 rolling_height_scale=.15, crouch_depth=.18, pitch_scale=.35,
                 smoothing_source_s=.10, output_fps=50.):
    """Return explicit task-adapted root intent, without modifying human data."""
    parameters = [time_scale, root_height, translation_scale, rolling_height_scale,
                  crouch_depth, pitch_scale, smoothing_source_s, output_fps]
    if not np.isfinite(parameters).all() or min(time_scale, root_height, output_fps) <= 0:
        raise ValueError("Finite parameters, positive time scale/root height/fps required")
    if min(translation_scale, rolling_height_scale, crouch_depth, pitch_scale, smoothing_source_s) < 0:
        raise ValueError("Adaptation scales and smoothing cannot be negative")
    if skill not in {"rolling_stop", "crouch"}:
        raise ValueError("Unknown wheel skill")
    frames = np.asarray(human["frame_numbers"])
    first = (160 if skill == "crouch" else int(frames[0])) if first_frame is None else first_frame
    last = (330 if skill == "crouch" else int(frames[-1])) if last_frame is None else last_frame
    selected = np.flatnonzero((frames >= first) & (frames <= last))
    if len(selected) < 3 or frames[selected[0]] != first or frames[selected[-1]] != last:
        raise ValueError("Requested inclusive source frame range must exist and contain >=3 frames")
    names = human["joint_names"].tolist()
    root_id = names.index("root")
    pitch, heading = calibrated_pitch_and_heading(human["joint_rotations_world_wxyz"][:, root_id])
    positions = human["joint_positions_world_m"][:, root_id] @ heading.T
    source_t = human["time_s"][selected] - human["time_s"][selected[0]]
    sigma = smoothing_source_s * float(human["fps"])
    selected_positions = positions[selected].copy()
    selected_pitch = pitch[selected].copy()
    if sigma > 0:
        selected_positions = gaussian_filter1d(selected_positions, sigma, axis=0, mode="nearest")
        selected_pitch = gaussian_filter1d(selected_pitch, sigma, mode="nearest")
    duration = float(source_t[-1] * time_scale)
    time_s = np.arange(int(np.floor(duration * output_fps + 1e-9)) + 1) / output_fps
    sampled_t = time_s / time_scale
    interp = lambda values: np.interp(sampled_t, source_t, values)
    root = np.zeros((len(time_s), 3))
    if skill == "rolling_stop":
        # Preserve the observed speed shape, but eliminate the lateral stepping
        # path. A 0.5 s final taper of speed provides an explicit stationary tail.
        x = translation_scale * interp(selected_positions[:, 0] - selected_positions[0, 0])
        raw_speed = np.gradient(x, time_s)
        taper = smoothstep((time_s[-1] - time_s) / .5)
        speed = raw_speed * taper
        root[1:, 0] = np.cumsum(.5 * (speed[1:] + speed[:-1]) * np.diff(time_s))
        root[:, 2] = root_height + rolling_height_scale * interp(selected_positions[:, 2] - selected_positions[0, 2])
        robot_pitch = pitch_scale * interp(selected_pitch - selected_pitch[0])
        # Remove endpoint drift in height/pitch velocity by holding these intent
        # channels through a smooth 0.5 s blend into their final pose.
        start = max(0., time_s[-1] - .5)
        blend = smoothstep((time_s - start) / max(time_s[-1] - start, 1e-9))
        root[:, 2] = (1 - blend) * root[:, 2] + blend * root[-1, 2]
        robot_pitch *= (1 - blend)
        detail = {"translation_scale": translation_scale, "height_scale": rolling_height_scale,
                  "discarded_human_channels": ["lateral translation", "yaw", "roll", "alternating foot steps"],
                  "stop_taper_output_s": .5,
                  "initial_motion": "Starts already rolling; reset must retain exported root and wheel velocities"}
    else:
        baseline = np.linspace(selected_positions[0, 2], selected_positions[-1, 2], len(selected))
        raw_drop = np.maximum(baseline - selected_positions[:, 2], 0)
        if raw_drop.max() < .05:
            raise ValueError("Selected human interval does not contain a >=5cm lowered-pelvis motion")
        envelope = smoothstep(sampled_t / .15) * smoothstep((source_t[-1] - sampled_t) / .15)
        profile = interp(raw_drop) * envelope
        if profile.max() <= 0:
            raise ValueError("No crouch intent remains after endpoint taper")
        root[:, 2] = root_height - crouch_depth * profile / profile.max()
        pitch_baseline = np.linspace(selected_pitch[0], selected_pitch[-1], len(selected))
        robot_pitch = pitch_scale * interp(selected_pitch - pitch_baseline) * envelope
        detail = {"depth_m": crouch_depth, "observed_smoothed_pelvis_drop_m": float(raw_drop.max()),
                  "height_mapping": "Subtract endpoint-linear height baseline, positive lowering only, normalize to explicit depth",
                  "endpoint_taper_source_s": .15,
                  "discarded_human_channels": ["all horizontal travel", "yaw", "roll", "alternating foot steps"],
                  "stationary_crouch": "Root XY fixed; wheel centers may roll fore/aft to maintain static COM support"}
    quat_xyzw = Rotation.from_euler("y", robot_pitch[:, None]).as_quat()
    metadata = {"source_first_frame_inclusive": int(first), "source_last_frame_inclusive": int(last),
                "source_selected_frames": len(selected), "source_selected_span_s": float(source_t[-1]),
                "output_sampled_source_end_s": float(sampled_t[-1]),
                "omitted_source_tail_s": float(source_t[-1] - sampled_t[-1]),
                "time_scale": time_scale, "nominal_root_height_m": root_height,
                "pitch_scale": pitch_scale, "gaussian_smoothing_sigma_source_s": smoothing_source_s,
                "root_orientation": "Initial-pose calibrated human pitch only; no yaw or roll",
                "world_floor_m": 0., "wheel_ground_center_height_m": .127,
                "constant_export_z_offset_m": 0., **detail}
    return {"time_s": time_s, "root_position": root, "root_quaternion": quat_xyzw[:, [3, 0, 1, 2]],
            "source_frame_numbers": first + sampled_t * float(human["fps"]),
            "metadata": metadata}


def solve_reference(intent, model_path=MODEL):
    """Bounded symmetric leg IK, then geometric wheel rates and all-frame audits."""
    import mujoco
    from retarget_mink import finite_difference_qvel
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    joint_names = [model.joint(i).name for i in range(1, model.njnt)]
    ids = {name: int(model.joint(name).qposadr[0]) for name in joint_names}
    h_l, k_l, h_r, k_r = [ids[name] for name in ("hip_L_Joint", "knee_L_Joint", "hip_R_Joint", "knee_R_Joint")]
    wheel_qids = [ids[name] for name in ("wheel_L_Joint", "wheel_R_Joint")]
    wheel_vids = [int(model.joint(name).dofadr[0]) for name in ("wheel_L_Joint", "wheel_R_Joint")]
    wheel_bids = [model.body(name).id for name in ("wheel_L_Link", "wheel_R_Link")]
    geom_ids = [model.geom(name).id for name in ("wheel_L_collision", "wheel_R_collision")]
    for gid in geom_ids:
        if not np.isclose(model.geom_size[gid, 0], .127):
            raise ValueError("Expected TRON1 0.127m cylindrical wheel geometry")
    lower = [-1.01229 + 1e-5, -.872665 + 1e-5]
    upper = [1.39626 - 1e-5, 1.36136 - 1e-5]
    qpos = np.repeat(model.qpos0[None], len(intent["time_s"]), axis=0)
    qpos[:, :3], qpos[:, 3:7] = intent["root_position"], intent["root_quaternion"]
    errors, centers, com = [], [], []
    warm = np.array([0., .1])
    for frame in range(len(qpos)):
        data.qpos[:] = qpos[frame]

        def residual(values):
            data.qpos[[h_l, k_l, h_r, k_r]] = [values[0], values[1], -values[0], -values[1]]
            mujoco.mj_forward(model, data)
            wheel_mid = data.xpos[wheel_bids].mean(axis=0)
            return np.array([wheel_mid[2] - .127, data.subtree_com[1, 0] - wheel_mid[0]])

        result = least_squares(residual, warm, bounds=(lower, upper), max_nfev=40,
                               ftol=1e-11, xtol=1e-11, gtol=1e-11)
        error = residual(result.x)
        if not result.success or np.max(np.abs(error)) > 1e-6:
            raise ValueError(f"Infeasible leg/support IK at frame {frame}: {error.tolist()}")
        warm = result.x
        qpos[frame] = data.qpos
        errors.append(error)
        centers.append(data.xpos[wheel_bids].copy())
        com.append(data.subtree_com[1].copy())
    times = intent["time_s"]
    qvel, _ = finite_difference_qvel(model, qpos, times)
    wheel_rates = np.empty((len(qpos), 2))
    for frame in range(len(qpos)):
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        for side, (gid, bid, vid) in enumerate(zip(geom_ids, wheel_bids, wheel_vids)):
            point = data.geom_xpos[gid] + [0., 0., -.127]
            jac = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jac, None, point, bid)
            # Includes root angular velocity AND hip/knee-induced axle rotation.
            if abs(jac[0, vid]) < .1:
                raise ValueError("Wheel contact Jacobian is not a sagittal rolling wheel")
            wheel_rates[frame, side] = -(jac @ qvel[frame])[0] / jac[0, vid]
    qpos[1:, wheel_qids] = np.cumsum(.5 * (wheel_rates[1:] + wheel_rates[:-1]) * np.diff(times)[:, None], axis=0)
    qvel, _ = finite_difference_qvel(model, qpos, times)
    ground_penetration, self_penetration, slip, nonwheel_ground = [], [], [], []
    site_names = ["pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_wheel", "right_wheel"]
    site_positions = []
    for frame in range(len(qpos)):
        data.qpos[:], data.qvel[:] = qpos[frame], qvel[frame]
        mujoco.mj_forward(model, data)
        ground = self_depth = other_depth = 0.
        for contact in data.contact:
            if contact.dist >= 0:
                continue
            bodies = model.geom_bodyid[[contact.geom1, contact.geom2]]
            if 0 in bodies:
                ground = max(ground, -contact.dist)
                if not any(g in geom_ids for g in (contact.geom1, contact.geom2)):
                    other_depth = max(other_depth, -contact.dist)
            else:
                self_depth = max(self_depth, -contact.dist)
        frame_slip = []
        for gid, bid in zip(geom_ids, wheel_bids):
            jac = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jac, None, data.geom_xpos[gid] + [0, 0, -.127], bid)
            frame_slip.append(jac @ qvel[frame])
        slip.append(frame_slip)
        ground_penetration.append(ground)
        self_penetration.append(self_depth)
        nonwheel_ground.append(other_depth)
        site_positions.append([data.site(name).xpos.copy() for name in site_names])
    qids = [ids[name] for name in joint_names]
    vids = [int(model.joint(name).dofadr[0]) for name in joint_names]
    ranges = np.array([model.joint(name).range.copy() for name in joint_names])
    limit_error = np.maximum(np.maximum(ranges[:, 0] - qpos[:, qids], qpos[:, qids] - ranges[:, 1]), 0)
    report = {"all_frames_checked": len(qpos), "max_ik_residual_m": float(np.max(np.abs(errors))),
              "max_joint_limit_violation_rad": float(limit_error.max()),
              "max_ground_penetration_m": float(np.max(ground_penetration)),
              "max_self_penetration_m": float(np.max(self_penetration)),
              "max_nonwheel_ground_penetration_m": float(np.max(nonwheel_ground)),
              "max_rolling_contact_tangent_velocity_residual_m_s": float(np.max(np.abs(np.asarray(slip)[:, :, :2]))),
              "max_leg_speed_rad_s": float(np.max(np.abs(qvel[:, [v for v in vids if v not in wheel_vids]]))),
              "max_wheel_speed_rad_s": float(np.max(np.abs(qvel[:, wheel_vids]))),
              "max_root_speed_m_s": float(np.max(np.linalg.norm(qvel[:, :3], axis=1))),
              "root_height_range_m": [float(qpos[:, 2].min()), float(qpos[:, 2].max())],
              "root_displacement_xyz_m": (qpos[-1, :3] - qpos[0, :3]).tolist(),
              "wheel_center_displacement_xyz_m": (np.asarray(centers)[-1] - np.asarray(centers)[0]).tolist(),
              "max_static_com_foreaft_support_error_m": float(np.max(np.abs(np.asarray(errors)[:, 1]))),
              "lateral_com_within_wheel_centers_all_frames": bool(np.all((np.asarray(com)[:, 1] >= np.asarray(centers)[:, :, 1].min(axis=1)) & (np.asarray(com)[:, 1] <= np.asarray(centers)[:, :, 1].max(axis=1)))),
              "contact_validation": "Geometry and Jacobian only, zero simulation steps; COM condition is static, not acceleration/force feasibility"}
    if limit_error.max() > 1e-6 or max(ground_penetration) > 1e-4 or max(self_penetration) > 1e-4:
        raise ValueError(f"Reference fails all-frame collision/range acceptance: {report}")
    arrays = {"fps": np.array(1 / (times[1] - times[0])), "time_s": times, "qpos": qpos, "qvel": qvel,
              "joint_names": np.array(joint_names), "joint_positions_rad": qpos[:, qids], "joint_velocities_rad_s": qvel[:, vids],
              "joint_ranges_rad": ranges, "joint_limited": np.ones(len(joint_names), dtype=bool),
              "site_names": np.array(site_names), "site_positions_world_m": np.asarray(site_positions),
              "root_target_position_m": intent["root_position"], "root_target_quat_wxyz": intent["root_quaternion"],
              "source_frame_numbers": intent["source_frame_numbers"], "ground_penetration_m": np.asarray(ground_penetration),
              "self_penetration_m": np.asarray(self_penetration), "rolling_contact_velocity_residual_m_s": np.asarray(slip),
              "retarget_method": np.array("human-inspired wheel-aware task adaptation + bounded symmetric leg IK + contact-Jacobian rolling")}
    return arrays, report


def prepare_reference(human_path, output_dir, skill, **options):
    from export_tracking_motion import convert_motion
    output_dir = Path(output_dir)
    targets = [output_dir / name for name in ("robot_reference.npz", "motion.npz", "motion.json", "adaptation.json")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Refusing to overwrite existing wheel-skill artifacts")
    human = load_human(human_path)
    intent = adapt_intent(human, skill, **options)
    arrays, report = solve_reference(intent)
    output_dir.mkdir(parents=True, exist_ok=True)
    with targets[0].open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    exported, metadata = convert_motion(targets[0], output_fps=50., append_s=1., z_offset=0.)
    metadata["wheel_spin_reference"] = "Geometric contact-Jacobian rolling estimate, not human observed; wheel angle/body orientation/rate remain masked"
    metadata["upstream_task_adaptation"] = intent["metadata"]
    metadata["warnings"] = [w for w in metadata["warnings"] if not w.startswith("No coordinate reset")]
    metadata["warnings"].append("Exporter performs no extra intent edits; source robot reference includes the explicit human-to-wheel task adaptation in adaptation.json.")
    with targets[1].open("xb") as stream:
        np.savez_compressed(stream, **exported)
    metadata["output_sha256"] = sha256(targets[1])
    targets[2].write_text(json.dumps(metadata, indent=2) + "\n")
    summary = {"schema_version": 1, "skill": skill, "kind": "human_inspired_wheel_task_adaptation_not_raw_retargeting",
               "source_human_npz": str(Path(human_path)), "source_human_sha256": sha256(human_path),
               "source_metadata": json.loads(str(human.get("metadata_json", "{}"))),
               "source_modified": False, "calibration_and_adaptation": intent["metadata"],
               "ik": "Two symmetric sagittal joint variables; six-joint mirrored pose, zero abduction; simultaneous wheel height and static COM-over-wheel constraint; bounded least squares, max 40 evaluations/frame",
               "rolling": "Wheel velocity cancels world-X contact-point velocity using full MuJoCo Jacobian, including root and leg angular velocity; trapezoidal spin integration",
               "wheel_spin_observed": False, "wheel_angle_orientation_rate_training_targets": False,
               "diagnostics": report, "motion_sha256": metadata["output_sha256"],
               "robot_reference_sha256": sha256(targets[0]), "model_sha256": sha256(MODEL),
               "fps": 50., "append_hold_s": 1., "motion_frames": len(exported["time_s"]),
               "sample_span_s": float(exported["time_s"][-1]), "playback_duration_s": len(exported["time_s"]) / 50.,
               "independent_urdf_fk_max_position_error_m": metadata["independent_urdf_fk_max_position_error_m"],
               "robot_retargeted": True, "physics_validated": False, "policy_trained": False, "hardware_ready": False,
               "warnings": ["Task adaptation explicitly removes human stepping and alters timing/amplitudes; this is not an untouched source trajectory.",
                            "Static support and geometric rolling do not prove acceleration, motor torque, friction, or balance feasibility.",
                            "Wheel rotations are unobserved kinematic estimates; training must honor masks.",
                            "A repeated endpoint hold is not a learned stabilization controller."]}
    targets[3].write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", required=True, choices=["rolling_stop", "crouch"])
    parser.add_argument("--human", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-frame", type=int)
    parser.add_argument("--last-frame", type=int)
    parser.add_argument("--time-scale", type=float, default=2.)
    parser.add_argument("--root-height", type=float, default=.89)
    parser.add_argument("--translation-scale", type=float, default=.35)
    parser.add_argument("--rolling-height-scale", type=float, default=.15)
    parser.add_argument("--crouch-depth", type=float, default=.18)
    parser.add_argument("--pitch-scale", type=float, default=.35)
    parser.add_argument("--smoothing-source-s", type=float, default=.10)
    args = vars(parser.parse_args())
    human, output, skill = args.pop("human"), args.pop("output_dir"), args.pop("skill")
    print(json.dumps(prepare_reference(human, output, skill, **args), indent=2))


if __name__ == "__main__":
    main()
