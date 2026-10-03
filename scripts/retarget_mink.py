"""CMU human reference -> constrained Mink IK for TRON1 WF (NO dynamics).

Floating-root motion is prescribed; only the six leg joints are solved. Wheel
spin is deliberately frozen, not guessed from ankle orientation or flight. This
exports a diagnostic kinematic baseline, NOT a robot-executable jump/policy.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]


def anchored_targets(human, robot_initial, scale):
    """One fixed offset per landmark; never re-ground individual frames."""
    return robot_initial[None] + scale * (human - human[:1])


def root_rotating_targets(human, human_root, robot_initial, robot_root_initial,
                          scale, root_quats_wxyz):
    """Transport initial morphology offsets with the relative root rotation.

    p_target = p_robot_root + s * (p_human - p_human_root) + R_relative * c,
    c = p_robot_initial - p_robot_root_initial - s * (p_human_0 - p_human_root_0).
    All inputs use the same heading-aligned world frame. Translation, source
    timing and pelvis rise remain unchanged; this is not framewise grounding.
    Unlike fixed-world anchoring, a rigid yaw of the entire human also rotates
    the complete robot landmark arrangement. Offset transport is a retargeting
    choice, not evidence of contact or angular-momentum feasibility.
    """
    human, human_root, robot_initial, robot_root_initial, quats = (
        np.asarray(value, dtype=float) for value in
        (human, human_root, robot_initial, robot_root_initial, root_quats_wxyz))
    if human.ndim != 3 or human.shape[-1] != 3 or len(human) < 1:
        raise ValueError("Human landmarks must have shape (frames, landmarks, 3)")
    frames, landmarks, _ = human.shape
    if (human_root.shape != (frames, 3) or robot_initial.shape != (landmarks, 3)
            or robot_root_initial.shape != (3,) or quats.shape != (frames, 4)):
        raise ValueError("Root, robot landmarks and quaternions must align with human frames")
    if (not np.isfinite(scale) or scale <= 0
            or not all(np.isfinite(v).all() for v in
                       (human, human_root, robot_initial, robot_root_initial, quats))
            or not np.allclose(np.linalg.norm(quats, axis=1), 1., atol=1e-8)):
        raise ValueError("Mapping requires finite arrays, positive scale and unit quaternions")
    rotation = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
    relative_rotation = rotation @ rotation[0].T
    offset = robot_initial - robot_root_initial - scale * (human[0] - human_root[0])
    root_target = anchored_targets(human_root, robot_root_initial, scale)
    return (root_target[:, None] + scale * (human - human_root[:, None])
            + np.einsum("fij,kj->fki", relative_rotation, offset))


def calibrated_root_rotations(quats_wxyz):
    """Remove ASF local axes and the initial robot mounting offset, not motion."""
    asf_to_world = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    human = Rotation.from_quat(quats_wxyz[:, [1, 2, 3, 0]]).as_matrix() @ asf_to_world.T
    heading = Rotation.from_euler("z", -Rotation.from_matrix(human[0]).as_euler("ZYX")[0]).as_matrix()
    result = heading @ human @ human[0].T @ heading.T
    quat = Rotation.from_matrix(result).as_quat()[:, [3, 0, 1, 2]]
    for i in range(1, len(quat)):
        if np.dot(quat[i - 1], quat[i]) < 0:
            quat[i] *= -1
    return quat, heading


def finite_difference_qvel(model, qpos, times):
    intervals = np.empty((len(qpos) - 1, model.nv))
    for f, dt in enumerate(np.diff(times)):
        mujoco.mj_differentiatePos(model, intervals[f], dt, qpos[f], qpos[f + 1])
    result = np.empty((len(qpos), model.nv))
    result[0], result[-1] = intervals[0], intervals[-1]
    result[1:-1] = .5 * (intervals[:-1] + intervals[1:])
    return result, intervals


def summarize_errors(errors):
    return {"rmse_m": float(np.sqrt(np.mean(errors ** 2))),
            "mean_m": float(np.mean(errors)), "p95_m": float(np.percentile(errors, 95)),
            "max_m": float(np.max(errors))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/mink_retarget_wf.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/mink-cmu-16_03")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output must be new or empty; existing results are preserved.")
    output.mkdir(parents=True, exist_ok=True)
    model_path, source_path = ROOT / settings["model"], ROOT / settings["source"]
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if model.nq != 15 or model.nv != 14 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Expected TRON1 free base + 8 hinge DOFs")
    with np.load(source_path, allow_pickle=False) as source:
        fps, times = float(source["fps"]), source["time_s"].copy()
        human = source["joint_positions_world_m"].copy()
        human_names, human_parents = source["joint_names"].copy(), source["parent_indices"].copy()
        root_quat, heading = calibrated_root_rotations(source["joint_rotations_world_wxyz"][:, 0])
        source_frames = source["frame_numbers"].copy()
    if len(times) < 2 or not np.allclose(np.diff(times), 1 / fps):
        raise ValueError("Expected uniform multi-frame reference")
    human = human @ heading.T
    human_index = {str(name): i for i, name in enumerate(human_names)}
    joint_names = [model.joint(i).name for i in range(1, model.njnt)]
    joint_qids = np.array([model.joint(n).qposadr[0] for n in joint_names])
    joint_vids = np.array([model.joint(n).dofadr[0] for n in joint_names])
    wheel_names = [n for n in joint_names if n.startswith("wheel_")]
    wheel_vids = [int(model.joint(n).dofadr[0]) for n in wheel_names]
    leg_names = [n for n in joint_names if n not in wheel_names]
    configuration = mink.Configuration(model)
    seed = model.qpos0.copy()
    seed[:7] = [0, 0, 0, 1, 0, 0, 0]
    for name, value in settings["seed_joint_positions_rad"].items():
        seed[model.joint(name).qposadr[0]] = value
    configuration.update(seed)
    site_names = ["pelvis", "left_hip", "right_hip", "left_knee", "right_knee",
                  "left_wheel", "right_wheel", "left_contact", "right_contact"]
    site_ids = np.array([model.site(n).id for n in site_names])
    target_names = settings["target_sites"]
    target_ids = np.array([model.site(n).id for n in target_names])
    # The official robot has two symmetric .127 m cylinders. No framewise lift.
    radius = float(json.loads((ROOT / "config/balance_wf.json").read_text())["model"]["radius"])
    initial_wheels = configuration.data.site_xpos[target_ids].copy()
    seed[2] = radius - float(np.mean(initial_wheels[:, 2]))
    configuration.update(seed)
    initial_wheels = configuration.data.site_xpos[target_ids].copy()
    source_ankles = human[:, [human_index[n] for n in settings["source_nodes"]]]
    robot_lengths, human_lengths = [], []
    for side, prefix in (("left", "l"), ("right", "r")):
        chain = configuration.data.site_xpos[[model.site(f"{side}_{part}").id
                                              for part in ("hip", "knee", "wheel")]]
        robot_lengths.append(float(np.linalg.norm(np.diff(chain, axis=0), axis=1).sum()))
        chain = human[0, [human_index[prefix + part] for part in ("hipjoint", "femur", "tibia")]]
        human_lengths.append(float(np.linalg.norm(np.diff(chain, axis=0), axis=1).sum()))
    scale = float(np.mean(robot_lengths) / np.mean(human_lengths))
    if not .3 < scale < 2:
        raise ValueError(f"Implausible scale {scale}")
    offset_mode = settings.get("landmark_offset_mode", "fixed_world")
    if offset_mode == "fixed_world":
        target_positions = anchored_targets(source_ankles, initial_wheels, scale)
        target_mapping = "p_target(t)=p_robot(0)+s*R_heading*(p_human(t)-p_human(0)); one constant world offset per ankle landmark"
    elif offset_mode == "root_rotating":
        target_positions = root_rotating_targets(
            source_ankles, human[:, human_index["root"]], initial_wheels, seed[:3], scale, root_quat)
        target_mapping = "p_target(t)=p_robot_root(t)+s*(p_human(t)-p_human_root(t))+R_robot(t)*c; initial per-landmark morphology offset c rotates with calibrated root; no grounding/retiming"
    else:
        raise ValueError(f"Unknown landmark_offset_mode: {offset_mode!r}")
    root_position = anchored_targets(human[:, human_index["root"]], seed[:3], scale)
    human_display = seed[None, None, :3] + scale * (human - human[:1, human_index["root"]:human_index["root"] + 1])
    target_tasks = [mink.FrameTask(n, "site", position_cost=settings["wheel_position_cost"],
                                   orientation_cost=0., lm_damping=0.) for n in target_names]
    posture = mink.PostureTask(model, cost=settings["posture_cost"])
    posture.set_target(seed)
    tasks = target_tasks + [posture]
    freeze = mink.DofFreezingTask(model, list(range(6)) + wheel_vids)
    speed_limit = float(settings["preview_joint_speed_limit_rad_s"])
    limits = [mink.ConfigurationLimit(model),
              mink.VelocityLimit(model, {n: speed_limit for n in leg_names})]
    substeps = int(settings["substeps"])
    if substeps < 1 or speed_limit <= 0:
        raise ValueError("substeps and speed limit must be positive")
    dt = 1 / (fps * substeps)
    orientation_at = Slerp(times, Rotation.from_quat(root_quat[:, [1, 2, 3, 0]]))
    frames = len(times)
    qpos = np.empty((frames, model.nq))
    sites = np.empty((frames, len(site_names), 3))
    errors = np.empty((frames, len(target_names)))
    contact_min = np.zeros(frames)
    self_contact_min = np.zeros(frames)
    ground_contact_count = np.zeros(frames, dtype=int)
    self_contact_count = np.zeros(frames, dtype=int)
    start = time.perf_counter()
    solves = 0
    for f in range(frames):
        for step in range(substeps if f else 0):
            alpha = (step + 1) / substeps
            target = (1 - alpha) * target_positions[f - 1] + alpha * target_positions[f]
            q = configuration.q.copy()
            q[:3] = (1 - alpha) * root_position[f - 1] + alpha * root_position[f]
            # Arithmetic on the last substep can exceed the final source time
            # by one ULP; do not ask SLERP to extrapolate that endpoint.
            sample_time = np.clip(times[f - 1] + alpha / fps, times[0], times[-1])
            q[3:7] = orientation_at(sample_time).as_quat()[[3, 0, 1, 2]]
            configuration.update(q)
            for task, name, point in zip(target_tasks, target_names, target):
                # Match the CURRENT frame orientation, making SE(3) rotational
                # residual zero. With isotropic XYZ cost this is pure position IK.
                rotation = configuration.get_transform_frame_to_world(name, "site").rotation()
                task.set_target(mink.SE3.from_rotation_and_translation(rotation, point))
            velocity = mink.solve_ik(configuration, tasks, dt, settings["solver"],
                                     damping=settings["damping"], safety_break=True,
                                     limits=limits, constraints=[freeze])
            if not np.isfinite(velocity).all():
                raise ValueError(f"Nonfinite velocity at frame {f}")
            configuration.integrate_inplace(velocity, dt)
            solves += 1
        qpos[f] = configuration.q
        sites[f] = configuration.data.site_xpos[site_ids]
        errors[f] = np.linalg.norm(configuration.data.site_xpos[target_ids] - target_positions[f], axis=1)
        # Read-only collision diagnostics. These are NOT IK constraints and do
        # not establish contact/force feasibility. No mj_step occurs anywhere.
        mujoco.mj_forward(model, configuration.data)
        for contact in configuration.data.contact:
            if contact.dist >= 0:
                continue
            bodies = model.geom_bodyid[[contact.geom1, contact.geom2]]
            if 0 in bodies:
                ground_contact_count[f] += 1
                contact_min[f] = min(contact_min[f], float(contact.dist))
            else:
                self_contact_count[f] += 1
                self_contact_min[f] = min(self_contact_min[f], float(contact.dist))
        if f % 100 == 0 or f == frames - 1:
            print(f"IK {f + 1}/{frames}: wheel errors {errors[f] * 1000} mm", flush=True)
    qvel, interval_vel = finite_difference_qvel(model, qpos, times)
    joint_pos = qpos[:, joint_qids]
    max_joint_speed = float(np.max(np.abs(interval_vel[:, joint_vids])))
    joint_ranges = model.jnt_range[1:].copy()
    limited = model.jnt_limited[1:].astype(bool)
    bound_violation = np.maximum(joint_ranges[:, 0] - joint_pos, joint_pos - joint_ranges[:, 1])
    max_violation = max(0., float(np.max(bound_violation[:, limited])))
    root_error = float(np.max(np.abs(qpos[:, :3] - root_position)))
    quat_error = float(np.max(1 - np.abs(np.sum(qpos[:, 3:7] * root_quat, axis=1))))
    if (max_violation > 1e-6 or max_joint_speed > speed_limit + 1e-5
            or root_error > 1e-7 or quat_error > 1e-9 or not np.isfinite(qpos).all()):
        raise ValueError("IK numerical/constraint validation failed")
    np.savez_compressed(output / "robot_reference.npz", fps=np.array(fps), time_s=times,
                        qpos=qpos, qvel=qvel, joint_names=np.array(joint_names),
                        joint_positions_rad=joint_pos, joint_velocities_rad_s=qvel[:, joint_vids],
                        joint_ranges_rad=joint_ranges, joint_limited=limited,
                        site_names=np.array(site_names), site_positions_world_m=sites,
                        target_names=np.array(target_names), target_positions_world_m=target_positions,
                        target_error_m=errors, root_target_position_m=root_position,
                        root_target_quat_wxyz=root_quat, human_joint_names=human_names,
                        human_parent_indices=human_parents, human_positions_world_m=human_display,
                        source_frame_numbers=source_frames,
                        ground_penetration_m=-contact_min, self_penetration_m=-self_contact_min)
    report = {
        "status": "completed_kinematic_preview", "robot_retargeted": True,
        "physics_validated": False, "policy_trained": False, "hardware_ready": False,
        "root_prescribed": True, "wheel_spin": "frozen_at_zero_not_rolling",
        "source": str(source_path), "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "model": str(model_path), "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "versions": {n: importlib.metadata.version(n) for n in ("mink", "mujoco", "qpsolvers", "daqp")},
        "settings": settings, "frames": frames, "fps": fps, "substeps": substeps,
        "qp_solves": solves, "elapsed_seconds": time.perf_counter() - start,
        "uniform_human_scale": scale, "robot_leg_chain_lengths_m": robot_lengths,
        "human_leg_chain_lengths_m": human_lengths, "initial_heading_alignment": heading.tolist(),
        "initial_root_position_m": seed[:3].tolist(), "initial_wheel_position_m": initial_wheels.tolist(),
        "landmark_offset_mode": offset_mode, "target_mapping": target_mapping,
        "root_rotation_mapping": "R_robot(t)=R_heading*R_human(t)*R_human(0)^T*R_heading^T; initial mounting calibrated to identity",
        "wheel_errors": {n: summarize_errors(errors[:, i]) for i, n in enumerate(target_names)},
        "all_wheel_errors": summarize_errors(errors), "max_joint_limit_violation_rad": max_violation,
        "max_interval_joint_speed_rad_s": max_joint_speed,
        "max_prescribed_root_position_error_m": root_error, "max_root_quat_dot_error": quat_error,
        "max_ground_penetration_m": float(-np.min(contact_min)),
        "frames_with_ground_penetration_gt_1mm": int(np.count_nonzero(contact_min < -.001)),
        "max_self_penetration_m": float(-np.min(self_contact_min)),
        "frames_with_self_penetration_gt_1mm": int(np.count_nonzero(self_contact_min < -.001)),
        "root_rise_from_initial_m": float(root_position[:, 2].max() - root_position[0, 2]),
        "warnings": [
            "Kinematic IK only: prescribed root motion can imply impossible forces/accelerations.",
            "Only wheel-center positions are tracked. Human forward-bending knees are NOT imposed on backward-bending TRON1 legs.",
            "Human ankle rotations, arm swing and whole-body angular momentum are not transferred.",
            "No rolling/contact/ground/self-collision constraints are enforced; penetrations are reported, not hidden.",
            "Ground and self-penetration diagnostics depend on enabled official collision geometry/masks.",
            "One fixed initial floor alignment; source end-of-clip foot height drift is retained.",
            "NPZ uses MuJoCo qpos/body order; not a drop-in BeyondMimic/Isaac reference."
        ]
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "scale": scale, "errors": report["all_wheel_errors"],
                      "max_ground_penetration_m": report["max_ground_penetration_m"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
