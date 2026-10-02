"""Convert GMR/Mink WF kinematics to the BeyondMimic 50 Hz motion contract.

This is an offline reference export, not simulation, PPO, or a dynamics claim.
The source world trajectory is never grounded per frame, rotated, or scaled.
An optional explicit uniform Z translation is recorded in the output metadata.
MuJoCo computes FK/Jacobians; an independent URDF FK checks every sample.
IsaacLab 2.1 combines link-frame poses with COM-frame linear velocities in
body_state_w, so the latter use URDF (unmerged Isaac) COM offsets explicitly.
Names and masks are additional metadata: unmodified upstream ignores them!
The training adapter MUST assert names and exclude unobserved wheel spin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "assets/robots/WF_TRON1A/mujoco/robot.xml"
DEFAULT_URDF = DEFAULT_MODEL.parent / "upstream/robot.urdf"
ISAAC_JOINT_NAMES = tuple(f"{part}_{side}_Joint" for part in ("abad", "hip", "knee", "wheel")
                          for side in ("L", "R"))
# Observed PhysX body order in outputs/sim2sim-isaac-gmr/report.json, not MJCF order.
ISAAC_BODY_NAMES = ("base_Link", "abad_L_Link", "hip_L_Link", "knee_L_Link", "wheel_L_Link",
                    "abad_R_Link", "hip_R_Link", "knee_R_Link", "wheel_R_Link", "limx_imu")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def continuous_quaternions(quats: np.ndarray) -> np.ndarray:
    """Unit wxyz quaternions with a continuous double-cover choice over time."""
    out = np.asarray(quats, dtype=np.float64).copy()
    lengths = np.linalg.norm(out, axis=-1, keepdims=True)
    if not np.isfinite(out).all() or np.any(lengths < 1e-12):
        raise ValueError("Quaternions must be finite and nonzero")
    out /= lengths
    for frame in range(1, len(out)):
        flip = np.sum(out[frame - 1] * out[frame], axis=-1) < 0
        out[frame] *= np.where(flip, -1.0, 1.0)[..., None]
    return out


def angular_velocity_world(quats: np.ndarray, dt: float) -> np.ndarray:
    """World-frame SO(3) central differences, one-sided at the two endpoints."""
    if len(quats) < 2 or dt <= 0:
        raise ValueError("Need two or more samples and a positive dt")
    rot = Rotation.from_quat(continuous_quaternions(quats)[:, [1, 2, 3, 0]])
    result = np.empty((len(quats), 3))
    result[0] = (rot[1] * rot[0].inv()).as_rotvec() / dt
    result[-1] = (rot[-1] * rot[-2].inv()).as_rotvec() / dt
    if len(quats) > 2:
        result[1:-1] = (rot[2:] * rot[:-2].inv()).as_rotvec() / (2 * dt)
    return result


def resample_qpos(times: np.ndarray, qpos: np.ndarray, output_fps: float,
                  prepend_s: float = 0.0, append_s: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Linear joint/translation + SLERP rotation interpolation without retiming.

    Like upstream csv_to_npz.py, samples lie on the output-frequency grid and
    do not extrapolate beyond the final source timestamp. Any sub-frame tail
    is explicitly reported. Optional holds repeat the resampled endpoint poses;
    they are NOT a landing recovery controller or a new settling transition.
    """
    times, qpos = np.asarray(times, float), np.asarray(qpos, float)
    if times.ndim != 1 or len(times) < 2 or qpos.shape[0] != len(times) or qpos.shape[1] < 7:
        raise ValueError("Expected aligned time samples and floating-base qpos")
    if not np.isfinite(times).all() or not np.isfinite(qpos).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Source must be finite and time strictly increasing")
    if not np.isfinite([output_fps, prepend_s, append_s]).all() or output_fps <= 0 or min(prepend_s, append_s) < 0:
        raise ValueError("FPS must be positive and endpoint hold durations nonnegative")
    duration = times[-1] - times[0]
    count = int(np.floor(duration * output_fps + 1e-9)) + 1
    if count < 2:
        raise ValueError("Source is too short for the requested output FPS")
    sample_t = np.arange(count, dtype=np.float64) / output_fps
    query = np.minimum(sample_t + times[0], times[-1])
    out = np.empty((count, qpos.shape[1]))
    for index in list(range(3)) + list(range(7, qpos.shape[1])):
        out[:, index] = np.interp(query, times, qpos[:, index])
    quat = continuous_quaternions(qpos[:, 3:7])
    out[:, 3:7] = Slerp(times, Rotation.from_quat(quat[:, [1, 2, 3, 0]]))(query).as_quat()[:, [3, 0, 1, 2]]
    out[:, 3:7] = continuous_quaternions(out[:, 3:7])
    n_pre, n_post = int(round(prepend_s * output_fps)), int(round(append_s * output_fps))
    out = np.concatenate((np.repeat(out[:1], n_pre, axis=0), out,
                          np.repeat(out[-1:], n_post, axis=0)), axis=0)
    source_times = np.r_[np.repeat(query[0], n_pre), query, np.repeat(query[-1], n_post)]
    output_times = np.arange(len(out), dtype=np.float64) / output_fps
    details = {"source_sample_span_s": float(duration), "resampled_sample_span_s": float(sample_t[-1]),
               "omitted_subframe_tail_s": max(0.0, float(duration - sample_t[-1])),
               "prepend_hold_frames": n_pre, "append_hold_frames": n_post,
               "prepend_hold_s": n_pre / output_fps, "append_hold_s": n_post / output_fps,
               "retimed": False, "framewise_ground_alignment": False}
    return output_times, source_times, out, details


class UrdfKinematics:
    """Independent link-frame FK, plus original (unmerged) link COM offsets."""
    def __init__(self, path: Path):
        root = ET.parse(path).getroot()
        self.com = {}
        self.joints = []
        self.parent = {}
        for link in root.findall("link"):
            origin = link.find("inertial/origin")
            self.com[link.get("name")] = self._xyz(origin, "xyz")
        for joint in root.findall("joint"):
            kind = joint.get("type")
            if kind not in ("fixed", "revolute", "continuous"):
                raise ValueError(f"Unsupported URDF joint type {kind}")
            origin, axis = joint.find("origin"), joint.find("axis")
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            self.parent[child] = (parent, kind)
            self.joints.append({"name": joint.get("name"), "parent": parent, "child": child,
                                "kind": kind, "translation": self._xyz(origin, "xyz"),
                                "rotation": Rotation.from_euler("xyz", self._xyz(origin, "rpy")).as_matrix(),
                                "axis": self._xyz(axis, "xyz", "1 0 0")})
        roots = set(self.com) - set(self.parent)
        if roots != {"base_Link"}:
            raise ValueError(f"Expected base_Link as the single URDF root, got {roots}")

    @staticmethod
    def _xyz(element, key, default="0 0 0"):
        return np.fromstring(default if element is None else element.get(key, default), sep=" ")

    def forward(self, base_position, base_quat, joints: dict[str, float]):
        poses = {"base_Link": (base_position, Rotation.from_quat(base_quat[[1, 2, 3, 0]]).as_matrix())}
        pending = list(self.joints)
        while pending:
            available = [joint for joint in pending if joint["parent"] in poses]
            if not available:
                raise ValueError("Disconnected URDF hierarchy")
            for joint in available:
                parent_pos, parent_rot = poses[joint["parent"]]
                pos = parent_pos + parent_rot @ joint["translation"]
                rot = parent_rot @ joint["rotation"]
                if joint["kind"] != "fixed":
                    rot = rot @ Rotation.from_rotvec(joint["axis"] * joints[joint["name"]]).as_matrix()
                poses[joint["child"]] = pos, rot
                pending.remove(joint)
        return poses

    def represented_parent(self, name: str, model: mujoco.MjModel) -> int:
        """Return a MuJoCo ancestor only when omitted links are fixed joints."""
        while mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
            if name not in self.parent or self.parent[name][1] != "fixed":
                raise ValueError(f"Missing non-fixed body in MuJoCo: {name}")
            name = self.parent[name][0]
        return model.body(name).id


def convert_motion(source_path: Path, model_path: Path = DEFAULT_MODEL, urdf_path: Path = DEFAULT_URDF,
                   output_fps: float = 50.0, prepend_s: float = 0.0, append_s: float = 0.0,
                   joint_names=ISAAC_JOINT_NAMES, body_names=ISAAC_BODY_NAMES,
                   z_offset: float = 0.0) -> tuple[dict, dict]:
    source_path, model_path, urdf_path = map(Path, (source_path, model_path, urdf_path))
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if model.nq != 15 or model.nv != 14 or model.njnt != 9 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Expected WF TRON1 floating base plus eight hinges")
    joint_names, body_names = tuple(joint_names), tuple(body_names)
    model_names = tuple(model.joint(index).name for index in range(1, model.njnt))
    if len(joint_names) != 8 or len(set(joint_names)) != 8 or set(joint_names) != set(model_names):
        raise ValueError("Isaac output joint names must be a permutation of all eight model joints")
    if not body_names or body_names[0] != "base_Link" or len(body_names) != len(set(body_names)):
        raise ValueError("Body order must be unique and base_Link must be first")
    with np.load(source_path, allow_pickle=False) as archive:
        source = {key: archive[key] for key in archive.files}
    if tuple(source["joint_names"]) != model_names:
        raise ValueError("Source qpos joint order differs from the named MuJoCo model")
    if source["qpos"].shape != (len(source["time_s"]), model.nq):
        raise ValueError("Source qpos shape does not match the model")
    qids = [int(model.joint(name).qposadr[0]) for name in model_names]
    if "joint_positions_rad" in source:
        np.testing.assert_allclose(source["qpos"][:, qids], source["joint_positions_rad"], atol=1e-10)
    source_fps = float(np.asarray(source["fps"]).item())
    if source_fps <= 0 or not np.allclose(np.diff(source["time_s"]), 1 / source_fps, rtol=1e-7, atol=1e-9):
        raise ValueError("Source fps and sample times disagree")
    if not np.allclose(np.linalg.norm(source["qpos"][:, 3:7], axis=1), 1.0, atol=1e-6):
        raise ValueError("Source floating-base quaternions are not normalized")
    time_s, source_time_s, qpos, sampling = resample_qpos(source["time_s"], source["qpos"],
                                                       output_fps, prepend_s, append_s)
    if not np.isfinite(z_offset):
        raise ValueError("Uniform Z offset must be finite")
    qpos[:, 2] += z_offset
    dt = 1 / output_fps
    qvel = np.empty((len(qpos), model.nv))
    qvel[:, :3] = np.gradient(qpos[:, :3], dt, axis=0)
    world_omega = angular_velocity_world(qpos[:, 3:7], dt)
    root_rotation = Rotation.from_quat(qpos[:, [4, 5, 6, 3]])
    # MuJoCo's free-joint angular qvel is local; its linear qvel is world.
    qvel[:, 3:6] = root_rotation.inv().apply(world_omega)
    for name in model_names:
        joint = model.joint(name)
        qvel[:, int(joint.dofadr[0])] = np.gradient(qpos[:, int(joint.qposadr[0])], dt)
    urdf = UrdfKinematics(urdf_path)
    if set(body_names) != set(urdf.com):
        raise ValueError("Body names must contain every original unmerged URDF link exactly once")
    data = mujoco.MjData(model)
    shape = (len(qpos), len(body_names))
    positions, quats = np.empty((*shape, 3)), np.empty((*shape, 4))
    linear_vel, angular_vel = np.empty((*shape, 3)), np.empty((*shape, 3))
    max_position_error = max_rotation_error = 0.0
    jac_position, jac_rotation = np.empty((3, model.nv)), np.empty((3, model.nv))
    represented_ids = [urdf.represented_parent(name, model) for name in body_names]
    for frame in range(len(qpos)):
        data.qpos[:], data.qvel[:] = qpos[frame], qvel[frame]
        mujoco.mj_forward(model, data)
        poses = urdf.forward(qpos[frame, :3], qpos[frame, 3:7],
                             {name: qpos[frame, qid] for name, qid in zip(model_names, qids)})
        for index, name in enumerate(body_names):
            urdf_pos, urdf_rot = poses[name]
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                pos, rot = data.xpos[body_id], data.xmat[body_id].reshape(3, 3)
                max_position_error = max(max_position_error, float(np.max(np.abs(pos - urdf_pos))))
                max_rotation_error = max(max_rotation_error, float(np.max(np.abs(rot - urdf_rot))))
            else:
                # Fixed IMU omitted by the official MJCF, retained by Isaac USD.
                pos, rot = urdf_pos, urdf_rot
            positions[frame, index] = pos
            quats[frame, index] = Rotation.from_matrix(rot).as_quat()[[3, 0, 1, 2]]
            com_world = pos + rot @ urdf.com[name]
            mujoco.mj_jac(model, data, jac_position, jac_rotation, com_world, represented_ids[index])
            linear_vel[frame, index] = jac_position @ qvel[frame]
            angular_vel[frame, index] = jac_rotation @ qvel[frame]
    if max(max_position_error, max_rotation_error) > 1e-9:
        raise ValueError(f"MuJoCo/URDF FK mismatch: {max_position_error}, {max_rotation_error}")
    quats = continuous_quaternions(quats)
    joint_qids = [int(model.joint(name).qposadr[0]) for name in joint_names]
    joint_vids = [int(model.joint(name).dofadr[0]) for name in joint_names]
    arrays = {"fps": np.array([output_fps], dtype=np.float64),
              "joint_pos": qpos[:, joint_qids].astype(np.float32),
              "joint_vel": qvel[:, joint_vids].astype(np.float32),
              "body_pos_w": positions.astype(np.float32), "body_quat_w": quats.astype(np.float32),
              "body_lin_vel_w": linear_vel.astype(np.float32), "body_ang_vel_w": angular_vel.astype(np.float32),
              "joint_names": np.array(joint_names), "body_names": np.array(body_names),
              "time_s": time_s, "source_time_s": source_time_s,
              "joint_tracking_mask": np.array([not name.startswith("wheel_") for name in joint_names]),
              "body_orientation_tracking_mask": np.array([not name.startswith("wheel_") for name in body_names]),
              "body_angular_velocity_tracking_mask": np.array([not name.startswith("wheel_") for name in body_names]),
              "qpos_mujoco": qpos, "qvel_mujoco": qvel}
    for key, array in arrays.items():
        if array.dtype.kind in "fiu" and not np.isfinite(array).all():
            raise ValueError(f"Nonfinite exported array: {key}")
    metadata = {
        "schema_version": 1, "status": "kinematic_training_reference_exported",
        "contract": "BeyondMimic MotionLoader / IsaacLab 2.1 or 2.3, full unmerged WF articulation",
        "source": str(source_path.resolve()), "source_sha256": sha256(source_path),
        "source_method": str(source.get("retarget_method", np.array("unspecified")).item()),
        "model": str(model_path.resolve()), "model_sha256": sha256(model_path),
        "urdf": str(urdf_path.resolve()), "urdf_sha256": sha256(urdf_path),
        "source_fps": source_fps, "fps": output_fps, "frames": len(time_s),
        "sample_span_s": float(time_s[-1]), "playback_duration_s": len(time_s) / output_fps,
        "sampling": sampling, "uniform_z_offset_m": float(z_offset),
        "joint_names": list(joint_names), "body_names": list(body_names),
        "source_joint_names": list(model_names), "root_body_index": 0,
        "positions": "world Z-up meters; rigid-body link/actor frame, not COM",
        "quaternions": "world-from-link orientation, wxyz, normalized and sign-continuous",
        "linear_velocity": "world-frame link COM velocity; MuJoCo Jacobian at original URDF COM",
        "angular_velocity": "world-frame angular velocity, rad/s; MuJoCo Jacobian",
        "joint_units": "radians and radians/second",
        "velocity_method": f"{output_fps:g} Hz generalized central differences (endpoints one-sided), analytic body Jacobians",
        "body_com_offsets_link_m": {name: urdf.com[name].tolist() for name in body_names},
        "independent_urdf_fk_samples": len(time_s),
        "independent_urdf_fk_max_position_error_m": max_position_error,
        "independent_urdf_fk_max_rotation_matrix_error": max_rotation_error,
        "root_rise_from_initial_m": float(positions[:, 0, 2].max() - positions[0, 0, 2]),
        "wheel_spin_observed": False,
        "wheel_spin_reference": "GMR placeholder; must not penalize wheel angle or zero-wheel-rate tracking",
        "wheel_orientation_is_training_target": False,
        "training_adapter_must_apply_masks": True,
        "training_runtime_name_order_assertion_required": True,
        "physics_validated": False, "policy_trained": False, "hardware_ready": False,
        "warnings": [
            "Unmodified upstream ignores name/mask arrays: validate actual simulator name order before training.",
            "The wheel orientation/rate is unobserved; track wheel-center positions but not wheel-body orientation/angular velocity.",
            "Endpoint holds merely repeat kinematic poses, do not provide physically feasible settling.",
            "No coordinate reset, height clipping, framewise floor correction, smoothing, or time/amplitude scaling; the sole optional change is recorded uniform_z_offset_m.",
            "URDF COM offsets match the existing unmerged USD; revalidate if a different asset/import is used.",
        ],
    }
    return arrays, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/gmr-cmu-16_03/robot_reference.npz")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, help="New NPZ path; omitted with --validate-only")
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--prepend-hold-s", type=float, default=0.0)
    parser.add_argument("--append-hold-s", type=float, default=0.0)
    parser.add_argument("--z-offset", type=float, default=0.0,
                        help="Explicit constant world Z translation in meters, applied equally to all frames/bodies")
    parser.add_argument("--isaac-order-json", type=Path,
                        help="Runtime report with body_names and joint_names or native_joint_names")
    parser.add_argument("--validate-only", action="store_true", help="Compute and validate without creating files")
    args = parser.parse_args()
    if not args.validate_only and args.output is None:
        parser.error("Specify --output or --validate-only")
    order = json.loads(args.isaac_order_json.read_text()) if args.isaac_order_json else {}
    joints = order.get("joint_names", order.get("native_joint_names", ISAAC_JOINT_NAMES))
    bodies = order.get("body_names", ISAAC_BODY_NAMES)
    if args.output is not None and not args.validate_only:
        if args.output.suffix != ".npz":
            parser.error("Output suffix must be .npz")
        for path in (args.output, args.output.with_suffix(".json")):
            if path.exists():
                parser.error(f"Refusing to replace existing artifact: {path}")
    arrays, metadata = convert_motion(args.source, args.model, args.urdf, args.output_fps,
                                      args.prepend_hold_s, args.append_hold_s, joints, bodies, args.z_offset)
    metadata["name_order_report"] = str(args.isaac_order_json.resolve()) if args.isaac_order_json else None
    if not args.validate_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        metadata["output_sha256"] = sha256(args.output)
        with args.output.with_suffix(".json").open("x") as stream:
            stream.write(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
