"""Identical reference sampling, torque law and reporting for both simulators.

This is a controller transfer DIAGNOSTIC. No RL policy is being evaluated and
no floating-root target is applied as a force, pose update, or constraint.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from balance_lqr import WIPParameters, design_lqr

ROOT = Path(__file__).resolve().parents[1]
JOINT_NAMES = tuple(f"{part}_{side}_Joint" for side in ("L", "R")
                    for part in ("abad", "hip", "knee", "wheel"))
WHEELS = np.array([3, 7])
LEGS = np.array([0, 1, 2, 4, 5, 6])


def load_config(path=None):
    return json.loads(Path(path or ROOT / "config/sim2sim_wf.json").read_text())


class ReferenceMotion:
    def __init__(self, path, case="motion", config=None):
        self.path = Path(path).resolve()
        self.config = load_config() if config is None else config
        if case not in ("motion", "standing"):
            raise ValueError("case must be motion or standing")
        self.case = case
        with np.load(self.path, allow_pickle=False) as source:
            self.times = source["time_s"].astype(float)
            self.fps = float(source["fps"])
            names = list(source["joint_names"])
            if set(names) != set(JOINT_NAMES):
                raise ValueError("Reference must contain exactly eight TRON1 joints")
            order = [names.index(name) for name in JOINT_NAMES]
            self.joints = source["joint_positions_rad"][:, order].astype(float)
            self.root_pos = source["qpos"][:, :3].astype(float)
            self.root_quat = source["qpos"][:, 3:7].astype(float)
            site_names = list(source["site_names"])
            wheel_sites = [site_names.index(n) for n in ("left_wheel", "right_wheel")]
            self.axles = source["site_positions_world_m"][:, wheel_sites].mean(axis=1)
        if len(self.times) < 2 or not np.allclose(np.diff(self.times), 1 / self.fps):
            raise ValueError("Expected uniformly sampled reference")
        if not np.isfinite(np.c_[self.joints, self.root_pos, self.root_quat, self.axles]).all():
            raise ValueError("Nonfinite motion reference")
        self.times -= self.times[0]
        self.joint_names = JOINT_NAMES
        self.clip_duration = float(self.times[-1])
        self.duration = (float(self.config["standing_seconds"]) if case == "standing" else
                         self.config["settle_seconds"] + self.clip_duration + self.config["recovery_seconds"])
        self.initial_qpos = np.r_[self.root_pos[0], self.root_quat[0], self.joints[0]]
        self.initial_qpos[2] += self.config["initial_clearance_m"]
        self.rotation = Slerp(self.times, Rotation.from_quat(self.root_quat[:, [1, 2, 3, 0]]))
        self.velocities = np.gradient(self.joints, self.times, axis=0, edge_order=2)
        self.axle_velocities = np.gradient(self.axles, self.times, axis=0, edge_order=2)

    def sample(self, time_s):
        if self.case == "standing":
            source_t, phase = 0., "standing"
        else:
            unclipped = float(time_s) - self.config["settle_seconds"]
            source_t = float(np.clip(unclipped, 0., self.clip_duration))
            phase = "settle" if unclipped < 0 else "recovery" if unclipped > self.clip_duration else "motion"
        def interp(values):
            return np.array([np.interp(source_t, self.times, values[:, i]) for i in range(values.shape[1])])
        moving = phase == "motion"
        return {"joint_pos": interp(self.joints),
                "joint_vel": interp(self.velocities) if moving else np.zeros(8),
                "root_pos": interp(self.root_pos),
                "root_quat_wxyz": self.rotation(source_t).as_quat()[[3, 0, 1, 2]],
                "axle_pos": interp(self.axles),
                "axle_vel": interp(self.axle_velocities) if moving else np.zeros(3),
                "source_time_s": source_t, "phase": phase}


class SharedController:
    def __init__(self, reference, config):
        self.reference, self.config = reference, config
        params = json.loads((ROOT / "config/balance_wf.json").read_text())["model"]
        self.K = design_lqr(WIPParameters(**params), config["physics_dt"], [20., 10., 500., 20.], .1).K
        self.previous = None

    def compute(self, time_s, state):
        joints = np.asarray(state["joint_pos"], dtype=float)
        axle = np.asarray(state["axle_pos"], dtype=float)
        delta = np.asarray(state["body_com"], dtype=float) - axle
        theta = float(np.arctan2(delta[0], delta[2]))
        velocity, axle_vx, theta_dot = np.zeros(8), 0., 0.
        if self.previous is not None:
            old_time, old_joints, old_axle, old_theta = self.previous
            elapsed = time_s - old_time
            if elapsed <= 0:
                raise ValueError("Controller time must increase")
            velocity = (joints - old_joints) / elapsed
            axle_vx = (axle[0] - old_axle[0]) / elapsed
            theta_dot = np.arctan2(np.sin(theta - old_theta), np.cos(theta - old_theta)) / elapsed
        self.previous = (time_s, joints.copy(), axle.copy(), theta)
        target = self.reference.sample(time_s)
        balance = np.array([axle[0] - target["axle_pos"][0], axle_vx - target["axle_vel"][0], theta, theta_dot])
        raw = self.config["leg_kp"] * (target["joint_pos"] - joints) + self.config["leg_kd"] * (target["joint_vel"] - velocity)
        raw[WHEELS] = -float((self.K @ balance).item()) / 2
        bounds = np.full(8, self.config["leg_torque_limit_nm"])
        bounds[WHEELS] = self.config["wheel_torque_limit_nm"]
        torque = np.clip(raw, -bounds, bounds)
        rotation = Rotation.from_quat(np.asarray(state["root_quat_wxyz"])[[1, 2, 3, 0]]).as_matrix()
        tilt = float(np.arccos(np.clip(rotation[2, 2], -1., 1.)))
        return {"torque": torque, "raw_torque": raw, "joint_vel": velocity,
                "joint_ref": target["joint_pos"], "root_ref": target["root_pos"],
                "balance_state": balance, "base_tilt_rad": tilt,
                "source_time_s": target["source_time_s"], "phase": target["phase"]}


def stop_reason(state, control, config):
    flat = np.r_[state["joint_pos"], state["root_pos"], state["root_quat_wxyz"], control["torque"]]
    if not np.isfinite(flat).all():
        return "nonfinite_state"
    if control["base_tilt_rad"] > np.deg2rad(config["fall_tilt_deg"]):
        return "fall_excessive_tilt"
    if state["root_pos"][2] < config["fall_root_height_m"]:
        return "fall_low_base"
    if np.linalg.norm(state["root_pos"][:2]) > config["max_horizontal_distance_m"]:
        return "left_test_area"
    return None


def save_run(output, engine, reference, config, samples, termination, extra=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not samples:
        raise ValueError("No simulation samples")
    arrays = {key: np.asarray([sample[key] for sample in samples]) for key in samples[0]}
    arrays["joint_names"] = np.array(JOINT_NAMES)
    np.savez_compressed(output / "rollout.npz", **arrays)
    motion_mask = arrays["phase"] == "motion"
    evaluation_mask = motion_mask if motion_mask.any() else np.ones(len(samples), dtype=bool)
    joint_error = arrays["joint_pos"][:, LEGS] - arrays["joint_ref"][:, LEGS]
    root_error = np.linalg.norm(arrays["root_pos"] - arrays["root_ref"], axis=1)
    bounds = np.full(8, config["leg_torque_limit_nm"])
    bounds[WHEELS] = config["wheel_torque_limit_nm"]
    # Airborne is a geometric observation here, not a contact-force measurement.
    elevated = arrays["axle_pos"][:, 2] > .127 + .02
    runs = np.diff(np.r_[False, elevated, False].astype(int))
    starts, ends = np.flatnonzero(runs == 1), np.flatnonzero(runs == -1)
    max_airtime = max(((end - start) * config["physics_dt"] for start, end in zip(starts, ends)), default=0.)
    report = {"status": "completed" if termination == "completed" else "terminated",
              "termination": termination, "engine": engine, "case": reference.case,
              "reference": str(reference.path),
              "reference_sha256": hashlib.sha256(reference.path.read_bytes()).hexdigest(),
              "controller_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "config": config, "physics_stepped": True, "root_prescribed_after_initialization": False,
              "trained_policy": False, "beyondmimic_training": False,
              "sample_count": len(samples), "simulated_seconds": float(arrays["time_s"][-1]),
              "expected_seconds": reference.duration,
              "leg_tracking_rmse_rad": float(np.sqrt(np.mean(joint_error[evaluation_mask] ** 2))),
              "leg_tracking_max_error_rad": float(np.max(np.abs(joint_error[evaluation_mask]))),
              "root_tracking_rmse_m": float(np.sqrt(np.mean(root_error[evaluation_mask] ** 2))),
              "max_base_tilt_deg": float(np.rad2deg(arrays["base_tilt_rad"].max())),
              "base_rise_from_initial_m": float(arrays["root_pos"][:, 2].max() - arrays["root_pos"][0, 2]),
              "max_axle_height_m": float(arrays["axle_pos"][:, 2].max()),
              "longest_axle_elevated_interval_s": float(max_airtime),
              "airborne_note": "Mean wheel axle > radius+2cm; not proof of simultaneous loss of both contacts.",
              "actuator_saturation_fraction": float(np.mean(np.abs(arrays["raw_torque"]) > bounds)),
              "max_torque_nm_by_joint": np.max(np.abs(arrays["torque"]), axis=0).tolist(),
              "warnings": config["known_limitations"]}
    if extra:
        report.update(extra)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
