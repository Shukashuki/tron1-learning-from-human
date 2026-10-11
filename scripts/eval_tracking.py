"""Evaluate one complete deterministic TRON1 tracking episode per environment.

Run with the IsaacLab Python environment.  This script captures terminal robot
state *before* IsaacLab's automatic reset, and never includes a second episode
in the metrics.  Contact-derived flight is reported separately from survival.
The saved TorchScript actor includes its learned observation normalization.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physics_clock_readback(env, cfg, contract):
    """Verify the live simulator and environment clocks before any rollout."""
    from training.tron1_physics_clock import validate_physics_clock
    expected_dt, expected_decimation = validate_physics_clock(contract)
    values = {
        "simulator_physics_dt_s": float(env.sim.get_physics_dt()),
        "environment_physics_dt_s": float(env.physics_dt),
        "policy_dt_s": float(env.step_dt),
        "net_contact_sensor_update_period_s": float(env.scene.sensors["contact_forces"].cfg.update_period),
    }
    if (any(not math.isfinite(value) for value in values.values())
            or type(cfg.decimation) is not int or cfg.decimation != expected_decimation
            or not all(math.isclose(values[key], expected_dt, abs_tol=1e-12, rel_tol=0)
                       for key in ("simulator_physics_dt_s", "environment_physics_dt_s",
                                   "net_contact_sensor_update_period_s"))
            or not math.isclose(values["policy_dt_s"], .02, abs_tol=1e-12, rel_tol=0)):
        raise RuntimeError("Live evaluation physics clock differs from its declared 50 Hz deployment contract")
    return {"status": "verified", "decimation": expected_decimation, **values}


class PhysicsTrajectoryRecorder:
    """First-episode physical states; independent of policy-rate recording.

    Scene-update callbacks may also occur during forward/reset operations.
    Only consecutive simulator-counter increments during a policy step are
    accepted. Observations/actions are the actual policy context held through
    its decimation interval; the recorder never computes new observations.
    """

    def __init__(self, num_envs, physics_dt, start_counter, initial_snapshot, initial_observation):
        self.num_envs = int(num_envs)
        self.physics_dt = float(physics_dt)
        if self.num_envs < 1 or not np.isfinite(self.physics_dt) or self.physics_dt <= 0:
            raise ValueError("Physics recorder needs positive environment count and timestep")
        self.start_counter = self.last_counter = int(start_counter)
        self.in_step = False
        self.valid = np.ones(self.num_envs, dtype=bool)
        self.records = {key: [] for key in initial_snapshot}
        self.records.update({key: [] for key in ("valid_mask", "terminal_sample", "action_observation",
                                                "actions_raw", "actions_clipped", "reference_frame_for_action",
                                                "action_was_applied")})
        self.counters = []
        self.observation = self._context(initial_observation, (self.num_envs, 51), "observation")
        self.raw_actions = np.zeros((self.num_envs, 8), dtype=np.float32)
        self.clipped_actions = np.zeros_like(self.raw_actions)
        self.action_frames = np.asarray(initial_snapshot["reference_frame"]).copy()
        self._append(initial_snapshot, applied=False)

    @staticmethod
    def _context(value, shape, label):
        result = np.asarray(value).copy()
        if result.shape != shape or not np.isfinite(result).all():
            raise ValueError(f"Physics recording {label} must have finite shape {shape}")
        return result

    def _masked(self, value):
        result = np.asarray(value).copy()
        if result.shape[0] != self.num_envs:
            raise ValueError("Physical sample has an inconsistent environment count")
        if result.dtype.kind == "b":
            result[~self.valid] = False
        elif np.issubdtype(result.dtype, np.integer):
            result[~self.valid] = -1
        else:
            result[~self.valid] = np.nan
        return result

    def _append(self, snapshot, applied):
        for key, value in snapshot.items():
            if key not in self.records:
                raise ValueError("Physical snapshot schema changed while recording")
            self.records[key].append(self._masked(value))
        self.records["valid_mask"].append(self.valid.copy())
        self.records["terminal_sample"].append(np.zeros(self.num_envs, dtype=bool))
        self.records["action_observation"].append(self._masked(self.observation))
        self.records["actions_raw"].append(self._masked(self.raw_actions))
        self.records["actions_clipped"].append(self._masked(self.clipped_actions))
        self.records["reference_frame_for_action"].append(self._masked(self.action_frames))
        self.records["action_was_applied"].append(self.valid.copy() if applied else np.zeros(self.num_envs, dtype=bool))
        self.counters.append(self.last_counter)

    def begin_step(self, valid, observation, raw_actions, clipped_actions, action_frames):
        if self.in_step:
            raise RuntimeError("Physics capture policy steps must not overlap")
        valid = np.asarray(valid)
        if valid.shape != (self.num_envs,) or valid.dtype.kind != "b":
            raise ValueError("Physics capture valid mask must be one boolean per environment")
        if np.any(valid & ~self.valid):
            raise ValueError("A completed first episode cannot become valid again")
        self.valid = valid.copy()
        self.observation = self._context(observation, (self.num_envs, 51), "observation")
        self.raw_actions = self._context(raw_actions, (self.num_envs, 8), "raw actions")
        self.clipped_actions = self._context(clipped_actions, (self.num_envs, 8), "clipped actions")
        self.action_frames = self._context(action_frames, (self.num_envs,), "action frames")
        self.step_start_counter = self.last_counter
        self.in_step = True

    def should_capture(self, counter):
        counter = int(counter)
        if not self.in_step or counter == self.last_counter:
            return False
        if counter != self.last_counter + 1:
            raise RuntimeError("Physical simulation counter skipped or moved backwards during capture")
        return True

    def capture(self, counter, snapshot):
        if not self.should_capture(counter):
            return False
        self.last_counter = int(counter)
        self._append(snapshot, applied=True)
        return True

    def finish_step(self, newly_done, expected_substeps):
        self.in_step = False
        if self.last_counter - self.step_start_counter != int(expected_substeps):
            raise RuntimeError("Physics recording missed a decimation substep")
        done = np.asarray(newly_done)
        if done.shape != (self.num_envs,) or done.dtype.kind != "b" or np.any(done & ~self.valid):
            raise ValueError("Physical terminal flags must belong to active first episodes")
        self.records["terminal_sample"][-1] |= done
        self.valid[done] = False

    def arrays(self, joint_names, body_names, wheel_body_names, policy_dt):
        arrays = {key: np.stack(value) for key, value in self.records.items()}
        arrays.update({
            "time_s": (np.asarray(self.counters, dtype=np.int64) - self.start_counter) * self.physics_dt,
            "simulation_step_counter": np.asarray(self.counters, dtype=np.int64),
            "policy_dt_s": np.array(policy_dt), "physics_dt_s": np.array(self.physics_dt),
            "joint_names": np.asarray(joint_names), "qpos_joint_names": np.asarray(joint_names),
            "body_names": np.asarray(body_names), "wheel_body_names": np.asarray(wheel_body_names),
            "action_observation_kind": np.asarray("actual policy pre-action observation held across decimation substeps"),
        })
        arrays["qpos"] = np.concatenate((arrays["root_position_m"], arrays["root_quaternion_wxyz"],
                                        arrays["joint_position_rad"]), axis=-1)
        return arrays


def true_runs(mask):
    """Half-open intervals of contiguous True samples."""
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    changes = np.diff(padded)
    return list(zip(np.flatnonzero(changes == 1).tolist(), np.flatnonzero(changes == -1).tolist()))


def contact_events(times, forces, threshold_n=5.0, support_s=0.1, flight_s=0.04):
    """Conservative force-based detection; initial reset contact absence is ignored.

    A flight needs prior sustained two-wheel support.  Landing needs subsequent
    sustained two-wheel support.  These are sampled contact events, not proof of
    dynamically stable recovery, and a no-contact segment alone is not a jump.
    """
    times, forces = np.asarray(times), np.asarray(forces)
    unavailable = {
        "available": False, "reason": "wheel contact forces were unavailable",
        "flight_intervals": [], "first_takeoff_s": None, "first_landing_s": None,
        "two_wheel_contact_fraction": None, "max_flight_duration_s": None,
    }
    if len(times) < 2 or forces.shape != (len(times), 2, 3) or not np.isfinite(forces).all():
        return unavailable
    dt = float(np.median(np.diff(times)))
    contact = np.linalg.norm(forces, axis=-1) >= threshold_n
    both_contact = np.all(contact, axis=-1)
    neither_contact = ~np.any(contact, axis=-1)
    support_count = max(2, int(math.ceil(support_s / dt - 1e-8)) + 1)
    flight_count = max(2, int(math.ceil(flight_s / dt - 1e-8)) + 1)
    supports = [(a, b) for a, b in true_runs(both_contact) if b - a >= support_count]
    intervals = []
    for start, stop in true_runs(neither_contact):
        if stop - start < flight_count or not any(end <= start for _, end in supports):
            continue
        landing = next((a for a, _ in supports if a >= stop), None)
        # A final airborne interval is right-censored: its observed duration is
        # a lower bound and there is no measured landing.
        end_time = float(times[stop]) if stop < len(times) else float(times[-1])
        intervals.append({
            "takeoff_s": float(times[start]),
            "no_contact_end_s": end_time,
            "no_contact_duration_s": end_time - float(times[start]),
            "landing_s": None if landing is None else float(times[landing]),
            "right_censored": stop == len(times),
        })
    return {
        "available": True, "force_threshold_n": threshold_n,
        "sampling_interval_s": dt, "minimum_prior_support_s": support_s,
        "minimum_flight_s": flight_s, "minimum_landing_support_s": support_s,
        "two_wheel_contact_fraction": float(np.mean(both_contact)),
        "left_contact_fraction": float(np.mean(contact[:, 0])),
        "right_contact_fraction": float(np.mean(contact[:, 1])),
        "flight_intervals": intervals,
        "first_takeoff_s": intervals[0]["takeoff_s"] if intervals else None,
        "first_landing_s": next((x["landing_s"] for x in intervals if x["landing_s"] is not None), None),
        "max_flight_duration_s": max((x["no_contact_duration_s"] for x in intervals), default=0.0),
        "reason": None if intervals else "no qualifying flight after sustained two-wheel support was detected",
    }


def summarize_episodes(arrays, outcomes, reference, contact_threshold_n):
    """Pure NumPy metrics over valid first-episode samples, including terminals."""
    dt = float(arrays["policy_dt_s"])
    reports = []
    ref_root_z = reference["body_pos_w"][:, 0, 2]
    for env_index, outcome in enumerate(outcomes):
        valid = arrays["valid_mask"][:, env_index]
        times = arrays["time_s"][valid]
        root = arrays["root_position_m"][valid, env_index]
        quat = arrays["root_quaternion_wxyz"][valid, env_index]
        ref_ids = arrays["reference_frame"][valid, env_index]
        if not len(root):
            raise RuntimeError(f"No first-episode state for environment {env_index}")
        quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
        cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        tilt = np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0)))
        ref_z = ref_root_z[np.clip(ref_ids, 0, len(ref_root_z) - 1)]
        events = contact_events(times, arrays["wheel_contact_force_w_n"][valid, env_index], contact_threshold_n)
        gain = float(np.max(root[:, 2]) - root[0, 2])
        full_clip = bool(outcome["timed_out"] and not outcome["early_terminated"]
                         and int(ref_ids[-1]) >= len(ref_root_z) - 1)
        landed_jump = bool(events["available"] and events["first_landing_s"] is not None and gain >= 0.05)
        reports.append({
            "environment": env_index, **outcome, "completed_full_reference": full_clip,
            "recorded_samples": len(times), "recorded_duration_s": float(times[-1]),
            "start_reference_frame": int(ref_ids[0]), "final_reference_frame": int(ref_ids[-1]),
            "base_link_initial_height_m": float(root[0, 2]),
            "base_link_peak_height_m": float(np.max(root[:, 2])), "base_link_height_gain_m": gain,
            "base_link_height_gain_from_lowest_pose_m": float(np.ptp(root[:, 2])),
            "reference_base_link_height_gain_m": float(np.max(ref_root_z) - ref_root_z[0]),
            "reference_height_rmse_m": float(np.sqrt(np.mean((root[:, 2] - ref_z) ** 2))),
            "max_base_tilt_deg": float(np.max(tilt)), "final_base_tilt_deg": float(tilt[-1]),
            "max_horizontal_displacement_m": float(np.linalg.norm(root[:, :2] - root[0, :2], axis=-1).max()),
            "contact": events,
            "jump_and_landing_detected": landed_jump,
            "full_clip_with_detected_jump_and_landing": bool(full_clip and landed_jump),
        })
    completed = [item["completed_full_reference"] for item in reports]
    return {
        "num_environments": len(reports),
        "full_reference_survival_rate": float(np.mean(completed)),
        "early_termination_rate": float(np.mean([item["early_terminated"] for item in reports])),
        "detected_jump_and_landing_rate": float(np.mean([item["jump_and_landing_detected"] for item in reports])),
        "full_clip_with_detected_jump_and_landing_rate": float(np.mean([
            item["full_clip_with_detected_jump_and_landing"] for item in reports
        ])),
        "mean_base_link_height_gain_m": float(np.mean([item["base_link_height_gain_m"] for item in reports])),
        "maximum_base_link_height_gain_m": float(max(item["base_link_height_gain_m"] for item in reports)),
        "mean_reference_height_rmse_m": float(np.mean([item["reference_height_rmse_m"] for item in reports])),
        "maximum_base_tilt_deg": float(max(item["max_base_tilt_deg"] for item in reports)),
        "policy_sampling_interval_s": dt, "episodes": reports,
    }


def main():
    # AppLauncher is the sole Isaac import before the application starts.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", required=True, type=Path)
    parser.add_argument("--asset-path", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--runner-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-envs", default=16, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--terrain-file", type=Path, help="Must match the terrain used for training")
    parser.add_argument("--task-profile", type=Path, help="Must match the checkpoint drift task/material contract")
    gain_diagnostics = parser.add_mutually_exclusive_group()
    gain_diagnostics.add_argument("--deployment-leg-kp", type=float, default=None,
                                 help="Explicit zero-shot common leg-Kp diagnostic (100–500); original profile must match training")
    gain_diagnostics.add_argument("--deployment-leg-kd", type=float, default=None,
                                 help="Explicit zero-shot common leg-Kd diagnostic (5–13); original profile must match training")
    parser.add_argument("--verify-unmasked-device", action="store_true",
                        help="Opt-in strict native PhysX/Torch/tensor CUDA-ordinal readback audit; requires no CUDA_VISIBLE_DEVICES mask")
    parser.add_argument("--contact-threshold-n", default=5.0, type=float)
    parser.add_argument("--record-physics", action="store_true",
                        help="Also save every physics substep in a separate first-episode physics_trajectory.npz")
    parser.add_argument("--physics-dt-s", type=float, choices=(.005, .0025, .00125), default=None,
                        help="Explicit evaluation-only numerical refinement; policy remains 50 Hz and training/task profile unchanged")
    parser.add_argument("--domain-parameters", type=Path,
                        help="One explicit fixed actuator-domain draw JSON; never resample during evaluation")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    deployment_leg_kp = getattr(args, "deployment_leg_kp", None)
    deployment_leg_kd = getattr(args, "deployment_leg_kd", None)
    if deployment_leg_kp is not None:
        if not math.isfinite(deployment_leg_kp) or not 100. <= deployment_leg_kp <= 500.:
            parser.error("deployment-leg-kp must be finite and between 100 and 500")
        if args.physics_dt_s is not None or args.domain_parameters is not None:
            parser.error("The single-factor leg-Kp diagnostic cannot combine physics refinement or actuator-domain overrides")
    if deployment_leg_kd is not None:
        if not math.isfinite(deployment_leg_kd) or not 5. <= deployment_leg_kd <= 13.:
            parser.error("deployment-leg-kd must be finite and between 5 and 13")
        if deployment_leg_kp is not None or args.physics_dt_s is not None or args.domain_parameters is not None:
            parser.error("The single-factor leg-Kd diagnostic cannot combine Kp, physics refinement or actuator-domain overrides")
    if args.num_envs < 1 or args.contact_threshold_n <= 0:
        parser.error("num-envs and contact-threshold-n must be positive")
    for name in ("motion_file", "asset_path", "checkpoint", "runner_config"):
        value = getattr(args, name).resolve()
        if not value.is_file():
            parser.error(f"{name} does not exist: {value}")
        setattr(args, name, value)
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Use an empty output directory; refusing to replace existing results: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    launcher = AppLauncher(args)
    app = launcher.app
    wrapped = None
    reward_accumulator = None
    original_scene_update = None
    physics_scene_update = None
    started = time.monotonic()
    try:
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit
        from rsl_rl.runners import OnPolicyRunner

        sys.path.insert(0, str(ROOT))
        from training.tron1_tracking import make_env_cfg, observation_action_contract, WHEEL_BODY_NAMES
        from training.tron1_domain_randomization import (
            validate_draw, apply_draw_to_contract, domain_randomization_audit,
        )
        domain_parameters = (validate_draw(json.loads(args.domain_parameters.read_text()))
                             if args.domain_parameters else None)
        from training.tron1_drift import load_task_profile
        task_profile = load_task_profile(args.task_profile) if args.task_profile else None
        controller_gain_diagnostic = None
        terrain_spec = None
        if args.terrain_file:
            from training.tron1_terrain import load_terrain
            terrain_spec = load_terrain(args.terrain_file)
        training_manifest_path = args.checkpoint.parent / "manifest.json"
        trained_manifest = None
        if training_manifest_path.is_file():
            trained_manifest = json.loads(training_manifest_path.read_text())
            trained_terrain = trained_manifest.get("terrain")
            if trained_terrain != terrain_spec:
                raise ValueError("Evaluation terrain differs from the checkpoint training manifest")
            if trained_manifest.get("task_profile") != task_profile:
                raise ValueError("Evaluation drift task/material profile differs from training")
        if deployment_leg_kp is not None:
            if trained_manifest is None:
                raise ValueError("The zero-shot leg-Kp diagnostic requires the original checkpoint training manifest")
            if trained_manifest.get("asset_sha256") != sha256(args.asset_path):
                raise ValueError("The zero-shot leg-Kp diagnostic must retain the trained robot asset")
            from training.tron1_controller_gain_diagnostic import (
                prepare_deployment_gain_diagnostic, apply_deployment_gain_diagnostic_to_contract,
                validate_deployment_gain_diagnostic,
            )
            from training.tron1_drift import apply_task_profile_to_contract
            nominal_contract = apply_task_profile_to_contract(observation_action_contract(), task_profile)
            task_profile, controller_gain_diagnostic = prepare_deployment_gain_diagnostic(
                task_profile, nominal_contract, trained_manifest, deployment_leg_kp,
                reference_sha256=sha256(args.motion_file))
            controller_gain_diagnostic.update(
                original_manifest_sha256=sha256(training_manifest_path), checkpoint_sha256=sha256(args.checkpoint))
            validate_deployment_gain_diagnostic(
                apply_deployment_gain_diagnostic_to_contract(nominal_contract, controller_gain_diagnostic),
                trained_manifest, original_manifest_sha256=sha256(training_manifest_path),
                checkpoint_sha256=sha256(args.checkpoint))
        if deployment_leg_kd is not None:
            if trained_manifest is None:
                raise ValueError("The zero-shot leg-Kd diagnostic requires the original checkpoint training manifest")
            if trained_manifest.get("asset_sha256") != sha256(args.asset_path):
                raise ValueError("The zero-shot leg-Kd diagnostic must retain the trained robot asset")
            from training.tron1_controller_damping_diagnostic import (
                prepare_deployment_gain_diagnostic, apply_deployment_gain_diagnostic_to_contract,
                validate_deployment_gain_diagnostic,
            )
            from training.tron1_drift import apply_task_profile_to_contract
            nominal_contract = apply_task_profile_to_contract(observation_action_contract(), task_profile)
            task_profile, controller_gain_diagnostic = prepare_deployment_gain_diagnostic(
                task_profile, nominal_contract, trained_manifest, deployment_leg_kd,
                reference_sha256=sha256(args.motion_file))
            controller_gain_diagnostic.update(
                original_manifest_sha256=sha256(training_manifest_path), checkpoint_sha256=sha256(args.checkpoint))
            validate_deployment_gain_diagnostic(
                apply_deployment_gain_diagnostic_to_contract(nominal_contract, controller_gain_diagnostic),
                trained_manifest, original_manifest_sha256=sha256(training_manifest_path),
                checkpoint_sha256=sha256(args.checkpoint))
        if args.physics_dt_s is not None:
            if trained_manifest is None or not isinstance(trained_manifest.get("contract"), dict):
                raise ValueError("Explicit physics refinement requires the checkpoint training manifest and its nominal contract")
            from training.tron1_physics_clock import apply_physics_clock_to_contract
            nominal_contract = observation_action_contract()
            if task_profile is not None:
                from training.tron1_drift import apply_task_profile_to_contract
                nominal_contract = apply_task_profile_to_contract(nominal_contract, task_profile)
            refined_contract = apply_physics_clock_to_contract(
                nominal_contract, args.physics_dt_s, trained_contract=trained_manifest["contract"])

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy().copy()

        def state_snapshot(env):
            robot = env.scene["robot"]
            command = env.command_manager.get_term("motion")
            sensor = env.scene.sensors.get("contact_forces")
            if sensor is not None and env.eval_wheel_sensor_ids is not None:
                forces = to_numpy(sensor.data.net_forces_w[:, env.eval_wheel_sensor_ids])
            else:
                forces = np.full((env.num_envs, 2, 3), np.nan, dtype=np.float32)
            snapshot = {
                "root_position_m": to_numpy(robot.data.root_pos_w - env.scene.env_origins),
                "root_quaternion_wxyz": to_numpy(robot.data.root_quat_w),
                "root_linear_velocity_w_m_s": to_numpy(robot.data.root_lin_vel_w),
                "root_angular_velocity_w_rad_s": to_numpy(robot.data.root_ang_vel_w),
                "joint_position_rad": to_numpy(robot.data.joint_pos),
                "joint_velocity_rad_s": to_numpy(robot.data.joint_vel),
                "applied_joint_torque_nm": to_numpy(robot.data.applied_torque),
                "wheel_position_m": to_numpy(robot.data.body_pos_w[:, env.eval_wheel_body_ids]
                                             - env.scene.env_origins[:, None, :]),
                "wheel_contact_force_w_n": forces,
                "reference_frame": to_numpy(command.time_steps),
            }
            if task_profile is not None:
                from training.tron1_contact_slip import sample_raw_isaac_ground_contacts, sample_nonwheel_ground_force_norm
                snapshot.update(sample_raw_isaac_ground_contacts(env, force_threshold_n=args.contact_threshold_n))
                names = tuple(name for name in env.scene.sensors if name.startswith("nonwheel_ground_"))
                snapshot.update(sample_nonwheel_ground_force_norm(env, names, force_threshold_n=1.))
            return snapshot

        class TerminalCaptureEnv(ManagerBasedRLEnv):
            def __init__(self, *positional, **keyword):
                self.capture_resets = False
                self.terminal_snapshots = {}
                self.eval_wheel_sensor_ids = None
                super().__init__(*positional, **keyword)

            def _reset_idx(self, env_ids):
                if self.capture_resets and len(env_ids):
                    snapshot = state_snapshot(self)
                    terms = {
                        name: to_numpy(self.termination_manager.get_term(name))
                        for name in self.termination_manager.active_terms
                    }
                    terminated = to_numpy(self.termination_manager.terminated)
                    timed_out = to_numpy(self.termination_manager.time_outs)
                    for env_index in to_numpy(torch.as_tensor(env_ids)).tolist():
                        self.terminal_snapshots[int(env_index)] = {
                            "state": {key: value[env_index].copy() for key, value in snapshot.items()},
                            "early_terminated": bool(terminated[env_index]),
                            "timed_out": bool(timed_out[env_index]),
                            "termination_terms": [name for name, value in terms.items() if value[env_index]],
                        }
                super()._reset_idx(env_ids)

        cfg = make_env_cfg(args.motion_file, args.asset_path, args.num_envs, args.device, eval_mode=True,
                           dr_profile={"mode": "fixed", "fixed_draw": domain_parameters} if domain_parameters else None,
                           terrain_spec=terrain_spec, task_profile=task_profile)
        if args.physics_dt_s is not None:
            from training.tron1_physics_clock import configure_physics_clock
            configure_physics_clock(cfg, args.physics_dt_s)
        cfg.seed = args.seed
        env = TerminalCaptureEnv(cfg=cfg)
        runtime_device_audit = None
        if getattr(args, "verify_unmasked_device", False):
            from training.tron1_runtime_device import validate_simulation_device
            runtime_device_audit = validate_simulation_device(env, requested_device=args.device)
        clock_readback = (physics_clock_readback(env, cfg, refined_contract)
                          if args.physics_dt_s is not None else None)
        contact_material_audit = None
        if task_profile is not None:
            from training.tron1_drift import audit_isaac_materials
            contact_material_audit = audit_isaac_materials(env, task_profile)
        terrain_audit = None
        if terrain_spec is not None:
            from training.tron1_terrain import terrain_runtime_audit
            terrain_audit = terrain_runtime_audit(env, terrain_spec)
        robot = env.scene["robot"]
        env.eval_wheel_body_ids = [robot.body_names.index(name) for name in WHEEL_BODY_NAMES]
        sensor = env.scene.sensors.get("contact_forces")
        if sensor is not None:
            env.eval_wheel_sensor_ids = [sensor.body_names.index(name) for name in WHEEL_BODY_NAMES]
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        runner_cfg = json.loads(args.runner_config.read_text())
        # OnPolicyRunner mutates nested class_name fields, hence use a copy.
        runner = OnPolicyRunner(wrapped, copy.deepcopy(runner_cfg), log_dir=None, device=args.device)
        runner.load(str(args.checkpoint), load_optimizer=False, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        if getattr(runner.alg.policy, "is_recurrent", False):
            raise NotImplementedError("This one-clip evaluation contract currently supports the trained feedforward actor")

        # The task's reset lifecycle refreshes relative targets without
        # advancing phase; verify that evaluation really starts at frame0.
        command = env.command_manager.get_term("motion")
        obs = wrapped.get_observations()
        if not torch.all(command.time_steps == 0):
            raise AssertionError("Deterministic evaluation did not begin at reference frame 0")
        if tuple(obs["policy"].shape) != (args.num_envs, 51):
            raise AssertionError(f"Unexpected actor observations: {obs['policy'].shape}")
        with np.load(args.motion_file, allow_pickle=False) as archive:
            reference = {key: archive[key].copy() for key in archive.files}
        frames = len(reference["joint_pos"])

        actor = runner.alg.policy
        normalizer = getattr(actor, "actor_obs_normalizer", None)
        if normalizer is None:
            raise RuntimeError("RSL3 actor observation normalizer is missing; refusing incomplete policy export")
        export_policy_as_jit(actor, normalizer=normalizer, path=str(args.output_dir), filename="actor_normalized.pt")
        exported = torch.jit.load(str(args.output_dir / "actor_normalized.pt"), map_location="cpu").eval()
        with torch.inference_mode():
            actor_input = actor.get_actor_obs(obs)
            export_error = float(torch.max(torch.abs(exported(actor_input.cpu()) - policy(obs).cpu())))
        if export_error > 1e-4:
            raise RuntimeError(f"Exported actor differs from runner by {export_error:g}")

        active = np.ones(args.num_envs, dtype=bool)
        outcomes = [
            {"early_terminated": False, "timed_out": False, "termination_terms": [], "end_reason": "not_finished"}
            for _ in range(args.num_envs)
        ]
        initial = state_snapshot(env)
        records = {key: [value] for key, value in initial.items()}
        records.update({
            "valid_mask": [active.copy()], "terminal_sample": [np.zeros(args.num_envs, dtype=bool)],
            "actions_raw": [np.zeros((args.num_envs, 8), dtype=np.float32)],
            "actions_clipped": [np.zeros((args.num_envs, 8), dtype=np.float32)],
            "action_observation": [to_numpy(obs["policy"])],
            "reference_frame_for_action": [to_numpy(command.time_steps)],
            "action_was_applied": [np.zeros(args.num_envs, dtype=bool)],
        })
        env.capture_resets = True
        physics_recorder = None
        original_scene_update = None
        if args.record_physics:
            physics_recorder = PhysicsTrajectoryRecorder(
                args.num_envs, env.physics_dt, env._sim_step_counter, initial, to_numpy(obs["policy"]),
            )
            original_scene_update = env.scene.update

            def physics_scene_update(*positional, **keyword):
                result = original_scene_update(*positional, **keyword)
                counter = env._sim_step_counter
                # Check before reading state: a same-counter reset/forward
                # update must not even fetch a post-reset terminal snapshot.
                if physics_recorder.should_capture(counter):
                    physics_recorder.capture(counter, state_snapshot(env))
                return result

            env.scene.update = physics_scene_update
        # Install AFTER the physical recorder so both observers chain once.
        # Sampling costs do not modify physics or add actor/observation calls.
        if task_profile is not None:
            from training.tron1_drift import install_drift_reward_sampling
            reward_accumulator = install_drift_reward_sampling(env, task_profile)
        export_validation_max_error = export_error
        max_steps = frames + int(math.ceil(0.2 / env.step_dt))
        with torch.inference_mode():
            for step_index in range(max_steps):
                if not active.any() or not app.is_running():
                    break
                valid = active.copy()
                policy_obs = to_numpy(obs["policy"])
                reference_for_action = to_numpy(command.time_steps)
                raw_actions = policy(obs)
                if not torch.isfinite(raw_actions).all():
                    raise FloatingPointError("Policy produced a nonfinite action")
                # Validate normalization and export on evolving observations,
                # not only a possibly all-zero reset sample.
                if step_index % 20 == 0:
                    check = exported(actor.get_actor_obs(obs).cpu())
                    error = float(torch.max(torch.abs(check - raw_actions.cpu())))
                    export_validation_max_error = max(export_validation_max_error, error)
                clipped_actions = torch.clamp(raw_actions, -1.0, 1.0)
                # Finished environments continue mechanically because Isaac is
                # vectorized, but cannot contribute a second episode's states.
                clipped_actions[torch.as_tensor(~active, device=env.device)] = 0.0
                env.terminal_snapshots.clear()
                if physics_recorder is not None:
                    physics_recorder.begin_step(valid, policy_obs, to_numpy(raw_actions),
                                                to_numpy(clipped_actions), reference_for_action)
                obs, rewards, dones, extras = wrapped.step(clipped_actions)
                del rewards, extras
                done = to_numpy(dones).astype(bool)
                newly_done = done & active
                if physics_recorder is not None:
                    physics_recorder.finish_step(newly_done, expected_substeps=cfg.decimation)
                state = state_snapshot(env)
                for env_index in np.flatnonzero(newly_done):
                    if int(env_index) not in env.terminal_snapshots:
                        raise RuntimeError("Missing pre-reset terminal snapshot; refusing contaminated evaluation")
                    terminal = env.terminal_snapshots[int(env_index)]
                    for key, value in terminal["state"].items():
                        state[key][env_index] = value
                    outcomes[env_index] = {
                        "early_terminated": terminal["early_terminated"],
                        "timed_out": terminal["timed_out"],
                        "termination_terms": terminal["termination_terms"],
                        "end_reason": "early_termination" if terminal["early_terminated"] else "timeout",
                    }
                for key, value in state.items():
                    # Invalid rows are explicit NaN/-1, not silently replayed
                    # reset trajectories; valid_mask is authoritative.
                    value = value.copy()
                    value[~valid] = -1 if np.issubdtype(value.dtype, np.integer) else np.nan
                    records[key].append(value)
                records["valid_mask"].append(valid)
                records["terminal_sample"].append(newly_done)
                records["actions_raw"].append(to_numpy(raw_actions))
                records["actions_clipped"].append(to_numpy(clipped_actions))
                records["action_observation"].append(policy_obs)
                records["reference_frame_for_action"].append(reference_for_action)
                records["action_was_applied"].append(valid)
                active[newly_done] = False

        for env_index in np.flatnonzero(active):
            outcomes[env_index]["end_reason"] = "application_interrupted" if not app.is_running() else "evaluation_step_limit"
        arrays = {key: np.stack(value) for key, value in records.items()}
        arrays.update({
            "time_s": np.arange(len(arrays["valid_mask"])) * env.step_dt,
            "policy_dt_s": np.array(env.step_dt), "physics_dt_s": np.array(env.physics_dt),
            "joint_names": np.asarray(robot.joint_names), "body_names": np.asarray(robot.body_names),
            "wheel_body_names": np.asarray(WHEEL_BODY_NAMES),
        })
        arrays["qpos"] = np.concatenate((arrays["root_position_m"], arrays["root_quaternion_wxyz"],
                                        arrays["joint_position_rad"]), axis=-1)
        summary = summarize_episodes(arrays, outcomes, reference, args.contact_threshold_n)
        contract = observation_action_contract()
        if controller_gain_diagnostic is not None:
            contract = apply_deployment_gain_diagnostic_to_contract(
                controller_gain_diagnostic["original_nominal_contract"], controller_gain_diagnostic)
        elif task_profile is not None:
            from training.tron1_drift import apply_task_profile_to_contract
            contract = apply_task_profile_to_contract(contract, task_profile)
        if args.physics_dt_s is not None:
            contract = apply_physics_clock_to_contract(
                contract, args.physics_dt_s, trained_contract=trained_manifest["contract"])
        contract["domain_randomization"] = "disabled during evaluation; fixed actuator parameters"
        if terrain_spec is not None:
            contract["terrain"] = terrain_spec
        if domain_parameters is not None:
            contract = apply_draw_to_contract(contract, domain_parameters)
        training_manifest = args.checkpoint.parent / "manifest.json"
        if training_manifest.is_file():
            metadata = json.loads(training_manifest.read_text())
            contract["training_domain_randomization"] = metadata.get("domain_randomization_profile")
        contract.update({
            "actor_file": "actor_normalized.pt", "actor_includes_observation_normalizer": True,
            "actor_output_is_unclipped": True, "actor_observation_groups": actor.obs_groups["policy"],
            "export_validation_max_abs_error": export_validation_max_error,
            "actual_joint_order": robot.joint_names, "actual_body_order": robot.body_names,
            "reference_file_sha256": sha256(args.motion_file), "checkpoint_sha256": sha256(args.checkpoint),
            "default_joint_position_rad": to_numpy(robot.data.default_joint_pos[0]).tolist(),
        })
        if controller_gain_diagnostic is not None:
            validate_deployment_gain_diagnostic(
                contract, trained_manifest, original_manifest_sha256=sha256(training_manifest_path),
                checkpoint_sha256=sha256(args.checkpoint))
        if export_validation_max_error > 1e-4:
            raise RuntimeError(f"Evolving-observation actor export validation failed: {export_validation_max_error}")
        report = {
            "schema_version": 1, "status": "evaluated" if not active.any() else "evaluation_incomplete",
            "terrain": terrain_spec,
            "terrain_runtime_audit": terrain_audit,
            "terrain_file_sha256": sha256(args.terrain_file) if args.terrain_file else None,
            "terrain_module_sha256": sha256(ROOT / "training/tron1_terrain.py") if terrain_spec is not None else None,
            "legacy_joint_friction_audit": env.legacy_joint_friction_audit,
            "domain_parameters": domain_parameters,
            "task_profile": task_profile,
            "contact_material_audit": contact_material_audit,
            "wheel_contact_slip_evidence": "force-weighted RMS from actual ground-filtered contact points and wheel COM twist" if task_profile else None,
            "wheel_lateral_contact_slip_evidence_kind": "actual_ground_contact_points" if task_profile else None,
            "contact_slip_helper_sha256": sha256(ROOT / "training/tron1_contact_slip.py") if task_profile else None,
            "evaluator_source_sha256": sha256(__file__),
            "drift_module_sha256": sha256(ROOT / "training/tron1_drift.py") if task_profile else None,
            "contact_slip_measurement": {"aggregation": "normal-force-weighted RMS over active actual ground contact points",
                                         "active_point_normal_force_threshold_n": args.contact_threshold_n,
                                         "nonwheel_active_point_threshold_n": 1.,
                                         "ground_is_stationary": True,
                                         "sample_dt_s": env.step_dt} if task_profile else None,
            "domain_parameters_file_sha256": sha256(args.domain_parameters) if args.domain_parameters else None,
            "domain_randomization_audit": domain_randomization_audit(env),
            "simulator": "IsaacLab/PhysX", "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": contract["checkpoint_sha256"],
            "runner_config": str(args.runner_config), "runner_config_sha256": sha256(args.runner_config),
            "motion_file": str(args.motion_file), "motion_file_sha256": contract["reference_file_sha256"],
            "asset_path": str(args.asset_path), "asset_root_file_sha256": sha256(args.asset_path),
            "seed": args.seed, "reference_frames": frames, "reference_sample_span_s": (frames - 1) * env.step_dt,
            "wall_time_seconds": time.monotonic() - started,
            "deterministic_start_frame": 0, "reference_state_initialization_during_evaluation": False,
            "observation_noise_enabled": False, "domain_randomization_enabled": False,
            "terminal_states_captured_before_auto_reset": True, "episodes_per_environment": 1,
            "hardware_ready": False, "summary": summary, "policy_export": contract,
            "limitations": [
                "Surviving the full clip is not by itself evidence of a successful jump.",
                "All environments begin from the same reference frame0 state without random perturbations; this is a deterministic smoke evaluation, not a robustness success probability.",
                "Reported height is the measured base_Link origin, not center-of-mass height, reference height, or wheel clearance.",
                "Contact events use wheel net-force norms sampled at policy frequency; self-contact can also contribute to these net forces.",
                "Flight needs at least 0.10s prior two-wheel support and 0.04s both-wheel contact absence; landing needs 0.10s subsequent two-wheel support.",
                "jump_and_landing_detected additionally requires at least0.05m measured base height gain; it does not certify stable recovery or hardware feasibility.",
                "Torque samples are the last physics substep, not the maximum over all decimation substeps.",
                "qpos arrays use named Isaac joint order, not native MuJoCo joint order; MuJoCo deployment must remap by name.",
                "Root linear velocity follows Isaac's COM-velocity convention; root pose is the link-frame pose.",
                "The exported actor emits raw actions; deployment must clamp all8 actions to[-1,1] before the documented action scaling.",
            ],
        }
        np.savez_compressed(args.output_dir / "trajectory.npz", **arrays)
        if runtime_device_audit is not None:
            report.update({"runtime_device_audit": runtime_device_audit,
                           "runtime_device_module_sha256": sha256(ROOT / "training/tron1_runtime_device.py")})
        if controller_gain_diagnostic is not None:
            report["controller_gain_diagnostic"] = copy.deepcopy(contract["controller_gain_diagnostic"])
        if reward_accumulator is not None:
            report.update({"reward_sampling_audit": reward_accumulator.audit(),
                           "substep_reward_module_sha256": sha256(ROOT / "training/tron1_substep_rewards.py")})
        if task_profile is not None and "wheel_ground_load_reserve" in task_profile:
            report["ground_load_reserve_module_sha256"] = sha256(ROOT / "training/tron1_ground_load_reserve.py")
        if args.physics_dt_s is not None:
            report.update({"physics_clock_refinement": contract["physics_clock_refinement"],
                           "physics_clock_readback": clock_readback})
        if physics_recorder is not None:
            physical_arrays = physics_recorder.arrays(robot.joint_names, robot.body_names,
                                                       WHEEL_BODY_NAMES, env.step_dt)
            physical_path = args.output_dir / "physics_trajectory.npz"
            np.savez_compressed(physical_path, **physical_arrays)
            report.update({
                "physics_trajectory_file": "physics_trajectory.npz",
                "physics_trajectory_sha256": sha256(physical_path),
                "physics_dt_s": env.physics_dt,
                "physics_recording": {
                    "sample_dt_s": env.physics_dt,
                    "samples": len(physical_arrays["time_s"]),
                    "time_source": "relative simulator physics-step counter",
                    "hook": "after original scene.update at each incremented simulator counter during wrapped.step",
                    "initial_state_samples": 1,
                    "terminal_states_before_reset": True,
                    "same_counter_reset_and_forward_updates_excluded": True,
                    "first_episode_only_valid_mask": True,
                    "observations": ("actual policy pre-action observations held across four physics substeps; validate actor only with trajectory.npz"
                                     if cfg.decimation == 4 else
                                     f"actual policy pre-action observations held across {cfg.decimation} physics substeps; validate actor only with trajectory.npz"),
                    "primary_policy_trajectory_file": "trajectory.npz",
                    "policy_sample_dt_s": env.step_dt,
                },
            })
        (args.output_dir / "policy_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"output_dir": str(args.output_dir), "status": report["status"],
                          "full_reference_survival_rate": summary["full_reference_survival_rate"],
                          "detected_jump_and_landing_rate": summary["detected_jump_and_landing_rate"],
                          "maximum_base_link_height_gain_m": summary["maximum_base_link_height_gain_m"]}, indent=2), flush=True)
    finally:
        # Undo observers in reverse installation order, even on an error.
        # Never replace an unknown wrapper added by another caller.
        try:
            if reward_accumulator is not None:
                reward_accumulator.close()
        finally:
            try:
                if original_scene_update is not None and env.scene.update is physics_scene_update:
                    env.scene.update = original_scene_update
                if wrapped is not None:
                    wrapped.close()
            finally:
                if args.headless:
                    # Isaac Sim 5.1's renderer/Replicator cleanup can wait indefinitely
                    # on this remote runtime. All NPZ/JSON/JIT outputs are closed above;
                    # no cameras or Replicator jobs exist in this headless evaluator.
                    try:
                        app.close(skip_cleanup=True)
                    except TypeError:  # Compatibility with older Sim versions.
                        app.close()
                else:
                    app.close()


if __name__ == "__main__":
    main()
