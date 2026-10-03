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
    parser.add_argument("--contact-threshold-n", default=5.0, type=float)
    parser.add_argument("--domain-parameters", type=Path,
                        help="One explicit fixed actuator-domain draw JSON; never resample during evaluation")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
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
            return {
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
                           dr_profile={"mode": "fixed", "fixed_draw": domain_parameters} if domain_parameters else None)
        cfg.seed = args.seed
        env = TerminalCaptureEnv(cfg=cfg)
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
                obs, rewards, dones, extras = wrapped.step(clipped_actions)
                del rewards, extras
                done = to_numpy(dones).astype(bool)
                state = state_snapshot(env)
                newly_done = done & active
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
        contract["domain_randomization"] = "disabled during evaluation; fixed actuator parameters"
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
        if export_validation_max_error > 1e-4:
            raise RuntimeError(f"Evolving-observation actor export validation failed: {export_validation_max_error}")
        report = {
            "schema_version": 1, "status": "evaluated" if not active.any() else "evaluation_incomplete",
            "legacy_joint_friction_audit": env.legacy_joint_friction_audit,
            "domain_parameters": domain_parameters,
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
        (args.output_dir / "policy_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"output_dir": str(args.output_dir), "status": report["status"],
                          "full_reference_survival_rate": summary["full_reference_survival_rate"],
                          "detected_jump_and_landing_rate": summary["detected_jump_and_landing_rate"],
                          "maximum_base_link_height_gain_m": summary["maximum_base_link_height_gain_m"]}, indent=2), flush=True)
    finally:
        if wrapped is not None:
            wrapped.close()
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
