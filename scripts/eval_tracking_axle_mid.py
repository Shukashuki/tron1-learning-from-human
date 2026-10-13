"""Evaluate trial021 axle-mid TRON1 policy with the original behavior gates.

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



# Reuse the exact original recorder/metrics, without patching their module.
if hashlib.sha256((ROOT / "scripts/eval_tracking_dense_ground_load.py").read_bytes()).hexdigest() != (
        "ba44bc4215f2913e59d6fa8c616d2d057c2d9c7fc1bc98b7fc9df68470d14185"):
    raise RuntimeError("Frozen baseline016 evaluator source mismatch")

if hashlib.sha256((ROOT / "scripts/eval_tracking_wheel_axle.py").read_bytes()).hexdigest() != (
        "c865d498c94ceafee1bf20f72ded62ac7ee417ec91e370c475ecbaa029fc1663"):
    raise RuntimeError("Frozen baseline017 evaluator source mismatch")

if hashlib.sha256((ROOT / "scripts/eval_tracking_worst_ground_load.py").read_bytes()).hexdigest() != (
        "3f0c80c4db0255a907bec6a1b9677e774dd313e421814d10ab8ffe1b7a2f2ecc"):
    raise RuntimeError("Frozen baseline018 evaluator source mismatch")

if hashlib.sha256((ROOT / "scripts/eval_tracking_tight_axle.py").read_bytes()).hexdigest() != (
        "252a378ef129746f65e78a08f10bd1e091adfae1bb345c962f6f077b4a06c176"):
    raise RuntimeError("Frozen baseline019 evaluator source mismatch")

from eval_tracking_dense_ground_load import (
    PhysicsTrajectoryRecorder, physics_clock_readback, sha256, summarize_episodes,
)

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
    if sha256(ROOT / "scripts/eval_tracking.py") != "fd90089810d3d14bfd96210723afdd3a36113a19f667ec60b2df617dd032f992":
        parser.error("Frozen evaluator baseline source mismatch")
    if (args.task_profile is None or args.num_envs != 1 or args.seed != 42
            or args.terrain_file is not None or args.domain_parameters is not None
            or args.deployment_leg_kp is not None or args.deployment_leg_kd is not None
            or args.physics_dt_s is not None or not args.record_physics
            or not args.verify_unmasked_device or args.contact_threshold_n != 5.0):
        parser.error("Trial021 requires one nominal seed42 200Hz episode, original gains and contact gates")
    sys.path.insert(0, str(ROOT))
    from training.tron1_axle_mid_trial import load_axle_mid_profile
    load_axle_mid_profile(args.task_profile)
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
        from training.tron1_wheel_axle_trial import baseline_dense_profile
        from training.tron1_axle_mid_trial import (
            load_axle_mid_profile, configure_axle_mid_trial,
        )
        task_profile = load_axle_mid_profile(args.task_profile)
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
                           terrain_spec=terrain_spec, task_profile=baseline_dense_profile())
        if args.physics_dt_s is not None:
            from training.tron1_physics_clock import configure_physics_clock
            configure_physics_clock(cfg, args.physics_dt_s)
        configure_axle_mid_trial(cfg, task_profile)
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
        from training.tron1_worst_ground_load_substeps import install_worst_ground_load_accumulator
        reward_accumulator = install_worst_ground_load_accumulator(env)
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
                           "substep_reward_module_sha256": sha256(ROOT / "training/tron1_substep_rewards.py"),
                           "dense_ground_load_module_sha256": sha256(ROOT / "training/tron1_ground_load_substeps.py"),
                           "dense_ground_load_trial_module_sha256": sha256(ROOT / "training/tron1_ground_load_substeps_trial.py"),
                           "baseline_evaluator_source_sha256": sha256(ROOT / "scripts/eval_tracking.py"),
                           "baseline_dense_evaluator_source_sha256": sha256(ROOT / "scripts/eval_tracking_dense_ground_load.py"),
                           "wheel_axle_tracking_source_sha256": sha256(ROOT / "training/tron1_wheel_axle_tracking.py"),
                           "wheel_axle_trial_source_sha256": sha256(ROOT / "training/tron1_wheel_axle_trial.py"),
                           "baseline_wheel_axle_evaluator_source_sha256": sha256(ROOT / "scripts/eval_tracking_wheel_axle.py"),
                           "worst_ground_load_module_sha256": sha256(ROOT / "training/tron1_worst_ground_load_substeps.py"),
                           "worst_ground_load_trial_module_sha256": sha256(ROOT / "training/tron1_worst_ground_load_trial.py"),
                           "tight_axle_tracking_source_sha256": sha256(ROOT / "training/tron1_tight_axle_tracking.py"),
                           "tight_axle_trial_source_sha256": sha256(ROOT / "training/tron1_tight_axle_trial.py"),
                           "baseline_worst_load_evaluator_source_sha256": sha256(ROOT / "scripts/eval_tracking_worst_ground_load.py"),
                           "axle_mid_trial_source_sha256": sha256(ROOT / "training/tron1_axle_mid_trial.py"),
                           "baseline_tight_axle_evaluator_source_sha256": sha256(ROOT / "scripts/eval_tracking_tight_axle.py"),
                           "only_factor_relative_to_trial019": "axle_reward_weight_1.0_to_2.0_std_0.2_held",
                           "dense_ground_load_sampling": "all_substeps_max_cost"})
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
