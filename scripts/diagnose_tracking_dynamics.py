"""Bounded, named-joint PhysX/MuJoCo single-step force-response diagnostic.

Default: phase-zero robot raised 2 m, all velocities zero, gravity retained,
one zero-torque case and eight joints driven by +/-1 and +/-4 Nm for one 5 ms step.
Isaac PD gains are disabled and exact torques are written directly to PhysX.
This is not a policy rollout, jump test, or a controller/trajectory change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_options(args):
    """Pure validation, before importing a simulator or creating output files."""
    for name in ("torque_nm", "extra_torque_nm"):
        value = getattr(args, name)
        if not np.isfinite(value) or not 0.0 < value <= 10.0:
            raise ValueError(f"{name} must be finite and in (0, 10] Nm")
    if args.torque_nm == args.extra_torque_nm:
        raise ValueError("Torque magnitudes must differ so case labels remain unique")
    if not np.isfinite(args.root_raise_m):
        raise ValueError("root_raise_m must be finite")
    angular_speed = getattr(args, "max_body_angular_speed_rad_s", None)
    if angular_speed is not None and (not np.isfinite(angular_speed) or not 0.0 < angular_speed <= 1000.0):
        raise ValueError("max_body_angular_speed_rad_s must be finite and in (0, 1000]")
    if isinstance(args.steps_per_case, bool) or not isinstance(args.steps_per_case, int) or not 1 <= args.steps_per_case <= 20:
        raise ValueError("Bounded diagnostics require 1 to 20 integer steps per case")
    if args.legacy_wheel_friction is not None:
        if not np.isfinite(args.legacy_wheel_friction) or not 0.0 <= args.legacy_wheel_friction <= 1.0:
            raise ValueError("legacy_wheel_friction must be finite and in [0, 1]")
        if args.zero_legacy_friction:
            raise ValueError("zero_legacy_friction and legacy_wheel_friction are mutually exclusive")


def override_legacy_friction(values, joint_names, wheel_value=None, zero_all=False):
    """Return an independent array; wheel-only mode preserves every other joint."""
    result = np.asarray(values).copy()
    names = list(joint_names)
    if result.ndim != 2 or result.shape[1] != len(names) or not np.isfinite(result).all():
        raise ValueError("Legacy friction must be a finite [num_envs, num_joints] array")
    if len(set(names)) != len(names):
        raise ValueError("Joint names must be unique")
    if zero_all and wheel_value is not None:
        raise ValueError("Legacy friction overrides are mutually exclusive")
    if zero_all:
        result[:] = 0.0
    elif wheel_value is not None:
        if not np.isfinite(wheel_value) or not 0.0 <= wheel_value <= 1.0:
            raise ValueError("Wheel friction must be finite and in [0, 1]")
        for name in ("wheel_L_Joint", "wheel_R_Joint"):
            if name not in names:
                raise ValueError(f"Missing wheel joint: {name}")
            result[:, names.index(name)] = wheel_value
    return result


def step_metadata(case_count, steps_per_case, dt):
    if isinstance(steps_per_case, bool) or not isinstance(steps_per_case, int) or not 1 <= steps_per_case <= 20:
        raise ValueError("steps_per_case must be an integer from 1 to 20")
    if not isinstance(case_count, int) or case_count < 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Case count and finite timestep must be positive")
    return {"case_count": case_count, "steps_per_case": steps_per_case,
            "physics_steps": case_count * steps_per_case, "dt_s": float(dt),
            "force_interval_s": steps_per_case * float(dt),
            "total_integrated_time_s": case_count * steps_per_case * float(dt)}


def force_cases(names, magnitude, extra_magnitude=None):
    if not names or len(set(names)) != len(names):
        raise ValueError("Nonempty unique joint names are required")
    for value in (magnitude,) + ((extra_magnitude,) if extra_magnitude is not None else ()):
        if not np.isfinite(value) or not 0.0 < value <= 10.0:
            raise ValueError("Torque magnitudes must be finite and in (0, 10]")
    if magnitude == extra_magnitude:
        raise ValueError("Torque magnitudes must be distinct")
    cases = [("gravity_only", np.zeros(len(names)))]
    for level in (magnitude,) + ((extra_magnitude,) if extra_magnitude is not None else ()):
        for index, name in enumerate(names):
            for sign in (1.0, -1.0):
                effort = np.zeros(len(names))
                effort[index] = sign * level
                cases.append((f"{name}:{sign * level:+g}Nm", effort))
    return cases


def run_isaac(args):
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(args)
    app = launcher.app
    env = None
    try:
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        sys.path.insert(0, str(ROOT))
        from training.tron1_tracking import make_env_cfg

        cfg = make_env_cfg(args.motion_file, args.asset_path, 1, args.device, eval_mode=True)
        if args.max_body_angular_speed_rad_s is not None:
            # USD/PhysX rigid-body schema uses degrees/s, not joint rad/s.
            cfg.scene.robot.spawn.rigid_props.max_angular_velocity = math.degrees(args.max_body_angular_speed_rad_s)
        for actuator in cfg.scene.robot.actuators.values():
            actuator.stiffness = 0.0
            actuator.damping = 0.0
        cfg.seed = 42
        env = ManagerBasedRLEnv(cfg=cfg)
        env.reset()
        robot = env.scene["robot"]
        names = list(robot.joint_names)
        view = robot.root_physx_view

        def arr(tensor):
            return tensor.detach().cpu().numpy().copy()

        def vector(tensor):
            return arr(tensor)[0].tolist()

        def state():
            return {
                "root_position_m": vector(robot.data.root_pos_w - env.scene.env_origins),
                "root_quaternion_wxyz": vector(robot.data.root_quat_w),
                "root_com_velocity_w_m_s": vector(robot.data.root_lin_vel_w),
                "root_angular_velocity_w_rad_s": vector(robot.data.root_ang_vel_w),
                "joint_position_rad": vector(robot.data.joint_pos),
                "joint_velocity_rad_s": vector(robot.data.joint_vel),
                "wheel_position_m": arr(robot.data.body_pos_w)[0,
                    [robot.body_names.index(n) for n in ("wheel_L_Link", "wheel_R_Link")]].tolist(),
            }

        props = {}
        from isaaclab.sim import get_current_stage
        props["usd_body_max_angular_velocity_deg_s"] = {
            prim.GetName(): float(prim.GetAttribute("physxRigidBody:maxAngularVelocity").Get())
            for prim in get_current_stage().Traverse()
            if prim.GetName() in robot.body_names and str(prim.GetPath()).startswith("/World/envs/env_0/")
            and prim.GetAttribute("physxRigidBody:maxAngularVelocity").IsValid()
            and prim.GetAttribute("physxRigidBody:maxAngularVelocity").Get() is not None
        }
        for name in ("get_dof_stiffnesses", "get_dof_dampings", "get_dof_armatures",
                     "get_dof_friction_coefficients", "get_dof_friction_properties",
                     "get_dof_max_forces", "get_dof_max_velocities", "get_masses", "get_coms",
                     "get_inertias", "get_dof_limits", "get_generalized_mass_matrices"):
            if hasattr(view, name):
                try:
                    props[name] = arr(getattr(view, name)()).tolist()
                except Exception as error:
                    props[name] = {"unavailable": str(error)}
        if not np.allclose(props["get_dof_stiffnesses"], 0.0) or not np.allclose(props["get_dof_dampings"], 0.0):
            raise AssertionError("PhysX still has implicit drive gains")
        props["available_friction_getters"] = [name for name in dir(view) if name.startswith("get_") and "friction" in name]
        props["resolved_actuators"] = {
            name: {attribute: arr(getattr(actuator, attribute)).tolist()
                   for attribute in ("friction", "dynamic_friction", "viscous_friction", "armature", "effort_limit", "velocity_limit")
                   if hasattr(actuator, attribute)}
            for name, actuator in robot.actuators.items()
        }
        original = view.get_dof_friction_coefficients().clone()
        new_properties_before = arr(view.get_dof_friction_properties())
        target = override_legacy_friction(arr(original), names, args.legacy_wheel_friction,
                                          zero_all=args.zero_legacy_friction)
        if args.zero_legacy_friction or args.legacy_wheel_friction is not None:
            view.set_dof_friction_coefficients(
                torch.as_tensor(target, dtype=original.dtype).cpu(), indices=torch.arange(1, dtype=torch.int32),
            )
        after_legacy = arr(view.get_dof_friction_coefficients())
        after_new = arr(view.get_dof_friction_properties())
        props["legacy_friction_override"] = {
            "mode": "zero_all" if args.zero_legacy_friction else (
                "wheel_only" if args.legacy_wheel_friction is not None else "unchanged"),
            "requested_wheel_value": args.legacy_wheel_friction,
            "before": arr(original).tolist(), "after": after_legacy.tolist(),
            "new_properties_before": new_properties_before.tolist(), "new_properties_after": after_new.tolist(),
        }
        if not np.allclose(after_legacy, target, atol=1e-9) or not np.array_equal(after_new, new_properties_before):
            raise AssertionError("Legacy override was not applied exactly or unexpectedly changed new friction properties")
        with np.load(args.motion_file, allow_pickle=False) as reference:
            source_names = reference["joint_names"].astype(str).tolist()
            joint0 = torch.tensor(reference["joint_pos"][0, [source_names.index(n) for n in names]],
                                  device=env.device, dtype=torch.float32).unsqueeze(0)
            root0 = torch.zeros((1, 13), device=env.device)
            root0[0, :3] = torch.tensor(reference["body_pos_w"][0, 0], device=env.device)
            root0[0, 2] += args.root_raise_m
            root0[0, 3:7] = torch.tensor(reference["body_quat_w"][0, 0], device=env.device)
        root0[:, :3] += env.scene.env_origins
        zero_joint = torch.zeros_like(joint0)
        env_ids = torch.arange(1, device=env.device, dtype=torch.int32)
        records = []
        for label, force in force_cases(names, args.torque_nm, args.extra_torque_nm):
            robot.reset()
            robot.write_root_state_to_sim(root0)
            robot.write_joint_state_to_sim(joint0, zero_joint)
            robot.set_joint_position_target(joint0)
            robot.set_joint_velocity_target(zero_joint)
            command = torch.tensor(force, device=env.device, dtype=torch.float32).unsqueeze(0)
            robot.set_joint_effort_target(command)
            env.scene.write_data_to_sim()
            # Exact force injection AFTER any explicit actuator processing;
            # native joint order is recorded above and verified by named tests.
            view.set_dof_actuation_forces(command, env_ids)
            before = state()
            for _ in range(args.steps_per_case):
                env.scene.write_data_to_sim()
                view.set_dof_actuation_forces(command, env_ids)
                env.sim.step(render=False)
                env.scene.update(cfg.sim.dt)
            after = state()
            contact = env.scene.sensors.get("contact_forces")
            contact_norm = float(torch.linalg.vector_norm(contact.data.net_forces_w, dim=-1).max()) if contact else None
            records.append({"case": label, "command_torque_nm": force.tolist(), "before": before,
                            "after": after, "max_body_contact_force_n": contact_norm})
        result = {
            "schema_version": 1, "simulator": "IsaacLab/PhysX", "joint_names": names,
            "body_names": list(robot.body_names), **step_metadata(len(records), args.steps_per_case, cfg.sim.dt),
            "gravity_m_s2": list(cfg.sim.gravity), "root_raise_m": args.root_raise_m,
            "torque_magnitude_nm": args.torque_nm,
            "extra_torque_magnitude_nm": args.extra_torque_nm,
            "zero_legacy_friction": args.zero_legacy_friction,
            "legacy_wheel_friction_requested": args.legacy_wheel_friction,
            "max_body_angular_speed_requested_rad_s": args.max_body_angular_speed_rad_s,
            "max_body_angular_speed_configured_deg_s": cfg.scene.robot.spawn.rigid_props.max_angular_velocity,
            "implicit_pd_disabled": True, "policy_used": False,
            "joint_velocity_limit_preserved": True, "joint_effort_limit_preserved": True,
            "actuator_properties": props, "cases": records,
            "provenance": {
                "diagnostic_source_sha256": sha256(__file__),
                "task_source_sha256": sha256(ROOT / "training/tron1_tracking.py"),
                "motion_file_sha256": sha256(args.motion_file), "asset_file_sha256": sha256(args.asset_path),
                "friction_override_applied_after_task_startup_events": True,
            },
            "notes": ["Every case resets identical pose and zero velocities before its bounded force interval.",
                      "Root pose is link-frame; root linear velocity is COM-frame velocity in world coordinates.",
                      "Generalized mass matrix is captured after environment init, before root elevation; exact poses are in each case.",
                      "No policy, trajectory tracking, success evaluation, or hardware command is performed."],
        }
        write_json(args.output_dir / "isaac_dynamics.json", result)
        print(json.dumps({"output": str(args.output_dir / "isaac_dynamics.json"),
                          "physics_steps": len(records) * args.steps_per_case, "max_contact_n": max(
                              item["max_body_contact_force_n"] or 0.0 for item in records)}, indent=2), flush=True)
    finally:
        if env is not None:
            env.close()
        if args.headless:
            try:
                app.close(skip_cleanup=True)
            except TypeError:
                app.close()
        else:
            app.close()


def run_mujoco(args):
    import mujoco
    from scipy.spatial.transform import Rotation
    from eval_tracking_mujoco import configure_model, load_model, read_contract

    source = json.loads(args.isaac_snapshot.read_text())
    names = source["joint_names"]
    model, _ = load_model(args.model, no_visual_mesh=True)
    configure_model(model, read_contract())
    model.opt.timestep = source["dt_s"]
    model.opt.gravity[:] = source["gravity_m_s2"]
    joint_ids = [model.joint(name).id for name in names]
    qids = [int(model.jnt_qposadr[index]) for index in joint_ids]
    vids = [int(model.jnt_dofadr[index]) for index in joint_ids]
    base_id = model.body("base_Link").id
    wheel_ids = [model.body(name).id for name in ("wheel_L_Link", "wheel_R_Link")]
    source_base = source["body_names"].index("base_Link")
    com_local = np.asarray(source["actuator_properties"]["get_coms"])[0, source_base, :3]

    def rotation(quat):
        return Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()

    def snapshot(data):
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        world_com = data.xpos[base_id] + data.xmat[base_id].reshape(3, 3) @ com_local
        mujoco.mj_jac(model, data, jacp, jacr, world_com, base_id)
        return {
            "root_position_m": data.xpos[base_id].tolist(),
            "root_quaternion_wxyz": data.xquat[base_id].tolist(),
            "root_com_velocity_w_m_s": (jacp @ data.qvel).tolist(),
            "root_angular_velocity_w_rad_s": (jacr @ data.qvel).tolist(),
            "joint_position_rad": data.qpos[qids].tolist(), "joint_velocity_rad_s": data.qvel[vids].tolist(),
            "wheel_position_m": data.xpos[wheel_ids].tolist(),
        }

    records = []
    for case in source["cases"]:
        data = mujoco.MjData(model)  # Fresh constraint warm-start for each case.
        before = case["before"]
        data.qpos[:3] = before["root_position_m"]
        data.qpos[3:7] = before["root_quaternion_wxyz"]
        data.qpos[qids] = before["joint_position_rad"]
        r = rotation(before["root_quaternion_wxyz"])
        omega = np.asarray(before["root_angular_velocity_w_rad_s"])
        data.qvel[:3] = before["root_com_velocity_w_m_s"] - np.cross(omega, r @ com_local)
        data.qvel[3:6] = r.T @ omega
        data.qvel[vids] = before["joint_velocity_rad_s"]
        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[vids] = case["command_torque_nm"]
        mujoco.mj_forward(model, data)
        initial = snapshot(data)
        for _ in range(source.get("steps_per_case", 1)):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        records.append({"case": case["case"], "command_torque_nm": case["command_torque_nm"],
                        "before": initial, "after": snapshot(data), "contacts": int(data.ncon)})
    zero_isaac, zero_mujoco = source["cases"][0]["after"], records[0]["after"]
    response_keys = ("joint_velocity_rad_s", "root_com_velocity_w_m_s", "root_angular_velocity_w_rad_s")
    comparisons = []
    for isaac, mj in zip(source["cases"], records):
        delta = {}
        for key in response_keys:
            measured_i = np.asarray(isaac["after"][key]) - np.asarray(zero_isaac[key])
            measured_m = np.asarray(mj["after"][key]) - np.asarray(zero_mujoco[key])
            err = measured_m - measured_i
            scale = max(float(np.linalg.norm(measured_i)), 1e-12)
            delta[key] = {"isaac_baseline_subtracted": measured_i.tolist(),
                          "mujoco_baseline_subtracted": measured_m.tolist(),
                          "max_abs_error": float(np.max(np.abs(err))),
                          "relative_l2_error": float(np.linalg.norm(err) / scale)}
        comparisons.append({"case": isaac["case"], "responses": delta})
    summary = {
        key: {"max_absolute_error": max(c["responses"][key]["max_abs_error"] for c in comparisons[1:]),
              "max_relative_l2_error": max(c["responses"][key]["relative_l2_error"] for c in comparisons[1:])}
        for key in response_keys
    }
    result = {"schema_version": 1, "simulator": "MuJoCo", "model": str(args.model.resolve()),
              "isaac_snapshot": str(args.isaac_snapshot.resolve()), "joint_names": names,
              **step_metadata(len(records), source.get("steps_per_case", 1), model.opt.timestep), "policy_used": False,
              "gravity_only": {"isaac": zero_isaac, "mujoco": zero_mujoco},
              "summary": summary, "baseline_subtracted_comparisons": comparisons, "cases": records,
              "provenance": {
                  "diagnostic_source_sha256": sha256(__file__),
                  "isaac_snapshot_sha256": sha256(args.isaac_snapshot), "model_sha256": sha256(args.model),
                  "mujoco_configuration_source_sha256": sha256(ROOT / "scripts/eval_tracking_mujoco.py"),
                  "isaac_recorded_provenance": source.get("provenance"),
              },
              "notes": ["Each engine uses the same saved step count and force interval; no feedback policy is involved.",
                        "Baseline subtraction removes common gravity acceleration before response comparison.",
                        "Relative errors are undefined for true zero response; absolute errors should dominate interpretation there.",
                        "Differences can include integration, joint constraints, or inertia, not only motor behavior.",
                        "MuJoCo force is injected directly as named generalized hinge torque, bypassing motor control range."]}
    write_json(args.output_dir / "mujoco_dynamics_comparison.json", result)
    print(json.dumps({"output": str(args.output_dir / "mujoco_dynamics_comparison.json"), "summary": summary}, indent=2))


def main():
    choice = argparse.ArgumentParser(add_help=False)
    choice.add_argument("--engine", required=True, choices=("isaac", "mujoco"))
    selected, _ = choice.parse_known_args()
    parser = argparse.ArgumentParser(description=__doc__, parents=[choice])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--motion-file", type=Path)
    parser.add_argument("--asset-path", type=Path)
    parser.add_argument("--isaac-snapshot", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "outputs/sim2sim-cmu-16_03/mujoco_model.xml")
    parser.add_argument("--root-raise-m", type=float, default=2.0)
    parser.add_argument("--torque-nm", type=float, default=1.0)
    parser.add_argument("--extra-torque-nm", type=float, default=4.0,
                        help="A second torque magnitude exposes static-friction dead zones")
    friction_group = parser.add_mutually_exclusive_group()
    friction_group.add_argument("--zero-legacy-friction", action="store_true",
                                help="Explicit A/B ablation of all legacy PhysX joint-friction coefficients")
    friction_group.add_argument("--legacy-wheel-friction", type=float, default=None,
                                help="Restore only the two wheel legacy coefficients after task startup (e.g.0.01 vs0)")
    parser.add_argument("--steps-per-case", type=int, default=1)
    parser.add_argument("--max-body-angular-speed-rad-s", type=float, default=None,
                        help="Override body angular speed limit, converted to USD degrees/s; default preserves task cfg")
    if selected.engine == "isaac":
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    try:
        validate_options(args)
    except ValueError as error:
        parser.error(str(error))
    for name in (("motion_file", "asset_path") if args.engine == "isaac" else ("isaac_snapshot", "model")):
        path = getattr(args, name)
        if path is None or not path.is_file():
            parser.error(f"A valid --{name.replace('_', '-')} file is required")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Choose a new empty output directory; diagnostic results will not be overwritten")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (run_isaac if args.engine == "isaac" else run_mujoco)(args)


if __name__ == "__main__":
    main()
