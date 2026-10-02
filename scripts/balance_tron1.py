"""Experimental, simulation-only TRON1 WF sagittal balance: leg PD + wheel LQR.

No fixture or root pose correction is used after initialization. The controller
uses simulator ground-truth COM states; an IMU/encoder estimator is future work.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
import traceback

from build_scene import DEFAULT_ASSET, ROOT, build_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--config", type=Path, default=ROOT / "config/balance_wf.json")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--initial-pitch-deg", type=float, default=2.0)
    parser.add_argument("--controller", choices=("lqr", "off"), default="lqr")
    parser.add_argument("--rate-source", choices=("pose", "tensor"), default="pose",
                        help="Pose finite differences avoid observed contact-solver velocity bias")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 3:
        parser.error("--seconds must be finite and >= 3")
    if not math.isfinite(args.initial_pitch_deg) or abs(args.initial_pitch_deg) > 10:
        parser.error("--initial-pitch-deg must be between -10 and 10")
    if args.headless and args.keep_open:
        parser.error("--keep-open requires a GUI")
    output = (args.output_dir or ROOT / "outputs" / ("balance-" + time.strftime("%Y%m%d-%H%M%S"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "balance_report.json"
    if report_path.exists():
        parser.error(f"Choose an unused output directory: {output}")
    report = {"status": "starting", "controller": args.controller,
              "initial_pitch_deg": args.initial_pitch_deg, "fixed_base": False,
              "state_source": "simulator ground-truth body COM and wheel axle",
              "rate_source": args.rate_source,
              "scope": "level ground, fixed leg posture, sagittal local balance"}

    def save_report():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def save_telemetry():
        if not samples:
            return
        with (output / "telemetry.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["time_s", "axle_x_m", "axle_vx_mps", "com_pitch_rad", "com_pitch_rate_radps",
                             "base_roll_rad", "base_yaw_rad", "max_leg_deflection_rad", "axle_z_m",
                             "total_torque_nm", "unclipped_torque_nm", "tensor_axle_vx_mps",
                             "tensor_com_pitch_rate_radps"])
            writer.writerows(samples)

    save_report()
    app = None
    samples = []
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": args.headless, "width": 960, "height": 720,
                             "renderer": "RayTracedLighting", "anti_aliasing": 0,
                             "multi_gpu": False, "sync_loads": True})
        import numpy as np
        from scipy.spatial.transform import Rotation
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, RigidPrim
        from isaacsim.core.utils.stage import open_stage, get_current_stage
        from isaacsim.core.utils.types import ArticulationAction
        from omni.kit.viewport.utility import get_active_viewport
        from balance_lqr import WIPParameters, design_lqr

        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        params = WIPParameters(**cfg["model"])
        dt = 1 / 120
        # Discrete stage cost: m, m/s, rad, rad/s, total motor torque in N m.
        design = design_lqr(params, dt, [20.0, 10.0, 500.0, 20.0], 0.1)
        pitch = cfg["equilibrium_base_pitch_rad"] + math.radians(args.initial_pitch_deg)
        initial_rotation = Rotation.from_euler("y", pitch)
        axle_local = np.asarray(cfg["axle_in_base"])
        initial_pos = np.array([0.0, 0.0, params.radius + 0.001]) - initial_rotation.apply(axle_local)
        scene = build_scene(args.asset, output / "balance_scene.usda", free_base=True)
        if not open_stage(str(scene)):
            raise RuntimeError(f"Could not open {scene}")
        stage = get_current_stage()
        root_xform = UsdGeom.Xformable(stage.GetPrimAtPath("/World/TRON1"))
        root_xform.ClearXformOpOrder()
        root_xform.AddTranslateOp().Set(Gf.Vec3d(*initial_pos.tolist()))
        q = initial_rotation.as_quat()  # scipy xyzw; Isaac/USD wxyz.
        root_xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(float(q[3]), Gf.Vec3d(*q[:3].tolist())))
        art_api = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath("/World/TRON1/base_Link"))
        art_api.CreateSolverPositionIterationCountAttr(8)
        art_api.CreateSolverVelocityIterationCountAttr(4)
        for _ in range(10):
            app.update()
        # Required in Isaac 4.5 for actual (not merely authored) 120 Hz stepping.
        stage.RemovePrim("/World/PhysicsScene")
        app.update()
        world = World(physics_prim_path="/World/PhysicsScene", stage_units_in_meters=1.0,
                      physics_dt=dt, rendering_dt=1 / 60, device="cpu")
        robot = world.scene.add(SingleArticulation(prim_path="/World/TRON1/base_Link", name="tron1"))
        # Construct views while stopped: adding contact APIs after reset can
        # recreate shapes and invalidate Isaac 4.5's live physics tensor view.
        paths = [str(p.GetPath()) for p in stage.Traverse()
                 if str(p.GetPath()).startswith("/World/TRON1/") and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        bodies = RigidPrim(prim_paths_expr=paths, name="balance_measurements",
                           reset_xform_properties=False, prepare_contact_sensors=False)
        world.reset()
        world.set_simulation_dt(physics_dt=dt, rendering_dt=1 / 60)
        if not np.isclose(world.get_physics_dt(), dt):
            raise RuntimeError("Effective physics timestep differs from LQR design")
        names = list(robot.dof_names)
        wheel_indices = np.array([names.index("wheel_L_Joint"), names.index("wheel_R_Joint")])
        leg_indices = np.array([i for i, n in enumerate(names) if not n.startswith("wheel_")])
        if len(leg_indices) != 6 or len(names) != 8:
            raise RuntimeError(f"Unexpected articulation: {names}")
        robot.set_joint_positions(np.zeros(8, dtype=np.float32))
        robot.set_joint_velocities(np.zeros(8, dtype=np.float32))
        controller = robot.get_articulation_controller()
        kp, kd = np.full(8, 500.0), np.full(8, 30.0)
        kp[wheel_indices] = kd[wheel_indices] = 0.0  # effort, not velocity control
        controller.set_gains(kps=kp, kds=kd, save_to_usd=True)
        controller.set_gains(kps=kp, kds=kd, save_to_usd=False)
        measured_kp, measured_kd = controller.get_gains()
        if not (np.allclose(measured_kp, kp) and np.allclose(measured_kd, kd)):
            raise RuntimeError("Runtime joint gains do not match leg-PD / wheel-effort configuration")
        controller.set_max_efforts(np.full(8, 80.0))
        robot.apply_action(ArticulationAction(joint_positions=np.zeros(6),
                                            joint_velocities=np.zeros(6), joint_indices=leg_indices))
        # Read actual articulated COM, so small leg deflections do not masquerade
        # as an unknown pitch bias. RigidPrim is only a read-only measurement view.
        bodies.initialize()
        paths = list(bodies.prim_paths)
        if paths != list(bodies._physics_view.prim_paths):
            raise RuntimeError("USD and PhysX measurement link orders disagree")
        wheel_bodies = np.array([i for i, p in enumerate(paths) if p.endswith(("/wheel_L_Link", "/wheel_R_Link"))])
        body_indices = np.array([i for i in range(len(paths)) if i not in wheel_bodies])
        if len(wheel_bodies) != 2:
            raise RuntimeError(f"Could not identify wheel links in {paths}")
        masses = np.asarray(bodies.get_masses()).reshape(-1)
        local_coms = np.asarray(bodies.get_coms()[0]).reshape(len(paths), 3)
        if not np.isclose(masses[body_indices].sum(), params.body_mass, rtol=1e-4):
            raise RuntimeError("Runtime body mass differs from pinned LQR parameters")
        viewport = get_active_viewport()
        if viewport:
            viewport.set_active_camera("/World/Camera")
        stage.GetRootLayer().Save()
        wheel_signs = np.array([cfg["wheel_signs_by_joint"][names[i]] for i in wheel_indices])
        total_torque_limit = 24.0  # deliberately below the asset's 80 N m / joint
        report.update({"physics_dt": dt, "model": cfg["model"], "K": design.K.tolist(),
                       "closed_loop_pole_magnitudes": np.abs(design.closed_loop_poles).tolist(),
                       "leg_kp": 500.0, "leg_kd": 30.0, "total_torque_limit_nm": total_torque_limit,
                       "joint_names": names, "measurement_links": paths,
                       "measured_stiffness_rad": np.asarray(measured_kp).tolist(),
                       "measured_damping_rad": np.asarray(measured_kd).tolist(),
                       "equilibrium_base_pitch_deg": math.degrees(cfg["equilibrium_base_pitch_rad"])})

        previous_measurement = None
        tensor_rates = np.zeros(2)

        def measure():
            nonlocal previous_measurement, tensor_rates
            positions, quats = bodies.get_world_poses()
            rotations = Rotation.from_quat(np.asarray(quats)[:, [1, 2, 3, 0]])
            coms = np.asarray(positions) + rotations.apply(local_coms)
            velocities = np.asarray(bodies.get_linear_velocities())  # COM velocities, world frame
            com = np.average(coms[body_indices], axis=0, weights=masses[body_indices])
            com_velocity = np.average(velocities[body_indices], axis=0, weights=masses[body_indices])
            axle = np.asarray(positions)[wheel_bodies].mean(axis=0)
            # Account for small nonzero wheel COM offsets when measuring axle speed.
            angular = np.asarray(bodies.get_angular_velocities())
            origin_vel = velocities - np.cross(angular, rotations.apply(local_coms))
            axle_vel = origin_vel[wheel_bodies].mean(axis=0)
            d, dv = com - axle, com_velocity - axle_vel
            theta = math.atan2(d[0], d[2])
            theta_dot = (d[2] * dv[0] - d[0] * dv[2]) / (d[0] ** 2 + d[2] ** 2)
            base_pos, base_quat = robot.get_world_pose()
            roll, _, yaw = Rotation.from_quat(np.asarray(base_quat)[[1, 2, 3, 0]]).as_euler("xyz")
            state = np.array([axle[0], axle_vel[0], theta, theta_dot])
            tensor_rates = state[[1, 3]].copy()
            if args.rate_source == "pose":
                now = world.current_time
                if previous_measurement is None:
                    state[[1, 3]] = 0.0
                else:
                    previous_time, previous_state = previous_measurement
                    elapsed = now - previous_time
                    if elapsed > dt / 2:
                        state[1] = (state[0] - previous_state[0]) / elapsed
                        state[3] = (state[2] - previous_state[2]) / elapsed
                    else:
                        state[[1, 3]] = previous_state[[1, 3]]
                previous_measurement = (now, state.copy())
            leg_error = float(np.max(np.abs(robot.get_joint_positions()[leg_indices])))
            if not np.isfinite(np.r_[state, roll, yaw, base_pos, leg_error]).all():
                raise RuntimeError("Non-finite simulation state")
            return state, roll, yaw, leg_error, axle[2]

        start_tick, start_time = world.current_time_step_index, world.current_time
        initial_state = measure()[0]
        if abs(initial_state[2] - math.radians(args.initial_pitch_deg)) > math.radians(0.2):
            raise RuntimeError("Measured initial COM pitch differs from requested disturbance")
        if stage.GetPrimAtPath("/World/TRON1/joints/inspection_fixture"):
            raise RuntimeError("Unexpected inspection fixture in balance scene")
        report["initial_state"] = initial_state.tolist()
        requested_steps = round(args.seconds / dt)
        report["requested_steps"] = requested_steps

        def advance(record=True):
            if not app.is_running():
                raise RuntimeError("Simulation closed during balance test")
            state, roll, yaw, leg_error, axle_z = measure()
            raw_torque = -float((design.K @ state).item()) if args.controller == "lqr" else 0.0
            torque = float(np.clip(raw_torque, -total_torque_limit, total_torque_limit))
            # No yaw controller yet: both +Y wheel axes have positive-forward sign.
            robot.apply_action(ArticulationAction(joint_efforts=wheel_signs * torque / 2,
                                                joint_indices=wheel_indices))
            if record:
                samples.append([world.current_time - start_time, *state.tolist(), roll, yaw,
                                leg_error, axle_z, torque, raw_torque, *tensor_rates.tolist()])
            if abs(state[2]) > math.radians(20) or abs(roll) > math.radians(15) or axle_z < 0.06:
                raise RuntimeError("Balance safety stop: excessive tilt or loss of support")
            if abs(state[0]) > 2.0:
                raise RuntimeError("Balance safety stop: left the local test area")
            world.step(render=False)
            if world.current_time_step_index % 4 == 0:
                world.render()

        for _ in range(requested_steps):
            advance()
        state, roll, yaw, leg_error, axle_z = measure()
        completed = world.current_time_step_index - start_tick
        elapsed = world.current_time - start_time
        if completed != requested_steps or not np.isclose(elapsed, requested_steps * dt, rtol=1e-5):
            raise RuntimeError("Physics tick/time mismatch")
        tail = np.asarray(samples)[-round(2 / dt):]
        criteria = {"tail_pitch_below_1_deg": bool(np.max(np.abs(tail[:, 3])) < math.radians(1)),
                    "tail_speed_below_005_mps": bool(np.max(np.abs(tail[:, 2])) < 0.05),
                    "tail_position_below_008_m": bool(np.max(np.abs(tail[:, 1])) < 0.08),
                    "tail_roll_below_3_deg": bool(np.max(np.abs(tail[:, 5])) < math.radians(3)),
                    "tail_yaw_below_5_deg": bool(np.max(np.abs(tail[:, 6])) < math.radians(5)),
                    "leg_deflection_below_01_rad": bool(np.max(np.abs(tail[:, 7])) < 0.1)}
        report.update({"criteria": criteria, "completed_steps": completed, "simulated_seconds": elapsed,
                       "final_state": state.tolist(), "final_roll_deg": math.degrees(roll),
                       "final_yaw_deg": math.degrees(yaw),
                       "max_abs_pitch_deg": float(np.degrees(np.max(np.abs(np.asarray(samples)[:, 3])))),
                       "max_abs_total_torque_nm": float(np.max(np.abs(np.asarray(samples)[:, 9])))})
        report["tail_max_abs_pitch_deg"] = float(np.degrees(np.max(np.abs(tail[:, 3]))))
        report["tail_max_abs_speed_mps"] = float(np.max(np.abs(tail[:, 2])))
        if not all(criteria.values()):
            raise RuntimeError("Simulation completed, but local balance acceptance criteria failed")
        report["status"] = "passed"
        save_telemetry()
        save_report()
        print("TRON1_BALANCE_PASSED " + str(report_path), flush=True)
        while args.keep_open and app.is_running():
            advance(record=False)
    except BaseException as exc:
        report.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        if samples:
            report["last_recorded_time_s"] = samples[-1][0]
            report["last_recorded_state"] = samples[-1][1:5]
        save_report()
        raise
    finally:
        save_telemetry()
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
