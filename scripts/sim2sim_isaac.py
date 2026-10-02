"""Native Isaac Sim 4.5 free-base torque rollout for sim2sim diagnostics.

Run using the Isaac runtime's python.bat, not the standalone motion environment.
Only initialization sets robot poses. Every recorded step thereafter is genuine
PhysX dynamics under the shared explicit PD + wheel LQR torque controller.
This is a controller/reference diagnostic, NOT a trained BeyondMimic policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

from build_scene import DEFAULT_ASSET, ROOT, build_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/sim2sim_wf.json")
    parser.add_argument("--case", choices=("motion", "standing"), default="motion")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output must be new or empty; preserving {output}")
    output.mkdir(parents=True, exist_ok=True)
    app = None
    samples = []
    began = time.monotonic()
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": args.headless, "width": 960, "height": 720,
                             "renderer": "RayTracedLighting", "anti_aliasing": 0,
                             "multi_gpu": False, "sync_loads": True})
        import numpy as np
        from scipy.spatial.transform import Rotation
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, RigidPrim
        from isaacsim.core.utils.stage import open_stage, get_current_stage
        from isaacsim.core.utils.types import ArticulationAction
        from sim2sim_common import (JOINT_NAMES, ReferenceMotion, SharedController,
                                    load_config, save_run, stop_reason)

        config = load_config(args.config)
        reference = ReferenceMotion(args.reference, case=args.case, config=config)
        shared = SharedController(reference, config)
        dt = float(config["physics_dt"])
        initial = reference.initial_qpos.copy()
        # ReferenceMotion already adds the common one-time contact clearance.
        scene = build_scene(args.asset, output / "scene.usda", free_base=True)
        if not open_stage(str(scene)):
            raise RuntimeError(f"Could not open derived scene {scene}")
        stage = get_current_stage()
        stage.SetTimeCodesPerSecond(1 / dt)
        xform = UsdGeom.Xformable(stage.GetPrimAtPath("/World/TRON1"))
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*initial[:3].tolist()))
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(float(initial[3]), Gf.Vec3d(*initial[4:7].tolist())))
        root_prim = stage.GetPrimAtPath("/World/TRON1/base_Link")
        articulation = PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
        articulation.CreateSolverPositionIterationCountAttr(8)
        articulation.CreateSolverVelocityIterationCountAttr(4)
        articulation.CreateEnabledSelfCollisionsAttr(False)

        # This material and all overrides belong to the derived scene, not USD
        # source assets. Parent collision groups already include their primitive
        # descendants, so do NOT add duplicate collider APIs to child geometry.
        material = UsdShade.Material.Define(stage, "/World/Sim2SimContactMaterial")
        physical = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physical.CreateStaticFrictionAttr(float(config["ground_friction"]))
        physical.CreateDynamicFrictionAttr(float(config["ground_friction"]))
        physical.CreateRestitutionAttr(float(config["restitution"]))
        material_physx = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        material_physx.CreateFrictionCombineModeAttr("average")
        material_physx.CreateRestitutionCombineModeAttr("average")
        collision_paths = []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
                collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                collision.CreateContactOffsetAttr(0.002)
                collision.CreateRestOffsetAttr(0.0)
                collision_paths.append(str(prim.GetPath()))
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                rigid.CreateLinearDampingAttr(0.0)
                rigid.CreateAngularDampingAttr(0.0)
            if prim.IsA(UsdPhysics.RevoluteJoint):
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.CreateStiffnessAttr(0.0)
                drive.CreateDampingAttr(0.0)
                drive.CreateTargetPositionAttr(0.0)
                drive.CreateTargetVelocityAttr(0.0)
                drive.CreateMaxForceAttr(80.0)
                joint_api = PhysxSchema.PhysxJointAPI.Apply(prim)
                joint_api.CreateJointFrictionAttr(0.0)
                joint_api.CreateArmatureAttr(0.0)
        if stage.GetPrimAtPath("/World/TRON1/joints/inspection_fixture"):
            raise RuntimeError("Fixture forbidden in sim2sim dynamics")
        for _ in range(10):
            app.update()
        # Required by this Isaac 4.5 runtime to avoid an effective 60 Hz scene.
        if not stage.RemovePrim("/World/PhysicsScene"):
            raise RuntimeError("Could not replace generated physics scene")
        app.update()
        world = World(physics_prim_path="/World/PhysicsScene", stage_units_in_meters=1.0,
                      physics_dt=dt, rendering_dt=1 / 60, device="cpu")
        world.get_physics_context().set_solver_type("TGS")
        robot = world.scene.add(SingleArticulation(prim_path="/World/TRON1/base_Link", name="tron1"))
        paths = [str(p.GetPath()) for p in stage.Traverse()
                 if str(p.GetPath()).startswith("/World/TRON1/") and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        bodies = RigidPrim(prim_paths_expr=paths, name="sim2sim_measurements",
                           reset_xform_properties=False, prepare_contact_sensors=False)
        world.reset()
        world.set_simulation_dt(physics_dt=dt, rendering_dt=1 / 60)
        if not np.isclose(world.get_physics_dt(), dt, rtol=1e-9, atol=1e-12):
            raise RuntimeError("Actual PhysX timestep differs from shared controller timestep")
        native_names = list(robot.dof_names)
        if set(native_names) != set(JOINT_NAMES) or len(native_names) != 8:
            raise RuntimeError(f"Unexpected articulation joints {native_names}")
        canonical_to_native = np.array([native_names.index(n) for n in JOINT_NAMES])
        native_to_canonical = np.array([list(JOINT_NAMES).index(n) for n in native_names])
        # World.reset() performs a physics warm-up. Reapply the requested root
        # exactly once here, before the first controller sample, so the engines
        # start identically rather than with a tiny Isaac-only gravity impulse.
        robot.set_world_pose(position=initial[:3].astype(np.float32),
                             orientation=initial[3:7].astype(np.float32))
        robot.set_joint_positions(np.asarray(initial[7:][native_to_canonical], dtype=np.float32))
        robot.set_joint_velocities(np.zeros(8, dtype=np.float32))
        robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
        robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
        controller = robot.get_articulation_controller()
        zeros = np.zeros(8, dtype=np.float32)
        controller.set_gains(kps=zeros, kds=zeros, save_to_usd=True)
        controller.set_gains(kps=zeros, kds=zeros, save_to_usd=False)
        gains = controller.get_gains()
        if not all(np.allclose(g, 0.0) for g in gains):
            raise RuntimeError(f"Native drives must be disabled, got {gains}")
        controller.set_max_efforts(np.full(8, 80.0, dtype=np.float32))
        view = robot._articulation_view
        view.set_friction_coefficients(np.zeros((1, 8), dtype=np.float32))
        view.set_armatures(np.zeros((1, 8), dtype=np.float32))
        if not (np.allclose(view.get_friction_coefficients(), 0)
                and np.allclose(view.get_armatures(), 0)):
            raise RuntimeError("Passive joint friction/armature not zero")
        world.physics_sim_view.update_articulations_kinematic()
        bodies.initialize()
        paths = list(bodies.prim_paths)
        if paths != list(bodies._physics_view.prim_paths):
            raise RuntimeError("USD and PhysX body measurement order differs")
        body_names = [Path(p).name for p in paths]
        wheel_ids = np.array([body_names.index(f"wheel_{side}_Link") for side in ("L", "R")])
        body_ids = np.array([i for i in range(len(paths)) if i not in wheel_ids])
        site_names = ["pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_wheel", "right_wheel"]
        site_body_names = ["base_Link", "hip_L_Link", "hip_R_Link", "knee_L_Link", "knee_R_Link", "wheel_L_Link", "wheel_R_Link"]
        site_ids = np.array([body_names.index(name) for name in site_body_names])
        masses = np.asarray(bodies.get_masses()).reshape(-1)
        local_com = np.asarray(bodies.get_coms()[0]).reshape(len(paths), 3)
        stage.GetRootLayer().Save()

        def measure():
            positions, quats = bodies.get_world_poses()
            positions, quats = np.asarray(positions), np.asarray(quats)
            rotations = Rotation.from_quat(quats[:, [1, 2, 3, 0]])
            coms = positions + rotations.apply(local_com)
            root_pos, root_quat = robot.get_world_pose()
            return {
                "joint_pos": np.asarray(robot.get_joint_positions())[canonical_to_native].copy(),
                "root_pos": np.asarray(root_pos).copy(),
                "root_quat_wxyz": np.asarray(root_quat).copy(),
                "axle_pos": positions[wheel_ids].mean(axis=0),
                "body_com": np.average(coms[body_ids], axis=0, weights=masses[body_ids]),
                "site_positions": positions[site_ids].copy(),
                "joint_vel_native": np.asarray(robot.get_joint_velocities())[canonical_to_native].copy(),
            }

        state = measure()
        np.testing.assert_allclose(state["root_pos"], initial[:3], atol=2e-6)
        np.testing.assert_allclose(state["joint_pos"], initial[7:], atol=2e-6)
        start_tick, start_time = world.current_time_step_index, world.current_time
        steps = int(np.ceil(reference.duration / dt))
        termination = "completed"
        for step in range(steps + 1):
            if not app.is_running():
                raise RuntimeError("Simulation closed before diagnostic completed")
            t = world.current_time - start_time
            state = measure()
            control = shared.compute(t, state)
            samples.append({"time_s": float(t), **state, **control})
            reason = stop_reason(state, control, config)
            if reason:
                termination = reason
                break
            if step == steps:
                break
            robot.apply_action(ArticulationAction(joint_efforts=np.asarray(
                control["torque"])[native_to_canonical].astype(np.float32)))
            world.step(render=False)
            # Render only for an explicitly requested GUI, never for telemetry.
            if not args.headless and step % 8 == 0:
                world.render()
            if step % 500 == 0:
                print(f"ISAAC_SIM2SIM t={t:.3f} / {reference.duration:.3f} tilt={np.degrees(control['base_tilt_rad']):.2f}", flush=True)
        actual_steps = world.current_time_step_index - start_tick
        elapsed = world.current_time - start_time
        if not np.isclose(elapsed, actual_steps * dt, rtol=1e-5, atol=1e-7):
            raise RuntimeError(f"Physics clock mismatch {elapsed}, {actual_steps}, {dt}")
        extra = {
            "runtime": "Isaac Sim 4.5 native Windows, PhysX CPU TGS",
            "native_joint_names": native_names, "canonical_to_native_indices": canonical_to_native.tolist(),
            "native_drive_stiffness": np.asarray(gains[0]).tolist(),
            "native_drive_damping": np.asarray(gains[1]).tolist(),
            "passive_joint_friction": np.asarray(view.get_friction_coefficients()).tolist(),
            "joint_armature": np.asarray(view.get_armatures()).tolist(),
            "body_names": body_names, "body_masses_kg": masses.tolist(),
            "site_names": site_names, "collision_groups": collision_paths,
            "nonwheel_mass_kg": float(masses[body_ids].sum()), "total_mass_kg": float(masses.sum()),
            "effective_physics_dt": world.get_physics_dt(), "physics_ticks": actual_steps,
            "physics_elapsed_s": elapsed, "scene": str(scene),
            "initial_root_clearance_m": config["initial_clearance_m"], "self_collision": False,
            "solver_position_iterations": 8, "solver_velocity_iterations": 4,
            "contact_offset_m": 0.002, "rest_offset_m": 0.0,
            "ground_static_dynamic_friction": config["ground_friction"], "restitution": config["restitution"],
            "fixed_base": False, "root_pose_writes_after_initialization": 0,
            "wall_time_s": time.monotonic() - began,
            "limitations": [
                "Shared untrained PD+LQR controller; not a BeyondMimic policy and not proof of robot jump feasibility.",
                "PhysX and MuJoCo contact models/solver implementations differ despite aligned nominal geometry/parameters.",
                "Native velocity telemetry is diagnostic only; shared control differentiates poses.",
            ],
        }
        save_run(output, "isaac", reference, config, samples, termination, extra=extra)
        print(f"ISAAC_SIM2SIM_RESULT termination={termination} elapsed={elapsed:.4f}s output={output}", flush=True)
    except BaseException as exc:
        (output / "isaac_error.json").write_text(json.dumps({
            "status": "execution_error", "error": str(exc), "traceback": traceback.format_exc(),
            "samples_recorded": len(samples), "wall_time_s": time.monotonic() - began,
        }, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
