"""Load TRON1 WF in Isaac Sim 4.5, verify articulation and save a preview."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
import traceback

from build_scene import DEFAULT_ASSET, JOINT_NAMES, ROOT, build_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--free-base", action="store_true")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--keep-open", action="store_true", help="Keep the GUI open after the smoke test")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.headless and args.keep_open:
        parser.error("--keep-open requires a GUI")
    output = (args.output_dir or ROOT / "outputs" / time.strftime("%Y%m%d-%H%M%S")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "import_report.json"
    if report_path.exists():
        parser.error(f"Output already contains a report; choose a new directory: {output}")
    report = {
        "status": "starting", "robot": "WF_TRON1A", "asset": str(args.asset.resolve()),
        "fixed_base_inspection": not args.free_base,
        "requested_physics_dt": 1 / 120, "physics_dt": None,
        "requested_steps": args.steps, "completed_steps": 0,
        "python": sys.version, "policy": None,
    }

    def write_report() -> None:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    write_report()
    app = None
    try:
        # All Omniverse / pxr imports happen after SimulationApp starts.
        from isaacsim import SimulationApp
        app = SimulationApp({
            "headless": args.headless, "width": 960, "height": 720,
            "renderer": "RayTracedLighting", "anti_aliasing": 0,
            "multi_gpu": False, "sync_loads": True,
        })
        import numpy as np
        from PIL import Image
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import open_stage, get_current_stage
        from isaacsim.core.utils.types import ArticulationAction
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

        scene = build_scene(args.asset, output / "tron1_wf.usda", free_base=args.free_base)
        if not open_stage(str(scene)):
            raise RuntimeError(f"Isaac could not open scene {scene}")
        for _ in range(10):
            app.update()
        # Isaac 4.5's SimulationManager can miss scenes present at stage-open.
        # Recreate our generated scene through World so it receives the add
        # notice; otherwise its manual-step getter may silently use 60 Hz.
        if not get_current_stage().RemovePrim("/World/PhysicsScene"):
            raise RuntimeError("Could not recreate the generated physics scene")
        app.update()
        world = World(
            physics_prim_path="/World/PhysicsScene", stage_units_in_meters=1.0,
            physics_dt=1 / 120, rendering_dt=1 / 60, device="cpu",
        )
        robot = world.scene.add(SingleArticulation(prim_path="/World/TRON1/base_Link", name="tron1"))
        world.reset()
        world.set_simulation_dt(physics_dt=1 / 120, rendering_dt=1 / 60)
        report["physics_dt"] = world.get_physics_dt()
        if not np.isclose(report["physics_dt"], 1 / 120):
            raise RuntimeError(f"Unexpected effective physics timestep: {report['physics_dt']}")
        get_current_stage().GetRootLayer().Save()
        actual = list(robot.dof_names)
        if set(actual) != set(JOINT_NAMES) or robot.num_dof != 8:
            raise RuntimeError(f"Expected 8 TRON1 joints, got {actual}")
        robot.set_joint_positions(np.zeros(8, dtype=np.float32))
        robot.set_joint_velocities(np.zeros(8, dtype=np.float32))
        robot.apply_action(ArticulationAction(
            joint_positions=np.zeros(8, dtype=np.float32),
            joint_velocities=np.zeros(8, dtype=np.float32),
        ))
        stiffness, damping = robot.get_articulation_controller().get_gains()
        expected_stiffness = [0.0 if n.startswith("wheel_") else 40.0 for n in actual]
        expected_damping = [0.8 if n.startswith("wheel_") else 2.5 for n in actual]
        if not (np.allclose(stiffness, expected_stiffness) and np.allclose(damping, expected_damping)):
            raise RuntimeError(f"Unexpected articulation gains: {stiffness}, {damping}")
        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("No render viewport was created")
        viewport.set_active_camera("/World/Camera")
        start_tick, start_time = world.current_time_step_index, world.current_time
        for step in range(args.steps):
            if not app.is_running():
                raise RuntimeError("Isaac closed before the requested simulation completed")
            world.step(render=False)
            if (step + 1) % 2 == 0 or step + 1 == args.steps:
                world.render()
            positions = np.asarray(robot.get_joint_positions())
            velocities = np.asarray(robot.get_joint_velocities())
            base_pos, base_quat = robot.get_world_pose()
            state = np.concatenate((positions, velocities, base_pos, base_quat))
            if not np.isfinite(state).all():
                raise RuntimeError(f"Non-finite robot state at step {step + 1}")
            report["completed_steps"] = world.current_time_step_index - start_tick
        if report["completed_steps"] != args.steps:
            raise RuntimeError(f"Expected {args.steps} physics ticks, got {report['completed_steps']}")
        elapsed_sim = world.current_time - start_time
        if not np.isclose(elapsed_sim, args.steps * report["physics_dt"], rtol=1e-5):
            raise RuntimeError(f"Physics clock mismatch: {elapsed_sim} seconds for {args.steps} ticks")
        report.update({
            "status": "capturing", "joint_names": actual, "num_dof": robot.num_dof,
            "stiffness_rad": np.asarray(stiffness).tolist(),
            "damping_rad": np.asarray(damping).tolist(),
            "final_joint_positions": positions.tolist(),
            "final_joint_velocities": velocities.tolist(),
            "final_base_position": np.asarray(base_pos).tolist(),
            "scene": str(scene), "simulated_seconds": elapsed_sim,
        })
        write_report()
        screenshot = output / "tron1_wf.png"
        capture = capture_viewport_to_file(viewport, str(screenshot))
        capture_task = asyncio.ensure_future(capture.wait_for_result())
        deadline = time.monotonic() + 60
        while not capture_task.done() and time.monotonic() < deadline:
            world.render()
        if not capture_task.done():
            capture_task.cancel()
            raise RuntimeError("Timed out waiting for rendered preview")
        capture_task.result()
        # Kit 106.5 completes the capture future before its background PNG
        # writer has necessarily flushed the file. Wait for a decodable image.
        image_size = None
        while time.monotonic() < deadline:
            try:
                with Image.open(screenshot) as preview:
                    preview.load()
                    image_size = list(preview.size)
                break
            except (OSError, ValueError):
                world.render()
        if image_size is None:
            raise RuntimeError("Simulation ran, but no complete rendered preview was saved")
        if world.current_time_step_index - start_tick != args.steps:
            raise RuntimeError("Physics advanced while saving the preview")
        report.update({
            "status": "passed", "joint_names": actual, "num_dof": robot.num_dof,
            "stiffness_rad": np.asarray(stiffness).tolist(),
            "damping_rad": np.asarray(damping).tolist(),
            "final_joint_positions": positions.tolist(),
            "final_joint_velocities": velocities.tolist(),
            "final_base_position": np.asarray(base_pos).tolist(),
            "scene": str(scene), "screenshot": str(screenshot),
            "screenshot_size": image_size,
            "simulated_seconds": elapsed_sim,
        })
        write_report()
        print("TRON1_IMPORT_PASSED " + str(report_path), flush=True)
        while args.keep_open and app.is_running():
            world.step(render=True)
    except BaseException as exc:
        report.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        write_report()
        raise
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
