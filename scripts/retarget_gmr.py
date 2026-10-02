"""Run the actual pinned upstream GMR two-stage IK on TRON1 WF.

GMR's unmodified GeneralMotionRetargeting.retarget is executed on every frame.
This adapter adds model/config registry entries, a CMU task-data conversion and
zero wheel velocity limits. The floating base is optimized, NOT prescribed.
No mj_step, force feasibility, RL policy, or simulation transfer is claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from retarget_mink import finite_difference_qvel, summarize_errors

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_upstream(settings, model_path, config_path):
    """Import only upstream core, avoiding unrelated viewer/SMPL/Torch imports.

    A private package namespace bypasses GMR's broad __init__.py. Both imported
    source files are hash-checked and remain unmodified. This is not a copied or
    reimplemented retargeter and does not install anything into Isaac Python.
    """
    repository = ROOT / "third_party/GMR"
    source = repository / "general_motion_retargeting"
    actual_commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != settings["upstream_commit"]:
        raise ValueError("GMR checkout does not match pinned upstream commit")
    for file, key in (("motion_retarget.py", "upstream_core_sha256"),
                      ("params.py", "upstream_params_sha256")):
        if sha256(source / file) != settings[key]:
            raise ValueError(f"Upstream {file} hash does not match source lock")
    package_name = "_tron1_gmr_upstream"
    package = types.ModuleType(package_name)
    package.__path__ = [str(source)]
    sys.modules[package_name] = package
    params = importlib.import_module(f"{package_name}.params")
    core = importlib.import_module(f"{package_name}.motion_retarget")
    # Upstream targets an older Mink positional API. In Mink 1.2 the sixth
    # parameter is safety_break, so upstream's list would silently be ignored
    # as a limit list. Adapt only that module's Mink reference, not global Mink.
    class MinkCompatibility:
        def __getattr__(self, name):
            return getattr(mink, name)

        @staticmethod
        def solve_ik(configuration, tasks, dt, solver, damping, limits):
            return mink.solve_ik(configuration=configuration, tasks=tasks, dt=dt,
                                 solver=solver, damping=damping, limits=limits,
                                 safety_break=True, primal_tol=1e-10, dual_tol=1e-10)

    if "safety_break" not in inspect.signature(mink.solve_ik).parameters:
        raise ValueError("This adapter expects the audited Mink safety_break API")
    core.mink = MinkCompatibility()
    params.ROBOT_XML_DICT["tron1_wf"] = model_path
    params.IK_CONFIG_DICT["cmu_anchored"] = {"tron1_wf": config_path}
    return core.GeneralMotionRetargeting, actual_commit


def gmr_frame_data(pelvis, wheels, root_quat, scale):
    """Prepare globally anchored data; upstream uniform scaling restores targets.

    The matched targets are already scaled. Dividing all 3 landmarks by s
    before GMR's global/root-relative uniform scaling yields exactly the same
    targets, not a second scale. Identity ankle orientation is a dummy with
    zero rotational task weight, not a human ankle pose or rolling command.
    """
    return {"Pelvis": [pelvis / scale, root_quat.copy()],
            "LeftAnkle": [wheels[0] / scale, np.array([1., 0., 0., 0.])],
            "RightAnkle": [wheels[1] / scale, np.array([1., 0., 0., 0.])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/gmr_retarget_wf.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/gmr-cmu-16_03")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    settings = config["_adapter"]
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output must be new or empty; previous results are preserved")
    if settings["offset_to_ground"]:
        raise ValueError("Per-frame grounding would destroy the jump reference")
    model_path = ROOT / settings["model"]
    baseline_path = ROOT / settings["matched_baseline"]
    source_path = ROOT / settings["human_source"]
    baseline_report = json.loads((baseline_path.parent / "report.json").read_text())
    if sha256(model_path) != baseline_report["model_sha256"]:
        raise ValueError("Model differs from the matched Mink baseline")
    if sha256(source_path) != baseline_report["source_sha256"]:
        raise ValueError("Human source differs from the matched Mink baseline")
    with np.load(baseline_path, allow_pickle=False) as archive:
        baseline = {key: archive[key].copy() for key in archive.files}
    scale = float(baseline_report["uniform_human_scale"])
    if not all(abs(value - scale) < 1e-14 for value in config["human_scale_table"].values()):
        raise ValueError("GMR scale differs from the matched baseline")
    times, fps = baseline["time_s"], float(baseline["fps"])
    if len(times) < 2 or not np.allclose(np.diff(times), 1 / fps):
        raise ValueError("Expected a regularly sampled multi-frame reference")
    GMR, commit = load_upstream(settings, model_path, args.config.resolve())
    retargeter = GMR(src_human="cmu_anchored", tgt_robot="tron1_wf",
                     actual_human_height=None, solver=settings["solver"],
                     damping=settings["damping"], verbose=False,
                     use_velocity_limit=False)
    model, configuration = retargeter.model, retargeter.configuration
    retargeter.max_iter = int(settings["max_iter_per_stage"])
    # dt is an IK step size, NOT simulated elapsed time: GMR may perform multiple
    # integrations within a single sample. Do not claim a leg velocity bound.
    model.opt.timestep = 1 / fps
    joint_names = [model.joint(i).name for i in range(1, model.njnt)]
    if joint_names != list(baseline["joint_names"]):
        raise ValueError("Baseline joint order differs from the GMR model")
    qids = np.array([int(model.joint(n).qposadr[0]) for n in joint_names])
    vids = np.array([int(model.joint(n).dofadr[0]) for n in joint_names])
    wheel_names = [n for n in joint_names if n.startswith("wheel_")]
    retargeter.ik_limits.append(mink.VelocityLimit(model, {n: 0.0 for n in wheel_names}))
    configuration.update(baseline["qpos"][0])
    retargeter.set_ground_offset(0.0)
    site_names = list(baseline["site_names"])
    target_names = list(baseline["target_names"])
    site_ids = [model.site(str(n)).id for n in site_names]
    target_ids = [model.site(str(n)).id for n in target_names]
    targets = baseline["target_positions_world_m"]
    root_targets = baseline["root_target_position_m"]
    root_quat = baseline["root_target_quat_wxyz"]
    frames = len(times)
    qpos = np.empty((frames, model.nq))
    sites = np.empty((frames, len(site_names), 3))
    errors = np.empty((frames, len(target_names)))
    ground_penetration = np.zeros(frames)
    self_penetration = np.zeros(frames)
    stage_errors = np.empty((frames, 2))
    start = time.perf_counter()
    for frame in range(frames):
        data = gmr_frame_data(root_targets[frame], targets[frame], root_quat[frame], scale)
        qpos[frame] = retargeter.retarget(data, offset_to_ground=False)
        # Validate exact target equivalence AFTER the upstream preprocessing.
        for name, expected in (("Pelvis", root_targets[frame]),
                               ("LeftAnkle", targets[frame, 0]),
                               ("RightAnkle", targets[frame, 1])):
            np.testing.assert_allclose(retargeter.scaled_human_data[name][0], expected, atol=1e-12)
        mujoco.mj_forward(model, configuration.data)
        sites[frame] = configuration.data.site_xpos[site_ids]
        errors[frame] = np.linalg.norm(configuration.data.site_xpos[target_ids] - targets[frame], axis=1)
        stage_errors[frame] = [retargeter.error1(), retargeter.error2()]
        for contact in configuration.data.contact:
            if contact.dist < 0:
                penetration = -float(contact.dist)
                body_ids = model.geom_bodyid[[contact.geom1, contact.geom2]]
                if 0 in body_ids:
                    ground_penetration[frame] = max(ground_penetration[frame], penetration)
                else:
                    self_penetration[frame] = max(self_penetration[frame], penetration)
        if frame % 100 == 0 or frame == frames - 1:
            print(f"GMR {frame + 1}/{frames}: wheel error {errors[frame] * 1000} mm", flush=True)
    elapsed = time.perf_counter() - start
    qvel, interval_vel = finite_difference_qvel(model, qpos, times)
    joint_positions = qpos[:, qids]
    limited = model.jnt_limited[1:].astype(bool)
    joint_ranges = model.jnt_range[1:].copy()
    violations = np.maximum(joint_ranges[:, 0] - joint_positions,
                            joint_positions - joint_ranges[:, 1])
    max_violation = max(0., float(np.max(violations[:, limited])))
    wheel_indices = [joint_names.index(n) for n in wheel_names]
    max_wheel_angle = float(np.max(np.abs(joint_positions[:, wheel_indices])))
    if not np.isfinite(qpos).all() or max_violation > 1e-6 or max_wheel_angle > 1e-8:
        raise ValueError(f"GMR checks failed: finite={np.isfinite(qpos).all()}, "
                         f"max_joint_violation={max_violation}, max_wheel_angle={max_wheel_angle}")
    root_errors = np.linalg.norm(qpos[:, :3] - root_targets, axis=1)
    angle_errors = (Rotation.from_quat(root_quat[:, [1, 2, 3, 0]]).inv()
                    * Rotation.from_quat(qpos[:, [4, 5, 6, 3]])).magnitude()
    # Keep the established NPZ schema for visualization and dynamic comparison.
    result = {key: value.copy() for key, value in baseline.items()}
    result.update(qpos=qpos, qvel=qvel, joint_positions_rad=joint_positions,
                  joint_velocities_rad_s=qvel[:, vids], site_positions_world_m=sites,
                  target_error_m=errors, ground_penetration_m=ground_penetration,
                  self_penetration_m=self_penetration, gmr_stage_final_errors=stage_errors,
                  retarget_method=np.array("gmr"), root_prescribed=np.array(False))
    report = {
        "status": "completed_kinematic_preview", "method": "gmr",
        "robot_retargeted": True, "physics_validated": False,
        "policy_trained": False, "hardware_ready": False, "root_prescribed": False,
        "wheel_spin": "zero_velocity_limit_not_rolling",
        "source": str(source_path), "source_sha256": sha256(source_path),
        "model": str(model_path), "model_sha256": sha256(model_path),
        "matched_baseline": str(baseline_path), "baseline_sha256": sha256(baseline_path),
        "upstream_url": settings["upstream_url"], "upstream_commit": commit,
        "upstream_core_sha256": settings["upstream_core_sha256"],
        "upstream_core_modified": False,
        "mink_api_compatibility_shim": "Route upstream sixth positional ik_limits by limits keyword; Mink 1.2 otherwise interprets it as safety_break and silently ignores explicit limits. Shim is scoped to imported GMR module.",
        "daqp_tolerance_override": {"primal_tol": 1e-10, "dual_tol": 1e-10, "reason": "Prevent accumulated zero-wheel-velocity inequality tolerance drift."},
        "upstream_entrypoint": "GeneralMotionRetargeting.retarget",
        "upstream_retarget_calls": frames, "both_upstream_stages_enabled": True,
        "gmr_package_import": "Load hash-verified params and motion_retarget only, bypass optional viewer/Torch imports",
        "versions": {n: importlib.metadata.version(n) for n in ("mink", "mujoco", "qpsolvers", "daqp", "rich")},
        "settings": config, "frames": frames, "fps": fps, "elapsed_seconds": elapsed,
        "uniform_human_scale": scale,
        "initial_heading_alignment": baseline_report["initial_heading_alignment"],
        "initial_root_position_m": baseline_report["initial_root_position_m"],
        "initial_wheel_position_m": baseline_report["initial_wheel_position_m"],
        "targets_identical_to_mink": True, "framewise_ground_alignment": False,
        "all_wheel_errors": summarize_errors(errors),
        "wheel_errors": {str(n): summarize_errors(errors[:, i]) for i, n in enumerate(target_names)},
        "root_position_errors": summarize_errors(root_errors),
        "max_root_orientation_error_rad": float(angle_errors.max()),
        "max_joint_limit_violation_rad": max_violation,
        "max_interval_joint_speed_rad_s": float(np.max(np.abs(interval_vel[:, vids]))),
        "max_ground_penetration_m": float(ground_penetration.max()),
        "frames_with_ground_penetration_gt_1mm": int(np.count_nonzero(ground_penetration > .001)),
        "max_self_penetration_m": float(self_penetration.max()),
        "root_rise_from_initial_m": float(qpos[:, 2].max() - qpos[0, 2]),
        "warnings": [
            "This is GMR kinematic retargeting, not BeyondMimic/PPO or a dynamics-validated jump.",
            "Free base is optimized by both GMR stages; it is not prescribed as in the baseline.",
            "TRON1-specific reduced mapping: no human knees, arms or ankle rotations are imposed.",
            "Both methods share the same human scale, input targets, neutral seed and official robot model.",
            "GMR has no inter-frame leg speed cap; baseline Mink has an 8 rad/s preview cap.",
            "No rolling, contact-force, ground/self-collision or momentum constraints; penetration is reported.",
            "The NPZ uses MuJoCo order, not a drop-in BeyondMimic/Isaac training reference."
        ]
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "robot_reference.npz", **result)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "wheel_errors": report["all_wheel_errors"],
                      "root_errors": report["root_position_errors"],
                      "max_joint_speed_rad_s": report["max_interval_joint_speed_rad_s"],
                      "max_ground_penetration_m": report["max_ground_penetration_m"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
