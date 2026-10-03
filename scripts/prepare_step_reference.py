"""Prepare CMU 83_03 as a TRON1 single-ledge diagnostic, not flat-floor stepping.

The human source, uniform length scale, one initial heading alignment and one
constant Z clearance are preserved explicitly. A static ledge is estimated from
foot support heights and terminal wheel targets: its bounds are NOT CMU ground
truth. Both retargeters use the same derived model containing that collider.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.tron1_terrain import augment_mujoco_xml, validate_terrain
from cmu_motion import load_motion, save_motion
from retarget_mink import calibrated_root_rotations

SOURCE_HASHES = {
    "asf": "b4eabfbcaeadf39faba73e6a5b2857830da73a8549266c95370863155e177093",
    "amc": "4a2e3d8cdde1fb07c93c98012519c46fd7f54764ead45f47ffa8ebd87d6d32b1",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def initial_targets(human, model_path):
    """Independent construction matching retarget_mink's unmodified anchoring."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    data.qpos[:7] = [0., 0, 0, 1, 0, 0, 0]
    data.qpos[7:] = 0.
    mujoco.mj_forward(model, data)
    names = human["joint_names"].astype(str).tolist()
    _, heading = calibrated_root_rotations(human["joint_rotations_world_wxyz"][:, names.index("root")])
    positions = human["joint_positions_world_m"] @ heading.T
    robot_lengths, human_lengths = [], []
    for side, prefix in (("left", "l"), ("right", "r")):
        robot_chain = data.site_xpos[[model.site(f"{side}_{part}").id for part in ("hip", "knee", "wheel")]]
        human_chain = positions[0, [names.index(prefix + part) for part in ("hipjoint", "femur", "tibia")]]
        robot_lengths.append(np.linalg.norm(np.diff(robot_chain, axis=0), axis=1).sum())
        human_lengths.append(np.linalg.norm(np.diff(human_chain, axis=0), axis=1).sum())
    scale = float(np.mean(robot_lengths) / np.mean(human_lengths))
    radius = float(json.loads((ROOT / "config/balance_wf.json").read_text())["model"]["radius"])
    wheel_ids = [model.site(f"{side}_wheel").id for side in ("left", "right")]
    initial = data.site_xpos[wheel_ids].copy()
    initial[:, 2] += radius - initial[:, 2].mean()
    ankles = positions[:, [names.index("ltibia"), names.index("rtibia")]]
    targets = initial[None] + scale * (ankles - ankles[:1])
    return targets, scale, radius, heading


def estimate_terrain(human, targets, scale, radius):
    """Fit one support level from feet, and conservative axis-aligned bounds.

    The final footprint identifies where support must exist, but does not
    uniquely locate the leading edge. The explicit edge/rear/side margins are
    modeling choices retained in provenance, never presented as measurements.
    """
    names = human["joint_names"].astype(str).tolist()
    positions, times = human["joint_positions_world_m"], human["time_s"]
    initial, terminal = times <= times[0] + .5, times >= times[-1] - .5
    foot_heights = np.column_stack([positions[:, [names.index(prefix + "foot"), names.index(prefix + "toes")], 2].min(axis=1)
                                    for prefix in ("l", "r")])
    first = np.percentile(foot_heights[initial], 5, axis=0)
    last = np.percentile(foot_heights[terminal], 5, axis=0)
    support_rises = last - first
    if min(support_rises) < .03 or abs(support_rises[0] - support_rises[1]) > .035:
        raise ValueError(f"A common elevated terminal support level is not identifiable: {support_rises}")
    if np.max(np.ptp(foot_heights[terminal], axis=0)) > .03:
        raise ValueError("Terminal foot heights are not stable enough to estimate a ledge")
    height = float(np.mean(support_rises) * scale)
    final_wheels = targets[terminal]
    travel = final_wheels.mean(axis=(0, 1)) - targets[0].mean(axis=0)
    if travel[0] < .3 or abs(travel[1]) > travel[0]:
        raise ValueError("This one-ledge adapter expects predominantly +X travel after source heading calibration")
    front_margin, rear_margin, side_margin = .12, .25, .175
    lower_x = float(np.min(final_wheels[..., 0]) - radius - front_margin)
    upper_x = float(np.max(final_wheels[..., 0]) + radius + rear_margin)
    lower_y = float(np.min(final_wheels[..., 1]) - side_margin)
    upper_y = float(np.max(final_wheels[..., 1]) + side_margin)
    if lower_x <= float(np.max(targets[0, :, 0])) + radius + .01:
        raise ValueError("Estimated ledge would overlap the initial wheels")
    # One constant clearance, based only on initial-floor and terminal-platform
    # support. No framewise grounding, trajectory scaling or dynamic fitting.
    clearance = max(.001, radius - float(targets[initial, :, 2].min()) + .001,
                    height + radius - float(final_wheels[..., 2].min()) + .001)
    spec = {
        "schema_version": 1, "frame": "world_z_up_m", "ground": {"z": 0., "friction": .6},
        "boxes": [{"name": "step_ledge", "center": [(lower_x + upper_x) / 2, (lower_y + upper_y) / 2, height / 2],
                   "size": [upper_x - lower_x, upper_y - lower_y, height], "friction": .6}],
        "provenance": {
            "geometry_estimated": True, "ground_truth_geometry": False,
            "source_clip": "CMU 83_03", "source_description": "large step to a short ledge",
            "source_index_url": "https://mocap.cs.cmu.edu/search.php?subjectnumber=83",
            "source_fps_verified": 120,
            "height_method": "mean of left/right terminal-minus-initial fifth-percentile foot/toe minimum heights, times the uniform robot/human leg-chain scale",
            "initial_terminal_window_s": .5, "source_initial_foot_support_m": first.tolist(),
            "source_terminal_foot_support_m": last.tolist(), "source_support_rise_by_foot_m": support_rises.tolist(),
            "uniform_human_scale": scale, "wheel_radius_m": radius,
            "robot_ledge_top_height_m": height,
            "terminal_target_wheel_center_range_m": [final_wheels.min(axis=(0, 1)).tolist(), final_wheels.max(axis=(0, 1)).tolist()],
            "horizontal_bounds_method": "terminal wheel-center footprint, X expanded by wheel radius plus stated front/rear margin; Y expanded by side margin",
            "front_margin_beyond_radius_m": front_margin, "rear_margin_beyond_radius_m": rear_margin,
            "lateral_margin_m": side_margin,
            "recommended_constant_reference_z_clearance_m": clearance,
            "limitations": ["Source contains no measured platform mesh or contact forces; front edge and depth are not uniquely identifiable.",
                           "This collider is a documented task hypothesis. It does not establish dynamic feasibility or successful stepping."]},
    }
    return validate_terrain(spec), clearance


def inspect_kinematics(model_path, reference_path, z_offset):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    with np.load(reference_path, allow_pickle=False) as archive:
        qpos = archive["qpos"].copy()
        times = archive["time_s"].copy()
        joint_speed = archive["joint_velocities_rad_s"].copy()
        joint_names = archive["joint_names"].astype(str).tolist()
    wheel_ids = [model.body(f"wheel_{side}_Link").id for side in ("L", "R")]
    wheels = np.empty((len(qpos), 2, 3))
    penetration = np.zeros(len(qpos))
    worst = None
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        data.qpos[2] += z_offset
        mujoco.mj_forward(model, data)
        wheels[frame] = data.xpos[wheel_ids]
        for contact in data.contact:
            if contact.dist < 0 and 0 in model.geom_bodyid[[contact.geom1, contact.geom2]]:
                value = -float(contact.dist)
                if value > float(penetration.max()):
                    worst = {"frame": frame, "time_s": float(times[frame]), "penetration_m": value,
                             "geom1": model.geom(contact.geom1).name, "geom2": model.geom(contact.geom2).name}
                penetration[frame] = max(penetration[frame], value)
    legs = [i for i, name in enumerate(joint_names) if not name.startswith("wheel_")]
    return {"model_includes_real_static_ledge_collider": True, "uniform_z_offset_m": z_offset,
            "max_robot_terrain_penetration_m": float(penetration.max()),
            "frames_terrain_penetration_gt_1cm": int(np.count_nonzero(penetration > .01)),
            "worst_terrain_contact": worst,
            "max_leg_reference_speed_rad_s": float(np.max(np.abs(joint_speed[:, legs]))),
            "zero_dynamics_steps": True, "physics_feasibility_established": False}, times, qpos, wheels


def render_diagnostic(output, terrain, human, times, qpos, wheels, z_offset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    box = terrain["boxes"][0]
    center, size = np.array(box["center"]), np.array(box["size"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1))
    for side, color in enumerate(("tab:blue", "tab:orange")):
        axes[0].plot(wheels[:, side, 0], wheels[:, side, 2], color=color, label=("left" if side == 0 else "right") + " wheel center")
        axes[1].plot(wheels[:, side, 0], wheels[:, side, 1], color=color)
        axes[2].plot(times, wheels[:, side, 2] - .127, color=color, label=("left" if side == 0 else "right") + " center - radius")
    axes[0].plot(qpos[:, 0], qpos[:, 2] + z_offset, color="black", linestyle="--", label="base origin")
    axes[0].add_patch(Rectangle((center[0] - size[0] / 2, 0), size[0], size[2], color="gray", alpha=.3))
    axes[1].add_patch(Rectangle(center[:2] - size[:2] / 2, size[0], size[1], color="gray", alpha=.3))
    axes[0].axhline(0, color="gray")
    axes[2].axhline(size[2], color="gray", linestyle="--", label="estimated platform top")
    axes[0].set(xlabel="X (m)", ylabel="Z (m)", title="GMR reference + actual task collider", aspect="equal")
    axes[1].set(xlabel="X (m)", ylabel="Y (m)", title="Estimated platform footprint (not CMU truth)", aspect="equal")
    axes[2].set(xlabel="Source time (s)", ylabel="Z (m)", title="Support-height diagnostic; no dynamics")
    for axis in axes:
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    fig.suptitle("CMU 83_03 / TRON1 WF: step-up reference with inferred ledge — NOT a successful robot step", fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asf", type=Path, default=ROOT / "assets/mocap/CMU/83.asf")
    parser.add_argument("--amc", type=Path, default=ROOT / "assets/mocap/CMU/83_03.amc")
    parser.add_argument("--human-motion", type=Path, help="Reuse an existing decoded 83_03 NPZ")
    parser.add_argument("--model", type=Path, default=ROOT / "outputs/sim2sim-cmu-16_03/mujoco_model.xml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/motion-suite-20261003/step_up")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    for name in ("terrain.json", "retarget_model.xml", "mink.json", "gmr.json", "summary.json"):
        if (output / name).exists():
            parser.error(f"Refusing to overwrite {output / name}")
    for kind in ("asf", "amc"):
        if sha256(getattr(args, kind)) != SOURCE_HASHES[kind]:
            raise ValueError(f"Unexpected {kind} source; this preparation is pinned to CMU83_03")
    output.mkdir(parents=True, exist_ok=True)
    if args.human_motion:
        human_path = args.human_motion.resolve()
    else:
        human_path = save_motion(load_motion(args.asf, args.amc, fps=120), output / "human")["npz"]
    with np.load(human_path, allow_pickle=False) as archive:
        human = {name: archive[name].copy() for name in archive.files}
    if len(human["time_s"]) != 669 or not np.isclose(float(human["fps"]), 120):
        raise ValueError("Expected all 669 unretimed 120Hz source samples")
    human_metadata = json.loads(str(human["metadata_json"].item()))
    for kind, expected in SOURCE_HASHES.items():
        if human_metadata["source"][kind + "_sha256"] != expected:
            raise ValueError("Decoded human reference source provenance is not the pinned CMU83_03 clip")
    targets, scale, radius, heading = initial_targets(human, args.model)
    terrain, z_offset = estimate_terrain(human, targets, scale, radius)
    terrain["provenance"]["source_sha256"] = {kind: sha256(getattr(args, kind)) for kind in SOURCE_HASHES}
    write_json(output / "terrain.json", terrain)
    xml = augment_mujoco_xml(args.model.read_text(), terrain)
    with (output / "retarget_model.xml").open("x") as stream:
        stream.write(xml)
    model_path = output / "retarget_model.xml"
    config = json.loads((ROOT / "config/mink_retarget_wf.json").read_text())
    config.update(source=str(human_path), model=str(model_path))
    write_json(output / "mink.json", config)
    print("Preparing step-up Mink IK with a static ledge collider", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/retarget_mink.py"), "--config", str(output / "mink.json"),
                    "--output-dir", str(output / "mink")], check=True)
    mink_report = json.loads((output / "mink/report.json").read_text())
    np.testing.assert_allclose(scale, mink_report["uniform_human_scale"], atol=1e-12)
    with np.load(output / "mink/robot_reference.npz", allow_pickle=False) as baseline:
        np.testing.assert_allclose(targets, baseline["target_positions_world_m"], atol=1e-10)
    gmr = json.loads((ROOT / "config/gmr_retarget_wf.json").read_text())
    gmr["human_scale_table"] = {name: scale for name in gmr["human_scale_table"]}
    gmr["_adapter"].update(model=str(model_path), matched_baseline=str(output / "mink/robot_reference.npz"),
                           human_source=str(human_path))
    write_json(output / "gmr.json", gmr)
    print("Preparing step-up GMR two-stage IK with the same targets and terrain", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/retarget_gmr.py"), "--config", str(output / "gmr.json"),
                    "--output-dir", str(output / "gmr")], check=True)
    diagnostics, times, qpos, wheels = inspect_kinematics(model_path, output / "gmr/robot_reference.npz", z_offset)
    render_diagnostic(output / "terrain_reference.png", terrain, human, times, qpos, wheels, z_offset)
    concerns = []
    if diagnostics["max_robot_terrain_penetration_m"] > .015:
        concerns.append("GMR kinematic reference intersects the actual ledge/ground by more than 15 mm; treat any training as a negative/repair diagnostic, not a feasible step reference.")
    if diagnostics["max_leg_reference_speed_rad_s"] > 15:
        concerns.append("Reference leg speed exceeds the pilot actuator's 15 rad/s setting; speed feasibility is not established.")
    if z_offset > .02:
        concerns.append("Required uniform support clearance exceeds 2 cm; support alignment needs review.")
    summary = {"task": "step_up", "status": "kinematic_reference_with_estimated_terrain",
               "source": "CMU83_03", "source_description": "large step to a short ledge", "source_fps": 120,
               "source_frames": 669, "source_duration_s": 669 / 120, "source_sha256": SOURCE_HASHES,
               "source_transport": "Official CMU HTTP downloads; SHA256 pins retrieved bytes, not vendor-signed authenticity",
               "source_page": "https://mocap.cs.cmu.edu/search.php?subjectnumber=83",
               "acknowledgement": "The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217.",
               "uniform_human_scale": scale, "initial_heading_alignment": heading.tolist(),
               "human_source": str(human_path), "gmr_reference": str(output / "gmr/robot_reference.npz"),
               "terrain_file": str(output / "terrain.json"), "terrain_sha256": sha256(output / "terrain.json"),
               "matched_retarget_model_sha256": sha256(model_path), "source_model_sha256": sha256(args.model),
               "recommended_export_constant_z_offset_m": z_offset,
               "terrain_top_height_m": terrain["boxes"][0]["size"][2], "kinematic_diagnostics": diagnostics,
               "concerns": concerns, "physics_validated": False, "policy_trained": False, "hardware_ready": False,
               "geometry_is_cmu_ground_truth": False,
               "downstream_requirement": "Pass this SAME terrain.json to Isaac training/evaluation and MuJoCo evaluation. Flat-floor training is not a step-up test."}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
