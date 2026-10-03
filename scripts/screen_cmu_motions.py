"""Screen decoded CMU human clips without retargeting or changing their motion.

Example: --motion 16_05=outputs/16_05/human_motion.npz --output-dir outputs/screen
Feet-clear intervals are endpoint-height heuristics, NOT measured contact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

ASF_TO_WORLD = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_human_motion(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    required = {"fps", "time_s", "frame_numbers", "joint_names", "parent_indices",
                "joint_positions_world_m", "metadata_json"}
    if missing := required - data.keys():
        raise ValueError(f"Missing decoded-human fields: {sorted(missing)}")
    names = data["joint_names"]
    if names.ndim != 1 or names.dtype.kind not in "US" or len(set(names.tolist())) != len(names):
        raise ValueError("Joint names must be unique strings")
    names = names.astype(str).tolist()
    if not {"root", "ltibia", "rtibia"}.issubset(names):
        raise ValueError("Need root and both ankle endpoints ltibia/rtibia")
    fps = np.asarray(data["fps"])
    if fps.size != 1 or not np.isfinite(fps).all() or float(fps.reshape(-1)[0]) <= 0:
        raise ValueError("FPS must be a finite positive scalar")
    fps = float(fps.reshape(-1)[0])
    positions = data["joint_positions_world_m"]
    times, frames, parents = data["time_s"], data["frame_numbers"], data["parent_indices"]
    count = len(positions)
    if count < 2 or positions.shape != (count, len(names), 3) or not np.isfinite(positions).all():
        raise ValueError("Positions must be finite (frames>=2, nodes, 3)")
    if times.shape != (count,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing")
    if frames.shape != (count,) or frames.dtype.kind not in "iu" or np.any(np.diff(frames.astype(float)) <= 0):
        raise ValueError("Frame numbers must be strictly increasing integers")
    if not np.allclose(np.diff(times), np.diff(frames.astype(float)) / fps, rtol=1e-7, atol=1e-9):
        raise ValueError("Timestamps disagree with frame numbers and FPS; no implicit retiming allowed")
    if parents.shape != (len(names),) or parents.dtype.kind not in "iu":
        raise ValueError("Parent indices must be an integer vector")
    root = names.index("root")
    if parents[root] != -1 or np.count_nonzero(parents == -1) != 1:
        raise ValueError("The root must be the only parentless node")
    for node in range(len(names)):
        seen, current = set(), node
        while current != -1:
            if current < 0 or current >= len(names) or current in seen:
                raise ValueError("Invalid/cyclic skeleton parent indices")
            seen.add(current)
            current = int(parents[current])
    if "joint_rotations_world_wxyz" in data:
        quats = data["joint_rotations_world_wxyz"]
        if (quats.shape != (count, len(names), 4) or not np.isfinite(quats).all()
                or not np.allclose(np.linalg.norm(quats, axis=-1), 1., atol=1e-5)):
            raise ValueError("Rotations must be finite normalized wxyz quaternions")
    try:
        metadata = json.loads(str(data["metadata_json"].item()))
    except (ValueError, TypeError) as exc:
        raise ValueError("metadata_json must contain one JSON object") from exc
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must contain an object")
    if metadata.get("output_up_axis", "+Z") != "+Z":
        raise ValueError("Expected decoded metric Z-up motion")
    return {**data, "metadata": metadata, "fps": fps, "names": names}


def calibrated_heading(quats_wxyz):
    """Same mounting calibration as retarget_mink, without importing Mink.

    Heading is the calibrated root's projected +X axis, not travel direction.
    Near-vertical forward axes have undefined heading; do not invent a yaw.
    """
    human = Rotation.from_quat(quats_wxyz[:, [1, 2, 3, 0]]).as_matrix() @ ASF_TO_WORLD.T
    if np.linalg.norm(human[0, :2, 0]) < 1e-6:
        return None
    yaw0 = np.arctan2(human[0, 1, 0], human[0, 0, 0])
    heading = Rotation.from_euler("z", -yaw0).as_matrix()
    calibrated = heading @ human @ human[0].T @ heading.T
    if np.any(np.linalg.norm(calibrated[:, :2, 0], axis=1) < .1):
        return None
    return np.unwrap(np.arctan2(calibrated[:, 1, 0], calibrated[:, 0, 0]))


def clear_intervals(mask, times, frame_numbers, minimum_span_s):
    """Observed clear sample spans only; never bridge missing source frames."""
    result, begin = [], None
    for index in range(len(mask) + 1):
        gap = index > 0 and index < len(mask) and frame_numbers[index] - frame_numbers[index - 1] != 1
        if begin is not None and (index == len(mask) or not mask[index] or gap):
            last = index - 1
            span = float(times[last] - times[begin])
            if span + 1e-12 >= minimum_span_s:
                result.append({"start_sample": begin, "last_clear_sample": last,
                               "start_time_s": float(times[begin]), "last_clear_time_s": float(times[last]),
                               "observed_clear_span_s": span,
                               "left_censored": bool(begin == 0 or frame_numbers[begin] - frame_numbers[begin - 1] != 1),
                               "right_censored": bool(index == len(mask) or gap)})
            begin = None
        if index < len(mask) and mask[index] and begin is None:
            begin = index
    return result


def analyze_motion(motion, motion_id, *, clearance_m=.05, initial_window_s=.25,
                   minimum_clear_span_s=.04, baseline_warning_m=.03):
    if any(not np.isfinite(x) or x <= 0 for x in (clearance_m, initial_window_s, baseline_warning_m)):
        raise ValueError("Clearance, baseline window and warning threshold must be positive and finite")
    if not np.isfinite(minimum_clear_span_s) or minimum_clear_span_s < 0:
        raise ValueError("Minimum clear span must be finite and nonnegative")
    p, t, names = motion["joint_positions_world_m"], motion["time_s"], motion["names"]
    root = p[:, names.index("root")]
    first = t < t[0] + initial_window_s
    last = t > t[-1] - initial_window_s
    warnings, sides = [], {}
    for side, prefix in (("left", "l"), ("right", "r")):
        foot_names = [name for name in (prefix + "foot", prefix + "toes") if name in names]
        if not foot_names:
            foot_names = [prefix + "tibia"]
            warnings.append(f"{side}: no forefoot/toe nodes; ankle-only floor proxy is not a sole surface")
        heights = p[:, [names.index(name) for name in foot_names], 2].min(axis=1)
        initial = float(np.percentile(heights[first], 5))
        final = float(np.percentile(heights[last], 5))
        sides[side] = {"nodes": foot_names, "initial_fifth_percentile_m": initial,
                       "final_fifth_percentile_m": final, "final_minus_initial_m": final - initial,
                       "height_min_max_m": [float(heights.min()), float(heights.max())]}
        if abs(final - initial) > baseline_warning_m:
            warnings.append(f"{side}: end/start foot baseline differs by more than {baseline_warning_m:g} m; preserved, not corrected")
    foot_heights = np.column_stack([
        p[:, [names.index(name) for name in sides[side]["nodes"]], 2].min(axis=1)
        for side in ("left", "right")])
    floor = float(np.percentile(foot_heights[first].min(axis=1), 5))
    if foot_heights.min() < floor - baseline_warning_m:
        warnings.append("Foot endpoints descend below the initial floor estimate; support level or initial-standing assumption may be wrong")
    if np.any(np.ptp(foot_heights[first], axis=0) > baseline_warning_m):
        warnings.append("Foot heights vary in the initial window; initial standing is not established")
    if abs(sides["left"]["initial_fifth_percentile_m"] - sides["right"]["initial_fifth_percentile_m"]) > baseline_warning_m:
        warnings.append("Initial left/right foot baselines differ; both feet may not share one support surface")
    gaps = int(np.count_nonzero(np.diff(motion["frame_numbers"]) != 1))
    if gaps:
        warnings.append("Missing source frames: intervals are split at gaps; plots join samples for display only")
    yaw = None
    if "joint_rotations_world_wxyz" in motion:
        yaw = calibrated_heading(motion["joint_rotations_world_wxyz"][:, names.index("root")])
    if yaw is None:
        warnings.append("Calibrated heading unavailable: rotations missing or root forward axis near vertical")
    clear = np.all(foot_heights > floor + clearance_m, axis=1)
    summary = {
        "id": motion_id, "frames": len(t), "fps": motion["fps"],
        "first_frame_number": int(motion["frame_numbers"][0]), "last_frame_number": int(motion["frame_numbers"][-1]),
        "time_start_s": float(t[0]), "time_end_s": float(t[-1]), "timestamp_span_s": float(t[-1] - t[0]),
        "source_frame_gaps": gaps,
        "pelvis": {"is_whole_body_com": False, "displacement_xyz_m": (root[-1] - root[0]).tolist(),
                   "displacement_xy_norm_m": float(np.linalg.norm(root[-1, :2] - root[0, :2])),
                   "coordinate_min_xyz_m": root.min(axis=0).tolist(), "coordinate_max_xyz_m": root.max(axis=0).tolist(),
                   "coordinate_range_xyz_m": np.ptp(root, axis=0).tolist(),
                   "height_excursion_m": float(np.ptp(root[:, 2])),
                   "max_rise_from_initial_m": float(root[:, 2].max() - root[0, 2])},
        "calibrated_root_heading": {"available": yaw is not None,
                    "change_deg": None if yaw is None else float(np.degrees(yaw[-1] - yaw[0])),
                    "excursion_deg": None if yaw is None else float(np.degrees(np.ptp(yaw))),
                    "method": "retarget_mink initial mounting calibration; unwrapped projected root +X heading; not travel direction"},
        "feet": sides,
        "kinematic_both_feet_clear": {"measured_contact": False, "floor_estimate_m": floor,
                    "floor_method": "5th percentile of lowest left/right foot endpoint in initial window; analysis-only estimate",
                    "initial_window_s": initial_window_s, "clearance_above_floor_m": clearance_m,
                    "minimum_observed_clear_span_s": minimum_clear_span_s,
                    "intervals": clear_intervals(clear, t, motion["frame_numbers"], minimum_clear_span_s),
                    "limitations": "Initial standing and one level floor are assumptions. Endpoints are not soles; no contact forces, takeoff/landing labels, or dynamics are measured."},
        "warnings": warnings, "source_metadata": motion["metadata"],
        "robot_retargeted": False, "physics_validated": False, "policy_trained": False,
        "motion_modified": False, "retimed": False, "scaled": False, "clipped": False,
    }
    return summary


def render_overview(clips, output):
    # Keep plotting caches, as well as artifacts, inside the requested output.
    previous = os.environ.get("MPLCONFIGDIR")
    with tempfile.TemporaryDirectory(prefix=".plot-cache-", dir=output.parent) as cache:
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(len(clips), 4, figsize=(16, 3.1 * len(clips)), squeeze=False)
            for row, (motion, summary) in enumerate(clips):
                p, t, names = motion["joint_positions_world_m"], motion["time_s"], motion["names"]
                pelvis = p[:, names.index("root")]
                xy, height, feet, skeleton = axes[row]
                xy.plot(pelvis[:, 0], pelvis[:, 1], color="black", linewidth=1)
                xy.scatter(pelvis[[0, -1], 0], pelvis[[0, -1], 1], c=["green", "red"], s=22)
                xy.set(title=f"{summary['id']}: pelvis XY (start green)", xlabel="X (m)", ylabel="Y (m)", aspect="equal", adjustable="datalim")
                height.plot(t, pelvis[:, 2], color="black")
                height.set(title="Pelvis height (not COM)", xlabel="Source time (s)", ylabel="Z (m)")
                for side, prefix, color in (("left", "l", "tab:blue"), ("right", "r", "tab:orange")):
                    feet.plot(t, p[:, names.index(prefix + "tibia"), 2], color=color, label=f"{side} ankle", linewidth=.9)
                    z = p[:, [names.index(n) for n in summary["feet"][side]["nodes"]], 2].min(axis=1)
                    feet.plot(t, z, color=color, linestyle="--", label=f"{side} foot min", linewidth=1)
                floor = summary["kinematic_both_feet_clear"]["floor_estimate_m"]
                feet.axhline(floor, color="gray", linestyle=":", linewidth=.8)
                for interval in summary["kinematic_both_feet_clear"]["intervals"]:
                    feet.axvspan(interval["start_time_s"], interval["last_clear_time_s"], color="green", alpha=.12)
                feet.set(title="Both feet: shade = height heuristic", xlabel="Source time (s)", ylabel="Z (m)")
                feet.legend(fontsize=6, ncol=2)
                indices = sorted(set([0, int(np.argmax(pelvis[:, 2])), len(t) - 1]))
                for index, color in zip(indices, ("green", "purple", "red")):
                    for child, parent in enumerate(motion["parent_indices"]):
                        if parent >= 0:
                            points = p[index, [int(parent), child]]
                            skeleton.plot(points[:, 0], points[:, 2], color=color, linewidth=.7, alpha=.65)
                    skeleton.plot([], [], color=color, label=f"t={t[index]:.2f}s")
                skeleton.set(title="Skeleton XZ: start / peak / end", xlabel="X (m)", ylabel="Z (m)", aspect="equal", adjustable="datalim")
                skeleton.legend(fontsize=6)
                for ax in axes[row]:
                    ax.grid(alpha=.2)
            fig.suptitle("Human mocap screening only — no robot retargeting, measured contact, or physics validation", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, .965))
            fig.savefig(output, dpi=130)
            plt.close(fig)
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def screen_motions(motions, output_dir, **analysis_options):
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Output must be new or empty; preserving existing results")
    if not motions or len({name for name, _ in motions}) != len(motions) or any(not name.strip() for name, _ in motions):
        raise ValueError("Supply at least one motion with a unique nonempty ID")
    clips = []
    for motion_id, path in motions:
        path = Path(path)
        before = sha256(path)
        motion = load_human_motion(path)
        summary = analyze_motion(motion, motion_id, **analysis_options)
        summary["input"] = {"filename": path.name, "npz_sha256": before}
        if sha256(path) != before:
            raise RuntimeError("Input motion changed while screening")
        clips.append((motion, summary))
    report = {"schema_version": 1, "kind": "human_kinematic_motion_screening",
              "robot_retargeted": False, "physics_validated": False, "policy_trained": False,
              "source_arrays_modified": False, "clips": [summary for _, summary in clips],
              "script_sha256": sha256(__file__)}
    # Validate JSON before making any output. Original embedded metadata is
    # retained verbatim, including source provenance and its floor assumptions.
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    output.mkdir(parents=True, exist_ok=True)
    render_overview(clips, output / "overview.png")
    (output / "summary.json").write_text(serialized + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", action="append", required=True, metavar="ID=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clearance-m", type=float, default=.05)
    parser.add_argument("--initial-window-s", type=float, default=.25)
    parser.add_argument("--minimum-clear-span-s", type=float, default=.04)
    args = parser.parse_args()
    motions = []
    for spec in args.motion:
        if "=" not in spec:
            parser.error("Each --motion must have the form ID=PATH")
        motion_id, path = spec.split("=", 1)
        if not path:
            parser.error("Motion path cannot be empty")
        motions.append((motion_id, Path(path)))
    try:
        report = screen_motions(motions, args.output_dir, clearance_m=args.clearance_m,
                                initial_window_s=args.initial_window_s, minimum_clear_span_s=args.minimum_clear_span_s)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Screening error: {exc}\n")
    print(json.dumps({"clips": len(report["clips"]), "summary": str(args.output_dir / "summary.json"),
                      "overview": str(args.output_dir / "overview.png")}, indent=2))


if __name__ == "__main__":
    main()
