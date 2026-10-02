"""Render recorded PhysX policy states beside their kinematic reference.

MuJoCo is only the common mesh renderer: mj_forward is used, NEVER mj_step.
The actual panel replays evaluation data up to the first episode's terminal
state.  An early-stopped panel freezes visibly while the reference continues.
An optional MuJoCo policy rollout is added only to the height chart, keeping
the visual comparison to two robot panels plus one simple time series.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
BG, PANEL, TEXT, MUTED = "#0a1220", "#101d31", "#e2e8f0", "#94a3b8"
ACTUAL, REFERENCE, OTHER = "#38bdf8", "#f6d56a", "#fb923c"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


class PoseClip:
    """Named joints and continuous wxyz root rotation at explicit timestamps."""

    def __init__(self, times, root_position, root_quaternion, joint_position, joint_names):
        self.times = np.asarray(times, dtype=float)
        self.root = np.asarray(root_position, dtype=float)
        self.quat = np.asarray(root_quaternion, dtype=float)
        self.joints = np.asarray(joint_position, dtype=float)
        self.joint_names = list(np.asarray(joint_names).astype(str))
        count = len(self.times)
        if count < 2 or np.any(np.diff(self.times) <= 0) or not np.isfinite(self.times).all():
            raise ValueError("A pose clip needs at least two finite, increasing timestamps")
        for name, value, shape in (
            ("root position", self.root, (count, 3)), ("root quaternion", self.quat, (count, 4)),
            ("joints", self.joints, (count, len(self.joint_names))),
        ):
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid {name}: expected finite array {shape}, got {value.shape}")
        norms = np.linalg.norm(self.quat, axis=-1, keepdims=True)
        if np.any(norms < 1e-8):
            raise ValueError("Root quaternions cannot be zero")
        self.quat = self.quat / norms
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("Duplicate joint names in pose clip")
        self.rotation = Slerp(self.times, Rotation.from_quat(self.quat[:, [1, 2, 3, 0]]))

    def sample(self, time_s):
        at = float(np.clip(time_s, self.times[0], self.times[-1]))
        root = np.array([np.interp(at, self.times, self.root[:, index]) for index in range(3)])
        joints = np.array([np.interp(at, self.times, self.joints[:, index]) for index in range(self.joints.shape[1])])
        quat = self.rotation(at).as_quat()[[3, 0, 1, 2]]
        return root, quat, joints


def load_rollout(arrays, environment=0):
    """Load evaluator [T,N,...] data or a named single MuJoCo [T,...] clip."""
    times = np.asarray(arrays["time_s"])
    # Prefer explicit named arrays: some MuJoCo exports additionally store a
    # native-order qpos while their joint_pos uses a named policy order.
    root_key = next((name for name in ("root_position_m", "root_pos") if name in arrays), None)
    quat_key = next((name for name in ("root_quaternion_wxyz", "root_quat_wxyz") if name in arrays), None)
    joint_key = next((name for name in ("joint_position_rad", "joint_pos") if name in arrays), None)
    if root_key is not None and quat_key is not None and joint_key is not None:
        qpos = np.concatenate((arrays[root_key], arrays[quat_key], arrays[joint_key]), axis=-1)
    elif "qpos" in arrays:
        qpos = np.asarray(arrays["qpos"])
    else:
        raise ValueError("Rollout needs qpos or explicit root position/quaternion and named joint positions")
    if qpos.ndim == 3:
        if not 0 <= environment < qpos.shape[1]:
            raise ValueError(f"Environment index {environment} not present in rollout")
        if "valid_mask" not in arrays:
            raise ValueError("Batched evaluation needs valid_mask to exclude auto-reset trajectories")
        valid = np.asarray(arrays["valid_mask"], dtype=bool)[:, environment]
        qpos = qpos[:, environment]
    elif qpos.ndim == 2:
        valid = np.asarray(arrays.get("valid_mask", np.ones(len(times), dtype=bool)), dtype=bool)
        if valid.ndim == 2:
            valid = valid[:, environment]
    else:
        raise ValueError(f"Unsupported rollout qpos shape {qpos.shape}")
    if valid.shape != times.shape or qpos.shape[0] != len(times):
        raise ValueError("Rollout timestamp, qpos and valid_mask shapes disagree")
    valid_indices = np.flatnonzero(valid)
    if not np.array_equal(valid_indices, np.arange(len(valid_indices))):
        raise ValueError("Only a contiguous first-episode valid prefix may be rendered")
    qpos, times = qpos[valid], times[valid]
    if qpos.shape[1] != 7 + len(arrays["joint_names"]):
        raise ValueError("qpos does not match floating root + named joints")
    return PoseClip(times, qpos[:, :3], qpos[:, 3:7], qpos[:, 7:], arrays["joint_names"])


def load_reference(arrays):
    names = list(np.asarray(arrays["body_names"]).astype(str))
    if "base_Link" not in names:
        raise ValueError("Reference does not contain the named base_Link body")
    base = names.index("base_Link")
    fps = float(np.asarray(arrays["fps"]).reshape(-1)[0])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid reference FPS")
    times = np.arange(len(arrays["joint_pos"])) / fps
    return PoseClip(times, arrays["body_pos_w"][:, base], arrays["body_quat_w"][:, base],
                    arrays["joint_pos"], arrays["joint_names"])


def named_qpos(model, clip, time_s):
    root, quat, joints = clip.sample(time_s)
    if model.nq != 7 + len(clip.joint_names) or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Rendering model must have a floating root and the same named hinge joints")
    result = model.qpos0.copy()
    result[:7] = np.r_[root, quat]
    for source_index, name in enumerate(clip.joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Named hinge joint is not present in rendering model: {name}")
        result[model.jnt_qposadr[joint_id]] = joints[source_index]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "assets/robots/WF_TRON1A/mujoco/robot.xml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-index", type=int, default=0)
    parser.add_argument("--mujoco-trajectory", type=Path, help="Optional additional actual rollout height curve only")
    parser.add_argument("--playback-speed", type=float, default=0.5)
    args = parser.parse_args()
    if args.playback_speed <= 0 or args.env_index < 0:
        parser.error("playback-speed must be positive and env-index nonnegative")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose an empty output directory; existing renders are preserved")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "render_report.json"
    report = {
        "status": "initializing", "renderer": f"MuJoCo {mujoco.__version__}",
        "gl_backend": os.environ["MUJOCO_GL"], "simulation_steps": 0,
        "method": "common-mesh visual reconstruction of recorded poses; no resimulation",
        "trajectory": str(args.trajectory.resolve()), "motion_file": str(args.motion_file.resolve()),
        "model": str(args.model.resolve()), "environment_index": args.env_index,
        "jump_success_claimed": False,
    }
    renderer = None
    writer = None
    fig = None
    try:
        actual = load_rollout(read_npz(args.trajectory), args.env_index)
        reference = load_reference(read_npz(args.motion_file))
        other = load_rollout(read_npz(args.mujoco_trajectory)) if args.mujoco_trajectory else None
        evaluation_path = args.trajectory.parent / "report.json"
        evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.is_file() else {}
        episodes = evaluation.get("summary", {}).get("episodes", [])
        episode = next((item for item in episodes if item.get("environment") == args.env_index), {})
        if evaluation and evaluation.get("simulator") != "IsaacLab/PhysX":
            raise ValueError("Primary panel must use the evaluator's recorded IsaacLab/PhysX trajectory")
        expected_motion_hash = evaluation.get("motion_file_sha256")
        motion_hash = sha256(args.motion_file)
        if expected_motion_hash and expected_motion_hash != motion_hash:
            raise ValueError("Evaluation and displayed reference have different hashes")
        end_time = float(max(actual.times[-1], reference.times[-1]))
        # At 50 Hz source and 25 fps video, every reference/control sample is
        # shown and playback is half speed. Physics data are not retimed.
        sample_fps = 50.0
        frame_times = np.arange(int(np.ceil(end_time * sample_fps)) + 1) / sample_fps
        video_fps = sample_fps * args.playback_speed
        model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
        if model.nmesh == 0:
            raise ValueError("The visualization model has no official robot meshes")
        for clip in (actual, reference) + ((other,) if other else ()):
            named_qpos(model, clip, clip.times[0])  # eager name/schema checks
        for geom in range(model.ngeom):
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH and model.geom(geom).name != "floor":
                model.geom_rgba[geom, 3] = 0.0
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, 680)
        model.vis.global_.offheight = max(model.vis.global_.offheight, 470)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, width=680, height=470)
        options = mujoco.MjvOption()
        options.sitegroup[:] = 0
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        all_roots = np.concatenate((actual.root, reference.root), axis=0)
        low, high = all_roots.min(0), all_roots.max(0)
        camera.lookat[:] = [0.5 * (low[0] + high[0]), 0.5 * (low[1] + high[1]),
                            max(0.55, 0.5 * high[2])]
        camera.distance = max(2.8, float(np.max(high - low)) * 1.8 + 1.8)
        camera.azimuth, camera.elevation = 125.0, -15.0

        def robot_image(clip, time_s):
            data.qpos[:] = named_qpos(model, clip, time_s)
            data.qvel[:] = 0.0
            # Kinematic reconstruction only; never advance simulation time.
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=options)
            return renderer.render().copy()

        plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT, "axes.labelcolor": MUTED})
        fig = plt.figure(figsize=(14, 9), dpi=100, facecolor=BG)
        grid = fig.add_gridspec(2, 2, left=0.045, right=0.97, top=0.84, bottom=0.15,
                               height_ratios=(2.2, 1.0), hspace=0.32, wspace=0.04)
        actual_ax = fig.add_subplot(grid[0, 0])
        ref_ax = fig.add_subplot(grid[0, 1])
        for ax in (actual_ax, ref_ax):
            ax.set_axis_off()
        actual_ax.set_title("RECORDED POLICY  /  Isaac PhysX", loc="left", color=ACTUAL, fontsize=12, pad=10)
        ref_ax.set_title("KINEMATIC TARGET  /  no dynamics", loc="left", color=REFERENCE, fontsize=12, pad=10)
        actual_artist = actual_ax.imshow(robot_image(actual, 0.0))
        reference_artist = ref_ax.imshow(robot_image(reference, 0.0))
        stopped_label = actual_ax.text(0.03, 0.93, "", transform=actual_ax.transAxes, va="top",
                                       color="#ff7777", fontsize=11, weight="bold",
                                       bbox={"facecolor": BG, "alpha": 0.85, "edgecolor": "none"})
        chart = fig.add_subplot(grid[1, :])
        chart.set_facecolor(PANEL)
        chart.tick_params(colors=MUTED, labelsize=9)
        for spine in chart.spines.values():
            spine.set_color("#334155")
        chart.grid(color="#64748b", alpha=0.16)
        chart.plot(reference.times, reference.root[:, 2], color=REFERENCE, lw=1.6, label="Kinematic reference")
        chart.plot(actual.times, actual.root[:, 2], color=ACTUAL, lw=2.0, label="Actual PhysX policy")
        if other:
            chart.plot(other.times, other.root[:, 2], color=OTHER, lw=1.5, label="Actual MuJoCo policy")
        chart.set_xlim(0.0, end_time)
        chart.set_xlabel("Simulation time (s)", fontsize=10)
        chart.set_ylabel("Base-link height (m)", fontsize=10)
        chart.legend(loc="best", fontsize=9, frameon=False, labelcolor=TEXT)
        cursor = chart.axvline(0.0, color=TEXT, lw=1, alpha=0.75)
        if actual.times[-1] < reference.times[-1] - 1e-6:
            chart.axvline(actual.times[-1], color="#ff7777", lw=1, ls="--", alpha=0.8)
        fig.text(0.045, 0.945, "TRON1 WF  /  learned motion tracking", fontsize=21, weight="bold")
        fig.text(0.045, 0.900, "Actual recorded robot state vs. GMR reference", fontsize=12, color=MUTED)
        clock = fig.text(0.81, 0.945, "0.00 s", fontsize=15, color=ACTUAL)
        measured_gain = float(np.max(actual.root[:, 2]) - actual.root[0, 2])
        reference_gain = float(np.max(reference.root[:, 2]) - reference.root[0, 2])
        stop_reason = episode.get("end_reason", "end of available recorded trajectory")
        outcome_note = "Full reference survived" if episode.get("completed_full_reference") else "Full reference not verified"
        fig.text(0.045, 0.083,
                 f"Actual base rise: {measured_gain:.3f} m  |  Reference rise: {reference_gain:.3f} m  |  {outcome_note}",
                 fontsize=10, color=TEXT)
        fig.text(0.045, 0.053,
                 f"{args.playback_speed:g}x playback. One first episode only; an early terminal pose freezes visibly. No jump-success claim.",
                 fontsize=9, color=MUTED)
        fig.text(0.045, 0.026,
                 "Both panels use MuJoCo meshes for visualization only (zero simulation steps); actual dynamics were recorded in PhysX.",
                 fontsize=8.5, color=MUTED)

        def update(time_s):
            actual_artist.set_data(robot_image(actual, time_s))
            reference_artist.set_data(robot_image(reference, time_s))
            cursor.set_xdata([time_s, time_s])
            clock.set_text(f"{time_s:.2f} s")
            stopped_label.set_text(f"RECORDING ENDED\n{stop_reason}" if time_s > actual.times[-1] + 1e-8 else "")
            fig.canvas.draw()

        preview_time = float(reference.times[np.argmax(reference.root[:, 2])])
        update(preview_time)
        fig.savefig(output / "overview.png", dpi=125, facecolor=BG)
        width, height = fig.canvas.get_width_height()
        writer = imageio_ffmpeg.write_frames(
            str(output / "tracking_comparison.mp4"), (width, height), fps=video_fps,
            codec="libx264", pix_fmt_out="yuv420p", quality=8, macro_block_size=1,
            output_params=["-movflags", "+faststart"],
        )
        writer.send(None)
        for frame_time in frame_times:
            update(float(frame_time))
            writer.send(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        writer.close()
        writer = None
        decode = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(output / "tracking_comparison.mp4"),
             "-f", "null", "-"], capture_output=True, text=True, check=False,
        )
        if decode.returncode:
            raise RuntimeError(f"Produced video failed full decode: {decode.stderr[-2000:]}")
        report.update({
            "status": "rendered", "trajectory_sha256": sha256(args.trajectory), "motion_sha256": motion_hash,
            "actual_samples": len(actual.times), "actual_sample_span_s": float(actual.times[-1] - actual.times[0]),
            "reference_samples": len(reference.times), "reference_sample_span_s": float(reference.times[-1]),
            "actual_base_link_height_gain_m": measured_gain, "reference_base_link_height_gain_m": reference_gain,
            "first_episode_only": True, "terminal_pose_included": True,
            "source_episode_result": episode, "camera_fixed_for_both_panels": True,
            "joint_mapping": "explicit names to MuJoCo qpos addresses, not positional copying",
            "video_frames": len(frame_times), "video_fps": video_fps, "playback_speed": args.playback_speed,
            "video_resolution": [width, height], "full_decode_verified": True,
            "optional_mujoco_height_curve": str(args.mujoco_trajectory.resolve()) if args.mujoco_trajectory else None,
        })
        page = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>TRON1 tracking</title>
<style>body{{margin:2rem auto;max-width:1400px;padding:0 1rem;background:{BG};color:{TEXT};font:16px system-ui}}video,img{{width:100%;border-radius:10px}}a{{color:{ACTUAL}}}p{{line-height:1.6}}</style>
<h1>TRON1：實際策略軌跡與參考動作</h1>
<p>左側為 Isaac／PhysX 實際策略紀錄；右側為 GMR 運動學參考。兩側僅以 MuJoCo 網格重建畫面，未重新模擬。</p>
<video controls autoplay muted loop playsinline src="tracking_comparison.mp4"></video>
<p>實際底座上升 {measured_gain:.3f} m；參考上升 {reference_gain:.3f} m。提前終止後明確凍結最後姿態，不拼接重置後回合。此預覽不宣稱跳躍成功。</p>
<p><a href="tracking_comparison.mp4">下載影片</a> · <a href="overview.png">關鍵畫面</a> · <a href="render_report.json">渲染紀錄</a></p>
<p>紀錄結束原因：{html.escape(str(stop_reason))}</p></html>'''
        (output / "index.html").write_text(page, encoding="utf-8")
        print(json.dumps({"output_directory": str(output), "video": "tracking_comparison.mp4",
                          "actual_base_link_height_gain_m": measured_gain, "simulation_steps": 0}, indent=2))
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        if fig is not None:
            plt.close(fig)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
