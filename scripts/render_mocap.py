"""Render real CMU ASF/AMC FK and export human reference trajectories.

This is a measured HUMAN kinematic reference, not a TRON1 retarget, policy,
physics simulation, or dynamically validated jump. No generative images are used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image
import imageio_ffmpeg

from cmu_motion import load_motion

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"background": "#0a1220", "panel": "#101d31", "text": "#e2e8f0",
          "muted": "#94a3b8", "left": "#38bdf8", "right": "#fb923c", "root": "#f6d56a"}


def intervals(mask):
    """Inclusive runs, with no smoothing or claimed force-sensor contact."""
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1) - 1))


def style_axis(ax):
    ax.set_facecolor(COLORS["panel"])
    ax.tick_params(colors=COLORS["muted"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.xaxis.label.set_color(COLORS["muted"])
    ax.yaxis.label.set_color(COLORS["muted"])
    ax.grid(color="#64748b", alpha=0.15)


def setup_3d(ax, points, small=False):
    style_axis(ax)
    low, high = np.min(points, axis=(0, 1)), np.max(points, axis=(0, 1))
    center = 0.5 * (low + high)
    half = max(0.75, (high[0] - low[0]) / 2 + 0.12, (high[1] - low[1]) / 2 + 0.12)
    top = max(2.2, high[2] + 0.15)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(-0.07, top)
    ax.set_box_aspect((2 * half, 2 * half, top + 0.07))
    ax.view_init(elev=16, azim=-62)
    floor = [[center[0] - half, center[1] - half, 0], [center[0] + half, center[1] - half, 0],
             [center[0] + half, center[1] + half, 0], [center[0] - half, center[1] + half, 0]]
    ax.add_collection3d(Poly3DCollection([floor], facecolors="#24364b", edgecolors="#52657d", alpha=0.35))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.06, 0.1, 0.17, 1))
        axis._axinfo["grid"]["color"] = (0.4, 0.5, 0.6, 0.16)
    ax.set_xlabel("X (m)", labelpad=7)
    ax.set_ylabel("Y (m)", labelpad=7)
    ax.set_zlabel("Z (m)", color=COLORS["muted"], labelpad=8)
    if small:
        ax.set_axis_off()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asf", type=Path, default=ROOT / "assets/mocap/CMU/16.asf")
    parser.add_argument("--amc", type=Path, default=ROOT / "assets/mocap/CMU/16_03.amc")
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/cmu-16_03")
    args = parser.parse_args()
    if args.fps != 120:
        parser.error("This pinned CMU 16_03 source is 120 Hz; changing fps would alter its timing")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "render_report.json"
    if report_path.exists():
        parser.error("Choose a new output directory; an existing render report will not be overwritten")
    report = {"status": "starting", "kind": "human_mocap_reference", "robot_retargeted": False,
              "physics_validated": False, "policy_trained": False}

    def save_report():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save_report()
    try:
        motion = load_motion(args.asf, args.amc, fps=args.fps)
        positions = np.asarray(motion["joint_positions_world_m"])
        names = list(motion["joint_names"])
        parents = np.asarray(motion["parent_indices"])
        times = np.asarray(motion["time_s"])
        fps = float(motion["fps"])
        frames = len(times)
        if positions.shape != (frames, len(names), 3) or not np.isfinite(positions).all():
            raise ValueError("Invalid motion positions")
        index = {name: i for i, name in enumerate(names)}
        root = positions[:, index["root"]]
        left = np.min(positions[:, [index["lfoot"], index["ltoes"]], 2], axis=1)
        right = np.min(positions[:, [index["rfoot"], index["rtoes"]], 2], axis=1)
        clearance_threshold = 0.06
        airborne = (left > clearance_threshold) & (right > clearance_threshold)
        flight_runs = [(int(a), int(b)) for a, b in intervals(airborne) if (b - a + 1) / fps >= 0.05]
        crouch_frame, apex_frame = int(np.argmin(root[:, 2])), int(np.argmax(root[:, 2]))
        longest = max(flight_runs, key=lambda run: run[1] - run[0], default=(apex_frame, apex_frame))
        report.update({"source_fps": fps, "frames": frames, "sample_span_s": float(times[-1] - times[0]),
                       "source_duration_s": frames / fps, "playback_speed": 0.5, "video_fps": fps / 4,
                       "joint_count": len(names), "metadata": motion["metadata"],
                       "pelvis_rise_from_first_frame_m": float(np.max(root[:, 2]) - root[0, 2]),
                       "pelvis_horizontal_net_displacement_m": float(np.linalg.norm(root[-1, :2] - root[0, :2])),
                       "estimated_flight_intervals_s": [[float(times[a]), float(times[b])] for a, b in flight_runs],
                       "flight_estimate_note": "Both foot/toe endpoint minima exceed 6 cm for >= 50 ms. Kinematic estimate, not measured contacts.",
                       "clearance_threshold_m": clearance_threshold,
                       "pose_note": "Root is pelvis, not whole-body center of mass. One constant floor alignment; original global motion retained."})
        save_report()

        # Preserve all 120 Hz input samples; the video alone is downsampled.
        arrays = {k: v for k, v in motion.items() if k != "metadata"}
        arrays["joint_velocities_world_mps"] = np.gradient(positions, times, axis=0, edge_order=2)
        arrays["foot_min_height_world_m"] = np.column_stack((left, right))
        arrays["airborne_estimate"] = airborne
        np.savez_compressed(output / "human_reference.npz", **arrays)
        (output / "motion_metadata.json").write_text(json.dumps(motion["metadata"], indent=2, allow_nan=False) + "\n", encoding="utf-8")
        selected = {"pelvis": "root", "left_hip": "lhipjoint", "left_knee": "lfemur", "left_ankle": "ltibia",
                    "left_toe": "ltoes", "right_hip": "rhipjoint", "right_knee": "rfemur", "right_ankle": "rtibia",
                    "right_toe": "rtoes", "chest": "thorax", "head": "head"}
        with (output / "reference_trajectories.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["source_frame", "time_s"] + [f"{name}_{axis}_m" for name in selected for axis in "xyz"]
                            + ["left_foot_min_z_m", "right_foot_min_z_m", "airborne_estimate"])
            for f in range(frames):
                writer.writerow([int(motion["frame_numbers"][f]), float(times[f])]
                                + positions[f, [index[v] for v in selected.values()]].ravel().tolist()
                                + [float(left[f]), float(right[f]), int(airborne[f])])
        sources = json.loads((ROOT / "config/mocap_sources.json").read_text(encoding="utf-8"))
        (output / "source_provenance.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
        for entry, path in zip(sources["cmu_16_03"]["files"], (args.asf, args.amc)):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != entry["sha256"]:
                raise ValueError(f"Source content does not match pinned CMU 16_03: {path}")

        plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": COLORS["text"], "axes.labelcolor": COLORS["muted"]})
        fig = plt.figure(figsize=(14, 8), dpi=100, facecolor=COLORS["background"])
        grid = fig.add_gridspec(2, 2, left=0.02, right=0.96, bottom=0.10, top=0.85,
                               width_ratios=(1.22, 1), height_ratios=(1, 1), wspace=0.18, hspace=0.40)
        ax = fig.add_subplot(grid[:, 0], projection="3d")
        setup_3d(ax, positions)
        pelvis_plot = fig.add_subplot(grid[0, 1])
        foot_plot = fig.add_subplot(grid[1, 1])
        for chart in (pelvis_plot, foot_plot):
            style_axis(chart)
            chart.set_xlim(times[0], times[-1])
            chart.set_xlabel("Source time (s)")
            for a, b in flight_runs:
                chart.axvspan(times[a], times[b], color="#818cf8", alpha=0.12)
        pelvis_plot.plot(times, root[:, 2], color=COLORS["root"], lw=2)
        pelvis_plot.set_ylabel("Pelvis height (m)")
        pelvis_plot.set_title("Pelvis trajectory  /  not whole-body COM", color=COLORS["text"], loc="left", fontsize=11, pad=13)
        pelvis_plot.set_ylim(max(0, root[:, 2].min() - 0.10), root[:, 2].max() + 0.10)
        foot_plot.plot(times, left, color=COLORS["left"], lw=1.8, label="Left foot / toe minimum")
        foot_plot.plot(times, right, color=COLORS["right"], lw=1.8, label="Right foot / toe minimum")
        foot_plot.axhline(0, color="#94a3b8", lw=1, alpha=0.5)
        foot_plot.axhline(clearance_threshold, color="#a5b4fc", lw=1, ls="--", alpha=0.5)
        foot_plot.set_ylabel("Foot keypoint height (m)")
        foot_plot.set_title("Foot trajectories  /  shaded = estimated flight", color=COLORS["text"], loc="left", fontsize=11, pad=13)
        foot_plot.set_ylim(min(-0.05, float(min(left.min(), right.min())) - 0.02), max(left.max(), right.max()) + 0.08)
        foot_plot.legend(loc="upper left", frameon=False, labelcolor=COLORS["muted"], fontsize=9)
        cursor_a = pelvis_plot.axvline(0, color="white", alpha=0.5, lw=1)
        cursor_b = foot_plot.axvline(0, color="white", alpha=0.5, lw=1)
        dot_a, = pelvis_plot.plot([], [], "o", color=COLORS["root"], ms=7)
        dot_l, = foot_plot.plot([], [], "o", color=COLORS["left"], ms=6)
        dot_r, = foot_plot.plot([], [], "o", color=COLORS["right"], ms=6)
        fig.text(0.045, 0.935, "CMU 16_03  /  HIGH JUMP", fontsize=21, fontweight="bold")
        fig.text(0.045, 0.892, "Recorded human motion  |  120 Hz source  |  0.5x playback  |  fixed world frame", fontsize=11, color=COLORS["muted"])
        status_text = fig.text(0.60, 0.93, "", fontsize=14, color=COLORS["root"])
        fig.text(0.045, 0.045, "Human FK reference only. No robot retarget, learned policy, or physics validation. Contact labels are estimates.",
                 fontsize=10, color=COLORS["muted"])
        visible = [i for i, name in enumerate(names) if not any(w in name for w in ("fingers", "thumb"))]
        edges = [(int(parents[i]), i) for i in visible if parents[i] >= 0]
        lines = []
        for parent, child in edges:
            name = names[child]
            color = COLORS["left"] if name.startswith("l") and name not in ("lowerback", "lowerneck") else COLORS["right"] if name.startswith("r") else COLORS["text"]
            line, = ax.plot([], [], [], color=color, lw=3.2, solid_capstyle="round")
            lines.append(line)
        pelvis_dot, = ax.plot([], [], [], "o", color=COLORS["root"], ms=8)
        head_dot, = ax.plot([], [], [], "o", color=COLORS["text"], ms=11)
        for name, color in (("root", COLORS["root"]), ("ltoes", COLORS["left"]), ("rtoes", COLORS["right"])):
            trace = positions[:, index[name]]
            ax.plot(trace[:, 0], trace[:, 1], trace[:, 2], color=color, lw=1, alpha=0.35)

        def update(frame):
            points = positions[frame]
            for line, (parent, child) in zip(lines, edges):
                pair = points[[parent, child]]
                line.set_data_3d(pair[:, 0], pair[:, 1], pair[:, 2])
            for artist, name in ((pelvis_dot, "root"), (head_dot, "head")):
                p = points[index[name]]
                artist.set_data_3d([p[0]], [p[1]], [p[2]])
            for cursor in (cursor_a, cursor_b):
                cursor.set_xdata([times[frame], times[frame]])
            dot_a.set_data([times[frame]], [root[frame, 2]])
            dot_l.set_data([times[frame]], [left[frame]])
            dot_r.set_data([times[frame]], [right[frame]])
            if airborne[frame]:
                phase = "FLIGHT (estimated)"
            elif frame < longest[0] and root[frame, 2] < root[0, 2] - 0.06:
                phase = "CROUCH / PUSH-OFF"
            elif frame > longest[1] and frame < min(frames - 1, longest[1] + round(0.6 * fps)):
                phase = "LAND / RECOVER"
            else:
                phase = "GROUND / PREPARATION"
            status_text.set_text(f"{times[frame]:.2f} s  |  {phase}")
            fig.canvas.draw()

        update(apex_frame)
        fig.savefig(output / "overview.png", dpi=125, facecolor=fig.get_facecolor())
        width, height = fig.canvas.get_width_height()
        writer = imageio_ffmpeg.write_frames(str(output / "mocap_reference.mp4"), (width, height),
                                             fps=fps / 4, codec="libx264", pix_fmt_out="yuv420p",
                                             quality=7, macro_block_size=2, output_params=["-movflags", "+faststart"])
        writer.send(None)
        gif_frames = []
        render_indices = np.arange(0, frames, 2)
        try:
            for number, frame in enumerate(render_indices):
                update(int(frame))
                rgb = np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
                writer.send(rgb)
                if number % 3 == 0:
                    thumb = Image.fromarray(rgb).resize((840, 480), Image.Resampling.LANCZOS)
                    gif_frames.append(thumb.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
                if number % 50 == 0:
                    print(f"RENDER {number + 1}/{len(render_indices)}", flush=True)
        finally:
            writer.close()
            plt.close(fig)
        gif_frames[0].save(output / "mocap_reference.gif", save_all=True, append_images=gif_frames[1:],
                           duration=round(6000 / fps * 2), loop=0, optimize=False)
        verify = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(output / "mocap_reference.mp4"),
                                 "-f", "null", "-"], capture_output=True, text=True, check=True)
        report["video_decode_verified"] = verify.returncode == 0
        report["video_frames"] = len(render_indices)
        report["video_duration_s"] = len(render_indices) / (fps / 4)
        report["video_size_px"] = [width, height]

        fig_key = plt.figure(figsize=(18, 5), dpi=100, facecolor=COLORS["background"])
        snapshots = [(0, "START"), (crouch_frame, "CROUCH"), (longest[0], "LIFT-OFF*"),
                     (apex_frame, "APEX"), (min(frames - 1, longest[1] + 1), "TOUCHDOWN*"), (frames - 1, "END")]
        for number, (frame, label) in enumerate(snapshots):
            subplot = fig_key.add_subplot(1, 6, number + 1, projection="3d")
            setup_3d(subplot, positions, small=True)
            for parent, child in edges:
                pair = positions[frame, [parent, child]]
                name = names[child]
                color = COLORS["left"] if name.startswith("l") and name not in ("lowerback", "lowerneck") else COLORS["right"] if name.startswith("r") else COLORS["text"]
                subplot.plot(pair[:, 0], pair[:, 1], pair[:, 2], color=color, lw=2.2)
            subplot.set_title(f"{label}\n{times[frame]:.2f} s", fontsize=11, color=COLORS["text"], pad=0)
        fig_key.suptitle("CMU 16_03 / recorded skeleton keyframes", fontsize=18, color=COLORS["text"], y=0.98)
        fig_key.text(0.04, 0.04, "* Timing estimated from foot keypoints, not measured ground contact. World scale and floor offset are fixed across all frames.", fontsize=10, color=COLORS["muted"])
        fig_key.subplots_adjust(left=0, right=1, top=0.88, bottom=0.12, wspace=-0.10)
        fig_key.savefig(output / "keyframes.png", dpi=125, facecolor=fig_key.get_facecolor())
        plt.close(fig_key)

        page = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CMU 16_03 動捕參考軌跡</title><style>body{margin:32px auto;padding:0 20px;max-width:1200px;background:#0a1220;color:#e2e8f0;font:16px/1.7 system-ui,sans-serif}h1{margin-bottom:0}p{color:#aebed2}video,img{width:100%;border-radius:12px;background:#101d31}a{color:#67d4ff}nav{display:flex;gap:24px;flex-wrap:wrap;margin:18px 0}.note{border-left:3px solid #f6d56a;padding:10px 18px;background:#101d31}small{color:#94a3b8}</style>
<h1>CMU 16_03：跳躍動捕參考軌跡</h1><p>410 幀 / 120 Hz 原始動捕。影片為 0.5 倍速；資料檔保留全部原始取樣。</p>
<video controls autoplay muted loop playsinline poster="overview.png" src="mocap_reference.mp4"></video>
<nav><a href="mocap_reference.mp4">MP4 影片</a><a href="mocap_reference.gif">GIF 動畫</a><a href="human_reference.npz">NPZ 軌跡</a><a href="reference_trajectories.csv">CSV 關鍵點</a><a href="render_report.json">驗證報告</a></nav>
<p class="note">這是人體骨架的運動學參考，尚未重定向到 TRON1，也不是訓練或物理模擬成果。骨盆不是全身質心。地板只整段對齊一次；離地與落地標記依腳部關鍵點高度估計。</p>
<h2>關鍵影格</h2><img alt="動捕跳躍關鍵影格" src="keyframes.png">
<p>來源：<a href="https://mocap.cs.cmu.edu/search.php?subjectnumber=16">CMU Motion Capture Database，subject 16 / motion 03</a>。原始檔雜湊、座標轉換與地板偏移記錄於 metadata 與來源檔案。</p>
<small>The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217.</small></html>'''
        (output / "index.html").write_text(page, encoding="utf-8")
        report["artifacts"] = {p.name: p.stat().st_size for p in output.iterdir() if p.is_file() and p.name != report_path.name}
        report["status"] = "passed"
        save_report()
        print("MOCAP_RENDER_PASSED " + str(output), flush=True)
    except BaseException as exc:
        report.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        save_report()
        raise


if __name__ == "__main__":
    main()
