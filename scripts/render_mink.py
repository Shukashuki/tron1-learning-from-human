"""Render TRON1 mesh poses from Mink IK; no integration, forces, or policy.

The human comparison is the yaw-aligned, uniformly scaled reference provided
by retarget_mink.py. MuJoCo is used only for forward kinematics and rendering.
"""

from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
BG, PANEL, TEXT, MUTED = "#0a1220", "#101d31", "#e2e8f0", "#94a3b8"
LEFT, RIGHT, GOLD = "#38bdf8", "#fb923c", "#f6d56a"


def style_axis(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.grid(color="#64748b", alpha=0.15)


def add_marker(scene, position, rgba, radius=0.023):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                       np.array([radius, 0.0, 0.0]), np.asarray(position),
                       np.eye(3).ravel(), np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


def add_connector(scene, start, end, rgba):
    if scene.ngeom >= scene.maxgeom or np.linalg.norm(np.asarray(end) - start) < 1e-6:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                       np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                       np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.003,
                        np.asarray(start), np.asarray(end))
    scene.ngeom += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path,
                        default=ROOT / "outputs/mink-cmu-16_03/robot_reference.npz")
    parser.add_argument("--model", type=Path,
                        default=ROOT / "assets/robots/WF_TRON1A/mujoco/robot.xml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/mink-cmu-16_03")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "render_report.json"
    if report_path.exists():
        parser.error("A render report already exists; choose an unused output directory")
    report = {
        "status": "starting", "kind": "mink_kinematic_ik_preview",
        "robot_retargeted": True, "physics_validated": False,
        "policy_trained": False, "root_prescribed": True,
        "human_reference": "yaw-aligned, uniformly scaled, fixed-offset human skeleton",
        "renderer": "MuJoCo " + mujoco.__version__,
        "gl_backend": os.environ["MUJOCO_GL"],
        "model": str(args.model.resolve()), "reference": str(args.reference.resolve()),
        "simulation_steps": 0,
    }

    def save_report():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save_report()
    renderer = None
    try:
        with np.load(args.reference, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        qpos = np.asarray(arrays["qpos"])
        times = np.asarray(arrays["time_s"])
        fps = float(arrays["fps"])
        human = np.asarray(arrays["human_positions_world_m"])
        human_names = list(arrays["human_joint_names"])
        parents = np.asarray(arrays["human_parent_indices"])
        site_names = list(arrays["site_names"])
        site_positions = np.asarray(arrays["site_positions_world_m"])
        target_names = list(arrays["target_names"])
        targets = np.asarray(arrays["target_positions_world_m"])
        errors = np.asarray(arrays["target_error_m"])
        root_targets = np.asarray(arrays["root_target_position_m"])
        count = len(times)
        if count < 2 or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Reference must contain at least two regularly timed frames")
        if not np.allclose(np.diff(times), 1 / fps, rtol=1e-5, atol=1e-9):
            raise ValueError("Reference timestamps do not match its declared sampling rate")
        expected = [(human, (count, len(human_names), 3)),
                    (site_positions, (count, len(site_names), 3)),
                    (targets, (count, len(target_names), 3)),
                    (errors, (count, len(target_names))), (root_targets, (count, 3))]
        for values, shape in expected:
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"Invalid reference array: expected {shape}, got {values.shape}")
        model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
        if qpos.shape != (count, model.nq) or not np.isfinite(qpos).all():
            raise ValueError(f"qpos must have shape {(count, model.nq)}")
        if model.nmesh == 0:
            raise ValueError("The TRON1 model has no mesh assets to render")
        # Hide collision proxies in this renderer's in-memory model. Keep the
        # official visual meshes and ground; do not alter the source MJCF.
        for geom_id in range(model.ngeom):
            if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) != "floor":
                    model.geom_rgba[geom_id, 3] = 0.0
        data = mujoco.MjData(model)
        site_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                             for name in site_names])
        if np.any(site_ids < 0):
            raise ValueError("A reference site name does not exist in the model")
        target_site_indices = np.array([site_names.index(name) for name in target_names])
        computed_errors = np.linalg.norm(site_positions[:, target_site_indices] - targets, axis=-1)
        if not np.allclose(errors, computed_errors, rtol=1e-4, atol=1e-7):
            raise ValueError("Reported IK errors do not match stored sites and targets")
        joint_names = list(arrays["joint_names"])
        wheel_joint_indices = [i for i, name in enumerate(joint_names) if name.startswith("wheel_")]
        wheel_spin_frozen = bool(np.allclose(arrays["joint_positions_rad"][:, wheel_joint_indices], 0.0, atol=1e-10))
        penetration = np.asarray(arrays["ground_penetration_m"])
        if penetration.shape != (count,) or not np.isfinite(penetration).all():
            raise ValueError("Missing or invalid ground-penetration diagnostics")
        max_penetration_mm = float(np.max(penetration)) * 1000
        penetrating_frames = int(np.count_nonzero(penetration > 0.001))
        wheel_rmse_mm = float(np.sqrt(np.mean(errors**2))) * 1000
        wheel_max_error_mm = float(errors.max()) * 1000
        spin_note = "Wheel spin fixed at 0" if wheel_spin_frozen else "Wheel angles supplied by reference"
        root_index = human_names.index("root")
        visible = [i for i, name in enumerate(human_names)
                   if not any(part in name for part in ("fingers", "thumb"))]
        edges = [(int(parents[i]), i) for i in visible if parents[i] >= 0]
        if parents.shape != (len(human_names),) or any(a >= len(human_names) for a, _ in edges):
            raise ValueError("Invalid human skeleton parent indices")

        # All frames use the same camera and coordinate scale.
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        low = np.min(np.r_[root_targets, targets.reshape(-1, 3)], axis=0)
        high = np.max(np.r_[root_targets, targets.reshape(-1, 3)], axis=0)
        camera.lookat[:] = [(low[0] + high[0]) / 2, (low[1] + high[1]) / 2,
                            max(0.55, (max(0.0, low[2]) + high[2]) / 2)]
        camera.distance = max(2.65, float(np.max(high - low)) * 2.0)
        camera.azimuth, camera.elevation = 132, -13
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), 1000)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), 800)
        renderer = mujoco.Renderer(model, width=1000, height=800)
        options = mujoco.MjvOption()
        options.sitegroup[:] = 0  # Explicit task markers below, not all model sites.
        max_site_difference = 0.0

        def robot_frame(frame):
            nonlocal max_site_difference
            data.qpos[:] = qpos[frame]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            difference = float(np.max(np.abs(data.site_xpos[site_ids] - site_positions[frame])))
            max_site_difference = max(max_site_difference, difference)
            if difference > 1e-5:
                raise ValueError(f"Rendered model sites differ from saved IK poses by {difference:.6g} m")
            renderer.update_scene(data, camera=camera, scene_option=options)
            for target_index, name in enumerate(target_names):
                rgba = [0.22, 0.74, 0.98, 0.8] if name.startswith("left") else [0.98, 0.56, 0.24, 0.8]
                goal = targets[frame, target_index]
                actual = site_positions[frame, target_site_indices[target_index]]
                add_marker(renderer.scene, goal, rgba)
                add_connector(renderer.scene, actual, goal, [0.96, 0.84, 0.41, 0.9])
            return renderer.render().copy()

        plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT,
                             "axes.labelcolor": MUTED})
        fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
        grid = fig.add_gridspec(3, 2, left=0.025, right=0.975, bottom=0.10, top=0.83,
                               width_ratios=(1.75, 1), height_ratios=(1.12, 0.7, 0.7),
                               wspace=0.16, hspace=0.50)
        robot_ax = fig.add_subplot(grid[:, 0])
        robot_ax.set_facecolor(PANEL)
        robot_ax.set_axis_off()
        robot_image = robot_ax.imshow(robot_frame(0))
        robot_ax.set_title("TRON1 WF  /  official mesh geometry", loc="left", fontsize=13, color=TEXT, pad=12)
        human_ax = fig.add_subplot(grid[0, 1], projection="3d")
        human_ax.set_facecolor(PANEL)
        human_low, human_high = human.min(axis=(0, 1)), human.max(axis=(0, 1))
        human_center = 0.5 * (human_low + human_high)
        human_half = max(0.6, float(max(human_high[:2] - human_low[:2])) / 2 + 0.08)
        human_ax.set_xlim(human_center[0] - human_half, human_center[0] + human_half)
        human_ax.set_ylim(human_center[1] - human_half, human_center[1] + human_half)
        human_ax.set_zlim(min(-0.05, human_low[2]), human_high[2] + 0.08)
        human_ax.set_box_aspect((2 * human_half, 2 * human_half, human_high[2] + 0.13))
        human_ax.view_init(elev=12, azim=-55)
        human_ax.set_axis_off()
        human_ax.set_title("Scaled human reference", loc="left", fontsize=11, color=TEXT, pad=0)
        human_lines = []
        for _, child in edges:
            name = human_names[child]
            color = LEFT if name.startswith("l") and name not in ("lowerback", "lowerneck") else RIGHT if name.startswith("r") else TEXT
            line, = human_ax.plot([], [], [], color=color, lw=2.2)
            human_lines.append(line)
        human_trace = human[:, root_index]
        human_ax.plot(human_trace[:, 0], human_trace[:, 1], human_trace[:, 2], color=GOLD, lw=1, alpha=0.4)
        height_ax, error_ax = fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[2, 1])
        for chart in (height_ax, error_ax):
            style_axis(chart)
            chart.set_xlim(times[0], times[-1])
            chart.set_xlabel("Source time (s)", fontsize=9)
        height_ax.plot(times, root_targets[:, 2], color=GOLD, lw=1.8, label="Prescribed pelvis")
        wheel_cols = [i for i, name in enumerate(target_names) if "wheel" in name]
        knee_cols = [i for i, name in enumerate(target_names) if "knee" in name]
        if wheel_cols:
            wheel_height = site_positions[:, target_site_indices[wheel_cols], 2].mean(axis=1)
            height_ax.plot(times, wheel_height, color=LEFT, lw=1.3, label="Mean wheel center")
        height_ax.set_ylabel("Height (m)", fontsize=9)
        height_ax.set_title("Root trajectory is prescribed, not simulated", loc="left", fontsize=10, color=TEXT, pad=9)
        height_ax.legend(frameon=False, labelcolor=MUTED, fontsize=8, loc="upper right")
        for cols, label, color in ((knee_cols, "Knee mean", RIGHT), (wheel_cols, "Wheel mean", LEFT)):
            if cols:
                error_ax.plot(times, errors[:, cols].mean(axis=1) * 100, color=color, lw=1.5, label=label)
        error_ax.set_ylabel("IK error (cm)", fontsize=9)
        error_ax.set_ylim(0, max(1.0, float(errors.max()) * 110))
        error_ax.set_title("Task-position residuals", loc="left", fontsize=10, color=TEXT, pad=9)
        error_ax.legend(frameon=False, labelcolor=MUTED, fontsize=8, loc="upper right")
        cursors = [chart.axvline(times[0], color="white", alpha=0.6, lw=1) for chart in (height_ax, error_ax)]
        fig.text(0.035, 0.94, "CMU 16_03  →  TRON1 WF  /  MINK IK", fontsize=22, fontweight="bold")
        fig.text(0.035, 0.89, "KINEMATIC IK  /  NO PHYSICS  /  ROOT PRESCRIBED", fontsize=13, color=GOLD)
        stamp = fig.text(0.78, 0.94, "", fontsize=13, color=GOLD)
        fig.text(0.035, 0.048, "Colored spheres: mapped wheel-center targets. Gold segments: IK residuals. 0.5x playback; data stay at source rate.", fontsize=10, color=MUTED)
        fig.text(0.035, 0.022, f"{spin_note}; no rolling/contact constraints. Max reported ground penetration: {max_penetration_mm:.2f} mm. Not a dynamically validated jump.", fontsize=9, color=MUTED)

        def update(frame):
            robot_image.set_data(robot_frame(frame))
            points = human[frame]
            for artist, (parent, child) in zip(human_lines, edges):
                pair = points[[parent, child]]
                artist.set_data_3d(pair[:, 0], pair[:, 1], pair[:, 2])
            for cursor in cursors:
                cursor.set_xdata([times[frame], times[frame]])
            stamp.set_text(f"{times[frame]:.2f} s  /  source")
            fig.canvas.draw()

        apex = int(np.argmax(root_targets[:, 2]))
        update(apex)
        fig.savefig(output / "overview.png", dpi=125, facecolor=BG)
        width, height = fig.canvas.get_width_height()
        # For 120 Hz input: retain every second sample at 30 fps => half speed.
        stride = max(1, round(fps / 60))
        video_fps = fps / stride * 0.5
        frame_indices = np.arange(0, count, stride)
        writer = imageio_ffmpeg.write_frames(str(output / "mink_retarget.mp4"), (width, height),
                                             fps=video_fps, codec="libx264", pix_fmt_out="yuv420p",
                                             quality=7, macro_block_size=2,
                                             output_params=["-movflags", "+faststart"])
        writer.send(None)
        try:
            for number, frame in enumerate(frame_indices):
                update(int(frame))
                rgb = np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
                writer.send(rgb)
                if number % 50 == 0:
                    print(f"MINK_RENDER {number + 1}/{len(frame_indices)}", flush=True)
        finally:
            writer.close()
            plt.close(fig)

        velocity = np.gradient(root_targets[:, 2], times)
        snapshots = [(0, "START"), (int(np.argmin(root_targets[:, 2])), "LOWEST ROOT"),
                     (int(np.argmax(velocity[:apex + 1])), "RISING"), (apex, "APEX"),
                     (apex + int(np.argmin(velocity[apex:])), "DESCENDING"), (count - 1, "END")]
        keyfig, axes = plt.subplots(1, len(snapshots), figsize=(18, 5.4), dpi=100, facecolor=BG)
        for ax, (frame, label) in zip(axes, snapshots):
            ax.imshow(robot_frame(frame))
            ax.set_axis_off()
            ax.set_title(f"{label}\n{times[frame]:.2f} s", fontsize=11, color=TEXT)
        keyfig.suptitle("TRON1 WF / Mink IK mesh poses", fontsize=18, color=TEXT, y=0.97)
        keyfig.text(0.03, 0.035, "KINEMATIC IK / NO PHYSICS / ROOT PRESCRIBED — colored markers are targets, not contacts.", fontsize=11, color=GOLD)
        keyfig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.11, wspace=0.02)
        keyfig.savefig(output / "keyframes.png", dpi=125, facecolor=BG)
        plt.close(keyfig)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(output / "mink_retarget.mp4"),
                        "-f", "null", "-"], capture_output=True, text=True, check=True)
        reference_link = html.escape(Path(os.path.relpath(args.reference.resolve(), output)).as_posix(), quote=True)
        ik_report_link = '<a href="report.json">IK 數值報告</a>' if (output / "report.json").exists() else ""
        page = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CMU 16_03 → TRON1 WF / Mink IK</title><style>body{{margin:32px auto;padding:0 20px;max-width:1400px;background:{BG};color:{TEXT};font:16px/1.7 system-ui,sans-serif}}h1{{margin-bottom:0}}p{{color:#aebed2}}video,img{{width:100%;border-radius:12px;background:{PANEL}}}a{{color:#67d4ff}}nav{{display:flex;gap:24px;flex-wrap:wrap;margin:18px 0}}.note{{border-left:3px solid {GOLD};padding:10px 18px;background:{PANEL}}}</style>
<h1>CMU 16_03 → TRON1 WF：Mink IK 動作預覽</h1>
<p>官方 TRON1 WF 網格模型，依 Mink IK 關節姿態逐幀顯示。影片半速；人體參考已旋轉對齊、統一縮放並施加固定平移。</p>
<video controls autoplay muted loop playsinline poster="overview.png" src="mink_retarget.mp4"></video>
<nav><a href="mink_retarget.mp4">MP4 影片</a><a href="{reference_link}">完整 IK 軌跡 NPZ</a>{ik_report_link}<a href="render_report.json">渲染驗證報告</a></nav>
<p class="note"><strong>KINEMATIC IK / NO PHYSICS / ROOT PRESCRIBED</strong><br>浮動基座位置與姿態由參考指定；沒有執行動力學積分、控制器訓練或實機驗證。這段影片不能證明機器人能維持平衡、滿足扭矩需求或完成真實跳躍。彩色球是人體目標，金色連線顯示 IK 誤差，並非接觸力。</p>
<p>輪心追蹤 RMSE：{wheel_rmse_mm:.3f} mm，最大誤差：{wheel_max_error_mm:.2f} mm。輪子自轉{'固定在零位' if wheel_spin_frozen else '使用參考角度'}，未加入滾動約束。依官方碰撞幾何診斷，最大穿地 {max_penetration_mm:.2f} mm；{penetrating_frames}/{count} 幀超過 1 mm。這些穿透保留顯示，並未透過修改基座軌跡隱藏。</p>
<h2>關鍵姿態</h2><img src="keyframes.png" alt="TRON1 WF Mink IK 關鍵姿態">
<p>來源：<a href="https://mocap.cs.cmu.edu/search.php?subjectnumber=16">CMU subject 16 / motion 03</a>。The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217.</p></html>'''
        (output / "index.html").write_text(page, encoding="utf-8")
        report.update({"status": "passed", "source_frames": count, "source_fps": fps,
                       "sample_span_s": float(times[-1] - times[0]), "playback_speed": 0.5,
                       "video_frames": len(frame_indices), "video_fps": video_fps,
                       "video_duration_s": len(frame_indices) / video_fps,
                       "video_size_px": [width, height], "video_decode_verified": True,
                       "mesh_count": int(model.nmesh), "actual_mesh_geometry": True,
                       "max_rendered_site_difference_m": max_site_difference,
                       "max_ik_target_error_m": float(errors.max()),
                       "wheel_spin_frozen_at_zero": wheel_spin_frozen,
                       "rolling_constraint_enforced": False,
                       "max_ground_penetration_m": max_penetration_mm / 1000,
                       "ground_penetration_frames_gt_1mm": penetrating_frames,
                       "keyframes": [{"frame_index": int(frame), "label": label,
                                      "source_time_s": float(times[frame])} for frame, label in snapshots],
                       "artifacts": {name: (output / name).stat().st_size for name in
                                     ("mink_retarget.mp4", "overview.png", "keyframes.png", "index.html")}})
        save_report()
        print("MINK_RENDER_PASSED " + str(output), flush=True)
    except BaseException as exc:
        report.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        save_report()
        raise
    finally:
        if renderer is not None:
            renderer.close()
        plt.close("all")


if __name__ == "__main__":
    main()
