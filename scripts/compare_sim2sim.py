"""Compare two retargets and two genuinely simulated free-base torque rollouts.

All movie panels use one MuJoCo mesh renderer. Isaac panels visualize recorded
native PhysX states; they are not native Isaac screenshots or MuJoCo resimulation.
"""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from sim2sim_common import JOINT_NAMES, LEGS, ROOT, ReferenceMotion, load_config

BG, TEXT, MUTED = "#0a1220", "#e2e8f0", "#94a3b8"
PALETTE = ("#f6d56a", "#38bdf8", "#fb923c")


def read_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def interpolate(times, values, at):
    return np.stack([np.interp(at, times, values[:, i]) for i in range(values.shape[1])], axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/sim2sim-cmu-16_03")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if (output / "comparison.json").exists():
        parser.error("Existing comparison is preserved; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    config = load_config()
    methods = ("mink", "gmr")
    engines = ("mujoco", "isaac")
    run_paths = {(m, e): ROOT / (f"outputs/sim2sim-cmu-16_03/mujoco-{m}" if e == "mujoco"
                                 else f"outputs/sim2sim-isaac-{m}") for m in methods for e in engines}
    reports = {key: json.loads((path / "report.json").read_text()) for key, path in run_paths.items()}
    rolls = {key: read_npz(path / "rollout.npz") for key, path in run_paths.items()}
    references = {m: ReferenceMotion(ROOT / f"outputs/{m}-cmu-16_03/robot_reference.npz", config=config)
                  for m in methods}
    ik_reports = {m: json.loads((ROOT / f"outputs/{m}-cmu-16_03/report.json").read_text()) for m in methods}
    source_hashes = {report["controller_source_sha256"] for report in reports.values()}
    if len(source_hashes) != 1:
        raise ValueError("Controllers differ between runs; not a matched sim2sim test")
    for method in methods:
        if reports[(method, "mujoco")]["reference_sha256"] != reports[(method, "isaac")]["reference_sha256"]:
            raise ValueError("Engines used different reference files")
    for key, report in reports.items():
        if report["config"] != config or not report["physics_stepped"] or report["root_prescribed_after_initialization"]:
            raise ValueError(f"Unmatched configuration or nonphysical playback: {key}")
        if not np.isfinite(rolls[key]["root_pos"]).all():
            raise ValueError(f"Nonfinite rollout: {key}")
        if list(rolls[key]["joint_names"]) != list(JOINT_NAMES):
            raise ValueError("Unexpected saved joint order")
    common_end = min(float(roll["time_s"][-1]) for roll in rolls.values())
    common_times = np.arange(config["settle_seconds"], common_end - config["physics_dt"] / 2, config["physics_dt"])
    if len(common_times) < 10:
        raise ValueError("Insufficient common motion horizon")
    metrics = []
    for method in methods:
        for engine in engines:
            key = method, engine
            roll, report = rolls[key], reports[key]
            actual = interpolate(roll["time_s"], roll["joint_pos"][:, LEGS], common_times)
            desired = interpolate(roll["time_s"], roll["joint_ref"][:, LEGS], common_times)
            root = interpolate(roll["time_s"], roll["root_pos"], common_times)
            root_ref = interpolate(roll["time_s"], roll["root_ref"], common_times)
            metrics.append({"method": method, "engine": engine,
                            "termination": report["termination"],
                            "stop_time_since_start_s": report["simulated_seconds"],
                            "stop_time_since_motion_start_s": report["simulated_seconds"] - config["settle_seconds"],
                            "common_prefix_leg_rmse_deg": float(np.rad2deg(np.sqrt(np.mean((actual - desired) ** 2)))),
                            "common_prefix_root_rmse_m": float(np.sqrt(np.mean(np.sum((root - root_ref) ** 2, axis=1)))),
                            "base_rise_from_initial_m": report["base_rise_from_initial_m"],
                            "max_base_tilt_deg": report["max_base_tilt_deg"],
                            "actuator_saturation_fraction": report["actuator_saturation_fraction"]})
    cross_engine = {}
    for method in methods:
        states = [interpolate(rolls[(method, e)]["time_s"], rolls[(method, e)]["root_pos"], common_times)
                  for e in engines]
        delta = np.linalg.norm(states[0] - states[1], axis=1)
        cross_engine[method] = {"root_position_rmse_m": float(np.sqrt(np.mean(delta ** 2))),
                                "root_position_max_difference_m": float(delta.max())}
    standing_paths = {"mujoco": ROOT / "outputs/sim2sim-cmu-16_03/mujoco-standing",
                      "isaac": ROOT / "outputs/sim2sim-isaac-standing"}
    standing = {name: json.loads((path / "report.json").read_text()) for name, path in standing_paths.items()}
    summary = {"status": "comparison_completed", "trained_policy": False, "beyondmimic_training": False,
               "type": "matched nonlearned controller/reference transfer diagnostic",
               "controller_source_sha256": next(iter(source_hashes)),
               "common_motion_interval_s": [float(common_times[0]), float(common_times[-1])],
               "metrics": metrics, "cross_engine_on_common_prefix": cross_engine,
               "standing_control": {k: {name: r[name] for name in ("termination", "simulated_seconds", "max_base_tilt_deg")}
                                    for k, r in standing.items()},
               "retarget_comparison": {m: {k: ik_reports[m][k] for k in
                    ("root_prescribed", "all_wheel_errors", "max_interval_joint_speed_rad_s", "max_ground_penetration_m")}
                    for m in methods},
               "warnings": ["GMR adapts upstream two-stage IK to WF, not an official TRON1 configuration.",
                            "GMR soft-root/unbounded inter-frame leg speed differs from frozen-root/8rad/s Mink baseline; residuals alone cannot establish method superiority.",
                            "No BeyondMimic PPO was trained. PD/LQR failures do not prove references are untrackable by a learned controller.",
                            "All dynamics are native to named engine; common renderer reconstructs their recorded poses without resimulation.",
                            "Terminated panels freeze visibly at last recorded pose; they are not extrapolated.",
                            "Contact solver and collision approximation differ despite parameter alignment."]}
    (output / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT, "axes.labelcolor": MUTED})
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/robots/WF_TRON1A/mujoco/robot.xml"))
    data = mujoco.MjData(model)
    for i in range(model.ngeom):
        if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH and model.geom(i).name != "floor":
            model.geom_rgba[i, 3] = 0.
    model.vis.global_.offwidth, model.vis.global_.offheight = 600, 400
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [.05, 0, .65]
    camera.distance, camera.azimuth, camera.elevation = 2.7, 125, -15
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    qids = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
    renderer = mujoco.Renderer(model, width=600, height=400)
    rotations = {key: Slerp(roll["time_s"], Rotation.from_quat(roll["root_quat_wxyz"][:, [1, 2, 3, 0]]))
                 for key, roll in rolls.items()}
    def pose(method, engine, t):
        if engine == "reference":
            sample = references[method].sample(t)
            position, quat, joints = sample["root_pos"], sample["root_quat_wxyz"], sample["joint_pos"]
            stopped = False
        else:
            key = method, engine
            roll = rolls[key]
            at = float(np.clip(t, roll["time_s"][0], roll["time_s"][-1]))
            position = interpolate(roll["time_s"], roll["root_pos"], at)
            joints = interpolate(roll["time_s"], roll["joint_pos"], at)
            quat = rotations[key](at).as_quat()[[3, 0, 1, 2]]
            stopped = t > roll["time_s"][-1] and reports[key]["termination"] != "completed"
        data.qpos[:7] = np.r_[position, quat]
        data.qpos[qids] = joints
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera, scene_option=options)
        return renderer.render().copy(), stopped

    fig = plt.figure(figsize=(18, 11.5), dpi=100, facecolor=BG)
    grid = fig.add_gridspec(3, 3, left=.025, right=.98, bottom=.10, top=.86,
                           height_ratios=(1, 1, .50), wspace=.035, hspace=.21)
    panels, labels = {}, {}
    for row, method in enumerate(methods):
        for col, engine in enumerate(("reference", *engines)):
            ax = fig.add_subplot(grid[row, col])
            ax.set_axis_off()
            name = "IK REFERENCE / NO PHYSICS" if engine == "reference" else "MuJoCo DYNAMICS" if engine == "mujoco" else "ISAAC / PhysX DYNAMICS"
            ax.set_title(f"{method.upper()}  |  {name}", fontsize=12, color=PALETTE[col], loc="left", pad=9)
            panels[(method, engine)] = ax.imshow(pose(method, engine, 0.)[0])
            labels[(method, engine)] = ax.text(.025, .94, "", transform=ax.transAxes, fontsize=12,
                                                color="#ff6969", weight="bold", va="top",
                                                bbox={"facecolor": BG, "alpha": .8, "edgecolor": "none"})
    chart_grid = grid[2, :].subgridspec(1, 2, wspace=.16)
    duration = max(reference.duration for reference in references.values())
    chart_cursors = []
    for index, method in enumerate(methods):
        ax = fig.add_subplot(chart_grid[0, index])
        ax.set_facecolor("#101d31")
        source_times = references[method].times + config["settle_seconds"]
        ax.plot(source_times, references[method].root_pos[:, 2], color=PALETTE[0], ls="--", label="IK reference")
        for color, engine in zip(PALETTE[1:], engines):
            roll = rolls[(method, engine)]
            ax.plot(roll["time_s"], roll["root_pos"][:, 2], color=color, label=engine.upper(), lw=1.6)
        ax.set_title(f"{method.upper()} / actual base height vs reference", fontsize=11, color=TEXT, loc="left")
        ax.set_xlim(0, duration)
        ax.set_ylim(.2, 1.4)
        ax.set_xlabel("Time including 1 s settling (s)")
        ax.set_ylabel("Base height (m)")
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(alpha=.15)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.legend(frameon=False, labelcolor=MUTED, ncol=3, fontsize=9, loc="upper right")
        chart_cursors.append(ax.axvline(0., color="white", alpha=.65))
    fig.text(.03, .95, "TRON1 WF  /  GMR vs MINK  /  MuJoCo ↔ ISAAC", fontsize=24, weight="bold")
    fig.text(.03, .905, "Same torque controller + matched physical parameters  |  FREE BASE  |  NO TRAINED PPO POLICY", fontsize=12, color=PALETTE[0])
    clock_label = fig.text(.86, .95, "", fontsize=15, color=PALETTE[0])
    fig.text(.03, .046, "Both engine columns are recorded native dynamics, shown using one common mesh renderer. Reference column is kinematic only.", color=MUTED, fontsize=10)
    fig.text(.03, .022, "Red STOP panels hold the last measured pose. These untrained PD + LQR trials do not establish BeyondMimic policy performance.", color=MUTED, fontsize=10)
    def update(t):
        for (method, engine), handle in panels.items():
            frame, stopped = pose(method, engine, t)
            handle.set_data(frame)
            labels[(method, engine)].set_text("STOP / EXCESSIVE TILT" if stopped else "")
        for cursor in chart_cursors:
            cursor.set_xdata([t, t])
        clock_label.set_text(f"{t:.2f} s")
        fig.canvas.draw()
    try:
        update(config["settle_seconds"] + 1.79)
        fig.savefig(output / "comparison.png", dpi=110, facecolor=BG)
        update(common_end)
        fig.savefig(output / "last_common_frame.png", dpi=110, facecolor=BG)
        if not args.no_video:
            writer = imageio_ffmpeg.write_frames(str(output / "sim2sim_comparison.mp4"), fig.canvas.get_width_height(),
                                                 fps=30, codec="libx264", pix_fmt_out="yuv420p", quality=7,
                                                 macro_block_size=2, output_params=["-movflags", "+faststart"])
            writer.send(None)
            video_times = np.arange(0., duration + 1e-9, 1 / 60)
            try:
                for i, t in enumerate(video_times):
                    update(float(t))
                    writer.send(np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]))
                    if i % 60 == 0:
                        print(f"COMPARISON_RENDER {i + 1}/{len(video_times)}", flush=True)
            finally:
                writer.close()
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(output / "sim2sim_comparison.mp4"),
                            "-f", "null", "-"], check=True, capture_output=True)
        table_rows = "".join(f'<tr><td>{r["method"].upper()}</td><td>{r["engine"].upper()}</td>'
                            f'<td>{r["stop_time_since_start_s"]:.3f}</td><td>{r["base_rise_from_initial_m"]*100:.2f}</td>'
                            f'<td>{r["common_prefix_leg_rmse_deg"]:.2f}</td><td>{r["common_prefix_root_rmse_m"]*100:.2f}</td>'
                            f'<td>{html.escape(r["termination"])}</td></tr>' for r in metrics)
        page = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TRON1 GMR / Mink sim2sim</title><style>body{{max-width:1600px;margin:32px auto;padding:0 20px;background:{BG};color:{TEXT};font:16px/1.7 system-ui}}video,img{{width:100%;border-radius:12px}}a{{color:#67d4ff}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #334155}}.note{{border-left:3px solid #f6d56a;padding:12px 20px;background:#101d31}}nav{{display:flex;gap:24px;margin:18px 0}}</style>
<h1>TRON1 輪足：GMR / Mink → MuJoCo / Isaac</h1>
<p>真實上游 GMR 兩階段重定向；同一份 PD＋輪式 LQR 力矩控制器，在 MuJoCo 與原生 Isaac PhysX 執行。不是 BeyondMimic 訓練結果。</p>
<video src="sim2sim_comparison.mp4" poster="comparison.png" controls autoplay muted loop playsinline></video>
<nav><a href="sim2sim_comparison.mp4">半速 MP4</a><a href="comparison.png">總覽圖</a><a href="last_common_frame.png">共同時間末幀</a><a href="comparison.json">完整比較數值</a></nav>
<p class="note">左欄僅播放 IK 參考；中、右欄分別是 MuJoCo／Isaac 原生物理計算的記錄，底座只在初始化設定。為方便比較，所有欄使用同一網格渲染器，右欄不是 Isaac 畫面截圖。超過 60° 傾斜會停止，之後明確顯示 STOP 並保留末姿態。</p>
<h2>物理結果</h2><p>前 1 秒為站立準備；追蹤 RMSE 使用所有試驗共同的 {common_times[0]:.3f}–{common_times[-1]:.3f} 秒區間。上升量是 base 高度，不是質心跳高；停止時間包含準備時間。</p>
<table><tr><th>參考</th><th>引擎</th><th>停止時間 s</th><th>Base 上升 cm</th><th>腿角 RMSE °</th><th>Base RMSE cm</th><th>結果</th></tr>{table_rows}</table>
<h2>如何解讀</h2><p>兩個引擎的 5 秒站立對照都完成。GMR 輪心幾何 RMS 約 0.049 mm，比原 Mink 的 0.555 mm 小；但 GMR 讓 root 自由最佳化，峰值腿速約 13.37 rad/s，而 Mink 有 8 rad/s 限制，因此不能只看殘差宣稱 GMR 全面較好。兩種參考都有約 4.5 mm 的穿地。</p>
<p>這次採用未訓練的共同控制器，輪式 LQR 原本只適用固定腿長的地面平衡，沒有跳躍推蹬、空中姿態與落地控制。倒下表示此控制器不能完成該跳躍，不代表 GMR 參考無法被 RL 學會。下一層才是依 BeyondMimic 作法訓練單一動作策略，再用同一 checkpoint 做 policy sim2sim。</p>
<p>已對齊質量、COM／慣性、碰撞尺寸、摩擦、時步、關節參數和扭矩上限；接觸／約束解算器仍不同。這是單一片段、確定性條件的診斷，不是統計 benchmark。</p>
<p>來源：<a href="https://github.com/YanjieZe/GMR">GMR 官方實作</a>；<a href="https://github.com/HybridRobotics/whole_body_tracking">BeyondMimic 官方追蹤訓練</a>；<a href="https://mocap.cs.cmu.edu/search.php?subjectnumber=16">CMU 16_03</a>。</p></html>'''
        (output / "index.html").write_text(page, encoding="utf-8")
        (output / "render_report.json").write_text(json.dumps({"status": "passed", "video_decode_verified": not args.no_video,
            "playback_speed": .5, "common_renderer": "MuJoCo EGL", "native_dynamics_sources": ["MuJoCo", "Isaac PhysX"],
            "physics_resimulation_in_renderer": False, "frozen_after_termination": True,
            "simulation_steps_in_renderer": 0, "root_prescribed_in_rollout": False}, indent=2) + "\n")
    finally:
        renderer.close()
        plt.close(fig)
    print(json.dumps(summary["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
