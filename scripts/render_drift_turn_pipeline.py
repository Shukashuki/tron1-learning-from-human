"""Four-panel replay of pinned CARLA turn, WF reference, Isaac and MuJoCo.

All robot panels use the same MJCF *renderer* and saved poses. No physics is
integrated; the title uses the separately recorded native-physics verdicts.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

from matplotlib.patches import Polygon

from assess_drift_turn import sha256
from prepare_drift_drl_turn_reference import decode_csv, selected_source
from render_car_drift_pipeline import CarClip
from render_tracking import (BG, MUTED, PANEL, TEXT, ACTUAL, OTHER, REFERENCE,
                             PoseClip, imageio_ffmpeg, load_rollout, mujoco,
                             named_qpos, np, plt, read_npz)
from render_motion_pipeline import source_clock

ROOT = Path(__file__).resolve().parents[1]
CAR = "#c4b5fd"


def prepare(reference_dir, isaac_dir, mujoco_dir, source_config):
    reference_dir, isaac_dir, mujoco_dir = map(Path, (reference_dir, isaac_dir, mujoco_dir))
    config = json.loads(Path(source_config).read_text())
    source_path = reference_dir / "source.csv"
    if sha256(source_path) != config["sha256"]:
        raise ValueError("Original CARLA CSV does not match its pinned hash")
    full = decode_csv(source_path.read_bytes())
    selected, _ = selected_source(full, config)
    start, end = config["source_row_start_inclusive"], config["source_row_end_inclusive"]
    selected["controls"] = full["controls"][start:end + 1]
    car = CarClip(selected)
    robot_path, motion_path = reference_dir / "robot_reference.npz", reference_dir / "motion.npz"
    robot, motion = read_npz(robot_path), read_npz(motion_path)
    adaptation = json.loads((reference_dir / "adaptation.json").read_text())
    export_report = json.loads((reference_dir / "motion.json").read_text())
    ia_report = json.loads((isaac_dir / "report.json").read_text())
    mj_report = json.loads((mujoco_dir / "report.json").read_text())
    ia_score = json.loads((isaac_dir.parent / "isaac_turn_assessment.json").read_text())
    mj_score = json.loads((mujoco_dir.parent / "mujoco_turn_assessment.json").read_text())
    motion_hash = sha256(motion_path)
    if (adaptation["provenance"]["source_configuration_sha256"] != sha256(source_config)
            or export_report["source_car_provenance"] != adaptation["provenance"]
            or export_report["output_sha256"] != motion_hash
            or adaptation["robot_reference_sha256"] != sha256(robot_path)
            or adaptation["motion_sha256"] != motion_hash
            or ia_report["motion_file_sha256"] != motion_hash
            or mj_report["motion_sha256"] != motion_hash
            or ia_score["rollout_sha256"] != sha256(isaac_dir / "physics_trajectory.npz")
            or mj_score["rollout_sha256"] != sha256(mujoco_dir / "rollout.npz")
            or ia_score["engine_report_sha256"] != sha256(isaac_dir / "report.json")
            or mj_score["engine_report_sha256"] != sha256(mujoco_dir / "report.json")
            or ia_score["status"] not in ("PASS", "FAIL")
            or mj_score["status"] not in ("PASS", "FAIL")
            or mj_report["policy_sha256"] != sha256(isaac_dir / "actor_normalized.pt")):
        raise ValueError("Source, reference, actor, rollout, or stored verdict identity changed")
    reference = PoseClip(robot["time_s"], robot["qpos"][:, :3], robot["qpos"][:, 3:7],
                         robot["joint_positions_rad"], robot["joint_names"])
    ia = load_rollout(read_npz(isaac_dir / "physics_trajectory.npz"))
    mj = load_rollout(read_npz(mujoco_dir / "rollout.npz"))
    task_t = np.asarray(robot["time_s"])
    source_t = np.asarray(robot["demonstration_time_s_estimated"])
    if not np.allclose(task_t, motion["time_s"]) or not np.allclose(source_t, motion["demonstration_time_s_estimated"]):
        raise ValueError("Source-to-robot clock mapping differs from exported motion")
    return {"car": car, "reference": reference, "isaac": ia, "mujoco": mj,
            "task_time": task_t, "source_time": source_t,
            "source_phase": robot["source_phase"], "reference_xy": robot["root_target_position_m"][:, :2],
            "ia_score": ia_score, "mj_score": mj_score,
            "hashes": {"source_csv": sha256(source_path), "reference": motion_hash,
                       "actor": sha256(isaac_dir / "actor_normalized.pt"),
                       "isaac_physics": sha256(isaac_dir / "physics_trajectory.npz"),
                       "mujoco_physics": sha256(mujoco_dir / "rollout.npz")}}


def render(prepared, model_path, output, fps=25., playback_speed=.5):
    from eval_tracking_mujoco import load_model
    if fps <= 0 or playback_speed <= 0 or not np.isfinite([fps, playback_speed]).all():
        raise ValueError("Finite positive render and playback rates required")
    model, _ = load_model(model_path, terrain=None)
    if model.nmesh == 0:
        raise ValueError("Official WF visual meshes are required")
    clips = [prepared[key] for key in ("reference", "isaac", "mujoco")]
    for clip in clips:
        named_qpos(model, clip, clip.times[0])
    for geom in range(model.ngeom):
        name = model.geom(geom).name or ""
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH and name != "floor":
            model.geom_rgba[geom, 3] = 0.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 360)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 360)
    data, option = mujoco.MjData(model), mujoco.MjvOption()
    option.sitegroup[:] = 0
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    all_roots = np.concatenate([clip.root for clip in clips])
    low, high = all_roots.min(axis=0), all_roots.max(axis=0)
    camera.lookat[:] = [(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, .38]
    camera.distance = 1.7
    camera.azimuth, camera.elevation = 125., -32.
    renderer, writer, fig = None, None, None
    try:
        renderer = mujoco.Renderer(model, width=360, height=360)
        fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
        fig.text(.025, .95, "High-speed sliding turn  /  TRON1 WF  /  same actor", fontsize=21, color=TEXT)
        fig.text(.025, .915,
                 f"CARLA reference → robot kinematics → Isaac {prepared['ia_score']['status']} → MuJoCo {prepared['mj_score']['status']}",
                 fontsize=13, color=MUTED)
        axes = [fig.add_axes([.02 + i * .245, .37, .23, .46]) for i in range(4)]
        source_ax = axes[0]
        source_ax.set_facecolor(PANEL)
        car = prepared["car"]
        source_ax.plot(car.position[:, 0], car.position[:, 1], color=MUTED, lw=1.5)
        source_ax.set_aspect("equal", adjustable="box")
        bounds = car.position.max(0) - car.position.min(0)
        center = (car.position.max(0) + car.position.min(0)) / 2
        span = max(bounds.max() * 1.25, 14.)
        source_ax.set(xlim=(center[0] - span / 2, center[0] + span / 2),
                      ylim=(center[1] - span / 2, center[1] + span / 2))
        source_ax.set_title("SIMULATED CARLA CAR\nOriginal-scale path", color=CAR, fontsize=11, loc="left")
        source_ax.tick_params(colors=MUTED, labelsize=8)
        patch = Polygon(np.zeros((4, 2)), color=CAR)
        source_ax.add_patch(patch)
        car_label = source_ax.text(.03, .97, "", transform=source_ax.transAxes, va="top", color=CAR,
                                   fontsize=8, bbox={"facecolor": BG, "alpha": .88, "edgecolor": "none"})
        titles = ("KINEMATIC REFERENCE\nno physical rollout", "ISAAC PHYSX\nrecorded 200 Hz",
                  "MUJOCO\nsame actor, recorded 200 Hz")
        colors = (REFERENCE, ACTUAL, OTHER)
        images, labels = [], []
        for ax, title, color in zip(axes[1:], titles, colors):
            ax.set_axis_off()
            ax.set_title(title, color=color, fontsize=11, loc="left")
            images.append(ax.imshow(np.zeros((360, 360, 3), dtype=np.uint8)))
            labels.append(ax.text(.03, .97, "", transform=ax.transAxes, va="top", color=color, fontsize=8,
                                  bbox={"facecolor": BG, "alpha": .88, "edgecolor": "none"}))
        path_ax = fig.add_axes([.05, .12, .39, .18], facecolor=PANEL)
        ref = prepared["reference"]
        path_ax.plot(ref.root[:, 0], ref.root[:, 1], color=REFERENCE, label="target")
        path_ax.plot(prepared["isaac"].root[:, 0], prepared["isaac"].root[:, 1], color=ACTUAL, label="Isaac")
        path_ax.plot(prepared["mujoco"].root[:, 0], prepared["mujoco"].root[:, 1], color=OTHER, label="MuJoCo")
        path_ax.set_title("Travel path (robot scale)", color=TEXT, fontsize=10)
        path_ax.set_aspect("equal", adjustable="datalim")
        path_ax.legend(frameon=False, labelcolor=TEXT, fontsize=8)
        time_ax = fig.add_axes([.53, .12, .40, .18], facecolor=PANEL)
        for key, color in (("isaac", ACTUAL), ("mujoco", OTHER)):
            clip = prepared[key]
            velocity = np.gradient(clip.root[:, :2], clip.times, axis=0)
            time_ax.plot(clip.times, np.linalg.norm(velocity, axis=1), color=color, label=key)
        time_ax.axhline(.8, ls=":", color=MUTED)
        time_ax.axvspan(1.44, 4., color=CAR, alpha=.09)
        time_ax.set(xlim=(0., max(clip.times[-1] for clip in clips)), ylabel="XY speed (m/s)", xlabel="Task time (s)")
        time_ax.set_title("Actual travel speed; shaded turn core", color=TEXT, fontsize=10)
        time_ax.legend(frameon=False, labelcolor=TEXT, fontsize=8)
        for ax in (path_ax, time_ax):
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.grid(alpha=.15)
        cursor = time_ax.axvline(0., color=TEXT, lw=1)
        stamp = fig.text(.73, .95, "", color=TEXT, fontsize=12)
        fig.text(.025, .035, "Source has an estimated clock; stop is synthetic. Robot cameras follow each root; path plot retains world XY. Meshes replay poses only.", color=MUTED, fontsize=10)
        end = max(prepared["task_time"][-1], *(clip.times[-1] for clip in clips))
        timeline = np.arange(int(math.ceil(end * fps)) + 1) / fps

        def update(at):
            sample = car.sample(source_clock(prepared["task_time"], prepared["source_time"], at))
            psi, xy = sample["yaw"], sample["position"]
            rotation = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
            patch.set_xy(np.array([[2.2, .9], [2.2, -.9], [-2.2, -.9], [-2.2, .9]]) @ rotation.T + xy)
            car_label.set_text(f"estimated {sample['estimated_time_s']:.2f}s\n"
                               f"speed {np.linalg.norm(sample['velocity']):.1f} m/s\n"
                               f"sideslip {np.rad2deg(sample['beta']):+.1f}°")
            for clip, image, label, key in zip(clips, images, labels, ("reference", "isaac", "mujoco")):
                data.qpos[:] = named_qpos(model, clip, at)
                data.qvel[:] = 0.
                mujoco.mj_forward(model, data)
                camera.lookat[:] = clip.sample(at)[0] + [0., 0., -.2]
                renderer.update_scene(data, camera=camera, scene_option=option)
                image.set_data(renderer.render().copy())
                if key == "reference":
                    note = "KINEMATIC ONLY"
                else:
                    score = prepared["ia_score"] if key == "isaac" else prepared["mj_score"]
                    note = (f"{score['status']}\n"
                            f"longest grounded drift {score['longest_qualifying_duration_s']:.3f}s\n"
                            f"turn {score['largest_qualifying_course_turn_deg']:.1f}°")
                label.set_text(note)
            cursor.set_xdata([at, at])
            stamp.set_text(f"task {at:.2f}s")
            fig.canvas.draw()

        output = Path(output)
        update(3.)
        fig.savefig(output / "overview.png", facecolor=BG)
        width, height = fig.canvas.get_width_height()
        video = output / "motion_pipeline.mp4"
        writer = imageio_ffmpeg.write_frames(str(video), (width, height), fps=fps * playback_speed,
                                              codec="libx264", pix_fmt_out="yuv420p", quality=7,
                                              macro_block_size=1, output_params=["-movflags", "+faststart"])
        writer.send(None)
        for at in timeline:
            update(float(at))
            writer.send(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        writer.close()
        writer = None
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(video), "-f", "null", "-"],
                       check=True, capture_output=True)
        return {"status": "rendered", "simulation_steps": 0, "frames": len(timeline),
                "video_resolution": [width, height], "video_fps": fps * playback_speed,
                "input_hashes": prepared["hashes"], "video_sha256": sha256(video),
                "overview_sha256": sha256(output / "overview.png"),
                "isaac_verdict": prepared["ia_score"]["status"],
                "mujoco_verdict": prepared["mj_score"]["status"], "hardware_ready": False}
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        if fig is not None:
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--isaac-dir", type=Path, required=True)
    parser.add_argument("--mujoco-dir", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, default=ROOT / "config/drift_turn_source.json")
    parser.add_argument("--model", type=Path, default=ROOT / "assets/robots/WF_TRON1A/mujoco/robot.xml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-fps", type=float, default=25.)
    parser.add_argument("--playback-speed", type=float, default=.5)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Preserving prior render output")
    adaptation = json.loads((args.reference_dir / "adaptation.json").read_text())
    if sha256(args.model) != adaptation["model_sha256"]:
        parser.error("Visual model differs from the pinned reference model")
    prepared = prepare(args.reference_dir, args.isaac_dir, args.mujoco_dir, args.source_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = render(prepared, args.model, args.output_dir, args.render_fps, args.playback_speed)
    (args.output_dir / "render_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
