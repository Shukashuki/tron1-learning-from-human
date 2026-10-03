"""Four-way source/kinematics/Isaac/MuJoCo visualization; NEVER simulate.

Uses the exported training reference, including its source-time map, clearance
and terminal hold. The human skeleton is the verified aligned/scaled GMR input,
not raw-size human data, wheel targets, a policy or measured contact evidence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

from render_tracking import (BG, PANEL, TEXT, MUTED, ACTUAL, REFERENCE, OTHER,
                             ROOT, PoseClip, load_rollout, named_qpos, read_npz,
                             sha256, terminal_annotation, mujoco, np, plt,
                             imageio_ffmpeg)
from render_mink import add_marker, add_connector
from compare_tracking import validate_provenance
from publish_motion_suite import sanitize

HUMAN = "#c4b5fd"


class KeypointClip:
    def __init__(self, times, positions, names, parents):
        self.times = np.asarray(times, float)
        self.positions = np.asarray(positions, float)
        self.names = np.asarray(names).astype(str).tolist()
        self.parents = np.asarray(parents)
        if (self.times.ndim != 1 or len(self.times) < 2 or not np.isfinite(self.times).all()
                or np.any(np.diff(self.times) <= 0)):
            raise ValueError("Keypoint timestamps must be finite and strictly increasing")
        if (self.positions.shape != (len(self.times), len(self.names), 3)
                or not np.isfinite(self.positions).all() or len(set(self.names)) != len(self.names)):
            raise ValueError("Keypoint positions/names must be finite, aligned and unique")
        n = len(self.names)
        if (self.parents.shape != (n,) or self.parents.dtype.kind not in "iu"
                or np.count_nonzero(self.parents == -1) != 1
                or np.any((self.parents < -1) | (self.parents >= n))):
            raise ValueError("Invalid skeleton parent indices or root count")
        for node in range(n):
            seen = set()
            while node >= 0:
                if node in seen:
                    raise ValueError("Skeleton parents contain a cycle")
                seen.add(node)
                node = int(self.parents[node])

    def sample(self, time_s):
        if not np.isfinite(time_s):
            raise ValueError("Sample time must be finite")
        flat = self.positions.reshape(len(self.times), -1)
        return np.array([np.interp(time_s, self.times, flat[:, i])
                         for i in range(flat.shape[1])]).reshape(-1, 3)


def source_clock(task_times, source_times, at):
    task_times, source_times = np.asarray(task_times, float), np.asarray(source_times, float)
    if (task_times.ndim != 1 or source_times.shape != task_times.shape or len(task_times) < 2
            or not np.isfinite(task_times).all() or not np.isfinite(source_times).all()
            or np.any(np.diff(task_times) <= 0) or np.any(np.diff(source_times) < 0)
            or not np.isfinite(at)):
        raise ValueError("Invalid task/source clock mapping")
    return float(np.interp(at, task_times, source_times))


def frame_times(end_s, fps=50.):
    if not np.isfinite([end_s, fps]).all() or min(end_s, fps) <= 0:
        raise ValueError("Timeline duration and FPS must be finite and positive")
    return np.arange(int(math.ceil((end_s - 1e-9) * fps)) + 1) / fps


def validate_hash_chain(human_path, gmr_path, mink_path, motion_path, model_path,
                        gmr_report, mink_report, export_report):
    checks = [(human_path, gmr_report, "source_sha256"), (human_path, mink_report, "source_sha256"),
              (mink_path, gmr_report, "baseline_sha256"), (gmr_path, export_report, "source_sha256"),
              (motion_path, export_report, "output_sha256"), (model_path, export_report, "model_sha256")]
    for path, record, key in checks:
        if record.get(key) != sha256(path):
            raise ValueError(f"Source-chain hash mismatch: {Path(path).name} / {key}")


def prepare(task_dir, model_path):
    task = Path(task_dir)
    paths = {"human": task / "human/human_motion.npz", "mink": task / "mink/robot_reference.npz",
             "gmr": task / "gmr/robot_reference.npz", "reference": task / "tracking/motion.npz",
             "isaac": task / "trial/isaac/trajectory.npz", "mujoco": task / "trial/mujoco/rollout.npz"}
    report_paths = {"mink": task / "mink/report.json", "gmr": task / "gmr/report.json",
                    "export": paths["reference"].with_suffix(".json"),
                    "isaac": task / "trial/isaac/report.json", "mujoco": task / "trial/mujoco/report.json"}
    reports = {key: json.loads(path.read_text()) for key, path in report_paths.items()}
    validate_hash_chain(*(paths[k] for k in ("human", "gmr", "mink", "reference")), model_path,
                        reports["gmr"], reports["mink"], reports["export"])
    provenance = validate_provenance(report_paths["isaac"].parent, report_paths["mujoco"].parent,
                                     reports["isaac"], reports["mujoco"])
    if provenance["reference_sha256"] != sha256(paths["reference"]):
        raise ValueError("Actual rollouts and kinematic playback use different references")
    if reports["isaac"].get("terrain") != reports["mujoco"].get("terrain"):
        raise ValueError("Actual rollouts use different terrain")
    if reports["export"]["sampling"]["retimed"]:
        raise ValueError("This renderer currently requires an unretimed GMR export")
    arrays = {key: read_npz(path) for key, path in paths.items()}
    raw, gmr, ref = (arrays[k] for k in ("human", "gmr", "reference"))
    human = KeypointClip(gmr["time_s"], gmr["human_positions_world_m"],
                         gmr["human_joint_names"], gmr["human_parent_indices"])
    if human.names != raw["joint_names"].astype(str).tolist():
        raise ValueError("Human skeleton names differ from the mocap source")
    np.testing.assert_array_equal(human.parents, raw["parent_indices"])
    np.testing.assert_allclose(human.times, raw["time_s"], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(gmr["source_frame_numbers"], raw["frame_numbers"])
    # Independently reconstruct the display transform from decoded raw points.
    mapping = reports["mink"]
    scale = float(mapping["uniform_human_scale"])
    aligned = raw["joint_positions_world_m"] @ np.asarray(mapping["initial_heading_alignment"]).T
    expected = (np.asarray(mapping["initial_root_position_m"])[None, None]
                + scale * (aligned - aligned[:1, human.names.index("root"):human.names.index("root") + 1]))
    np.testing.assert_allclose(human.positions, expected, atol=1e-10, rtol=0)
    human_display_floor = float(mapping["initial_root_position_m"][2]
                                - scale * aligned[0, human.names.index("root"), 2])
    qpos = ref["qpos_mujoco"]
    kinematic = PoseClip(ref["time_s"], qpos[:, :3], qpos[:, 3:7], qpos[:, 7:],
                         reports["export"]["source_joint_names"])
    np.testing.assert_allclose(kinematic.root, ref["body_pos_w"][:, 0], atol=1e-6, rtol=0)
    task_times, source_times = ref["time_s"], ref["source_time_s"]
    source_clock(task_times, source_times, 0.)
    if source_times[0] < human.times[0] - 1e-9 or source_times[-1] > human.times[-1] + 1e-9:
        raise ValueError("Export source-time map exceeds human source coverage")
    if reports["export"]["sampling"]["prepend_hold_frames"] != 0:
        raise ValueError("Prepend holds need explicit labels; not supported by this renderer")
    for task_time, source_time in zip(task_times, source_times):
        if not np.isclose(source_time, min(task_time, source_times[-1]), atol=1e-8):
            raise ValueError("Unexpected time warp in the unretimed reference")
    isaac, mj = load_rollout(arrays["isaac"]), load_rollout(arrays["mujoco"])
    metadata = json.loads(raw["metadata_json"].item())
    return {"human": human, "reference": kinematic, "isaac": isaac, "mujoco": mj,
            "task_times": task_times, "source_times": source_times, "reports": reports,
            "source_clip": Path(metadata["source"]["amc_path"]).stem, "scale": scale,
            "human_display_floor_m": human_display_floor,
            "human_heading_alignment": mapping["initial_heading_alignment"],
            "human_seed_root_m": mapping["initial_root_position_m"],
            "provenance": sanitize(provenance),
            "hashes": {**{k: sha256(p) for k, p in paths.items()},
                       **{k + "_report": sha256(p) for k, p in report_paths.items()}, "model": sha256(model_path),
                       "renderer_script": sha256(Path(__file__))}}


def render(prepared, model_path, output, playback_speed):
    human, reference, isaac, mj = (prepared[k] for k in ("human", "reference", "isaac", "mujoco"))
    end_s = float(max(reference.times[-1], isaac.times[-1], mj.times[-1]))
    timeline = frame_times(end_s)
    task_times, source_times = prepared["task_times"], prepared["source_times"]
    hold_start = float(source_times[-1])
    from eval_tracking_mujoco import load_model
    model, _ = load_model(model_path, terrain=prepared["reports"]["isaac"].get("terrain"))
    if not model.nmesh:
        raise ValueError("A mesh visualization model is required")
    for i in range(model.ngeom):
        name = model.geom(i).name or ""
        if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH and name != "floor" and not name.startswith("terrain_"):
            model.geom_rgba[i, 3] = 0
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 460)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 520)
    data, options = mujoco.MjData(model), mujoco.MjvOption()
    options.sitegroup[:] = 0
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    points = np.concatenate((human.positions.reshape(-1, 3), reference.root, isaac.root, mj.root))
    low, high = points.min(0), points.max(0)
    camera.lookat[:] = [(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, high[2] * .46]
    camera.distance = max(2.8, (high[2] - min(0., low[2])) * 1.7)
    camera.azimuth, camera.elevation = 125, -13
    renderer, writer, fig = None, None, None
    try:
        renderer = mujoco.Renderer(model, width=460, height=520)
        visible = [i for i, name in enumerate(human.names) if not any(s in name for s in ("fingers", "thumb"))]
        landmarks = {human.names.index(name) for name in ("root", "ltibia", "rtibia")}
        def panel_image(clip, time_s, skeleton=False):
            data.qpos[:] = named_qpos(model, clip, time_s)
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)  # Kinematics only: NO dynamics integration.
            renderer.update_scene(data, camera=camera, scene_option=options)
            if skeleton:
                for geom in renderer.scene.geoms[:renderer.scene.ngeom]:
                    if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM:
                        geom.rgba[3] = 0
                # Pelvis alignment moves the human's original display floor by
                # one constant amount. Show that floor, NOT robot ground z=0.
                # This is a visual reference plane, never measured contact.
                floor = prepared["human_display_floor_m"]
                geom = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_PLANE, np.array([3., 3., .1]),
                                   np.array([0., 0., floor]), np.eye(3).ravel(), np.array([.15, .20, .26, 1.], np.float32))
                renderer.scene.ngeom += 1
                for grid_coordinate in np.arange(-1., 2.1, .5):
                    add_connector(renderer.scene, [grid_coordinate, -1., floor + .001], [grid_coordinate, 2., floor + .001], [.27, .34, .42, 1.])
                    add_connector(renderer.scene, [-1., grid_coordinate, floor + .001], [2., grid_coordinate, floor + .001], [.27, .34, .42, 1.])
                p = human.sample(source_clock(task_times, source_times, time_s))
                for i in visible:
                    parent = int(human.parents[i])
                    if parent >= 0:
                        add_connector(renderer.scene, p[parent], p[i], [.77, .70, .99, 1.])
                    add_marker(renderer.scene, p[i], [1., .83, .30, 1.] if i in landmarks else [.77, .70, .99, 1.],
                               radius=.026 if i in landmarks else .012)
            return renderer.render().copy()

        plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT, "axes.labelcolor": MUTED})
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
        titles = [("HUMAN KEYPOINTS", f"CMU {prepared['source_clip']} | aligned, scale {prepared['scale']:.3f}", HUMAN),
                  ("GMR KINEMATICS", "Exported training reference | NO DYNAMICS", REFERENCE),
                  ("ISAAC POLICY", "Recorded PhysX rollout | NOT reference", ACTUAL),
                  ("MUJOCO POLICY", "Same actor | recorded physics rollout", OTHER)]
        artists, labels = [], []
        clips = (reference, reference, isaac, mj)
        for i, (title, subtitle, color) in enumerate(titles):
            ax = fig.add_axes([.025 + i * .242, .34, .23, .49])
            ax.set_axis_off()
            ax.set_title(title + "\n" + subtitle, fontsize=11, color=color, loc="left", pad=12)
            artists.append(ax.imshow(panel_image(clips[i], 0, skeleton=i == 0)))
            labels.append(ax.text(.04, .95, "", va="top", transform=ax.transAxes, color=color,
                                  fontsize=10, bbox={"facecolor": BG, "alpha": .88, "edgecolor": "none"}))
            if i == 0:
                ax.text(.03, .03, f"Display floor z={prepared['human_display_floor_m']:.3f} m\nConstant alignment offset / NOT contact", transform=ax.transAxes,
                        fontsize=9, color=HUMAN, bbox={"facecolor": BG, "alpha": .9, "edgecolor": "none"})
        chart = fig.add_axes([.05, .135, .92, .16], facecolor=PANEL)
        chart.tick_params(colors=MUTED)
        for spine in chart.spines.values():
            spine.set_color("#334155")
        chart.grid(alpha=.16)
        human_z = np.array([human.sample(source_clock(task_times, source_times, t))[human.names.index("root"), 2]
                            for t in timeline])
        chart.plot(timeline, human_z - human_z[0], color=HUMAN, ls="--", label="Human pelvis (scaled)", lw=2)
        for clip, color, name in ((reference, REFERENCE, "GMR reference"), (isaac, ACTUAL, "Isaac"), (mj, OTHER, "MuJoCo")):
            chart.plot(clip.times, clip.root[:, 2] - clip.root[0, 2], color=color, label=name, lw=2)
        chart.axvline(hold_start, color=MUTED, ls=":", lw=1)
        chart.set(xlim=(0, end_s), xlabel="Shared task time (s); no phase alignment or time warping",
                  ylabel="Pelvis / base rise (m)")
        chart.legend(loc="upper right", ncol=4, frameon=False, labelcolor=TEXT, fontsize=10)
        cursor = chart.axvline(0, color=TEXT, lw=1)
        fig.text(.035, .95, f"CMU {prepared['source_clip']}  /  TRON1 WF  /  one-motion comparison", fontsize=23, weight="bold")
        fig.text(.035, .905, "Source keypoints  >  kinematic retargeting  >  learned control in two physics engines", fontsize=15, color=MUTED)
        stamp = fig.text(.73, .95, "", fontsize=15, color=TEXT)
        fig.text(.035, .073, f"Human: one heading alignment + uniform scale. Gold points: pelvis / ankles (not wheel targets).  |  Playback {playback_speed:g}x", fontsize=12, color=MUTED)
        fig.text(.035, .047, f"Robot reference: {prepared['reports']['export']['uniform_z_offset_m'] * 1000:.1f} mm global Z offset; terminal hold from {hold_start:.2f} s. Wheel spin is unobserved.", fontsize=12, color=MUTED)
        fig.text(.035, .021, "All panels use one fixed camera and metric scale. Zero physics steps here; right panels replay recorded dynamics. Pelvis/base height is not COM height.", fontsize=11, color=MUTED)
        episode = prepared["reports"]["isaac"]["summary"]["episodes"][0]
        reasons = (episode["end_reason"], prepared["reports"]["mujoco"]["termination"])
        def update(t):
            for i, clip in enumerate(clips):
                artists[i].set_data(panel_image(clip, t, skeleton=i == 0))
                labels[i].set_text(("SOURCE / REFERENCE\nENDPOINT HOLD" if t > hold_start + 1e-8 else "") if i < 2
                                   else terminal_annotation(clip, t, reasons[i - 2]))
            cursor.set_xdata([t, t])
            stamp.set_text(f"Task {t:.2f} s   |   Source {source_clock(task_times, source_times, t):.2f} s")
            fig.canvas.draw()
        update(float(reference.times[np.argmax(reference.root[:, 2])]))
        fig.savefig(output / "overview.png", dpi=100, facecolor=BG)
        width, height = fig.canvas.get_width_height()
        video = output / "motion_pipeline.mp4"
        writer = imageio_ffmpeg.write_frames(str(video), (width, height), fps=50 * playback_speed,
                    codec="libx264", pix_fmt_out="yuv420p", quality=8, macro_block_size=1,
                    output_params=["-movflags", "+faststart"])
        writer.send(None)
        for t in timeline:
            update(float(t))
            writer.send(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        writer.close()
        writer = None
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(video), "-f", "null", "-"],
                       check=True, capture_output=True)
        return {"status": "rendered", "simulation_steps": 0, "full_decode_verified": True,
                "method": "four synchronized panels: transformed human keypoints, exported kinematics, two recorded physics rollouts",
                "source_clip": prepared["source_clip"], "source_hashes": prepared["hashes"],
                "same_actor_provenance": prepared["provenance"], "video_resolution": [width, height],
                "video_frames": len(timeline), "video_fps": 50 * playback_speed, "playback_speed": playback_speed,
                "task_end_s": end_s, "source_sample_span_s": float(human.times[-1]),
                "source_hold_at_s": hold_start, "source_time_mapping": "exported source_time_s, no inferred phase alignment",
                "human_uniform_scale": prepared["scale"], "human_transform_verified_against_raw_source": True,
                "human_heading_alignment": prepared["human_heading_alignment"], "human_seed_root_m": prepared["human_seed_root_m"],
                "human_display_floor_m": prepared["human_display_floor_m"], "human_floor_measured_contact": False,
                "reference_uniform_z_offset_m": prepared["reports"]["export"]["uniform_z_offset_m"],
                "fixed_shared_camera": True, "camera": {"lookat": camera.lookat.tolist(), "distance": float(camera.distance),
                "azimuth": float(camera.azimuth), "elevation": float(camera.elevation)},
                "video_sha256": sha256(video), "overview_sha256": sha256(output / "overview.png"),
                "limitations": ["Human skeleton is aligned/scaled for visual comparison; ankles are not robot wheel targets.",
                    "Human display floor follows its single constant pelvis-alignment offset; it is not the robot ground or contact evidence.",
                    "Kinematic reference prescribes poses; it proves no contact, force or balance feasibility.",
                    "Terminal reference/human hold is explicitly labeled; source final 0.01 s is omitted by the 50 Hz export.",
                    "Recorded rollouts retain their real timestamps and freeze visibly if their recording ends.",
                    "Source human pelvis and robot base link are not whole-body COM; no new training or dynamics simulation."]}
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        if fig is not None:
            plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True, help="Unretimed suite task with human/mink/gmr/tracking/trial evidence")
    parser.add_argument("--model", type=Path, default=ROOT / "assets/robots/WF_TRON1A/mujoco/robot.xml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--playback-speed", type=float, default=.5)
    args = parser.parse_args(argv)
    if not math.isfinite(args.playback_speed) or args.playback_speed <= 0:
        parser.error("playback-speed must be finite and positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output must be new or empty; prior renders are preserved")
    prepared = prepare(args.task_dir, args.model.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "rendering", "simulation_steps": 0}
    try:
        report = render(prepared, args.model.resolve(), args.output_dir, args.playback_speed)
        (args.output_dir / "index.html").write_text('''<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TRON1 單動作四欄比較</title>
<style>body{background:#0a1220;color:#e2e8f0;font:16px system-ui;margin:2rem}video,img{width:100%;max-width:1920px}a{color:#38bdf8}</style>
<h1>人體關鍵點 → 純運動學 → Isaac → MuJoCo</h1>
<p>左側骨架已做朝向對齊與等比例縮放；第二欄只播放訓練參考姿態、沒有物理模擬。右側兩欄為同政策的實際物理紀錄。</p>
<video src="motion_pipeline.mp4" controls autoplay muted loop playsinline></video>
<p>同一時間軸、固定相機。末端停留明確標示；本影片沒有重新訓練或重新執行動力學。</p>
<p><a href="motion_pipeline.mp4">影片</a> · <a href="overview.png">預覽</a> · <a href="render_report.json">來源與渲染紀錄</a></p></html>''', encoding="utf-8")
        print(json.dumps({"status": report["status"], "video": str(args.output_dir / "motion_pipeline.mp4"), "simulation_steps": 0}))
    except BaseException as exc:
        report.update(status="failed", error=sanitize(f"{type(exc).__name__}: {exc}"))
        raise
    finally:
        (args.output_dir / "render_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
