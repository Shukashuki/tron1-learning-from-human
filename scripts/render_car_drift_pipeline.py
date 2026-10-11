"""Replay CARLA source, robot kinematics and two recorded policy rollouts.

MuJoCo is a common mesh renderer only: this script uses mj_forward and never
integrates dynamics. Source time is the disclosed estimated car clock mapped
through the original uniform retiming. Synthetic recovery is explicitly marked;
the car source freezes while the robot's synthetic reference continues.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

from matplotlib.patches import Polygon

from render_tracking import (ACTUAL, BG, MUTED, OTHER, PANEL, REFERENCE, ROOT, TEXT,
                             PoseClip, imageio_ffmpeg, load_rollout, mujoco,
                             named_qpos, np, plt, read_npz, sha256, terminal_annotation)
from render_motion_pipeline import frame_times, source_clock
from render_mink import add_connector
from assess_drift import assess_runs, sideslip
from publish_motion_suite import sanitize
from compare_tracking import validate_provenance

CAR = "#c4b5fd"
TRAVEL = "#f472b6"


class CarClip:
    """Original world XY, yaw, velocity and controls on distinct estimated times."""
    def __init__(self, arrays):
        times = np.asarray(arrays["time_s_estimated"], float)
        ids = np.asarray(arrays["unique_pose_indices"])
        if (times.ndim != 1 or len(times) < 3 or not np.isfinite(times).all()
                or np.any(np.diff(times) < 0) or ids.ndim != 1 or ids.dtype.kind not in "iu"
                or len(ids) < 3 or ids[0] != 0 or ids[-1] >= len(times) or np.any(np.diff(ids) <= 0)):
            raise ValueError("Car source requires ordered distinct-pose indices and an estimated clock")
        self.times = times[ids]
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("Selected car pose timestamps must strictly increase")
        self.position, self.velocity, self.yaw, self.controls = [np.asarray(arrays[key], float)[ids].copy()
            for key in ("position_xy_m", "velocity_xy_m_s", "yaw_rad", "controls")]
        count = len(self.times)
        if (self.position.shape != (count, 2) or self.velocity.shape != (count, 2)
                or self.yaw.shape != (count,) or self.controls.shape != (count, 4)
                or not all(np.isfinite(v).all() for v in (self.position, self.velocity, self.yaw, self.controls))):
            raise ValueError("Car pose, velocity and control shapes must be finite and aligned")
        self.yaw = np.unwrap(self.yaw)
        self.beta = np.arctan2(np.sin(np.arctan2(self.velocity[:, 1], self.velocity[:, 0]) - self.yaw),
                              np.cos(np.arctan2(self.velocity[:, 1], self.velocity[:, 0]) - self.yaw))

    def sample(self, time_s):
        if not np.isfinite(time_s):
            raise ValueError("Car sample time must be finite")
        at = float(np.clip(time_s, self.times[0], self.times[-1]))
        interpolate = lambda values: np.array([np.interp(at, self.times, values[:, j])
                                               for j in range(values.shape[1])])
        control_index = max(0, int(np.searchsorted(self.times, at, side="right")) - 1)
        return {"position": interpolate(self.position), "velocity": interpolate(self.velocity),
                "yaw": float(np.interp(at, self.times, self.yaw)),
                "beta": float(np.interp(at, self.times, np.unwrap(self.beta))),
                "controls": self.controls[control_index].copy(), "estimated_time_s": at}


def validate_mapping(car, robot, exported, adaptation):
    """Verify the original adaptation; never fit a new alignment to policy results."""
    times = np.asarray(robot["time_s"], float)
    source = np.asarray(robot["demonstration_time_s_estimated"], float)
    phase = np.asarray(robot["source_phase"])
    source_clock(times, source, times[0])
    scale, time_scale = float(adaptation["spatial_scale"]), float(adaptation["time_scale"])
    if not np.isfinite([scale, time_scale]).all() or min(scale, time_scale) <= 0:
        raise ValueError("Declared car adaptation scales must be finite and positive")
    if (phase.shape != times.shape or phase.dtype.kind not in "iu" or phase[0] != 0
            or not np.isin(phase, [0, 1, 2]).all() or np.any(np.diff(phase.astype(int)) < 0)):
        raise ValueError("Source phases must monotonically mark capture, synthetic stop and rest")
    captured = phase == 0
    if np.count_nonzero(captured) < 3:
        raise ValueError("Need at least three captured reference poses")
    np.testing.assert_allclose(source[captured], car.times[0] + times[captured] / time_scale, atol=1e-10, rtol=0)
    np.testing.assert_allclose(source[~captured], source[captured][-1], atol=1e-10, rtol=0)
    if source[0] < car.times[0] - 1e-9 or source[-1] > car.times[-1] + 1e-9:
        raise ValueError("Declared source clock exceeds captured car coverage")
    for key in ("time_s", "demonstration_time_s_estimated", "source_phase"):
        np.testing.assert_allclose(exported[key], robot[key], atol=1e-10, rtol=0)
    heading = np.asarray(adaptation["heading_alignment_matrix_xy"], float)
    initial_yaw = car.yaw[0]
    expected_heading = np.array([[np.cos(initial_yaw), np.sin(initial_yaw)], [-np.sin(initial_yaw), np.cos(initial_yaw)]])
    np.testing.assert_allclose(heading, expected_heading, atol=1e-10, rtol=0)
    sampled_xy = np.array([car.sample(at)["position"] for at in source[captured]])
    expected_xy = scale * (sampled_xy - car.position[0]) @ heading.T
    np.testing.assert_allclose(robot["qpos"][captured, :2], expected_xy, atol=1e-8, rtol=0)
    sampled_yaw = np.array([car.sample(at)["yaw"] for at in source[captured]]) - initial_yaw
    np.testing.assert_allclose(robot["root_yaw_rad"][captured], sampled_yaw, atol=1e-10, rtol=0)
    return {"source_end_task_s": float(times[captured][-1]), "source_hold_estimated_s": float(source[captured][-1]),
            "synthetic_stop_end_task_s": float(times[captured][-1] + adaptation["recovery_duration_s"]),
            "source_time_scale": time_scale, "source_spatial_scale": scale}


def phase_annotation(times, phases, at):
    index = max(0, int(np.searchsorted(times, at, side="right")) - 1)
    phase = int(phases[min(index, len(phases) - 1)])
    return ("CAPTURED CAR MOTION", "SYNTHETIC STOP\nNOT CAR CAPTURE", "SYNTHETIC STATIONARY TAIL")[phase]


def _telemetry(arrays, key, engine, count, trailing_shape):
    if key not in arrays:
        return None
    values = np.asarray(arrays[key], float)
    if engine == "Isaac":
        valid = np.asarray(arrays["valid_mask"])
        if values.shape[:2] != valid.shape:
            raise ValueError(f"Malformed {key} time/environment shape")
        values = values[valid[:, 0], 0]
    if values.shape != (count, *trailing_shape) or not np.isfinite(values).all():
        raise ValueError(f"Malformed finite first-episode {key} telemetry")
    return values


def assessment_equal(stored, fresh):
    """Exact identities/gates plus roundoff-tolerant metrics across NumPy builds."""
    if isinstance(stored, dict) and isinstance(fresh, dict):
        return stored.keys() == fresh.keys() and all(assessment_equal(stored[key], fresh[key]) for key in stored)
    if isinstance(stored, list) and isinstance(fresh, list):
        return len(stored) == len(fresh) and all(assessment_equal(a, b) for a, b in zip(stored, fresh))
    if isinstance(stored, (int, float)) and isinstance(fresh, (int, float)) and not isinstance(stored, bool) and not isinstance(fresh, bool):
        return bool(np.isfinite([stored, fresh]).all() and np.isclose(stored, fresh, rtol=1e-12, atol=1e-12))
    return type(stored) is type(fresh) and stored == fresh


def prepare(captured_source, robot_reference, isaac_dir, mujoco_dir, model_path, assessment_path=None):
    captured_source, robot_reference, isaac_dir, mujoco_dir, model_path = map(Path,
        (captured_source, robot_reference, isaac_dir, mujoco_dir, model_path))
    paths = {"captured_source": captured_source, "robot_reference": robot_reference,
             "motion": robot_reference.parent / "motion.npz", "adaptation": robot_reference.parent / "adaptation.json",
             "export_report": robot_reference.parent / "motion.json", "isaac": isaac_dir / "trajectory.npz",
             "mujoco": mujoco_dir / "rollout.npz", "isaac_report": isaac_dir / "report.json",
             "mujoco_report": mujoco_dir / "report.json", "model": model_path,
             "assessment": Path(assessment_path) if assessment_path is not None else isaac_dir.parent / "assessment.json"}
    reports = {key: json.loads(paths[key].read_text()) for key in ("adaptation", "export_report", "isaac_report", "mujoco_report")}
    hashes = {key: sha256(path) for key, path in paths.items()}
    adaptation, exported = reports["adaptation"], reports["export_report"]
    if (adaptation.get("provenance", {}).get("source_npz_sha256") != hashes["captured_source"]
            or exported.get("source_car_provenance", {}).get("source_npz_sha256") != hashes["captured_source"]
            or adaptation.get("robot_reference_sha256") != hashes["robot_reference"]
            or exported.get("source_sha256") != hashes["robot_reference"]
            or adaptation.get("motion_sha256") != hashes["motion"]
            or exported.get("output_sha256") != hashes["motion"]
            or adaptation.get("model_sha256") != hashes["model"] or exported.get("model_sha256") != hashes["model"]):
        raise ValueError("Captured car / robot / export / model source-chain hash mismatch")
    if exported.get("upstream_task_adaptation") != adaptation.get("calibration_and_adaptation"):
        raise ValueError("Retarget and export adaptation metadata differ")
    provenance = validate_provenance(isaac_dir, mujoco_dir, reports["isaac_report"], reports["mujoco_report"])
    if provenance["reference_sha256"] != hashes["motion"]:
        raise ValueError("Actual policy trajectories use a different reference")
    if reports["isaac_report"].get("terrain") != reports["mujoco_report"].get("terrain"):
        raise ValueError("Recorded policy trajectories use different terrain")
    source, robot, motion, ia, ma = [read_npz(paths[key]) for key in ("captured_source", "robot_reference", "motion", "isaac", "mujoco")]
    car = CarClip(source)
    mapping = validate_mapping(car, robot, motion, adaptation["calibration_and_adaptation"])
    reference = PoseClip(motion["time_s"], motion["qpos_mujoco"][:, :3], motion["qpos_mujoco"][:, 3:7],
                         motion["qpos_mujoco"][:, 7:], exported["source_joint_names"])
    original = PoseClip(robot["time_s"], robot["qpos"][:, :3], robot["qpos"][:, 3:7],
                        robot["joint_positions_rad"], robot["joint_names"])
    np.testing.assert_allclose(reference.root, original.root, atol=1e-8, rtol=0)
    np.testing.assert_allclose(reference.quat, original.quat, atol=1e-8, rtol=0)
    for index, name in enumerate(reference.joint_names):
        np.testing.assert_allclose(reference.joints[:, index], original.joints[:, original.joint_names.index(name)], atol=1e-8, rtol=0)
    isaac, mj = load_rollout(ia), load_rollout(ma)
    series = {}
    for key, engine, arrays, clip in (("isaac", "Isaac", ia, isaac), ("mujoco", "MuJoCo", ma, mj)):
        velocity_key = "root_linear_velocity_w_m_s" if engine == "Isaac" else "root_com_lin_vel_w"
        velocity = _telemetry(arrays, velocity_key, engine, len(clip.times), (3,))
        if velocity is None:
            raise ValueError("Measured COM velocity is required to display policy sideslip")
        speed, beta = sideslip(clip.quat, velocity)
        series[key] = {"velocity": velocity, "speed": speed, "beta_deg": np.where(speed >= .1, np.degrees(beta), np.nan),
                       "slip": _telemetry(arrays, "wheel_lateral_contact_slip_m_s", engine, len(clip.times), (2,)),
                       "contacts": _telemetry(arrays, "wheel_ground_contact_count", engine, len(clip.times), (2,))}
    ref_speed, ref_beta = sideslip(reference.quat, motion["body_lin_vel_w"][:, 0])
    assessment = assess_runs(isaac_dir, mujoco_dir, paths["motion"])
    stored_assessment = json.loads(paths["assessment"].read_text())
    if not assessment_equal(sanitize(stored_assessment), sanitize(assessment)):
        raise ValueError("Stored assessment differs from the current recorded policy evidence or assessor")
    hashes.update(renderer=sha256(__file__), drift_assessor=sha256(Path(__file__).with_name("assess_drift.py")),
                  pose_helpers=sha256(Path(__file__).with_name("render_tracking.py")),
                  clock_helpers=sha256(Path(__file__).with_name("render_motion_pipeline.py")))
    return {"car": car, "reference": reference, "isaac": isaac, "mujoco": mj, "series": series,
            "reference_speed": ref_speed, "reference_beta_deg": np.where(ref_speed >= .1, np.degrees(ref_beta), np.nan),
            "reference_velocity": motion["body_lin_vel_w"][:, 0], "task_times": motion["time_s"],
            "source_times": motion["demonstration_time_s_estimated"], "source_phases": motion["source_phase"],
            "reference_action_duration_s": len(motion["time_s"]) / float(motion["fps"].item()),
            "mapping": mapping, "reports": reports, "assessment": assessment, "provenance": provenance, "hashes": hashes}


def _style_chart(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(alpha=.16)
    for spine in ax.spines.values():
        spine.set_color("#334155")


def _charts(prepared, axes, end_s):
    beta_ax, speed_ax, slip_ax = axes
    reference = prepared["reference"]
    source_end = prepared["mapping"]["source_end_task_s"]
    stop_end = prepared["mapping"]["synthetic_stop_end_task_s"]
    captured_times = prepared["task_times"][prepared["source_phases"] == 0]
    source_samples = [prepared["car"].sample(source_clock(prepared["task_times"], prepared["source_times"], t)) for t in captured_times]
    beta_ax.plot(captured_times, [np.degrees(v["beta"]) for v in source_samples], color=CAR, ls="--", label="Car source")
    speed_factor = prepared["mapping"]["source_spatial_scale"] / prepared["mapping"]["source_time_scale"]
    speed_ax.plot(captured_times, [np.linalg.norm(v["velocity"]) * speed_factor for v in source_samples],
                  color=CAR, ls="--", label=f"Car speed x{speed_factor:g}")
    for ax, values in ((beta_ax, prepared["reference_beta_deg"]), (speed_ax, prepared["reference_speed"])):
        ax.plot(reference.times, values, color=REFERENCE, label="Kinematic target", lw=1.8)
    for key, clip, color in (("isaac", prepared["isaac"], ACTUAL), ("mujoco", prepared["mujoco"], OTHER)):
        values = prepared["series"][key]
        beta_ax.plot(clip.times, values["beta_deg"], color=color, label=key.title(), lw=1.5)
        speed_ax.plot(clip.times, values["speed"], color=color, label=key.title(), lw=1.5)
        if values["slip"] is not None and values["contacts"] is not None:
            supported = np.all(values["contacts"] > 0, axis=1)
            slip_ax.plot(clip.times, np.where(supported, np.abs(values["slip"]).mean(axis=1), np.nan),
                         color=color, label=key.title() + " (both grounded)", lw=1.5)
        else:
            slip_ax.text(.02, .87 if key == "isaac" else .72, key.title() + ": CONTACT EVIDENCE MISSING",
                         transform=slip_ax.transAxes, color=color, fontsize=9)
        for ax in axes:
            if clip.times[-1] < end_s - 1e-8:
                ax.axvline(clip.times[-1], color=color, ls=":", lw=1.2)
    beta_ax.axhline(8., color=MUTED, ls=":", lw=.8)
    beta_ax.axhline(-8., color=MUTED, ls=":", lw=.8)
    speed_ax.axhline(.4, color=MUTED, ls=":", lw=.8)
    slip_ax.axhline(.04, color=MUTED, ls=":", lw=.8)
    for ax in axes:
        _style_chart(ax)
        ax.axvspan(source_end, stop_end, color=REFERENCE, alpha=.06)
        ax.axvspan(stop_end, end_s, color=MUTED, alpha=.08)
        ax.axvline(source_end, color=MUTED, ls=":", lw=1)
        ax.set(xlim=(0., end_s), xlabel="Shared task time (s)")
        ax.legend(frameon=False, labelcolor=TEXT, fontsize=8, loc="upper right")
    beta_ax.set_ylabel("COM sideslip beta (deg)", color=MUTED)
    speed_ax.set_ylabel("Planar COM speed (m/s)", color=MUTED)
    slip_ax.set_ylabel("Contact-patch lateral slip (m/s)", color=MUTED)


def render(prepared, model_path, output, playback_speed=.5):
    from eval_tracking_mujoco import load_model
    output = Path(output)
    reference, isaac, mj, car = [prepared[key] for key in ("reference", "isaac", "mujoco", "car")]
    end_s = float(max(prepared["reference_action_duration_s"], isaac.times[-1], mj.times[-1]))
    timeline = frame_times(end_s)
    model, _ = load_model(model_path, terrain=prepared["reports"]["isaac_report"].get("terrain"))
    if not model.nmesh:
        raise ValueError("Official robot visualization meshes are required")
    for clip in (reference, isaac, mj):
        named_qpos(model, clip, clip.times[0])
    for geom in range(model.ngeom):
        name = model.geom(geom).name or ""
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH and name != "floor" and not name.startswith("terrain_"):
            model.geom_rgba[geom, 3] = 0.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 460)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 520)
    data, options = mujoco.MjData(model), mujoco.MjvOption()
    options.sitegroup[:] = 0
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    all_roots = np.concatenate((reference.root, isaac.root, mj.root))
    low, high = all_roots.min(0), all_roots.max(0)
    camera.lookat[:] = [(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, max(.45, high[2] * .48)]
    camera.distance = max(2.8, float(np.max(high[:2] - low[:2])) * 1.5 + 1.4)
    camera.azimuth, camera.elevation = 125., -32.
    close_camera = mujoco.MjvCamera()
    close_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    close_camera.distance, close_camera.azimuth, close_camera.elevation = 1.6, 125., -20.
    renderer, writer, fig = None, None, None
    try:
        renderer = mujoco.Renderer(model, width=460, height=520)
        plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT, "axes.labelcolor": MUTED})
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
        assessment = prepared["assessment"]
        fig.text(.03, .95, "Captured car drift  /  TRON1 WF  /  same-policy sim2sim", fontsize=23, weight="bold")
        fig.text(.03, .91, f"Fixed task gates: {assessment['verdict'].upper()}  |  Isaac {assessment['isaac']['verdict'].upper()}  /  MuJoCo {assessment['mujoco']['verdict'].upper()}", fontsize=15, color=MUTED)
        stamp = fig.text(.72, .95, "", fontsize=13)
        axes = [fig.add_axes([.025 + index * .242, .345, .23, .47]) for index in range(4)]
        car_ax = axes[0]
        _style_chart(car_ax)
        car_ax.set_title("CARLA CAR DEMONSTRATION\nOriginal metric XY/yaw | estimated clock", color=CAR, loc="left", fontsize=11, pad=12)
        car_ax.plot(car.position[:, 0], car.position[:, 1], color=MUTED, lw=1.8, alpha=.7)
        car_ax.set_aspect("equal", adjustable="box")
        car_low, car_high = car.position.min(0) - 4., car.position.max(0) + 4.
        car_center = (car_low + car_high) / 2
        car_span = car_high - car_low
        # Preserve metric aspect while filling the allotted panel, avoiding a
        # thin axis whose labels spill into the neighboring robot panel.
        panel_aspect = (.23 * 1920) / (.47 * 1080)
        car_span[0] = max(car_span[0], car_span[1] * panel_aspect)
        car_span[1] = max(car_span[1], car_span[0] / panel_aspect)
        car_ax.set(xlim=(car_center[0] - car_span[0] / 2, car_center[0] + car_span[0] / 2),
                   ylim=(car_center[1] - car_span[1] / 2, car_center[1] + car_span[1] / 2),
                   xlabel="Source world X (m)", ylabel="Source world Y (m)")
        car_patch = Polygon(np.zeros((4, 2)), color=CAR, alpha=.9)
        car_ax.add_patch(car_patch)
        heading = car_ax.quiver([0.], [0.], [0.], [0.], color=REFERENCE, angles="xy", scale_units="xy", scale=1)
        travel = car_ax.quiver([0.], [0.], [0.], [0.], color=TRAVEL, angles="xy", scale_units="xy", scale=1)
        car_label = car_ax.text(.03, .97, "", transform=car_ax.transAxes, va="top", fontsize=9, color=CAR,
                               bbox={"facecolor": BG, "alpha": .9, "edgecolor": "none"})
        artists, inset_artists, robot_labels = [], [], []
        clips = (reference, isaac, mj)
        titles = [("ROBOT KINEMATICS", "Exported training reference | NO DYNAMICS", REFERENCE),
                  (f"ISAAC POLICY | {assessment['isaac']['verdict'].upper()}", "Recorded PhysX episode", ACTUAL),
                  (f"MUJOCO POLICY | {assessment['mujoco']['verdict'].upper()}", "Same actor | recorded physics episode", OTHER)]
        for ax, clip, (title, subtitle, color) in zip(axes[1:], clips, titles):
            ax.set_axis_off()
            ax.set_title(title + "\n" + subtitle, color=color, loc="left", fontsize=11, pad=12)
            artists.append(ax.imshow(np.zeros((520, 460, 3), dtype=np.uint8)))
            inset = ax.inset_axes([.62, .57, .36, .40])
            inset.set_axis_off()
            inset_artists.append(inset.imshow(np.zeros((520, 460, 3), dtype=np.uint8)))
            inset.text(.04, .96, "POSE CLOSE-UP\nroot-following camera", transform=inset.transAxes, va="top",
                       fontsize=7, color=TEXT, bbox={"facecolor": BG, "alpha": .85, "edgecolor": "none"})
            robot_labels.append(ax.text(.03, .97, "", transform=ax.transAxes, va="top", fontsize=9, color=color,
                                        bbox={"facecolor": BG, "alpha": .9, "edgecolor": "none"}))
        chart_axes = [fig.add_axes([.05 + i * .315, .115, .285, .16], facecolor=PANEL) for i in range(3)]
        _charts(prepared, chart_axes, end_s)
        cursors = [ax.axvline(0., color=TEXT, lw=1) for ax in chart_axes]
        mapping = prepared["mapping"]
        fig.text(.03, .067, f"Original adaptation: XY x{mapping['source_spatial_scale']:g}; time x{mapping['source_time_scale']:g}. Captured task segment ends {mapping['source_end_task_s']:.2f}s; source then holds.", fontsize=11, color=MUTED)
        fig.text(.03, .041, "Gold shading: synthetic stop; gray shading: synthetic rest. Gold arrows: heading; pink arrows: travel direction. Source controls are car inputs.", fontsize=11, color=MUTED)
        fig.text(.03, .015, f"Robot world views share one fixed camera; labeled pose insets follow each root. Playback {playback_speed:g}x; no new physics. Failures freeze at recorded termination.", fontsize=11, color=MUTED)
        isaac_outcomes = [v for v in prepared["reports"]["isaac_report"].get("summary", {}).get("episodes", []) if v.get("environment") == 0]
        reasons = (isaac_outcomes[0].get("end_reason", "unknown") if len(isaac_outcomes) == 1 else "unknown",
                   prepared["reports"]["mujoco_report"].get("termination", "unknown"))

        def robot_image(clip, at, velocity, *, closeup=False):
            data.qpos[:] = named_qpos(model, clip, at)
            data.qvel[:] = 0.
            mujoco.mj_forward(model, data)  # Pose replay only; never integrate dynamics.
            root, quat, _ = clip.sample(at)
            close_camera.lookat[:] = root + [0., 0., -.28]
            renderer.update_scene(data, camera=close_camera if closeup else camera, scene_option=options)
            w, x, y, z = quat
            psi = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            origin = root + [0., 0., .13]
            add_connector(renderer.scene, origin, origin + .4 * np.array([np.cos(psi), np.sin(psi), 0.]), [1., .83, .3, 1.])
            v = np.array([np.interp(at, clip.times, velocity[:, j]) for j in range(3)])
            planar_speed = np.linalg.norm(v[:2])
            if planar_speed >= .1:
                add_connector(renderer.scene, origin, origin + .4 * np.r_[v[:2] / planar_speed, 0.], [1., .4, .7, 1.])
            return renderer.render().copy()

        def update(at):
            source_t = source_clock(prepared["task_times"], prepared["source_times"], at)
            sample = car.sample(source_t)
            psi, position = sample["yaw"], sample["position"]
            rotation = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
            car_patch.set_xy(np.array([[2.2, .9], [2.2, -.9], [-2.2, -.9], [-2.2, .9]]) @ rotation.T + position)
            heading.set_offsets(position[None])
            heading.set_UVC(4 * np.cos(psi), 4 * np.sin(psi))
            velocity = sample["velocity"]
            travel.set_offsets(position[None])
            travel.set_UVC(*(4 * velocity / max(np.linalg.norm(velocity), 1e-9)))
            controls = sample["controls"]
            source_note = "SOURCE ENDPOINT HOLD\nROBOT RECOVERY IS SYNTHETIC" if at > mapping["source_end_task_s"] + 1e-8 else "CAPTURED CAR STATE"
            car_label.set_text(f"{source_note}\nSource {source_t:.2f}s (estimated)\nbeta {np.degrees(sample['beta']):+.1f} deg | speed {np.linalg.norm(velocity):.1f} m/s\nsteer {controls[0]:+.2f} | throttle {controls[1]:.2f}")
            for index, (clip, artist, label) in enumerate(zip(clips, artists, robot_labels)):
                key = "isaac" if index == 1 else "mujoco"
                v = prepared["reference_velocity"] if index == 0 else prepared["series"][key]["velocity"]
                artist.set_data(robot_image(clip, at, v))
                inset_artists[index].set_data(robot_image(clip, at, v, closeup=True))
                if index == 0:
                    note = phase_annotation(prepared["task_times"], prepared["source_phases"], at)
                    if at > reference.times[-1] + 1e-8:
                        note += "\nREFERENCE ENDPOINT HOLD"
                else:
                    note = terminal_annotation(clip, at, reasons[index - 1])
                    if note:
                        note = assessment[key]["verdict"].upper() + " | TERMINAL FREEZE\n" + note
                label.set_text(note)
            for cursor in cursors:
                cursor.set_xdata([at, at])
            stamp.set_text(f"Task {at:.2f}s | source {source_t:.2f}s*")
            fig.canvas.draw()

        source_peak = float(car.times[np.argmax(np.abs(car.beta))])
        source_index = int(np.argmin(np.abs(prepared["source_times"] - source_peak)))
        update(float(prepared["task_times"][source_index]))
        fig.savefig(output / "overview.png", dpi=100, facecolor=BG)
        update(end_s)
        fig.savefig(output / "final.png", dpi=100, facecolor=BG)
        width, height = fig.canvas.get_width_height()
        video = output / "motion_pipeline.mp4"
        writer = imageio_ffmpeg.write_frames(str(video), (width, height), fps=50 * playback_speed,
                    codec="libx264", pix_fmt_out="yuv420p", quality=8, macro_block_size=1,
                    output_params=["-movflags", "+faststart"])
        writer.send(None)
        for at in timeline:
            update(float(at))
            writer.send(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        writer.close()
        writer = None
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(video), "-f", "null", "-"],
                       check=True, capture_output=True)
        plt.close(fig)
        fig = plt.figure(figsize=(16, 10), dpi=120, facecolor=BG)
        charts = [fig.add_axes([.08, .67 - i * .28, .85, .22]) for i in range(3)]
        _charts(prepared, charts, end_s)
        fig.text(.08, .955, f"Recorded drift evidence | {assessment['verdict'].upper()}", fontsize=22, weight="bold")
        fig.text(.08, .92, "Actual COM sideslip, speed and wheel-ground contact-patch slip; terminal states are not extrapolated", fontsize=12, color=MUTED)
        fig.text(.08, .045, "Source clock is estimated. Gold band is synthetic stop; gray band is rest. Slip curves include only both-wheel-grounded samples.", fontsize=11, color=MUTED)
        fig.savefig(output / "contact_overview.png", facecolor=BG)
        return sanitize({"schema_version": 1, "status": "rendered", "simulation_steps": 0,
            "full_decode_verified": True, "method": "captured CARLA topview / exported robot kinematics / recorded Isaac / same-actor recorded MuJoCo",
            "task_verdict": assessment["verdict"], "assessment": assessment, "same_actor_provenance": prepared["provenance"],
            "source_hashes": prepared["hashes"], "source_provenance": prepared["reports"]["adaptation"]["provenance"],
            "assessment_sha256": prepared["hashes"]["assessment"],
            "original_adaptation": prepared["reports"]["adaptation"]["calibration_and_adaptation"],
            "mapping": mapping, "task_end_s": end_s, "source_time_mapping": "original demonstration_time_s_estimated; task/time_scale during capture, then constant source endpoint; no fitted alignment",
            "source_panel": "original captured world XY/yaw; native metric scale; schematic car outline; its own fixed topview",
            "robot_reference": "exact exported training reference; source mapping verified against original capture; no physical feasibility claim",
            "recorded_policy_end_s": {"isaac": float(isaac.times[-1]), "mujoco": float(mj.times[-1])},
            "policy_terminal_freeze_labels": True, "robot_panels_share_fixed_camera": True,
            "pose_closeup_insets": {"labeled": True, "camera": "independently follows each recorded/reference root; no trajectory coordinates changed",
                                     "distance": float(close_camera.distance), "azimuth": float(close_camera.azimuth), "elevation": float(close_camera.elevation)},
            "camera": {"lookat": camera.lookat.tolist(), "distance": float(camera.distance), "azimuth": float(camera.azimuth), "elevation": float(camera.elevation)},
            "video_resolution": [width, height], "video_frames": len(timeline), "video_fps": 50 * playback_speed,
            "playback_speed": playback_speed, "video_sha256": sha256(video),
            "overview_sha256": sha256(output / "overview.png"), "final_sha256": sha256(output / "final.png"),
            "contact_overview_sha256": sha256(output / "contact_overview.png"), "hardware_ready": False,
            "limitations": ["Source is a simulated CARLA car; its original CSV has no original timestamps.",
                "Schematic source topview uses captured poses, not reconstructed physical car dynamics.",
                "Synthetic stop/rest are robot task adaptations, not demonstrated source motion.",
                "Source controls remain car evidence and are not robot action targets.",
                "Near-rest COM sideslip is hidden below0.1m/s; the angle is undefined near rest.",
                "Contact-patch slip is drawn only when both recorded wheels contact the ground; missing telemetry remains missing.",
                "Early-ended actual trajectories visibly hold their recorded terminal pose; no phase shift or extrapolated dynamics.",
                "One recorded first episode per engine does not establish statistical robustness."]})
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        if fig is not None:
            plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captured-source", type=Path, required=True)
    parser.add_argument("--robot-reference", type=Path, required=True)
    parser.add_argument("--isaac-dir", type=Path, required=True)
    parser.add_argument("--mujoco-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "assets/robots/WF_TRON1A/mujoco/robot.xml")
    parser.add_argument("--assessment", type=Path, help="Default: Isaac directory's parent/assessment.json; must match fresh offline assessment")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--playback-speed", type=float, default=.5)
    args = parser.parse_args(argv)
    if not math.isfinite(args.playback_speed) or args.playback_speed <= 0:
        parser.error("playback-speed must be finite and positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use a new or empty output directory; existing renders are preserved")
    prepared = prepare(args.captured_source, args.robot_reference, args.isaac_dir, args.mujoco_dir, args.model, args.assessment)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "rendering", "simulation_steps": 0}
    try:
        report = render(prepared, args.model, args.output_dir, args.playback_speed)
        (args.output_dir / "index.html").write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TRON1 captured car drift comparison</title>
<style>body{background:#0a1220;color:#e2e8f0;font:16px system-ui;margin:2rem}video,img{width:100%;max-width:1920px}a{color:#38bdf8}</style>
<h1>Captured car → robot kinematics → Isaac → MuJoCo</h1>
<p>The original car demonstration has an estimated clock. Its stored spatial/time adaptation is preserved; the synthetic stop and stationary tail are labeled. Recorded policy failures freeze visibly at their terminal states.</p>
<video src="motion_pipeline.mp4" controls autoplay muted loop playsinline></video>
<p>One normalized actor and one exported reference; actual COM sideslip, speed and wheel-ground contact-patch slip are plotted on the shared task clock. Rendering integrates no physics.</p>
<p><a href="motion_pipeline.mp4">Video</a> · <a href="overview.png">Peak overview</a> · <a href="final.png">Final state</a> · <a href="contact_overview.png">Contact evidence</a> · <a href="render_report.json">Report</a></p></html>''', encoding="utf-8")
        print(json.dumps({"status": report["status"], "verdict": report["task_verdict"], "simulation_steps": 0}))
    except BaseException as exc:
        report.update(status="failed", error=sanitize(f"{type(exc).__name__}: {exc}"))
        raise
    finally:
        (args.output_dir / "render_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
