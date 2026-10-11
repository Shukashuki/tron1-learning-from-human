"""Capture pinned car drift demonstrations and audit an explicitly estimated clock.

No third-party code is executed. No robot reference, training, or physics steps
are produced. Raw states/controls and duplicated rows remain in private outputs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ("world_x", "world_y", "world_heading", "world_vx", "world_vy", "slip_angle", "yaw_rate",
           "steer", "throttle", "hand_brake", "brake", "world_z", "world_vz", "pitch")


def wrap(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def decode_csv(payload):
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    if reader.fieldnames is None or len(reader.fieldnames) != len(COLUMNS) or set(reader.fieldnames) != set(COLUMNS):
        raise ValueError("Expected the 14-column DOA expert schema")
    rows = list(reader)
    if len(rows) < 3:
        raise ValueError("Need at least three source rows")
    if any(set(row) != set(COLUMNS) or None in row.values() for row in rows):
        raise ValueError("Missing or extra CSV values")
    values = {k: np.asarray([float(r[k]) for r in rows]) for k in COLUMNS if k != "hand_brake"}
    if not all(np.isfinite(v).all() for v in values.values()):
        raise ValueError("Nonfinite source values")
    if any(r["hand_brake"] not in ("True", "False") for r in rows):
        raise ValueError("Invalid hand_brake boolean")
    for k, low, high in (("steer", -1., 1.), ("throttle", 0., 1.), ("brake", 0., 1.)):
        if np.any((values[k] < low) | (values[k] > high)):
            raise ValueError(f"Invalid {k} control")
    pos = np.c_[values["world_x"], -values["world_y"]]
    vel = np.c_[values["world_vx"], -values["world_vy"]]
    speed = np.linalg.norm(vel, axis=1)
    if np.any(speed < .1):
        raise ValueError("This moving-only source clock cannot handle near-zero speed")
    yaw = np.unwrap(-np.deg2rad(values["world_heading"]))
    delta = np.diff(pos, axis=0)
    midpoint_vel = .5 * (vel[:-1] + vel[1:])
    repeated = np.linalg.norm(delta, axis=1) < 1e-7
    if np.any(repeated & (np.abs(np.diff(yaw)) > 1e-6)):
        raise ValueError("Pure rotation has no XY support for the estimated clock")
    dt = np.sum(delta * midpoint_vel, axis=1) / np.maximum(np.sum(midpoint_vel ** 2, axis=1), 1e-12)
    if np.any(dt < -1e-10):
        raise ValueError("Position increments oppose reported velocity")
    dt = np.maximum(dt, 0.)
    if np.any((~repeated) & (dt <= 1e-10)):
        raise ValueError("Moving pose has no positive estimated time increment")
    dt[repeated] = 0.
    if np.count_nonzero(~repeated) < 2:
        raise ValueError("Need three distinct moving poses")
    t = np.r_[0., np.cumsum(dt)]
    residual = delta - dt[:, None] * midpoint_vel
    return {"source_row": np.arange(len(rows)), "position_xy_m": pos, "velocity_xy_m_s": vel,
            "source_z_m": values["world_z"], "source_vz_m_s": values["world_vz"],
            "source_pitch_deg": values["pitch"], "yaw_rad": yaw,
            "beta_rad": wrap(np.arctan2(vel[:, 1], vel[:, 0]) - yaw),
            "reported_beta_rad": wrap(-np.deg2rad(values["slip_angle"])),
            "logged_yaw_rate_rad_s": -np.deg2rad(values["yaw_rate"]),
            "controls": np.c_[values["steer"], values["throttle"], values["brake"],
                              [r["hand_brake"] == "True" for r in rows]],
            "time_s_estimated": t, "unique_pose_indices": np.r_[0, np.flatnonzero(~repeated) + 1],
            "clock_displacement_residual_m": np.linalg.norm(residual, axis=1)}


def summarize(a):
    ids = a["unique_pose_indices"]
    t, beta = a["time_s_estimated"][ids], a["beta_rad"][ids]
    yaw_rate = np.gradient(a["yaw_rad"][ids], t)
    tail = t >= t[-1] - .25
    speed = np.linalg.norm(a["velocity_xy_m_s"], axis=1)
    control = a["controls"]
    # Estimated duration of a sideslip interval, not a measured contact event.
    intervals = (np.abs(beta[:-1]) >= np.deg2rad(10)) & (np.abs(beta[1:]) >= np.deg2rad(10))
    return {"source_rows": len(a["source_row"]), "distinct_pose_rows": len(ids),
            "duplicate_pose_intervals": len(a["source_row"]) - len(ids),
            "clock_status": "reconstructed_estimate_not_original_timestamps", "estimated_duration_s": float(t[-1]),
            "clock_formula": "dt = dot(delta_xy, average_velocity_xy) / squared_norm(average_velocity_xy); duplicate poses dt=0",
            "clock_max_displacement_residual_m": float(a["clock_displacement_residual_m"].max()),
            "peak_abs_sideslip_deg": float(np.rad2deg(np.abs(beta).max())),
            "estimated_time_abs_sideslip_ge_10deg_s": float(np.diff(t)[intervals].sum()),
            "final_sideslip_deg": float(np.rad2deg(beta[-1])),
            "estimated_last_025s_median_abs_sideslip_deg": float(np.rad2deg(np.median(np.abs(beta[tail])))),
            "estimated_last_025s_median_abs_yaw_rate_deg_s": float(np.rad2deg(np.median(np.abs(yaw_rate[tail])))),
            "logged_final_yaw_rate_deg_s": float(np.rad2deg(a["logged_yaw_rate_rad_s"][-1])),
            "max_reported_vs_recomputed_sideslip_error_deg": float(np.rad2deg(np.abs(wrap(a["beta_rad"] - a["reported_beta_rad"]))).max()),
            "speed_range_m_s": [float(speed.min()), float(speed.max())],
            "max_abs_steer": float(np.abs(control[:, 0]).max()), "max_throttle": float(control[:, 1].max()),
            "max_brake": float(control[:, 2].max()), "hand_brake_rows": int(control[:, 3].sum()),
            "control_evidence": "steering_input_recorded; car commands are not robot actuator targets",
            "stable_exit_certified": False, "wheel_contact_slip_recorded": False,
            "source_filename_success_used_as_validation": False}


def render_preview(a, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    import imageio_ffmpeg
    ids = a["unique_pose_indices"]
    t, yaw = a["time_s_estimated"][ids], a["yaw_rad"][ids]
    initial_yaw = yaw[0]
    rot = np.array([[np.cos(initial_yaw), np.sin(initial_yaw)], [-np.sin(initial_yaw), np.cos(initial_yaw)]])
    xy = (a["position_xy_m"][ids] - a["position_xy_m"][0]) @ rot.T
    vel = a["velocity_xy_m_s"][ids] @ rot.T
    psi = yaw - initial_yaw
    beta = np.rad2deg(a["beta_rad"][ids])
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor="#0a1220")
    axes = [fig.add_axes(rect, facecolor="#132033") for rect in
            ([.06, .18, .48, .67], [.62, .60, .34, .23], [.62, .31, .34, .23], [.62, .08, .34, .16])]
    for ax in axes:
        ax.tick_params(colors="#cbd5e1"); ax.grid(alpha=.18)
        for spine in ax.spines.values(): spine.set_color("#64748b")
        ax.xaxis.label.set_color("#cbd5e1"); ax.yaxis.label.set_color("#cbd5e1")
    route, angles, speeds, controls = axes
    route.plot(xy[:, 0], xy[:, 1], color="#64748b", lw=2)
    route.set(xlabel="Forward-aligned X (m)", ylabel="Y (m); right-handed, source metric scale", aspect="equal")
    route.set_xlim(xy[:, 0].min() - 4, xy[:, 0].max() + 4)
    route.set_ylim(xy[:, 1].min() - 6, xy[:, 1].max() + 6)
    car = Polygon(np.zeros((4, 2)), color="#38bdf8", alpha=.8)
    route.add_patch(car)
    heading = route.quiver([0.], [0.], [0.], [0.], color="#fbbf24", angles="xy", scale_units="xy", scale=1)
    travel = route.quiver([0.], [0.], [0.], [0.], color="#f472b6", angles="xy", scale_units="xy", scale=1)
    angles.plot(t, beta, label="Sideslip beta", color="#f472b6")
    angles.plot(t, np.rad2deg(psi), label="Body yaw", color="#fbbf24")
    angles.set(ylabel="Angle (deg)")
    angles.legend(frameon=False, labelcolor="#e2e8f0", fontsize=9)
    speeds.plot(t, np.linalg.norm(vel, axis=1), color="#38bdf8")
    speeds.set(ylabel="Speed (m/s)")
    for index, label, color in ((0, "Steer", "#a78bfa"), (1, "Throttle", "#fbbf24"), (2, "Brake", "#38bdf8")):
        controls.plot(t, a["controls"][ids, index], label=label, color=color)
    controls.set(xlabel="Estimated source time (s)", ylabel="Source controls", ylim=(-1.1, 1.1))
    controls.legend(frameon=False, labelcolor="#e2e8f0", fontsize=9, ncol=3)
    cursors = [ax.axvline(0, color="white", lw=1) for ax in axes[1:]]
    fig.text(.06, .94, "Captured car expert demonstration / active steering drift", color="#e2e8f0", fontsize=21)
    fig.text(.06, .895, "DOA / CARLA 0.9.14 / simulated car / source clock reconstructed from position and velocity", color="#cbd5e1", fontsize=13)
    fig.text(.06, .08, "Gold: body heading   Pink: travel direction\nCar outline is schematic; no robot or new physics simulation.", color="#cbd5e1", fontsize=11)
    stamp = fig.text(.06, .035, "", color="#cbd5e1", fontsize=12)
    def update(at):
        p = np.array([np.interp(at, t, xy[:, j]) for j in range(2)])
        angle = np.interp(at, t, psi)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        car.set_xy(np.array([[2.2, .9], [2.2, -.9], [-2.2, -.9], [-2.2, .9]]) @ rotation.T + p)
        v = np.array([np.interp(at, t, vel[:, j]) for j in range(2)])
        heading.set_offsets(p[None]); heading.set_UVC(4 * np.cos(angle), 4 * np.sin(angle))
        travel.set_offsets(p[None]); travel.set_UVC(*(4 * v / max(np.linalg.norm(v), 1e-9)))
        for cursor in cursors: cursor.set_xdata([at, at])
        stamp.set_text(f"Estimated time {at:.3f}s / sideslip {np.interp(at, t, beta):+.1f}deg / playback 0.5x")
        fig.canvas.draw()
    writer = None
    try:
        update(t[np.argmax(np.abs(beta))])
        fig.savefig(folder / "overview.png", facecolor=fig.get_facecolor())
        width, height = fig.canvas.get_width_height()
        writer = imageio_ffmpeg.write_frames(str(folder / "source_preview.mp4"), (width, height), fps=30,
                   codec="libx264", pix_fmt_out="yuv420p", quality=8, macro_block_size=1, output_params=["-movflags", "+faststart"])
        writer.send(None)
        for at in np.arange(0., t[-1] + 1e-9, 1 / 60):
            update(at); writer.send(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    finally:
        if writer is not None: writer.close()
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use a new or empty output directory")
    lock_path = ROOT / "config/drift_sources.json"
    config = json.loads(lock_path.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = args.output_dir / "raw"; raw.mkdir(exist_ok=True)
    report = {"status": "source_capture", "robot_training_performed": False, "simulation_steps": 0,
              "source_repository": config["repository"], "source_commit": config["commit"],
              "configuration_sha256": sha256(lock_path.read_bytes()), "capture_script_sha256": sha256(Path(__file__).read_bytes()),
              "selected_for_first_reference": config["selected_for_first_reference"], "captures": {},
              "limitations": config["limitations"], "clock": config["clock"], "validation_extension": config["validation_extension"]}
    for entry in config["files"]:
        url = f"https://raw.githubusercontent.com/ustcly/DOA/{config['commit']}/{entry['path']}"
        with urllib.request.urlopen(url, timeout=30) as response: payload = response.read(2_000_001)
        if len(payload) > 2_000_000 or sha256(payload) != entry["sha256"]:
            raise ValueError(f"Source hash/size mismatch: {entry['id']}")
        (raw / Path(entry["path"]).name).write_bytes(payload)
        if not entry["path"].endswith(".csv"): continue
        arrays = decode_csv(payload)
        folder = args.output_dir / entry["id"]; folder.mkdir()
        np.savez_compressed(folder / "captured_source.npz", **arrays)
        summary = {**summarize(arrays), "source_path": entry["path"], "source_url": url, "source_sha256": entry["sha256"]}
        report["captures"][entry["id"]] = summary
        if entry["id"] == config["selected_for_first_reference"]:
            render_preview(arrays, folder)
            summary.update(overview_sha256=sha256((folder / "overview.png").read_bytes()),
                           video_sha256=sha256((folder / "source_preview.mp4").read_bytes()))
    report["status"] = "captured_and_screened"
    (args.output_dir / "source_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
