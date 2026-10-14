"""Pin and adapt a high-speed CARLA drift_drl turn to the TRON1 WF motion contract.

The source CSV records a car, not a robot or contact patches. Its time is
estimated from displacement and velocity, not treated as an original clock.
No dynamics simulation, policy learning, or behavior validation occurs here.
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

from export_tracking_motion import convert_motion
from prepare_car_drift_reference import adapt_source, sha256, solve_reference, MODEL

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/drift_turn_source.json"
COLUMNS = ("world_x", "world_y", "world_heading", "local_vx", "local_vy",
           "slip_angle", "yaw_rate", "steer", "throttle", "hand_brake", "brake")


def wrap(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def decode_csv(payload):
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    if reader.fieldnames is None or tuple(reader.fieldnames) != COLUMNS:
        raise ValueError("Unexpected drift_drl car CSV schema")
    rows = list(reader)
    if len(rows) < 3 or any(set(row) != set(COLUMNS) or None in row.values() for row in rows):
        raise ValueError("Missing or extra car source rows")
    values = {name: np.asarray([float(row[name]) for row in rows], dtype=float)
              for name in COLUMNS if name != "hand_brake"}
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError("Nonfinite car source value")
    if any(row["hand_brake"] not in ("0", "1") for row in rows):
        raise ValueError("Unexpected drift_drl hand-brake encoding")
    for name, low, high in (("steer", -1., 1.), ("throttle", 0., 1.), ("brake", 0., 1.)):
        if np.any((values[name] < low) | (values[name] > high)):
            raise ValueError(f"Out-of-range {name} source control")
    yaw_left = np.deg2rad(values["world_heading"])
    forward, lateral = values["local_vx"], values["local_vy"]
    world_vx = forward * np.cos(yaw_left) - lateral * np.sin(yaw_left)
    world_vy = forward * np.sin(yaw_left) + lateral * np.cos(yaw_left)
    position = np.c_[values["world_x"], -values["world_y"]]
    velocity = np.c_[world_vx, -world_vy]
    speed = np.linalg.norm(velocity, axis=1)
    if np.any(speed < .1):
        raise ValueError("Source includes speed too low for position-derived clock")
    yaw = np.unwrap(-yaw_left)
    beta = wrap(np.arctan2(velocity[:, 1], velocity[:, 0]) - yaw)
    reported_beta = wrap(-np.deg2rad(values["slip_angle"]))
    if np.max(np.abs(wrap(beta - reported_beta))) > 1e-5:
        raise ValueError("Source local velocity and reported slip angle disagree")
    delta = np.diff(position, axis=0)
    midpoint = .5 * (velocity[:-1] + velocity[1:])
    repeated = np.linalg.norm(delta, axis=1) < 1e-7
    dt = np.sum(delta * midpoint, axis=1) / np.maximum(np.sum(midpoint**2, axis=1), 1e-12)
    if np.any(dt < -1e-8) or np.any((~repeated) & (dt <= 1e-10)):
        raise ValueError("Source displacement opposes velocity or has no positive clock")
    if np.any(repeated & (np.abs(np.diff(yaw)) > 1e-6)):
        raise ValueError("Pure-rotation source row has no clock support")
    dt = np.maximum(dt, 0.)
    dt[repeated] = 0.
    residual = delta - dt[:, None] * midpoint
    return {
        "position_xy_m": position, "velocity_xy_m_s": velocity,
        "yaw_rad": yaw, "beta_rad": beta, "speed_m_s": speed,
        "time_s_estimated": np.r_[0., np.cumsum(dt)],
        "unique_pose_indices": np.r_[0, np.flatnonzero(~repeated) + 1],
        "controls": np.c_[values["steer"], values["throttle"], values["brake"],
                          np.array([row["hand_brake"] == "1" for row in rows])],
        "max_clock_displacement_residual_m": float(np.linalg.norm(residual, axis=1).max()),
    }


def selected_source(full, config):
    start, end = config["source_row_start_inclusive"], config["source_row_end_inclusive"]
    drift_start, drift_end = config["drift_row_start_inclusive"], config["drift_row_end_inclusive"]
    if not (isinstance(start, int) and isinstance(end, int) and isinstance(drift_start, int)
            and isinstance(drift_end, int) and 0 <= start < drift_start < drift_end < end
            and end < len(full["time_s_estimated"])):
        raise ValueError("Invalid preselected source and drift row intervals")
    core = slice(drift_start, drift_end + 1)
    velocity = full["velocity_xy_m_s"]
    course = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    turn_deg = float(np.rad2deg(course[drift_end] - course[drift_start]))
    speed = full["speed_m_s"][core]
    slip = np.rad2deg(full["beta_rad"][core])
    controls = full["controls"][core]
    if (abs(turn_deg) < 60. or np.min(speed) < 20.
            or np.min(np.abs(slip)) < 8. or np.max(np.abs(slip)) < 20.
            or np.min(controls[:, 1]) < .99 or np.max(controls[:, 2:]) > 0.):
        raise ValueError("Pinned car window no longer matches its high-speed sliding-turn evidence")
    indices = full["unique_pose_indices"]
    local_unique = indices[(indices >= start) & (indices <= end)] - start
    if len(local_unique) < 3 or local_unique[0] != 0 or local_unique[-1] != end - start:
        raise ValueError("Crop endpoints must be distinct original car poses")
    selected = {name: full[name][start:end + 1].copy()
                for name in ("position_xy_m", "velocity_xy_m_s", "yaw_rad")}
    selected["time_s_estimated"] = (full["time_s_estimated"][start:end + 1]
                                     - full["time_s_estimated"][start])
    selected["unique_pose_indices"] = local_unique
    evidence = {
        "drift_rows_inclusive": [drift_start, drift_end],
        "car_turn_course_deg": turn_deg,
        "car_speed_start_m_s": float(speed[0]),
        "car_speed_min_m_s": float(speed.min()),
        "car_peak_abs_sideslip_deg": float(np.max(np.abs(slip))),
        "car_min_abs_sideslip_deg": float(np.min(np.abs(slip))),
        "car_min_throttle": float(controls[:, 1].min()),
        "car_max_brake": float(controls[:, 2].max()),
        "car_handbrake_rows": int(np.count_nonzero(controls[:, 3])),
        "source_window_duration_s_estimated": float(selected["time_s_estimated"][-1]),
        "drift_duration_s_estimated": float(full["time_s_estimated"][drift_end]
                                            - full["time_s_estimated"][drift_start]),
        "full_source_max_clock_displacement_residual_m": full["max_clock_displacement_residual_m"],
    }
    return selected, evidence


def reference_turn_evidence(arrays, config):
    """Check the exported kinematic path, not a simulated or learned motion."""
    times = arrays["time_s"]
    xy = arrays["root_target_position_m"][:, :2]
    rows = arrays["source_frame_numbers"]
    phase = arrays["source_phase"]
    velocity = np.gradient(xy, times, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    course = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    beta = np.rad2deg(np.abs(wrap(course - arrays["root_yaw_rad"])))
    core = np.flatnonzero((rows >= config["drift_row_start_inclusive"])
                          & (rows <= config["drift_row_end_inclusive"])
                          & (phase == 0))
    if len(core) < 3 or np.any(np.diff(core) != 1):
        raise ValueError("The exported reference lost the selected drift interval")
    turn = float(np.rad2deg(course[core[-1]] - course[core[0]]))
    minimum_speed = float(speed[core].min())
    minimum_slip = float(beta[core].min())
    if abs(turn) < 45. or minimum_speed < .8 or minimum_slip < 8.:
        raise ValueError("The exported reference lost the high-speed sliding turn")
    return {
        "measurement": "finite-difference root-target XY at 50 Hz; no simulated contact or policy",
        "sample_count": len(core), "time_interval_s": [float(times[core[0]]), float(times[core[-1]])],
        "source_row_mapping_interval": [float(rows[core[0]]), float(rows[core[-1]])],
        "course_turn_deg": turn, "minimum_speed_m_s": minimum_speed,
        "minimum_abs_sideslip_deg": minimum_slip,
        "peak_abs_sideslip_deg": float(beta[core].max()),
    }


def prepare(config_path, output_dir, raw_csv=None):
    config_path, output_dir = Path(config_path), Path(output_dir)
    config = json.loads(config_path.read_text())
    if (config["schema_version"] != 1 or config["repository"] != "https://github.com/caipeide/drift_drl"
            or not config["commit"] or config["path"] != "code/ref_trajectory/traj_4.csv"):
        raise ValueError("Unexpected source pin")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Preserving prior turn reference output")
    url = ("https://raw.githubusercontent.com/caipeide/drift_drl/"
           + config["commit"] + "/" + config["path"])
    payload = Path(raw_csv).read_bytes() if raw_csv is not None else urllib.request.urlopen(url, timeout=30).read()
    source_hash = hashlib.sha256(payload).hexdigest()
    if source_hash != config["sha256"]:
        raise ValueError("Car source CSV differs from pinned SHA-256")
    full = decode_csv(payload)
    selected, evidence = selected_source(full, config)
    intent = adapt_source(selected, spatial_scale=config["spatial_scale"],
                          time_scale=config["time_scale"], root_height=config["root_height_m"],
                          recovery_s=config["recovery_s"],
                          stationary_tail_s=config["stationary_tail_s"])
    intent["source_row_numbers"] += config["source_row_start_inclusive"]
    arrays, diagnostics = solve_reference(intent)
    turn_evidence = reference_turn_evidence(arrays, config)
    provenance = {
        "source_repository": config["repository"], "source_commit": config["commit"],
        "source_csv_path": config["path"], "source_csv_sha256": source_hash,
        "source_configuration_sha256": sha256(config_path),
        "source_is_physical_car_capture": False,
        "source_clock_status": "reconstructed_estimate_not_original_timestamps",
        "source_row_interval_inclusive": [config["source_row_start_inclusive"],
                                           config["source_row_end_inclusive"]],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source.csv").write_bytes(payload)
    robot_path = output_dir / "robot_reference.npz"
    with robot_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    exported, metadata = convert_motion(robot_path, output_fps=50., append_s=0., z_offset=0.)
    for key in ("demonstration_time_s_estimated", "source_phase", "root_yaw_rad"):
        if len(exported["time_s"]) != len(arrays[key]):
            raise ValueError("Exporter changed the 50 Hz motion sample count")
        exported[key] = arrays[key]
    metadata.update({
        "wheel_spin_reference": "Rolling-direction geometric Jacobian; lateral slip retained; unobserved wheel spin remains masked",
        "upstream_task_adaptation": intent["metadata"], "source_car_provenance": provenance,
        "source_time_mapping": "Source times are estimated relative to the selected car crop; source_frame_numbers retain full-CSV row indices",
        "recovery_is_demonstrated": False, "source_controls_are_robot_targets": False,
    })
    metadata["warnings"] = [warning for warning in metadata["warnings"]
                            if not warning.startswith(("No coordinate reset", "Endpoint holds"))]
    metadata["warnings"].append("This is a cropped drift_drl/CARLA reference with an estimated clock and a synthetic stop; no wheel contact is recorded.")
    motion_path = output_dir / "motion.npz"
    with motion_path.open("xb") as stream:
        np.savez_compressed(stream, **exported)
    metadata["output_sha256"] = sha256(motion_path)
    (output_dir / "motion.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    report = {
        "schema_version": 1, "skill": "car_drift_turn", "status": "kinematic_reference_ready_for_physics_training",
        "kind": "car_demonstration_turn_adaptation", "provenance": provenance,
        "source_selection_evidence": evidence, "calibration_and_adaptation": intent["metadata"],
        "exported_reference_turn_evidence": turn_evidence,
        "diagnostics": diagnostics, "model_sha256": sha256(MODEL),
        "generator_sha256": sha256(Path(__file__)), "robot_reference_sha256": sha256(robot_path),
        "motion_sha256": sha256(motion_path), "frames": len(exported["time_s"]),
        "sample_span_s": float(exported["time_s"][-1]),
        "physics_validated": False, "policy_trained": False, "sim2sim_evaluated": False,
        "hardware_ready": False,
    }
    (output_dir / "adaptation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=CONFIG)
    parser.add_argument("--source-csv", type=Path, help="Offline copy; same pinned hash required")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_config, args.output_dir, args.source_csv),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
