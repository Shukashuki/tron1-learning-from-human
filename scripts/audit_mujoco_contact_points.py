"""Instrumented, deterministic MuJoCo replay of wheel/ground contact points.

The historical rollout did not archive raw data.contact. This script never
changes its policy, gate, or physics. It captures points in a new episode and
requires every numerical replay rollout array to match the original exactly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import eval_tracking_mujoco as evaluator
from training.tron1_contact_slip import contact_point_lateral_slip_m_s


def point_snapshot(state):
    mj, model, data = state.mj, state.model, state.data
    floor = model.geom("floor").id
    plane_origin = data.geom_xpos[floor]
    plane_normal = data.geom_xmat[floor].reshape(3, 3)[:, 2]
    wheels = []
    for side, body in enumerate(state.wheels):
        geom = model.geom(("wheel_L_collision", "wheel_R_collision")[side]).id
        center = data.geom_xpos[geom]
        rotation = data.geom_xmat[geom].reshape(3, 3)
        axis = rotation[:, 2]
        radius, halfwidth = model.geom_size[geom, :2]
        axis_normal = float(axis @ plane_normal)
        support = halfwidth * abs(axis_normal) + radius * np.sqrt(max(0., 1. - axis_normal**2))
        clearance = float(plane_normal @ (center - plane_origin) - support)
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mj.mj_jacBodyCom(model, data, jacp, jacr, int(body))
        com_velocity, angular_velocity = jacp @ data.qvel, jacr @ data.qvel
        points = []
        for index, contact in enumerate(data.contact):
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = model.geom_bodyid[[g1, g2]]
            if not (0 in (b1, b2) and body in (b1, b2)):
                continue
            wrench = np.zeros(6)
            mj.mj_contactForce(model, data, index, wrench)
            normal = contact.frame[:3].copy()
            if b1 == body:
                normal = -normal
            position = contact.pos.copy()
            velocity = com_velocity + np.cross(angular_velocity, position - data.xipos[body])
            lateral_slip = contact_point_lateral_slip_m_s(
                position[None], normal[None], data.xipos[body], com_velocity,
                angular_velocity, data.xquat[body],
            )[0]
            local = rotation.T @ (position - center)
            points.append({
                "geom_pair": [model.geom(g1).name, model.geom(g2).name],
                "position_w_m": position.tolist(),
                "position_wheel_geom_local_m": local.tolist(),
                "signed_contact_distance_m": float(contact.dist),
                "normal_on_wheel_w": normal.tolist(),
                "normal_force_n": float(wrench[0]),
                "tangent_force_contact_n": wrench[1:3].tolist(),
                "point_velocity_w_m_s": velocity.tolist(),
                "lateral_slip_m_s": float(lateral_slip),
                "bearing_above_5n": bool(wrench[0] > 5.),
            })
        wheels.append({
            "raw_manifold_count": len(points),
            "bearing_point_count_above_5n": sum(p["bearing_above_5n"] for p in points),
            "total_normal_force_n": float(sum(p["normal_force_n"] for p in points)),
            "analytic_finite_cylinder_clearance_m": clearance,
            "wheel_geom_center_w_m": center.tolist(),
            "wheel_geom_axis_w": axis.tolist(),
            "axis_dot_ground_normal": axis_normal,
            "points": points,
        })
    return {"time_s": float(data.time), "wheels": wheels}


def compare_rollouts(original, replay):
    with np.load(original, allow_pickle=False) as a, np.load(replay, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise AssertionError(f"Rollout field mismatch: {set(a.files) ^ set(b.files)}")
        differences = []
        for key in a.files:
            x, y = a[key], b[key]
            if not np.array_equal(x, y):
                differences.append(key)
        if differences:
            raise AssertionError(f"Replay differs from archived rollout: {differences}")
        return {"field_count": len(a.files), "physical_samples": int(len(a["time_s"])),
                "policy_samples": int(len(a["actions_raw"])), "all_arrays_exactly_equal": True}


def validate_point_aggregates(original, records):
    with np.load(original, allow_pickle=False) as archive:
        times = np.array([row["time_s"] for row in records])
        raw = np.array([[wheel["raw_manifold_count"] for wheel in row["wheels"]]
                        for row in records])
        bearing = np.array([[wheel["bearing_point_count_above_5n"] for wheel in row["wheels"]]
                            for row in records])
        forces = np.array([[wheel["total_normal_force_n"] for wheel in row["wheels"]]
                           for row in records])
        if not np.array_equal(times, archive["time_s"]):
            raise AssertionError("Raw point timestamps do not match the archive")
        if not np.array_equal(bearing, archive["wheel_ground_contact_count"]):
            raise AssertionError("Raw bearing-point counts do not match the archive")
        expected_force = archive["wheel_ground_contact_force_w_n"][:, :, 2]
        force_error = float(np.max(np.abs(forces - expected_force)))
        if force_error > 1e-9:
            raise AssertionError(f"Raw normal force mismatch: {force_error} N")
        hidden_below_threshold = int(np.count_nonzero((raw > 0) & (bearing == 0)))
        return {"wheel_side_samples": int(raw.size),
                "zero_manifold_samples": int(np.count_nonzero(raw == 0)),
                "raw_points_but_no_bearing_points": hidden_below_threshold,
                "bearing_counts_match_exactly": True,
                "max_normal_force_error_n": force_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-report", type=Path, required=True)
    parser.add_argument("--event-decomposition", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Preserving prior audit: {args.output_dir}")
    prior = json.loads(args.historical_report.read_text())
    events = json.loads(args.event_decomposition.read_text())
    original_rollout = args.historical_report.parent / "rollout.npz"
    if evaluator.sha256(original_rollout) != events["rollout_sha256"]:
        raise ValueError("Event decomposition is not from the supplied historical rollout")
    records = []
    original_method = evaluator.MujocoState.ground_contact_slip

    def instrumented(self, force_threshold_n=5.):
        if force_threshold_n != 5.:
            raise ValueError("This audit expects the unchanged >5 N bearing threshold")
        records.append(point_snapshot(self))
        return original_method(self, force_threshold_n)

    evaluator.MujocoState.ground_contact_slip = instrumented
    replay_dir = args.output_dir / "replay"
    evaluator.run_evaluation(
        evaluator.load_torchscript(prior["policy"]), prior["motion_file"],
        prior["model"], replay_dir, policy_path=prior["policy"],
        contact_threshold_n=5., no_visual_mesh=prior["no_visual_mesh"],
        contract_path=prior["contract_source"],
    )
    comparison = compare_rollouts(original_rollout, replay_dir / "rollout.npz")
    if len(records) != comparison["physical_samples"]:
        raise AssertionError("Point snapshots do not align with physical samples")
    aggregate_validation = validate_point_aggregates(original_rollout, records)
    support_events = [e for e in events["events"] if e["support_failed"]]
    event_windows = []
    for event in support_events:
        index = event["sample_index"]
        failed = [side for side, count in enumerate(event["bearing_contact_count"]) if count == 0]
        window = []
        for row in range(max(0, index - 2), min(len(records), index + 3)):
            entry = records[row]
            window.append({"index": row, "time_s": entry["time_s"],
                           "failed_wheel_side": [entry["wheels"][side] for side in failed],
                           "other_wheel_side": [entry["wheels"][1 - side] for side in failed]})
        event_windows.append({"index": index, "time_s": event["time_s"],
                              "failed_wheel_index": failed, "window": window})
    summary = {
        "kind": "instrumented_MuJoCo_contact_point_replay",
        "historical_rollout_sha256": evaluator.sha256(original_rollout),
        "historical_report_sha256": evaluator.sha256(args.historical_report),
        "event_decomposition_sha256": evaluator.sha256(args.event_decomposition),
        "model_sha256": evaluator.sha256(prior["model"]),
        "actor_sha256": evaluator.sha256(prior["policy"]),
        "contract_sha256": evaluator.sha256(prior["contract_source"]),
        "motion_sha256": evaluator.sha256(prior["motion_file"]),
        "diagnostic_source_sha256": evaluator.sha256(__file__),
        "replay_rollout_sha256": evaluator.sha256(replay_dir / "rollout.npz"),
        "comparison": comparison,
        "aggregate_validation": aggregate_validation,
        "event_count": len(event_windows), "event_windows": event_windows,
        "limitations": [
            "Raw point data are from an exact-array-matched replay, not archived in the historical rollout.",
            "Contact indices have no persistent physical identity across time steps.",
            "No actual contact position or force exists in a zero-manifold sample; the finite-cylinder clearance is an analytic geometric proxy.",
            "The contact solver and discretized cylinder surface may make transitions abrupt; this audit does not identify a unique root cause or prove an Isaac-to-MuJoCo mechanism.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "all_contact_points.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "contact_point_audit.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"exact_replay": True, "samples": len(records), "support_events": len(event_windows)}))


if __name__ == "__main__":
    main()
