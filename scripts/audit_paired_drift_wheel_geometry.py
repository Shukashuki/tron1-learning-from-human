"""Project paired 200 Hz Isaac/MuJoCo poses into a common wheel geometry.

The common MJCF finite-cylinder clearance is a geometric counterfactual for
Isaac poses, NOT a measurement of PhysX collision separation or contact force.
No dynamics, actor inference, force solve or gate changes occur in this audit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

import eval_tracking_mujoco as evaluator
from training.tron1_drift import configure_mujoco_materials


def projected_geometry(model, data, qpos, joint_ids=None):
    if joint_ids is None:
        data.qpos[:] = qpos
    else:
        data.qpos[:7] = qpos[:7]
        data.qpos[joint_ids] = qpos[7:]
    data.qvel[:] = 0.
    mujoco.mj_fwdPosition(model, data)
    floor = model.geom("floor").id
    normal = data.geom_xmat[floor].reshape(3, 3)[:, 2]
    point = data.geom_xpos[floor]
    result = []
    for side in ("L", "R"):
        geom = model.geom(f"wheel_{side}_collision").id
        body = model.body(f"wheel_{side}_Link").id
        center = data.geom_xpos[geom]
        axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
        radius, halfwidth = model.geom_size[geom, :2]
        u = float(axis @ normal)
        support = radius * np.sqrt(max(0., 1. - u*u)) + halfwidth * abs(u)
        result.append((float(normal @ (center - point) - support),
                       float(data.xpos[body, 2]), float(np.degrees(np.arcsin(np.clip(u, -1., 1.))))))
    return np.asarray(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-dir", type=Path, required=True)
    parser.add_argument("--mujoco-dir", type=Path, required=True)
    parser.add_argument("--event-decomposition", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Preserving prior audit: {args.output_dir}")
    ir = json.loads((args.isaac_dir / "report.json").read_text())
    mr = json.loads((args.mujoco_dir / "report.json").read_text())
    events = json.loads(args.event_decomposition.read_text())
    isaac_path = args.isaac_dir / "physics_trajectory.npz"
    mujoco_path = args.mujoco_dir / "rollout.npz"
    if (events["rollout_sha256"] != evaluator.sha256(mujoco_path)
            or ir["physics_trajectory_sha256"] != evaluator.sha256(isaac_path)
            or ir["physics_dt_s"] != .005 or mr["contract"]["physics_dt"] != .005):
        raise ValueError("Expected hash-pinned, paired 200 Hz physical archives")
    if mr["model_sha256"] != evaluator.sha256(mr["model"]):
        raise ValueError("Physical MJCF bytes changed")
    model, _ = evaluator.load_model(mr["model"], mr["no_visual_mesh"],
                                    terrain=mr["contract"].get("terrain"))
    evaluator.configure_model(model, mr["contract"])
    configure_mujoco_materials(model, mr["contract"]["task_profile"])
    data = mujoco.MjData(model)
    with np.load(isaac_path, allow_pickle=False) as a, np.load(mujoco_path, allow_pickle=False) as b:
        t = np.asarray(b["time_s"])
        if (len(t) != 1153 or len(a["time_s"]) != len(t)
                or not np.allclose(a["time_s"], t, rtol=0, atol=1e-9)
                or not np.all(a["valid_mask"])
                or not np.array_equal(a["reference_frame"][:, 0], b["reference_frame"])):
            raise ValueError("Physical telemetry clocks, validity or reference phases disagree")
        names = a["qpos_joint_names"].astype(str).tolist()
        if set(names) != set(b["qpos_joint_names"].astype(str).tolist()) or len(names) != 8:
            raise ValueError("Named generalized-coordinate layouts differ")
        qids = [int(model.joint(name).qposadr[0]) for name in names]
        if sorted(qids) != list(range(7, 15)):
            raise ValueError("Unexpected MJCF free-root/joint coordinate layout")
        projected = np.empty((2, len(t), 2, 3), float)
        for i in range(len(t)):
            projected[0, i] = projected_geometry(model, data, a["qpos"][i, 0], qids)
            projected[1, i] = projected_geometry(model, data, b["qpos"][i])
        isaac_wheel_z = a["wheel_position_m"][:, 0, :, 2]
        mujoco_wheel_z = b["wheel_pos"][:, :, 2]
        isaac_z_error = float(np.max(np.abs(projected[0, :, :, 1] - isaac_wheel_z)))
        mujoco_z_error = float(np.max(np.abs(projected[1, :, :, 1] - mujoco_wheel_z)))
        if isaac_z_error > 5e-4 or mujoco_z_error > 1e-9:
            raise AssertionError("Common FK does not reproduce archived wheel-link heights")
        rows = []
        for event in events["events"]:
            if not event["support_failed"]:
                continue
            i = event["sample_index"]
            failed = [side for side, count in enumerate(event["bearing_contact_count"]) if count == 0]
            for side in failed:
                rows.append({
                    "index": i, "time_s": float(t[i]), "wheel": ("L", "R")[side],
                    "isaac_bearing_count": int(a["wheel_ground_contact_count"][i, 0, side]),
                    "mujoco_bearing_count": int(b["wheel_ground_contact_count"][i, side]),
                    "isaac_normal_force_n": float(a["wheel_ground_contact_force_w_n"][i, 0, side, 2]),
                    "mujoco_normal_force_n": float(b["wheel_ground_contact_force_w_n"][i, side, 2]),
                    "isaac_common_mjcf_clearance_um": float(projected[0, i, side, 0] * 1e6),
                    "mujoco_common_mjcf_clearance_um": float(projected[1, i, side, 0] * 1e6),
                    "isaac_wheel_link_z_m": float(isaac_wheel_z[i, side]),
                    "mujoco_wheel_link_z_m": float(mujoco_wheel_z[i, side]),
                    "wheel_link_z_mujoco_minus_isaac_mm": float((mujoco_wheel_z[i, side] - isaac_wheel_z[i, side]) * 1e3),
                    "root_z_mujoco_minus_isaac_mm": float((b["root_pos"][i, 2] - a["root_position_m"][i, 0, 2]) * 1e3),
                    "isaac_common_mjcf_camber_deg": float(projected[0, i, side, 2]),
                    "mujoco_common_mjcf_camber_deg": float(projected[1, i, side, 2]),
                })
    summary = {
        "kind": "paired_saved_pose_common_MJCF_wheel_geometry_not_native_PhysX_collision",
        "isaac_archive_sha256": evaluator.sha256(isaac_path),
        "mujoco_archive_sha256": evaluator.sha256(mujoco_path),
        "event_decomposition_sha256": evaluator.sha256(args.event_decomposition),
        "model_sha256": evaluator.sha256(mr["model"]),
        "diagnostic_source_sha256": evaluator.sha256(__file__),
        "physical_samples": len(t),
        "isaac_wheel_link_z_fk_max_abs_error_m": isaac_z_error,
        "mujoco_wheel_link_z_fk_max_abs_error_m": mujoco_z_error,
        "moving_drift_support_failure_rows": rows,
        "limitations": [
            "Applying MJCF geometry to Isaac qpos is a common-frame geometric proxy, not PhysX contact separation.",
            "Isaac contact counts/forces are actual archived native PhysX telemetry and are not derived from the proxy.",
            "A pose/clearance difference at a contact gap does not by itself identify solver, geometry or controller causation.",
            "Only saved poses are evaluated; no actor, force solve, physics step, gap fill or new paired behavior assessment occurs.",
        ],
    }
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir / "projected_geometry.npz", time_s=t, projected=projected)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"events": len(rows), "isaac_fk_error_m": isaac_z_error,
                      "mujoco_fk_error_m": mujoco_z_error}))


if __name__ == "__main__":
    main()
