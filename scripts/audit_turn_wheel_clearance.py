"""Audit saved high-speed-turn poses against the ACTUAL paired MuJoCo model.

This is analytic cylinder/plane clearance at recorded qpos, not an additional
physics rollout, a contact-force measurement, or a PhysX separation estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cylinder_clearance(center, axis, plane_point, plane_normal, radius, halfwidth):
    dot = np.dot(axis, plane_normal)
    support = radius * np.sqrt(max(0., 1. - dot * dot)) + halfwidth * abs(dot)
    return float(np.dot(center - plane_point, plane_normal) - support)


def audit(report_file, rollout_file, assessment_file):
    import mujoco

    report = json.loads(Path(report_file).read_text())
    assessment = json.loads(Path(assessment_file).read_text())
    model_file = Path(report["model"])
    if digest(model_file) != report["model_sha256"]:
        raise ValueError("Reported paired physical model changed")
    if digest(rollout_file) != assessment["rollout_sha256"]:
        raise ValueError("Physical rollout differs from its frozen assessment")
    model = mujoco.MjModel.from_xml_path(str(model_file))
    data = mujoco.MjData(model)
    wheel_names = ("wheel_L_collision", "wheel_R_collision")
    wheel_geoms = [model.geom(name).id for name in wheel_names]
    wheel_bodies = [model.body(name).id for name in ("wheel_L_Link", "wheel_R_Link")]
    floor = model.geom("floor").id
    if any(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_CYLINDER for g in wheel_geoms):
        raise ValueError("Expected both finite-cylinder wheel collisions")
    with np.load(rollout_file, allow_pickle=False) as source:
        time, qpos = source["time_s"], source["qpos"]
        wheel_pos = source["wheel_pos"]
        count = source["wheel_ground_contact_count"]
        force = source["wheel_ground_contact_force_w_n"]
    if (qpos.shape != (len(time), model.nq) or wheel_pos.shape != (len(time), 2, 3)
            or count.shape != (len(time), 2) or force.shape != (len(time), 2, 3)):
        raise ValueError("Invalid physical pose/contact array shape")
    start, stop = assessment["reference_core_time_s"]
    core = (time >= start - 1e-8) & (time <= stop + 1e-8)
    if not core.any():
        raise ValueError("No recorded physical turn-core samples")
    clearance = np.empty((int(core.sum()), 2))
    pose_error = 0.
    for j, index in enumerate(np.flatnonzero(core)):
        data.qpos[:] = qpos[index]
        mujoco.mj_kinematics(model, data)
        pose_error = max(pose_error, float(np.abs(data.xpos[wheel_bodies] - wheel_pos[index]).max()))
        normal = data.geom_xmat[floor].reshape(3, 3)[:, 2]
        if not np.allclose(normal, [0., 0., 1.], atol=1e-12):
            raise ValueError("Expected horizontal recorded ground plane")
        for side, geom in enumerate(wheel_geoms):
            axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
            clearance[j, side] = cylinder_clearance(
                data.geom_xpos[geom], axis, data.geom_xpos[floor], normal,
                *model.geom_size[geom, :2],
            )
    if pose_error > 1e-9:
        raise ValueError("Recorded wheel poses differ from model FK")
    missing = count[core] == 0
    weak = (count[core] == 0) | (np.linalg.norm(force[core], axis=-1) < 5.)
    sides = []
    for side in range(2):
        selected = clearance[missing[:, side], side]
        sides.append({
            "wheel": wheel_names[side],
            "raw_zero_contact_samples": int(missing[:, side].sum()),
            "weak_below_5n_samples": int(weak[:, side].sum()),
            "fraction_positive_analytic_clearance_when_raw_zero": float(np.mean(selected > 0)) if len(selected) else None,
            "missing_contact_clearance_quantiles_mm": (
                (1000 * np.quantile(selected, [.05, .5, .95])).tolist() if len(selected) else None
            ),
            "radius_m": float(model.geom_size[wheel_geoms[side], 0]),
            "halfwidth_m": float(model.geom_size[wheel_geoms[side], 1]),
            "compiled_contact_margin_m": float(model.geom_margin[wheel_geoms[side]]),
        })
    return {
        "kind": "recorded_pose_actual_paired_model_analytic_clearance",
        "model_sha256": report["model_sha256"],
        "rollout_sha256": digest(rollout_file),
        "assessment_sha256": digest(assessment_file),
        "core_samples": int(core.sum()),
        "maximum_recorded_wheel_pose_fk_error_m": pose_error,
        "floor_compiled_contact_margin_m": float(model.geom_margin[floor]),
        "wheels": sides,
        "new_physics_steps": 0,
        "limitations": "Cylinder/plane support is a saved-pose geometric proxy, not solved contact, native PhysX clearance, or a unique-cause diagnosis.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "rollout", "assessment", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserving previous clearance audit")
    result = audit(args.report, args.rollout, args.assessment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"model_sha256": result["model_sha256"],
                      "core_samples": result["core_samples"],
                      "maximum_recorded_wheel_pose_fk_error_m": result["maximum_recorded_wheel_pose_fk_error_m"]}))


if __name__ == "__main__":
    main()
