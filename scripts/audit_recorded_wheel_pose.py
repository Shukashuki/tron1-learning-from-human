"""Recorded MuJoCo wheel tilt/plane clearance; FK only, never forces or rollout.

Use the unchanged compiled physical model and recorded native qpos ordering.
Cylinder support against the existing horizontal plane is analytic. Archived
forces/slip remain separate. No mj_step, force solve, actor, reset or fitting.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

MODEL_SHA = "bd92264b72b4d09084732888dbd81fd94a64a1678978e58a2ab0dd191887f513"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cylinder_bottom_z(center_z, axial_z, radius, half_width):
    a = np.asarray(axial_z, float)
    center = np.asarray(center_z, float)
    require(np.isfinite(center).all() and np.isfinite(a).all() and np.all(np.abs(a) <= 1. + 1e-10)
            and np.isfinite([radius, half_width]).all() and radius > 0 and half_width > 0,
            "Finite unit cylinder axis and positive geometry")
    a = np.clip(a, -1., 1.)
    return center - radius * np.sqrt(1. - a * a) - half_width * np.abs(a)


def audit(rollout, model_file, breaks_file):
    require(sha(model_file) == MODEL_SHA, "Unchanged original physical model")
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(model_file.resolve()))
    data = mujoco.MjData(model)
    with np.load(rollout, allow_pickle=False) as archive:
        a = {k: archive[k].copy() for k in archive.files}
    qpos, wheel_pos = a["qpos"], a["wheel_pos"]
    require(qpos.shape == (1153, model.nq) and wheel_pos.shape == (1153, 2, 3)
            and np.isfinite(qpos).all(), "Complete recorded physical poses")
    require(a["qpos_joint_names"].tolist() == [model.joint(i).name for i in range(1, model.njnt)], "Original native joint ordering")
    events = json.loads(breaks_file.read_text())["events"]
    wheels = [model.body("wheel_" + side + "_Link").id for side in ("L", "R")]
    geoms = [model.geom("wheel_" + side + "_collision").id for side in ("L", "R")]
    require(all(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER for g in geoms), "Original two cylinders")
    plane = next(i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE)
    clear, tilt, errors = [], [], []
    for i in range(1153):
        data.qpos[:] = qpos[i]
        mujoco.mj_kinematics(model, data)  # poses only; no collision/constraint solve
        require(np.allclose(data.geom_xmat[plane].reshape(3, 3)[:, 2], [0., 0., 1.], atol=1e-12), "Same horizontal ground")
        errors.append(float(np.abs(data.xpos[wheels] - wheel_pos[i]).max()))
        bottom = [cylinder_bottom_z(data.geom_xpos[g, 2], data.geom_xmat[g].reshape(3, 3)[2, 2],
                                   model.geom_size[g, 0], model.geom_size[g, 1]) - data.geom_xpos[plane, 2] for g in geoms]
        clear.append(bottom)
        tilt.append(np.degrees(np.arcsin(np.clip([data.xmat[b].reshape(3, 3)[2, 1] for b in wheels], -1., 1.))))
    require(max(errors) <= 1e-10, "FK differs from actual recorded wheel-link poses")
    clear, tilt = np.asarray(clear), np.asarray(tilt)
    result_events = []
    for e in events:
        i = e["sample_index"]
        require(abs(float(a["time_s"][i]) - e["time_s"]) <= 1e-12, "Original event clock")
        result_events.append({**e, "analytic_cylinder_plane_clearance_um": (clear[i] * 1e6).tolist(),
                              "wheel_link_axle_tilt_deg": tilt[i].tolist()})
    return {"schema_version": 1, "status": "recorded_pose_FK_verified", "source_sha256": sha(__file__),
            "rollout_sha256": sha(rollout), "model_sha256": MODEL_SHA, "breaks_sha256": sha(breaks_file),
            "recorded_physical_poses": 1153, "maximum_FK_wheel_position_error_m": max(errors),
            "events": result_events, "wheel_axle_tilt_peak_abs_deg": np.max(np.abs(tilt), axis=0).tolist(),
            "new_physics_steps": 0, "new_actor_calls": 0, "force_solves": 0,
            "limitations": ["Analytic cylinder/horizontal-plane support is geometric, not a solved contact force.",
                            "Only MuJoCo recorded FK is reconstructed; PhysX manifolds are not inferred.",
                            "No original drift gate, force, slip or interval is recomputed from geometry."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rollout", "model", "breaks", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "New diagnostic only; never overwrite")
    result = audit(args.rollout, args.model, args.breaks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": result["status"], "FK_error_m": result["maximum_FK_wheel_position_error_m"], "new_physics_steps": 0}))


if __name__ == "__main__": main()
