"""Genuine torque-driven free-base MuJoCo test; never prescribes root after reset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from sim2sim_common import JOINT_NAMES, ROOT, ReferenceMotion, SharedController, load_config, save_run, stop_reason


def run(reference_path, output, model_path, case="motion"):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    config = load_config()
    reference = ReferenceMotion(reference_path, case=case, config=config)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    if not np.isclose(model.opt.timestep, config["physics_dt"]):
        raise ValueError("Model/control timestep mismatch")
    qids = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
    aids = [model.actuator(name).id for name in JOINT_NAMES]
    wheel_bodies = [model.body(f"wheel_{side}_Link").id for side in ("L", "R")]
    nonwheel = [i for i in range(1, model.nbody) if i not in wheel_bodies]
    base = model.body("base_Link").id
    data.qpos[:7] = reference.initial_qpos[:7]
    data.qpos[qids] = reference.initial_qpos[7:]
    data.qvel[:] = 0.
    mujoco.mj_forward(model, data)
    controller = SharedController(reference, config)
    samples = []
    termination = "completed"
    requested_steps = int(np.ceil(reference.duration / config["physics_dt"]))
    for step in range(requested_steps + 1):
        state = {"joint_pos": data.qpos[qids].copy(), "root_pos": data.xpos[base].copy(),
                 "root_quat_wxyz": data.xquat[base].copy(),
                 "axle_pos": data.xpos[wheel_bodies].mean(axis=0),
                 "body_com": np.average(data.xipos[nonwheel], weights=model.body_mass[nonwheel], axis=0)}
        control = controller.compute(float(data.time), state)
        samples.append({"time_s": float(data.time), **state, **control})
        reason = stop_reason(state, control, config)
        if reason:
            termination = reason
            break
        if step == requested_steps:
            break
        data.ctrl[aids] = control["torque"]
        mujoco.mj_step(model, data)
        # Refresh xpos to the POST-step qpos. No force/pose overwrite occurs.
        mujoco.mj_forward(model, data)
    report = save_run(output, "MuJoCo", reference, config, samples, termination,
                      {"engine_version": mujoco.__version__, "model": str(Path(model_path).resolve()),
                       "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
                       "physics_steps": step, "requested_steps": requested_steps,
                       "initial_qpos": reference.initial_qpos.tolist(), "controller_K": controller.K.tolist(),
                       "self_collisions": False, "root_state_writes": 1})
    print(json.dumps({key: report[key] for key in ("engine", "case", "termination", "simulated_seconds",
                      "leg_tracking_rmse_rad", "max_base_tilt_deg", "base_rise_from_initial_m")}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "outputs/sim2sim-cmu-16_03/mujoco_model.xml")
    parser.add_argument("--case", choices=("standing", "motion"), default="motion")
    args = parser.parse_args()
    run(args.reference, args.output_dir, args.model, args.case)
