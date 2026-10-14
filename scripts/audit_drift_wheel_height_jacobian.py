"""Read-only kinematic wheel-height response to hip/knee joint targets.

This identifies leg-action signs for a prospective physical-rate feedback
experiment. It does not simulate torques, contact forces, or policy behavior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from eval_tracking_mujoco import (MotionReference, MujocoState, configure_model,
                                  load_model, read_contract, sha256)
from training.tron1_drift import configure_mujoco_materials


def audit(rollout, motion, model_file, contract_file):
    contract = read_contract(contract_file)
    reference = MotionReference(motion, contract)
    model, _ = load_model(model_file, no_visual_mesh=True)
    configure_model(model, contract)
    configure_mujoco_materials(model, contract["task_profile"])
    data = mujoco.MjData(model)
    state = MujocoState(model, data, reference)
    with np.load(rollout, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    selected = np.flatnonzero((arrays["time_s"] >= 1.44 - 1e-8)
                              & (arrays["time_s"] <= 4.0 + 1e-8))
    pairs = (("L", "hip_L_Joint", "knee_L_Joint", state.wheels[0]),
             ("R", "hip_R_Joint", "knee_R_Joint", state.wheels[1]))
    values = {side: [] for side, *_ in pairs}
    for index in selected:
        data.qpos[:] = arrays["qpos"][index]
        mujoco.mj_forward(model, data)
        for side, hip, knee, wheel in pairs:
            jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, int(wheel))
            values[side].append([float(jacp[2, model.joint(name).dofadr[0]])
                                 for name in (hip, knee)])
    return {"physical_samples": len(selected),
            "d_wheel_link_height_d_hip_knee_m_per_rad_p05_p50_p95": {
                side: np.quantile(np.asarray(rows), [.05, .5, .95], axis=0).tolist()
                for side, rows in values.items()},
            "rollout_sha256": sha256(rollout), "motion_sha256": sha256(motion),
            "model_sha256": sha256(model_file), "contract_sha256": sha256(contract_file),
            "source_sha256": sha256(__file__),
            "limitations": ["Root-fixed kinematic Jacobian only; closed-loop contact and slip require a new simulation.",
                            "Wheel-link origin height is not exact finite-cylinder clearance."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rollout", "motion-file", "model", "contract", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserving previous wheel-height Jacobian audit")
    result = audit(args.rollout, args.motion_file, args.model, args.contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
