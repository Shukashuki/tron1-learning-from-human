"""Replay each Isaac turn-core state for exactly one MuJoCo physics step.

The root pose, COM twist, joint positions/velocities and the *recorded next-step
Isaac joint effort* are supplied to MuJoCo. No policy is called, no trajectory
is spliced, and no success gate is regraded. This distinguishes immediate
contact-model response from long closed-loop policy divergence.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np

from audit_drift_one_step_action_sensitivity import restored_state, signed_clearance
from eval_tracking_mujoco import (DEFAULT_MODEL, MotionReference, MujocoState,
                                  configure_model, load_model, read_contract, sha256)
from training.tron1_drift import configure_mujoco_materials


JOINT_NAMES = ("abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint",
               "knee_L_Joint", "knee_R_Joint", "wheel_L_Joint", "wheel_R_Joint")


def replay_step(model, reference, arrays, index):
    """Reset to one observed PhysX state; inject its next applied torque."""
    data = mujoco.MjData(model)
    state = MujocoState(model, data, reference)
    data.qpos[state.root_qid:state.root_qid + 7] = np.r_[
        arrays["root_position_m"][index, 0], arrays["root_quaternion_wxyz"][index, 0]]
    data.qpos[state.qids] = arrays["joint_position_rad"][index, 0]
    data.qvel[:] = 0.
    data.qvel[state.vids] = arrays["joint_velocity_rad_s"][index, 0]
    mujoco.mj_forward(model, data)
    jacp, jacr = state.root_jacobians()
    jac = np.vstack((jacp, jacr))
    root_columns = slice(state.root_vid, state.root_vid + 6)
    desired = np.r_[arrays["root_linear_velocity_w_m_s"][index, 0],
                    arrays["root_angular_velocity_w_rad_s"][index, 0]]
    data.qvel[root_columns] = np.linalg.solve(jac[:, root_columns], desired - jac @ data.qvel)
    mujoco.mj_forward(model, data)
    before = state.measure()
    initial_ground = state.ground_contact_slip()
    initial_clearance = [signed_clearance(state, wheel) for wheel in range(2)]
    initial_wheel_error = before["wheel_pos"] - arrays["wheel_position_m"][index, 0]
    initial_twist_error = np.r_[before["root_com_lin_vel_w"], before["root_ang_vel_w"]] - desired
    torque = arrays["applied_joint_torque_nm"][index + 1, 0]
    data.ctrl[state.aids] = torque
    mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    after = state.measure()
    ground = state.ground_contact_slip()
    force = ground["wheel_ground_contact_force_w_n"][:, 2]
    return {"time_s": float(arrays["time_s"][index + 1]),
            "initial_wheel_position_max_abs_error_m": float(np.max(np.abs(initial_wheel_error))),
            "initial_twist_max_abs_error": float(np.max(np.abs(initial_twist_error))),
            "initial_mujoco_signed_wheel_clearance_m": initial_clearance,
            "initial_mujoco_bearing_count": initial_ground["wheel_ground_contact_count"].tolist(),
            "same_applied_joint_torque_nm": torque.tolist(),
            "isaac_next_wheel_z_m": arrays["wheel_position_m"][index + 1, 0, :, 2].tolist(),
            "mujoco_next_wheel_z_m": after["wheel_pos"][:, 2].tolist(),
            "isaac_next_ground_normal_force_n": arrays["wheel_ground_contact_force_w_n"][index + 1, 0, :, 2].tolist(),
            "mujoco_next_ground_normal_force_n": force.tolist(),
            "isaac_next_bearing_count": arrays["wheel_ground_contact_count"][index + 1, 0].tolist(),
            "mujoco_next_bearing_count": ground["wheel_ground_contact_count"].tolist()}


def summarize(rows):
    if not rows:
        raise ValueError("No matched turn-core steps")
    first_error = max(row["initial_wheel_position_max_abs_error_m"] for row in rows)
    twist_error = max(row["initial_twist_max_abs_error"] for row in rows)
    if first_error > .002 or twist_error > 1e-8:
        raise ValueError(f"Initial cross-engine state mismatch: wheel={first_error}, twist={twist_error}")
    ia = np.asarray([row["isaac_next_ground_normal_force_n"] for row in rows])
    mj = np.asarray([row["mujoco_next_ground_normal_force_n"] for row in rows])
    ia_count = np.asarray([row["isaac_next_bearing_count"] for row in rows])
    mj_count = np.asarray([row["mujoco_next_bearing_count"] for row in rows])
    ia_support = ((ia_count > 0) & (ia >= 5.)).all(axis=1)
    mj_support = ((mj_count > 0) & (mj >= 5.)).all(axis=1)
    zi = np.asarray([row["isaac_next_wheel_z_m"] for row in rows])
    zm = np.asarray([row["mujoco_next_wheel_z_m"] for row in rows])
    clearance = np.asarray([row["initial_mujoco_signed_wheel_clearance_m"] for row in rows])
    initial_count = np.asarray([row["initial_mujoco_bearing_count"] for row in rows])
    return {"steps": len(rows), "initial_wheel_position_max_abs_error_m": first_error,
            "initial_twist_max_abs_error": twist_error,
            "initial_mujoco_both_wheels_bearing_fraction": float(np.mean((initial_count > 0).all(axis=1))),
            "initial_mujoco_signed_clearance_p05_p50_p95_m": np.quantile(clearance, [.05, .5, .95], axis=0).tolist(),
            "isaac_both_wheels_supported_fraction": float(ia_support.mean()),
            "mujoco_both_wheels_supported_fraction_from_same_isaac_state_and_torque": float(mj_support.mean()),
            "fraction_isaac_supported_but_mujoco_not": float(np.mean(ia_support & ~mj_support)),
            "wheel_center_next_step_z_rmse_m": np.sqrt(np.mean((zi - zm)**2, axis=0)).tolist(),
            "isaac_ground_normal_force_p95_n": np.quantile(ia, .95, axis=0).tolist(),
            "mujoco_ground_normal_force_p95_n": np.quantile(mj, .95, axis=0).tolist()}


def validate_same_model_replay(model, reference, path):
    """Check that fresh MuJoCo MjData can reproduce its own archived 5 ms steps."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    errors, support_matches = [], []
    for index, time in enumerate(arrays["time_s"][:-1]):
        if not 1.44 <= time < 4.0:
            continue
        state = restored_state(model, reference, arrays, index)
        data = state.data
        data.ctrl[state.aids] = arrays["joint_torque_nm"][index + 1]
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        errors.append(float(np.max(np.abs(data.qpos - arrays["qpos"][index + 1]))))
        observed = state.ground_contact_slip()
        recorded = arrays["wheel_ground_contact_count"][index + 1]
        support_matches.append(bool(np.array_equal(observed["wheel_ground_contact_count"], recorded)))
    if not errors:
        raise ValueError("No archived MuJoCo control steps in the turn core")
    result = {"steps": len(errors), "max_next_qpos_abs_error": max(errors),
              "bearing_contact_count_match_fraction": float(np.mean(support_matches)),
              "rollout_sha256": sha256(path)}
    if result["max_next_qpos_abs_error"] > 1e-8 or result["bearing_contact_count_match_fraction"] < 1.:
        raise ValueError(f"Fresh MuJoCo state did not reproduce its own recorded next step: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-physics", type=Path, required=True)
    parser.add_argument("--isaac-report", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mujoco-baseline-rollout", type=Path, required=True,
                        help="Same actor's nominal MuJoCo rollout, used only to validate one-step fresh-state replay")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserving existing one-step audit")
    contract = read_contract(args.contract)
    report = json.loads(args.isaac_report.read_text())
    task_profile = copy.deepcopy(contract.get("task_profile"))
    if task_profile is None:
        raise ValueError("Saved drift task profile required")
    diagnostic_solref = task_profile.pop("mujoco_contact_solref", None)
    if (contract.get("reference_file_sha256") != sha256(args.motion_file)
            or report.get("motion_file_sha256") != sha256(args.motion_file)
            or report.get("task_profile") != task_profile):
        raise ValueError("Saved Isaac report, contract, and motion must match except explicit MuJoCo solref")
    reference = MotionReference(args.motion_file, contract)
    model, _ = load_model(args.model, no_visual_mesh=True)
    configure_model(model, contract)
    configure_mujoco_materials(model, contract["task_profile"])
    if not np.isclose(model.opt.timestep, .005) or contract["decimation"] != 4:
        raise ValueError("Exact 200/50 Hz physical and policy clocks required")
    with np.load(args.isaac_physics, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if (list(arrays["joint_names"].astype(str)) != list(JOINT_NAMES)
            or list(arrays["qpos_joint_names"].astype(str)) != list(JOINT_NAMES)
            or arrays["valid_mask"].shape != (len(arrays["time_s"]), 1)
            or not np.all(arrays["valid_mask"][:, 0])
            or not np.allclose(np.diff(arrays["time_s"]), .005, rtol=0, atol=1e-9)):
        raise ValueError("Aligned, complete, named-joint Isaac physical episode required")
    rows = []
    for index, time in enumerate(arrays["time_s"][:-1]):
        if 1.44 <= time < 4.0:
            rows.append(replay_step(model, reference, arrays, index))
    result = {"kind": "matched_isaac_state_and_next_applied_torque_one_mujoco_step_diagnostic",
              "scope": "Each 5 ms step resets MuJoCo to its recorded Isaac predecessor; no actor calls or trajectory splicing",
              "isaac_physics_sha256": sha256(args.isaac_physics),
              "isaac_report_sha256": sha256(args.isaac_report),
              "policy_contract_sha256": sha256(args.contract),
              "motion_sha256": sha256(args.motion_file), "mujoco_input_model_sha256": sha256(args.model),
              "evaluation_only_mujoco_contact_solref": diagnostic_solref,
              "summary": summarize(rows),
              "same_model_replay_control": validate_same_model_replay(
                  model, reference, args.mujoco_baseline_rollout),
              "rows": rows,
              "limitations": ["This is not a closed-loop MuJoCo policy rollout or a sim2sim success test.",
                              "Matching generalized state and joint effort cannot make the two solvers identical.",
                              "The injected torque is Isaac's recorded next-step effort, not MuJoCo's DC motor calculation."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
