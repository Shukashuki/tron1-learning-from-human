"""Four-step (one 20 ms action window) matched-state contact replay.

MuJoCo is reset once per policy-aligned window to Isaac's measured state.
The next four *recorded Isaac joint efforts* are applied at their original
5 ms times, without another reset or actor call. A MuJoCo-own-rollout replay
control verifies the identical reconstruction procedure over four steps.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np

from audit_drift_one_step_action_sensitivity import restored_state
from eval_tracking_mujoco import (DEFAULT_MODEL, MotionReference, MujocoState,
                                  configure_model, load_model, read_contract, sha256)
from training.tron1_drift import configure_mujoco_materials


HORIZON = 4
DT = .005


def restore_isaac_state(model, reference, arrays, index):
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
    measured = state.measure()
    wheel_error = float(np.max(np.abs(measured["wheel_pos"] - arrays["wheel_position_m"][index, 0])))
    twist_error = float(np.max(np.abs(np.r_[measured["root_com_lin_vel_w"], measured["root_ang_vel_w"]] - desired)))
    if wheel_error > .002 or twist_error > 1e-8:
        raise ValueError(f"Initial matched state differs: wheel={wheel_error}, twist={twist_error}")
    return state, wheel_error


def replay_window(model, reference, arrays, index):
    state, initial_error = restore_isaac_state(model, reference, arrays, index)
    results = []
    for k in range(1, HORIZON + 1):
        state.data.ctrl[state.aids] = arrays["applied_joint_torque_nm"][index + k, 0]
        mujoco.mj_step(model, state.data)
        mujoco.mj_forward(model, state.data)
        measured = state.measure()
        ground = state.ground_contact_slip()
        mu_force = ground["wheel_ground_contact_force_w_n"][:, 2]
        ia_force = arrays["wheel_ground_contact_force_w_n"][index + k, 0, :, 2]
        results.append({"substep": k, "time_s": float(arrays["time_s"][index + k]),
                        "isaac_wheel_z_m": arrays["wheel_position_m"][index + k, 0, :, 2].tolist(),
                        "mujoco_wheel_z_m": measured["wheel_pos"][:, 2].tolist(),
                        "isaac_wheel_ground_normal_force_n": ia_force.tolist(),
                        "mujoco_wheel_ground_normal_force_n": mu_force.tolist(),
                        "isaac_both_wheels_supported": bool(np.all(
                            (arrays["wheel_ground_contact_count"][index + k, 0] > 0) & (ia_force >= 5.))),
                        "mujoco_both_wheels_supported": bool(np.all(
                            (ground["wheel_ground_contact_count"] > 0) & (mu_force >= 5.))),
                        "root_position_error_m": float(np.linalg.norm(
                            measured["root_pos"] - arrays["root_position_m"][index + k, 0]))})
    return {"start_time_s": float(arrays["time_s"][index]),
            "initial_wheel_position_max_abs_error_m": initial_error,
            "substeps": results}


def validate_mujoco_self_replay(model, reference, path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    position_error, count_match, steps = 0., 0, 0
    for index in range(0, len(arrays["time_s"]) - HORIZON, HORIZON):
        if not 1.44 <= arrays["time_s"][index] < 4.0:
            continue
        state = restored_state(model, reference, arrays, index)
        for k in range(1, HORIZON + 1):
            state.data.ctrl[state.aids] = arrays["joint_torque_nm"][index + k]
            mujoco.mj_step(model, state.data)
            mujoco.mj_forward(model, state.data)
            position_error = max(position_error, float(np.max(np.abs(
                state.data.qpos - arrays["qpos"][index + k]))))
            ground = state.ground_contact_slip()
            count_match += int(np.array_equal(ground["wheel_ground_contact_count"],
                                              arrays["wheel_ground_contact_count"][index + k]))
            steps += 1
    if steps == 0 or position_error > 1e-8 or count_match != steps:
        raise ValueError(f"Four-step MuJoCo self-replay failed: {steps=}, {position_error=}, {count_match=}")
    return {"steps": steps, "max_qpos_abs_error": position_error,
            "bearing_contact_count_match_fraction": count_match / steps,
            "rollout_sha256": sha256(path)}


def summarize(rows):
    if not rows:
        raise ValueError("No matched policy windows")
    substeps = [sample for row in rows for sample in row["substeps"]]
    ia = np.asarray([sample["isaac_both_wheels_supported"] for sample in substeps])
    mj = np.asarray([sample["mujoco_both_wheels_supported"] for sample in substeps])
    height = np.asarray([np.asarray(sample["mujoco_wheel_z_m"]) - sample["isaac_wheel_z_m"]
                         for sample in substeps]).reshape(len(rows), HORIZON, 2)
    root_error = np.asarray([sample["root_position_error_m"] for sample in substeps]).reshape(len(rows), HORIZON)
    mj_window = mj.reshape(len(rows), HORIZON)
    ia_window = ia.reshape(len(rows), HORIZON)
    return {"policy_windows": len(rows), "native_steps": len(substeps),
            "max_initial_wheel_position_abs_error_m": max(row["initial_wheel_position_max_abs_error_m"] for row in rows),
            "isaac_two_wheel_support_fraction": float(ia.mean()),
            "mujoco_two_wheel_support_fraction": float(mj.mean()),
            "isaac_all_four_steps_supported_window_fraction": float(np.mean(ia_window.all(1))),
            "mujoco_all_four_steps_supported_window_fraction": float(np.mean(mj_window.all(1))),
            "mujoco_two_wheel_support_fraction_by_substep": mj_window.mean(0).tolist(),
            "wheel_z_rmse_m_by_substep_and_side": np.sqrt(np.mean(height**2, axis=0)).tolist(),
            "root_position_rmse_m_by_substep": np.sqrt(np.mean(root_error**2, axis=0)).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-physics", type=Path, required=True)
    parser.add_argument("--isaac-report", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mujoco-baseline-rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserving previous 20 ms contact-window audit")
    contract = read_contract(args.contract)
    report = json.loads(args.isaac_report.read_text())
    profile = copy.deepcopy(contract["task_profile"])
    solref = profile.pop("mujoco_contact_solref", None)
    if (profile != report.get("task_profile")
            or contract.get("reference_file_sha256") != sha256(args.motion_file)
            or report.get("motion_file_sha256") != sha256(args.motion_file)):
        raise ValueError("Isaac report, contract and motion must match except explicit MuJoCo solref")
    model, _ = load_model(args.model, no_visual_mesh=True)
    configure_model(model, contract)
    configure_mujoco_materials(model, contract["task_profile"])
    if model.opt.timestep != DT or contract["decimation"] != HORIZON:
        raise ValueError("Expected original 200/50 Hz physical/control clock")
    reference = MotionReference(args.motion_file, contract)
    with np.load(args.isaac_physics, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if (not np.all(arrays["valid_mask"][:, 0])
            or not np.allclose(np.diff(arrays["time_s"]), DT, atol=1e-9, rtol=0)):
        raise ValueError("Complete aligned Isaac physical episode required")
    rows = [replay_window(model, reference, arrays, index)
            for index in range(0, len(arrays["time_s"]) - HORIZON, HORIZON)
            if 1.44 <= arrays["time_s"][index] < 4.0]
    result = {"kind": "matched_state_20ms_four_substep_contact_response_not_policy_rollout",
              "isaac_physics_sha256": sha256(args.isaac_physics),
              "isaac_report_sha256": sha256(args.isaac_report),
              "contract_sha256": sha256(args.contract), "motion_sha256": sha256(args.motion_file),
              "mujoco_model_sha256": sha256(args.model), "mujoco_contact_solref": solref,
              "mujoco_self_replay_control": validate_mujoco_self_replay(
                  model, reference, args.mujoco_baseline_rollout),
              "summary": summarize(rows), "rows": rows,
              "limitations": ["No actor calls; all four joint torques per window are Isaac's recorded efforts.",
                              "Each 20 ms window is reset independently, so this is not a full closed-loop transfer test.",
                              "A matching short window cannot prove a continuous high-speed sliding turn."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
