"""Diagnostic same-actor MuJoCo turn continuation from an exact Isaac state.

The 0 s reseed is a full-episode parity control. Later reseeds start from the
recorded Isaac physical state, then let the same actor run MuJoCo closed-loop.
They are not full-reference policy-transfer episodes or assessor PASS results.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np

from audit_drift_matched_contact_window import DT, HORIZON, restore_isaac_state
from eval_tracking_mujoco import (MotionReference, action_torques, build_observation,
                                  configure_model, load_model, load_torchscript,
                                  read_contract, sha256)
from training.tron1_drift import configure_mujoco_materials


def closed_loop_segment(model, reference, contract, isaac, policy, start_frame, end_frame, baseline=None):
    start = start_frame * HORIZON
    state, wheel_error = restore_isaac_state(model, reference, isaac, start)
    state.data.time = float(isaac["time_s"][start])
    previous = (np.zeros(8) if start == 0 else isaac["actions_clipped"][start, 0].copy())
    rows = []
    first_observation_error = None
    first_action_error = None
    parity = {"max_qpos_abs_error": 0., "max_joint_torque_abs_error_nm": 0.,
              "max_actor_raw_action_abs_error": 0., "contact_count_matches": 0,
              "max_ground_normal_force_abs_error_n": 0.} if baseline is not None else None
    for frame in range(start_frame, end_frame):
        measured = state.measure()
        observation = build_observation(measured, reference.sample(frame), reference.q0, previous)
        action_raw = np.asarray(policy(observation), dtype=float)
        if parity is not None:
            parity["max_actor_raw_action_abs_error"] = max(
                parity["max_actor_raw_action_abs_error"],
                float(np.max(np.abs(action_raw - baseline["actions_raw"][frame]))))
        if frame == start_frame:
            first_observation_error = float(np.max(np.abs(
                observation - isaac["action_observation"][start + 1, 0])))
            first_action_error = float(np.max(np.abs(
                action_raw - isaac["actions_raw"][start + 1, 0])))
            if first_observation_error > 2e-4 or first_action_error > 2e-3:
                raise ValueError("Reseeded actor input/action does not match Isaac at start")
        action = np.clip(action_raw, *contract["raw_action_clip"])
        for k in range(1, HORIZON + 1):
            torque = action_torques(action, measured["joint_pos"], measured["joint_vel"],
                                    reference.q0, contract)
            state.data.ctrl[state.aids] = torque
            mujoco.mj_step(model, state.data)
            mujoco.mj_forward(model, state.data)
            measured = state.measure()
            ground = state.ground_contact_slip()
            index = frame * HORIZON + k
            normal = ground["wheel_ground_contact_force_w_n"][:, 2]
            if parity is not None:
                parity["max_qpos_abs_error"] = max(parity["max_qpos_abs_error"], float(np.max(np.abs(
                    state.data.qpos - baseline["qpos"][index]))))
                parity["max_joint_torque_abs_error_nm"] = max(
                    parity["max_joint_torque_abs_error_nm"], float(np.max(np.abs(
                        torque - baseline["joint_torque_nm"][index]))))
                parity["max_ground_normal_force_abs_error_n"] = max(
                    parity["max_ground_normal_force_abs_error_n"], float(np.max(np.abs(
                        normal - baseline["wheel_ground_contact_force_w_n"][index, :, 2]))))
                parity["contact_count_matches"] += int(np.array_equal(
                    ground["wheel_ground_contact_count"],
                    baseline["wheel_ground_contact_count"][index]))
            rows.append({"time_s": float(state.data.time),
                         "both_wheels_supported": bool(np.all(
                             (ground["wheel_ground_contact_count"] > 0) & (normal >= 5.))),
                         "wheel_ground_normal_force_n": normal.tolist(),
                         "root_position_error_m": float(np.linalg.norm(
                             measured["root_pos"] - isaac["root_position_m"][index, 0])),
                         "max_wheel_position_error_m": float(np.max(np.linalg.norm(
                             measured["wheel_pos"] - isaac["wheel_position_m"][index, 0], axis=1))),
                         "max_leg_joint_position_error_rad": float(np.max(np.abs(
                             measured["joint_pos"][:6] - isaac["joint_position_rad"][index, 0, :6]))),
                         "max_joint_torque_error_nm": float(np.max(np.abs(
                             torque - isaac["applied_joint_torque_nm"][index, 0])))})
        previous = action.copy()
    support = np.asarray([row["both_wheels_supported"] for row in rows])
    gap_onsets = np.flatnonzero(~support & np.r_[True, support[:-1]])
    supported_runs = np.diff(np.flatnonzero(np.r_[True, np.diff(support.astype(int)) != 0, True]))
    support_runs = supported_runs[support[np.flatnonzero(np.r_[True, np.diff(support.astype(int)) != 0])]]
    if parity is not None:
        parity["contact_count_match_fraction"] = parity.pop("contact_count_matches") / len(rows)
    return {"start_frame": start_frame, "start_time_s": float(isaac["time_s"][start]),
            "end_time_s": float(isaac["time_s"][end_frame * HORIZON]),
            "initial_wheel_position_error_m": wheel_error,
            "initial_actor_observation_max_abs_error": first_observation_error,
            "initial_actor_raw_action_max_abs_error": first_action_error,
            "native_steps": len(rows), "both_wheel_support_fraction": float(support.mean()),
            "gap_onsets": int(len(gap_onsets)),
            "first_gap_time_s": None if len(gap_onsets) == 0 else rows[int(gap_onsets[0])]["time_s"],
            "longest_supported_duration_s": float(max(support_runs, default=0) * DT),
            "final_root_position_error_m": rows[-1]["root_position_error_m"],
            "final_max_wheel_position_error_m": rows[-1]["max_wheel_position_error_m"],
            "zero_time_baseline_parity": parity,
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-physics", type=Path, required=True)
    parser.add_argument("--isaac-report", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--baseline-rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserving previous turn-core reseed audit")
    contract = read_contract(args.contract)
    report = json.loads(args.isaac_report.read_text())
    profile = copy.deepcopy(contract["task_profile"])
    profile.pop("mujoco_contact_solref", None)
    if (profile != report["task_profile"]
            or contract["reference_file_sha256"] != sha256(args.motion_file)
            or report["motion_file_sha256"] != sha256(args.motion_file)):
        raise ValueError("Frozen Isaac report, contract, and reference disagree")
    model, _ = load_model(args.model, no_visual_mesh=True)
    configure_model(model, contract)
    configure_mujoco_materials(model, contract["task_profile"])
    if model.opt.timestep != DT or contract["decimation"] != HORIZON:
        raise ValueError("Expected original 200/50 Hz clock")
    reference = MotionReference(args.motion_file, contract)
    with np.load(args.isaac_physics, allow_pickle=False) as archive:
        isaac = {key: archive[key].copy() for key in archive.files}
    with np.load(args.baseline_rollout, allow_pickle=False) as archive:
        baseline = {key: archive[key].copy() for key in archive.files}
    if (not np.all(isaac["valid_mask"][:, 0])
            or not np.allclose(isaac["time_s"], baseline["time_s"], atol=1e-9, rtol=0)):
        raise ValueError("Complete aligned Isaac and MuJoCo episodes required")
    end_frame = 200  # 4.00 s, same frozen turn-core endpoint
    runs = []
    for start_frame in (0, 72, 100, 150):  # 0, 1.44, 2.00, 3.00 s
        policy = load_torchscript(args.policy)
        run = closed_loop_segment(model, reference, contract, isaac, policy, start_frame, end_frame,
                                  baseline if start_frame == 0 else None)
        if start_frame == 0:
            baseline_support = np.all(
                (baseline["wheel_ground_contact_count"][1:end_frame * HORIZON + 1] > 0)
                & (baseline["wheel_ground_contact_force_w_n"][1:end_frame * HORIZON + 1, :, 2] >= 5.), axis=1)
            baseline_fraction = float(baseline_support.mean())
            if abs(run["both_wheel_support_fraction"] - baseline_fraction) > .005:
                raise ValueError("0 s reseed did not reproduce baseline support fraction")
            run["baseline_full_rollout_support_fraction_to_4s"] = baseline_fraction
        runs.append(run)
        print({key: value for key, value in run.items() if key != "rows"}, flush=True)
    result = {"kind": "same_actor_closed_loop_turn_continuation_from_exact_isaac_state_not_full_transfer",
              "input_sha256": {str(path): sha256(path) for path in
                               (args.isaac_physics, args.isaac_report, args.contract,
                                args.motion_file, args.model, args.policy, args.baseline_rollout)},
              "clock_s": DT, "decimation": HORIZON, "turn_endpoint_s": 4.0,
              "runs": runs,
              "limitations": ["The later runs bypass earlier trajectory history and cannot pass the full-reference drift gate.",
                              "A finite seeded continuation diagnoses accumulation but is not evidence of paired deployment."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
