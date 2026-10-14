"""Local 20 ms actor-action branch around historical MuJoCo wheel gaps.

The original four native substeps must replay exactly from the archived policy
decision state. Each branch changes one clipped action coordinate for the same
whole 20 ms window, with unchanged per-step PD/DC-motor torque recomputation.
This is a local counterfactual, not a modified actor or full-episode verdict.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from audit_drift_one_step_action_sensitivity import (
    DELTA_MAGNITUDES, restored_state, signed_clearance,
)
import eval_tracking_mujoco as evaluator
from training.tron1_drift import configure_mujoco_materials


def run_window(model, reference, arrays, frame, action):
    start = frame * 4
    state = restored_state(model, reference, arrays, start)
    data = state.data
    rows = []
    for substep in range(4):
        before = state.measure()
        torque = evaluator.action_torques(action, before["joint_pos"], before["joint_vel"],
                                          reference.q0, reference.contract)
        data.ctrl[state.aids] = torque
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        contact = state.ground_contact_slip()
        rows.append({
            "sample_index": start + substep + 1,
            "time_s": float(data.time),
            "qpos": data.qpos.copy(),
            "torque_nm": torque.copy(),
            "bearing_count": contact["wheel_ground_contact_count"].copy(),
            "normal_force_n": contact["wheel_ground_contact_force_w_n"][:, 2].copy(),
            "clearance_m": np.array([signed_clearance(state, wheel) for wheel in range(2)]),
            "root_pos": data.xpos[state.base].copy(),
        })
    return rows


def verify_original(rows, arrays):
    maximum = 0.
    for row in rows:
        i = row["sample_index"]
        qpos_error = float(np.max(np.abs(row["qpos"] - arrays["qpos"][i])))
        torque_error = float(np.max(np.abs(row["torque_nm"] - arrays["joint_torque_nm"][i])))
        force_error = float(np.max(np.abs(row["normal_force_n"]
                                          - arrays["wheel_ground_contact_force_w_n"][i, :, 2])))
        if (qpos_error > 1e-12 or torque_error > 1e-10 or force_error > 1e-8
                or not np.array_equal(row["bearing_count"], arrays["wheel_ground_contact_count"][i])
                or abs(row["time_s"] - arrays["time_s"][i]) > 1e-10):
            raise AssertionError(f"Four-substep original replay mismatches at physical sample {i}")
        maximum = max(maximum, qpos_error)
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-report", type=Path, required=True)
    parser.add_argument("--event-decomposition", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Preserving prior policy-window audit: {args.output_dir}")
    prior = json.loads(args.historical_report.read_text())
    events = json.loads(args.event_decomposition.read_text())
    rollout = args.historical_report.parent / "rollout.npz"
    if (events["rollout_sha256"] != evaluator.sha256(rollout)
            or prior["model_sha256"] != evaluator.sha256(prior["model"])
            or prior["motion_sha256"] != evaluator.sha256(prior["motion_file"])
            or prior["policy_sha256"] != evaluator.sha256(prior["policy"])
            or prior["contract_source_sha256"] != evaluator.sha256(prior["contract_source"])):
        raise ValueError("Historical rollout or inputs changed")
    with np.load(rollout, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    contract = evaluator.read_contract(prior["contract_source"])
    if contract != prior["contract"] or contract["physics_dt"] != .005 or contract["decimation"] != 4:
        raise ValueError("Expected unchanged 200/50 Hz deployment contract")
    reference = evaluator.MotionReference(prior["motion_file"], contract)
    model, _ = evaluator.load_model(prior["model"], prior["no_visual_mesh"],
                                    terrain=contract.get("terrain"))
    evaluator.configure_model(model, contract)
    configure_mujoco_materials(model, contract["task_profile"])
    results = []
    for event in events["events"]:
        if not event["support_failed"]:
            continue
        index = int(event["sample_index"])
        frame = (index - 1) // 4
        start = frame * 4
        if (int(arrays["reference_frame_for_action"][frame]) != int(arrays["reference_frame"][index])
                or abs(float(arrays["policy_time_s"][frame]) - float(arrays["time_s"][start])) > 1e-10):
            raise AssertionError("Policy decision window is misaligned")
        original_action = arrays["actions_clipped"][frame]
        baseline = run_window(model, reference, arrays, frame, original_action)
        exact_error = verify_original(baseline, arrays)
        for wheel, count in enumerate(event["bearing_contact_count"]):
            if count != 0:
                continue
            row_at_gap = baseline[index - start - 1]
            if row_at_gap["bearing_count"][wheel] != 0:
                raise AssertionError("Original failed wheel is loaded in the exact window replay")
            branches = []
            for action_index, name in enumerate(reference.action_names):
                for magnitude in DELTA_MAGNITUDES:
                    for sign in (-1., 1.):
                        changed = original_action.copy()
                        changed[action_index] = np.clip(changed[action_index] + sign*magnitude,
                                                        *contract["raw_action_clip"])
                        delta = float(changed[action_index] - original_action[action_index])
                        if abs(delta) < 1e-12:
                            continue
                        branch = run_window(model, reference, arrays, frame, changed)
                        gap = branch[index - start - 1]
                        full_support = all(np.all(item["bearing_count"] > 0) for item in branch)
                        branches.append({
                            "joint": name, "action_index": action_index, "action_delta": delta,
                            "gap_bearing_points": int(gap["bearing_count"][wheel]),
                            "gap_normal_force_n": float(gap["normal_force_n"][wheel]),
                            "gap_clearance_um": float(gap["clearance_m"][wheel] * 1e6),
                            "all_four_substeps_both_wheels_supported": bool(full_support),
                            "contact_counts_by_substep": [item["bearing_count"].tolist() for item in branch],
                            "normal_force_n_by_substep": [item["normal_force_n"].tolist() for item in branch],
                            "torque_change_at_gap_nm": float(gap["torque_nm"][action_index]
                                                             - row_at_gap["torque_nm"][action_index]),
                            "end_root_position_change_mm": ((branch[-1]["root_pos"]
                                                             - baseline[-1]["root_pos"]) * 1e3).tolist(),
                        })
            rescued = [branch for branch in branches if branch["all_four_substeps_both_wheels_supported"]]
            rescued.sort(key=lambda item: (abs(item["action_delta"]), abs(item["torque_change_at_gap_nm"])))
            results.append({
                "gap_index": index, "time_s": float(arrays["time_s"][index]),
                "wheel": ("L", "R")[wheel], "policy_index": frame,
                "policy_window_start_s": float(arrays["time_s"][start]),
                "policy_window_end_s": float(arrays["time_s"][start + 4]),
                "exact_baseline_qpos_error": exact_error,
                "baseline_gap_clearance_um": float(row_at_gap["clearance_m"][wheel] * 1e6),
                "branch_count": len(branches), "full_window_rescue_count": len(rescued),
                "smallest_full_window_rescue": rescued[0] if rescued else None,
                "branches": branches,
            })
    result = {
        "kind": "exact_local_policy_window_counterfactual_not_full_episode",
        "historical_rollout_sha256": evaluator.sha256(rollout),
        "event_decomposition_sha256": evaluator.sha256(args.event_decomposition),
        "actor_sha256": evaluator.sha256(prior["policy"]),
        "model_sha256": evaluator.sha256(prior["model"]),
        "diagnostic_source_sha256": evaluator.sha256(__file__),
        "action_perturbation_magnitudes": list(DELTA_MAGNITUDES),
        "event_count": len(results), "events": results,
        "limitations": [
            "Each branch begins at an archived policy decision state and changes one action coordinate for only that 20 ms window.",
            "A policy trained to emit a different action may change all earlier states; this local branch is not a full-rollout rescue.",
            "The branch tests fixed nominal MuJoCo physics only; Isaac and randomized-domain safety remain untested.",
            "No original rollout, actuator, contact gate, threshold or policy weight is changed.",
        ],
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "one_window_action_sensitivity.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"events": len(results),
                      "rescued_by_one_coordinate_full_window": sum(e["full_window_rescue_count"] > 0 for e in results)}))


if __name__ == "__main__":
    main()
