"""Counterfactual one-step policy-action sensitivity at MuJoCo wheel gaps.

Each saved pre-gap state first reproduces its original next step exactly.
Branches then perturb one already-clipped actor action coordinate only for that
single 5 ms step and recompute the unchanged DC-motor/PD torque law. This is
NOT a deployable 50 Hz policy rollout or evidence that the full gap is solved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from audit_drift_wheel_clearance import recover_generalized_velocity
import eval_tracking_mujoco as evaluator
from training.tron1_drift import configure_mujoco_materials


DELTA_MAGNITUDES = (.002, .005, .01, .02, .04, .08)


def signed_clearance(state, wheel):
    model, data = state.model, state.data
    side = ("L", "R")[wheel]
    geom = model.geom(f"wheel_{side}_collision").id
    floor = model.geom("floor").id
    normal = data.geom_xmat[floor].reshape(3, 3)[:, 2]
    axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
    radius, halfwidth = model.geom_size[geom, :2]
    u = float(axis @ normal)
    return float(normal @ (data.geom_xpos[geom] - data.geom_xpos[floor])
                 - radius * np.sqrt(max(0., 1. - u*u)) - halfwidth * abs(u))


def restored_state(model, reference, arrays, pre_index):
    data = mujoco.MjData(model)
    data.qpos[:] = arrays["qpos"][pre_index]
    data.time = float(arrays["time_s"][pre_index])
    mujoco.mj_forward(model, data)
    state = evaluator.MujocoState(model, data, reference)
    jacp, jacr = state.root_jacobians()
    twist = np.r_[arrays["root_com_lin_vel_w"][pre_index], arrays["root_ang_vel_w"][pre_index]]
    data.qvel[:] = recover_generalized_velocity(
        np.vstack((jacp, jacr)), arrays["joint_vel"][pre_index], state.vids,
        np.arange(state.root_vid, state.root_vid + 6), twist)
    mujoco.mj_forward(model, data)
    measured = state.measure()
    for name in ("joint_pos", "joint_vel", "root_pos", "root_com_lin_vel_w", "root_ang_vel_w", "wheel_pos"):
        if not np.allclose(measured[name], arrays[name][pre_index], atol=1e-9, rtol=0):
            raise AssertionError(f"Archived pre-gap state reconstruction mismatch: {name}")
    return state


def one_step(model, reference, arrays, pre_index, action, wheel):
    state = restored_state(model, reference, arrays, pre_index)
    data = state.data
    torque = evaluator.action_torques(action, arrays["joint_pos"][pre_index],
                                      arrays["joint_vel"][pre_index], reference.q0,
                                      reference.contract)
    data.ctrl[state.aids] = torque
    mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    contacts = state.ground_contact_slip()
    return {
        "torque_nm": torque.tolist(),
        "qpos": data.qpos.copy(),
        "root_pos": data.xpos[state.base].copy(),
        "wheel_link_z_m": float(data.xpos[state.wheels[wheel], 2]),
        "wheel_clearance_m": signed_clearance(state, wheel),
        "bearing_points": int(contacts["wheel_ground_contact_count"][wheel]),
        "normal_force_n": float(contacts["wheel_ground_contact_force_w_n"][wheel, 2]),
        "time_s": float(data.time),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-report", type=Path, required=True)
    parser.add_argument("--event-decomposition", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Preserving prior action-sensitivity audit: {args.output_dir}")
    prior = json.loads(args.historical_report.read_text())
    events = json.loads(args.event_decomposition.read_text())
    rollout = args.historical_report.parent / "rollout.npz"
    if (events["rollout_sha256"] != evaluator.sha256(rollout)
            or prior["model_sha256"] != evaluator.sha256(prior["model"])
            or prior["motion_sha256"] != evaluator.sha256(prior["motion_file"])
            or prior["policy_sha256"] != evaluator.sha256(prior["policy"])
            or prior["contract_source_sha256"] != evaluator.sha256(prior["contract_source"])):
        raise ValueError("Historical input hashes or event decomposition changed")
    with np.load(rollout, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    contract = evaluator.read_contract(prior["contract_source"])
    reference = evaluator.MotionReference(prior["motion_file"], contract)
    model, _ = evaluator.load_model(prior["model"], prior["no_visual_mesh"],
                                    terrain=contract.get("terrain"))
    evaluator.configure_model(model, contract)
    configure_mujoco_materials(model, contract["task_profile"])
    if float(model.opt.timestep) != .005 or contract["decimation"] != 4:
        raise ValueError("Expected the original 200/50 Hz physical/controller clocks")
    rows = []
    for event in events["events"]:
        if not event["support_failed"]:
            continue
        index = event["sample_index"]
        pre = index - 1
        if pre < 0:
            raise ValueError("No preceding physical state for an event")
        policy_index = pre // 4
        if int(arrays["reference_frame_for_action"][policy_index]) != int(arrays["reference_frame"][index]):
            raise AssertionError("Policy action is not aligned with the failed physical step")
        original_action = arrays["actions_clipped"][policy_index]
        for wheel, count in enumerate(event["bearing_contact_count"]):
            if count != 0:
                continue
            baseline = one_step(model, reference, arrays, pre, original_action, wheel)
            if (not np.allclose(baseline["qpos"], arrays["qpos"][index], atol=1e-12, rtol=0)
                    or not np.allclose(baseline["torque_nm"], arrays["joint_torque_nm"][index], atol=1e-10, rtol=0)
                    or baseline["bearing_points"] != 0
                    or baseline["normal_force_n"] != 0.
                    or abs(baseline["time_s"] - arrays["time_s"][index]) > 1e-10):
                raise AssertionError(f"Original one-step gap is not exactly reproducible at index {index}")
            branches = []
            for action_index, name in enumerate(reference.action_names):
                for magnitude in DELTA_MAGNITUDES:
                    for sign in (-1., 1.):
                        changed = original_action.copy()
                        changed[action_index] = np.clip(changed[action_index] + sign*magnitude,
                                                        *contract["raw_action_clip"])
                        actual_delta = float(changed[action_index] - original_action[action_index])
                        if abs(actual_delta) < 1e-12:
                            continue
                        result = one_step(model, reference, arrays, pre, changed, wheel)
                        branches.append({
                            "joint": name, "action_index": action_index,
                            "action_delta": actual_delta,
                            "torque_delta_nm": float(result["torque_nm"][action_index] - baseline["torque_nm"][action_index]),
                            "wheel_clearance_um": float(result["wheel_clearance_m"] * 1e6),
                            "normal_force_n": result["normal_force_n"],
                            "bearing_points": result["bearing_points"],
                            "wheel_link_height_change_um": float((result["wheel_link_z_m"] - baseline["wheel_link_z_m"])*1e6),
                            "root_position_change_um": (1e6*(result["root_pos"] - baseline["root_pos"])).tolist(),
                        })
            rescued = [branch for branch in branches if branch["bearing_points"] > 0
                       and branch["normal_force_n"] > 5.]
            rescued.sort(key=lambda item: (abs(item["action_delta"]), abs(item["torque_delta_nm"])))
            rows.append({
                "sample_index": index, "time_s": float(arrays["time_s"][index]),
                "wheel": ("L", "R")[wheel], "policy_index": policy_index,
                "original_action": original_action.tolist(),
                "baseline_clearance_um": baseline["wheel_clearance_m"] * 1e6,
                "baseline_qpos_max_abs_error": float(np.max(np.abs(baseline["qpos"] - arrays["qpos"][index]))),
                "branch_count": len(branches), "single_coordinate_rescue_count": len(rescued),
                "smallest_single_coordinate_rescue": rescued[0] if rescued else None,
                "branches": branches,
            })
    result = {
        "kind": "MuJoCo_exact_pre_gap_one_step_action_counterfactual_not_a_policy_rollout",
        "historical_rollout_sha256": evaluator.sha256(rollout),
        "event_decomposition_sha256": evaluator.sha256(args.event_decomposition),
        "model_sha256": evaluator.sha256(prior["model"]),
        "actor_sha256": evaluator.sha256(prior["policy"]),
        "diagnostic_source_sha256": evaluator.sha256(__file__),
        "action_perturbation_magnitudes": list(DELTA_MAGNITUDES),
        "event_count": len(rows), "events": rows,
        "limitations": [
            "All branches start from the saved state one native step before the gap; this is not a complete 50 Hz action-held rollout.",
            "Only one action coordinate changes at a time, through the unchanged motor and PD torque law.",
            "The policy cannot know the future gap or change its action solely at an arbitrary substep; rescue is local controllability evidence only.",
            "All historical states, force gates and original paired verdicts remain untouched.",
        ],
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "one_step_action_sensitivity.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"events": len(rows), "rescued_by_one_coordinate": sum(row["single_coordinate_rescue_count"] > 0 for row in rows)}))


if __name__ == "__main__":
    main()
