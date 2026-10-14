"""Prove the independent physical-rate controller evaluator is inert at zero gain."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval_tracking_mujoco import sha256


def compare(baseline, zero):
    with np.load(baseline, allow_pickle=False) as archive:
        original = {key: archive[key].copy() for key in archive.files}
    with np.load(zero, allow_pickle=False) as archive:
        candidate = {key: archive[key].copy() for key in archive.files}
    fields = ("time_s", "qpos", "root_pos", "root_com_lin_vel_w", "root_quat_wxyz",
              "wheel_pos", "joint_torque_nm", "wheel_ground_contact_force_w_n",
              "wheel_lateral_contact_slip_m_s", "action_observation",
              "actions_raw", "actions_clipped")
    errors = {}
    for key in fields:
        if original[key].shape != candidate[key].shape:
            raise AssertionError(f"Different {key} shapes in zero-feedback control")
        errors[key] = float(np.max(np.abs(original[key] - candidate[key])))
    counts_match = bool(np.array_equal(original["wheel_ground_contact_count"],
                                       candidate["wheel_ground_contact_count"]))
    if max(errors.values()) > 1e-10 or not counts_match:
        raise AssertionError(f"Zero feedback changes original physics: {errors}, contacts={counts_match}")
    return {"baseline_rollout_sha256": sha256(baseline),
            "zero_feedback_rollout_sha256": sha256(zero),
            "field_max_abs_errors": errors, "all_ground_contact_counts_match": counts_match,
            "physical_steps": len(original["time_s"]) - 1,
            "policy_steps": len(original["actions_clipped"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--zero", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserving previous parity audit")
    result = compare(args.baseline, args.zero)
    result["kind"] = "exact_zero_feedback_full_rollout_parity"
    result["audit_source_sha256"] = sha256(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
