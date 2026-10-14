"""Score saved MuJoCo episodes under the proposed residual-training reward.

Read-only audit: the frozen acceptance assessment remains authoritative.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from assess_drift_turn import assess, physical_episode
from eval_tracking_mujoco import MotionReference, read_contract, sha256
from training.drift_mujoco_residual_reward import reward_components
from training.drift_mujoco_streak import DriftStreakTracker


def audit(rollout, reference, robot_reference, residual):
    residual = np.asarray(residual, dtype=float)
    if residual.shape != (8,):
        raise ValueError("Expected eight raw-action residuals")
    with np.load(rollout, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    terms = []
    streak = DriftStreakTracker()
    streak_bonus = 0.
    for frame in range(len(arrays["actions_clipped"])):
        endpoint = (frame + 1) * 4
        state = {key: arrays[key][endpoint] for key in (
            "root_pos", "root_com_lin_vel_w", "root_quat_wxyz")}
        substeps = [{key: arrays[key][index] for key in (
            "time_s", "root_com_lin_vel_w", "root_quat_wxyz",
            "wheel_ground_contact_count", "wheel_ground_contact_force_w_n",
            "wheel_lateral_contact_slip_m_s")}
                    for index in range(endpoint - 3, endpoint + 1)]
        terms.append(reward_components(reference.sample(frame), state, substeps, residual))
        streak_bonus += 10. * sum(streak.observe(sample) for sample in substeps)
    streak_bonus += 10. * streak.finish()
    verdict = assess(physical_episode(arrays, "mujoco"), robot_reference)
    if (not np.isclose(streak.largest_qualifying_turn_deg,
                       verdict["largest_qualifying_course_turn_deg"], rtol=0, atol=1e-7)
            or not np.isclose(streak.longest_qualifying_duration_s,
                              verdict["longest_qualifying_duration_s"], rtol=0, atol=1e-10)):
        raise AssertionError("Training streak tracker disagrees with frozen assessor")
    core = [entry for entry in terms if entry["core"]]
    return {"policy_steps": len(terms), "core_policy_steps": len(core),
            "sum_reward": float(sum(entry["reward"] for entry in terms)),
            "streak_bonus": streak_bonus,
            "sum_with_streak_bonus": float(sum(entry["reward"] for entry in terms) + streak_bonus),
            "largest_qualifying_course_turn_deg": streak.largest_qualifying_turn_deg,
            "longest_qualifying_duration_s": streak.longest_qualifying_duration_s,
            "core_mean_joint_qualifying_soft_fraction": float(np.mean([
                entry["joint_qualifying_soft_fraction"] for entry in core])),
            "core_mean_support_fraction": float(np.mean([entry["support_fraction"] for entry in core])),
            "rollout_sha256": sha256(rollout)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--robot-reference", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, action="append", required=True)
    parser.add_argument("--residual", type=float, nargs=8, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserving existing reward audit")
    if len(args.rollout) != len(args.residual):
        parser.error("Each rollout needs its corresponding eight-action residual")
    reference = MotionReference(args.motion_file, read_contract(args.contract))
    with np.load(args.robot_reference, allow_pickle=False) as archive:
        robot_reference = {key: archive[key].copy() for key in archive.files}
    result = {"kind": "read_only_reward_alignment_audit_not_gate",
              "motion_sha256": sha256(args.motion_file),
              "robot_reference_sha256": sha256(args.robot_reference),
              "contract_sha256": sha256(args.contract),
              "reward_sha256": sha256(Path(__file__).resolve().parents[1] / "training/drift_mujoco_residual_reward.py"),
              "streak_sha256": sha256(Path(__file__).resolve().parents[1] / "training/drift_mujoco_streak.py"),
              "episodes": [dict(audit(path, reference, robot_reference, residual), residual=residual)
                           for path, residual in zip(args.rollout, args.residual)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
