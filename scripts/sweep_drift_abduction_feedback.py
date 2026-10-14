"""Bounded six-episode MuJoCo abduction-feedback feasibility sweep.

This is evaluation-only; the same feedback still needs an Isaac implementation
and paired gate before any sim2sim claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from assess_drift_turn import assess, physical_episode, sha256
from eval_drift_predictive_abduction_mujoco import PredictiveAbductionFeedback, evaluate
from eval_tracking_mujoco import load_torchscript


THRESHOLDS_M = (.0012, .0015, .0018)
GAINS = (.5, 1.)
LOOKAHEAD_S = .005
MAX_BIAS_RAD = .08
MAX_CHANGE_RAD_PER_STEP = .04


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("policy", "motion-file", "model", "contract", "robot-reference", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        parser.error("Preserving previous abduction-feedback sweep")
    with np.load(args.robot_reference, allow_pickle=False) as archive:
        reference = {key: archive[key].copy() for key in archive.files}
    policy = load_torchscript(args.policy)
    candidates = []
    for threshold in THRESHOLDS_M:
        for gain in GAINS:
            name = f"t{int(round(threshold*1e6)):04d}-g{int(round(gain*100)):03d}"
            output = args.output_root / name
            controller = PredictiveAbductionFeedback(threshold_m=threshold,
                lookahead_s=LOOKAHEAD_S, gain=gain, max_bias_rad=MAX_BIAS_RAD,
                max_change_rad_per_step=MAX_CHANGE_RAD_PER_STEP)
            report = evaluate(policy, args.motion_file, args.model, args.contract,
                              output, controller, actor_file=args.policy)
            with np.load(output / "rollout.npz", allow_pickle=False) as archive:
                episode = physical_episode(archive, "mujoco")
            score = assess(episode, reference)
            score["turn_window_status"] = score["status"]
            score["full_reference_completed"] = report["completed_full_reference"]
            score["status"] = "PASS" if score["status"] == "PASS" and report["completed_full_reference"] else "FAIL"
            score.update(engine="mujoco", rollout_sha256=sha256(output / "rollout.npz"),
                         robot_reference_sha256=sha256(args.robot_reference),
                         engine_report_sha256=sha256(output / "report.json"),
                         assessor_sha256=sha256(Path(__file__).with_name("assess_drift_turn.py")))
            (output / "turn_assessment.json").write_text(json.dumps(score, indent=2, allow_nan=False) + "\n")
            summary = {"name": name, "threshold_m": threshold, "gain": gain,
                       "status": score["status"], "full_reference_completed": report["completed_full_reference"],
                       "minimum_core_speed_m_s": score["whole_core_min_speed_m_s"],
                       "core_path_rmse_m": score["core_path_rmse_m"],
                       "two_wheel_support_fraction": score["core_fraction_two_wheel_support_ok"],
                       "longest_qualifying_duration_s": score["longest_qualifying_duration_s"],
                       "largest_qualifying_course_turn_deg": score["largest_qualifying_course_turn_deg"],
                       "max_abs_abduction_bias_rad": report["max_abs_abduction_bias_rad"],
                       "rollout_sha256": score["rollout_sha256"],
                       "report_sha256": score["engine_report_sha256"]}
            candidates.append(summary)
            print(json.dumps(summary, allow_nan=False), flush=True)
    ledger = {"kind": "bounded_mujoco_only_abduction_feedback_sweep_not_paired_sim2sim",
              "frozen_input_sha256": {str(path): sha256(path) for path in
                                      (args.policy, args.motion_file, args.model,
                                       args.contract, args.robot_reference)},
              "candidate_grid": {"threshold_m": THRESHOLDS_M, "gain": GAINS,
                                 "lookahead_s": LOOKAHEAD_S, "max_bias_rad": MAX_BIAS_RAD,
                                 "max_change_rad_per_step": MAX_CHANGE_RAD_PER_STEP},
              "assessor_sha256": sha256(Path(__file__).with_name("assess_drift_turn.py")),
              "candidates": candidates,
              "limitations": ["MuJoCo-only bounded feasibility sweep; no candidate is paired Isaac-tested.",
                              "The feedback intervention changes the deployed controller, not the frozen outer actor."]}
    (args.output_root / "sweep.json").write_text(json.dumps(ledger, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
