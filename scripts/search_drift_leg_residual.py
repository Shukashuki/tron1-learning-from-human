"""Evaluation-only symmetric leg-action residual grid for the frozen WF actor.

Each candidate uses the same normalized actor, 51D observation, reference,
controller, MuJoCo model, physical clock and *unchanged* drift assessor. Only
the actor's raw hip/knee actions receive a constant, left/right-mirrored offset.
The search is MuJoCo-only; even a PASS here is not a sim2sim PASS until the
identical actor wrapper is exported and independently evaluated in Isaac.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from assess_drift_turn import assess, completed_from_report, physical_episode, sha256
from eval_tracking_mujoco import load_torchscript, run_evaluation


def mirrored_leg_bias(hip, knee):
    hip, knee = float(hip), float(knee)
    if not np.isfinite([hip, knee]).all() or max(abs(hip), abs(knee)) > .15:
        raise ValueError("Only finite small hip/knee action residuals are allowed")
    return np.array([0., 0., hip, -hip, knee, -knee, 0., 0.], dtype=np.float32)


def candidate_name(hip, knee):
    return f"hip{hip:+.3f}_knee{knee:+.3f}".replace("+", "p").replace("-", "m").replace(".", "d")


def run_grid(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Preserving existing residual search: {args.output_dir}")
    base_policy = load_torchscript(args.policy)
    with np.load(args.robot_reference, allow_pickle=False) as archive:
        reference = {key: archive[key].copy() for key in archive.files}
    rows = []
    for hip in args.hip:
        for knee in args.knee:
            bias = mirrored_leg_bias(hip, knee)
            path = args.output_dir / candidate_name(hip, knee)

            def policy(observation):
                return base_policy(observation) + bias

            report = run_evaluation(policy, args.motion_file, args.model, path,
                                    policy_path=None, contract_path=args.contract,
                                    no_visual_mesh=True)
            with np.load(path / "rollout.npz", allow_pickle=False) as archive:
                outcome = assess(physical_episode(archive, "mujoco"), reference)
            completed = completed_from_report(report, "mujoco")
            outcome["full_reference_completed"] = completed
            outcome["status"] = "PASS" if outcome["status"] == "PASS" and completed else "FAIL"
            row = {"hip_action_offset": float(hip), "knee_action_offset": float(knee),
                   "raw_action_bias": bias.tolist(), "status": outcome["status"],
                   "completed_full_reference": completed,
                   "minimum_core_speed_m_s": outcome["whole_core_min_speed_m_s"],
                   "core_path_rmse_m": outcome["core_path_rmse_m"],
                   "core_two_wheel_support_fraction": outcome["core_fraction_two_wheel_support_ok"],
                   "longest_qualifying_duration_s": outcome["longest_qualifying_duration_s"],
                   "largest_qualifying_turn_deg": outcome["largest_qualifying_course_turn_deg"],
                   "rollout_sha256": sha256(path / "rollout.npz"), "output_dir": str(path)}
            (path / "turn_assessment.json").write_text(json.dumps(outcome, indent=2, allow_nan=False) + "\n")
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = {"kind": "mujoco_only_frozen_actor_constant_symmetric_leg_residual_search_not_sim2sim",
              "policy_sha256": sha256(args.policy), "contract_sha256": sha256(args.contract),
              "motion_sha256": sha256(args.motion_file), "model_sha256": sha256(args.model),
              "robot_reference_sha256": sha256(args.robot_reference),
              "residual_source_sha256": sha256(__file__), "candidates": rows,
              "limitations": ["These are separate full closed-loop MuJoCo episodes; no Isaac residual-actor evaluation is included.",
                              "Candidate search on the same reference is exploratory, not a generalization test.",
                              "The raw actor output is biased before the original evaluator's action clipping."]}
    (args.output_dir / "search.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--robot-reference", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hip", type=float, nargs="+", required=True)
    parser.add_argument("--knee", type=float, nargs="+", required=True)
    args = parser.parse_args()
    for value in (*args.hip, *args.knee):
        if not np.isfinite(value) or abs(value) > .15:
            parser.error("Residual action grid must remain within [-0.15, 0.15]")
    if len(args.hip) * len(args.knee) > 49:
        parser.error("At most 49 exploratory candidates per search")
    run_grid(args)


if __name__ == "__main__":
    main()
