"""Verify and summarize every frozen-actor predictive-contact trial."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_tracking_mujoco import sha256


TRIALS = ("p01", "p02", "p03", "p04", "p05", "p06", "p07",
          "c01", "c02", "c03", "c04", "c05", "c06")


def summarize(root):
    rows = []
    identities = set()
    for trial in TRIALS:
        folder = root / f"feedback_{trial}"
        report = json.loads((folder / "report.json").read_text())
        assessment = json.loads((folder / "turn_assessment.json").read_text())
        rollout = folder / "rollout.npz"
        if (assessment["engine"] != "mujoco" or assessment["engine_report_sha256"] != sha256(folder / "report.json")
                or assessment["rollout_sha256"] != sha256(rollout)
                or assessment["full_reference_completed"] != report["completed_full_reference"]):
            raise ValueError(f"Mismatched physical assessment or rollout: {trial}")
        identity = tuple(report[key] for key in ("policy_sha256", "motion_sha256", "model_sha256", "contract_sha256"))
        identities.add(identity)
        rows.append({"trial": trial, "controller": report["controller"],
                     "status": assessment["status"],
                     "full_reference_completed": assessment["full_reference_completed"],
                     "minimum_core_speed_m_s": assessment["whole_core_min_speed_m_s"],
                     "core_path_rmse_m": assessment["core_path_rmse_m"],
                     "two_wheel_support_fraction": assessment["core_fraction_two_wheel_support_ok"],
                     "patch_slip_fraction": assessment["core_fraction_patch_slip_ok"],
                     "sideslip_fraction": assessment["core_fraction_sideslip_ok"],
                     "longest_qualifying_duration_s": assessment["longest_qualifying_duration_s"],
                     "largest_qualifying_course_turn_deg": assessment["largest_qualifying_course_turn_deg"],
                     "max_abs_knee_bias_rad": report["max_abs_knee_bias_rad"],
                     "rollout_sha256": sha256(rollout),
                     "report_sha256": sha256(folder / "report.json"),
                     "assessment_sha256": sha256(folder / "turn_assessment.json")})
    if len(identities) != 1:
        raise ValueError("Feedback trials changed actor, motion, model, or contract")
    return {"kind": "frozen_actor_predictive_contact_feedback_sweep",
            "actor_motion_model_contract_sha256": list(next(iter(identities))),
            "trial_count": len(rows), "all_status_fail": all(row["status"] == "FAIL" for row in rows),
            "all_full_reference_completed": all(row["full_reference_completed"] for row in rows),
            "source_sha256": sha256(__file__), "trials": rows,
            "limitations": ["All feedback trials are MuJoCo-only; none proves the same controller in Isaac.",
                            "The frozen assessor is used without gap bridging or changed thresholds."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserving previous sweep summary")
    result = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("trial_count", "all_status_fail", "all_full_reference_completed")},
                     indent=2))


if __name__ == "__main__":
    main()
