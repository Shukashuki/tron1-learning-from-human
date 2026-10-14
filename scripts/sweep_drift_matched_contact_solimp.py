"""Predeclared MuJoCo contact-impedance sweep against Isaac 20 ms windows.

This ranks short-horizon physics only. It does not call or optimize the actor.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from audit_drift_matched_contact_window import DT, HORIZON, replay_window, summarize
from eval_tracking_mujoco import MotionReference, configure_model, load_model, read_contract, sha256
from training.tron1_drift import configure_mujoco_materials


IMPEDANCES = ((.5, .7), (.5, .9), (.7, .85), (.7, .95),
              (.9, .95), (.9, .99), (.95, .99))
WIDTHS = (.0005, .001, .002)
FORCE_P95_RELATIVE_TOLERANCE = .2
SUPPORT_MINIMUM = .99


def force_percentiles(rows):
    steps = [step for row in rows for step in row["substeps"]]
    ia = np.asarray([step["isaac_wheel_ground_normal_force_n"] for step in steps])
    mj = np.asarray([step["mujoco_wheel_ground_normal_force_n"] for step in steps])
    ia_p95 = np.percentile(ia, 95, axis=0)
    mj_p95 = np.percentile(mj, 95, axis=0)
    return {"isaac_p95_n": ia_p95.tolist(), "mujoco_p95_n": mj_p95.tolist(),
            "maximum_p95_relative_error": float(np.max(np.abs(mj_p95 / ia_p95 - 1)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-physics", type=Path, required=True)
    parser.add_argument("--isaac-report", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserving previous impedance sweep")
    contract = read_contract(args.contract)
    isaac_report = json.loads(args.isaac_report.read_text())
    profile = copy.deepcopy(contract["task_profile"])
    solref = profile.pop("mujoco_contact_solref", None)
    if (profile != isaac_report.get("task_profile")
            or contract["reference_file_sha256"] != sha256(args.motion_file)
            or isaac_report["motion_file_sha256"] != sha256(args.motion_file)
            or solref != [.05, 2]):
        raise ValueError("Frozen Isaac report, motion and MuJoCo contact contract must match")
    model, _ = load_model(args.model, no_visual_mesh=True)
    configure_model(model, contract)
    configure_mujoco_materials(model, contract["task_profile"])
    if model.opt.timestep != DT or contract["decimation"] != HORIZON:
        raise ValueError("Expected original 200/50 Hz clock")
    colliding = np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    baseline = model.geom_solimp[colliding].copy()
    if not np.allclose(baseline, [.9, .95, .001, .5, 2]):
        raise ValueError(f"Unexpected compiled impedance: {baseline[0]}")
    reference = MotionReference(args.motion_file, contract)
    with np.load(args.isaac_physics, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if (not np.all(arrays["valid_mask"][:, 0])
            or not np.allclose(np.diff(arrays["time_s"]), DT, atol=1e-9, rtol=0)):
        raise ValueError("Complete aligned Isaac physical episode required")
    indices = [i for i in range(0, len(arrays["time_s"]) - HORIZON, HORIZON)
               if 1.44 <= arrays["time_s"][i] < 4.0]
    records = []
    for d0, dwidth in IMPEDANCES:
        for width in WIDTHS:
            solimp = [d0, dwidth, width, .5, 2]
            model.geom_solimp[colliding] = solimp
            rows = [replay_window(model, reference, arrays, i) for i in indices]
            summary = summarize(rows)
            forces = force_percentiles(rows)
            last_rmse = float(np.sqrt(np.mean(
                np.square(summary["wheel_z_rmse_m_by_substep_and_side"][-1]))))
            admissible = (summary["mujoco_all_four_steps_supported_window_fraction"] >= SUPPORT_MINIMUM
                          and forces["maximum_p95_relative_error"] <= FORCE_P95_RELATIVE_TOLERANCE)
            records.append({"solimp": solimp, "summary": summary, "forces": forces,
                            "last_step_wheel_z_rmse_m": last_rmse,
                            "physics_admissible": bool(admissible)})
            print(f"solimp={solimp} support={summary['mujoco_all_four_steps_supported_window_fraction']:.3f} "
                  f"z20={last_rmse*1000:.3f}mm force_p95_error={forces['maximum_p95_relative_error']:.3f}",
                  flush=True)
    candidates = [record for record in records if record["physics_admissible"]]
    selected = min(candidates, key=lambda item: item["last_step_wheel_z_rmse_m"]) if candidates else None
    result = {"kind": "matched_state_20ms_physics_calibration_not_policy_transfer",
              "input_sha256": {str(path): sha256(path) for path in
                               (args.isaac_physics, args.isaac_report, args.contract,
                                args.motion_file, args.model)},
              "frozen": {"clock_s": DT, "action_decimation": HORIZON,
                         "solref": solref, "contact_friction": profile["contact_friction"],
                         "margin_model": str(args.model), "isaac_window_count": len(indices)},
              "ranking_rule": {"all_four_support_at_least": SUPPORT_MINIMUM,
                               "p95_force_relative_error_at_most": FORCE_P95_RELATIVE_TOLERANCE,
                               "then_minimize": "20ms left-right RMS wheel-center-height error"},
              "records": records, "selected_solimp": selected["solimp"] if selected else None,
              "limitations": ["Independent 20 ms resets; no actor calls or policy-transfer result.",
                              "Parameter sweep is exploratory, not a unique PhysX contact-law identification."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"selected={result['selected_solimp']}", flush=True)


if __name__ == "__main__":
    main()
