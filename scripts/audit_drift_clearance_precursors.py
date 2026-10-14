"""Read-only finite-cylinder ground clearance before wheel contact gaps.

Saved MuJoCo qpos is forwarded through the exact audited collision MJCF;
there is no force solve, actor call, action change or physics step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from audit_drift_one_step_action_sensitivity import signed_clearance
from eval_tracking_mujoco import load_model, sha256


def quantiles(values):
    values = np.asarray(values, dtype=float)
    return None if not len(values) else np.quantile(values, [.05, .5, .95]).tolist()


def audit(rollout, model_file):
    with np.load(rollout, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    model, _ = load_model(model_file, no_visual_mesh=True)
    data = mujoco.MjData(model)
    # signed_clearance only needs model/data and the named collision geoms.
    state = type("Geometry", (), {"model": model, "data": data})()
    clearance = np.zeros((len(arrays["time_s"]), 2))
    for index, qpos in enumerate(arrays["qpos"]):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        clearance[index] = [signed_clearance(state, side) for side in range(2)]
    force = arrays["wheel_ground_contact_force_w_n"][:, :, 2]
    count = arrays["wheel_ground_contact_count"]
    core = (arrays["time_s"] >= 1.44 - 1e-8) & (arrays["time_s"] <= 4.0 + 1e-8)
    values = {}
    for name, side in (("L", 0), ("R", 1)):
        bearing = (count[:, side] > 0) & (force[:, side] >= 5.)
        onset = np.flatnonzero(core & ~bearing & np.r_[True, bearing[:-1]])
        prior = onset[onset > 0] - 1
        stable = np.flatnonzero(core & bearing)
        p90 = float(np.quantile(clearance[stable, side], .9))
        values[name] = {"gap_onsets": int(len(onset)),
                        "stable_clearance_p05_p50_p95_m": quantiles(clearance[stable, side]),
                        "one_step_before_gap_clearance_p05_p50_p95_m": quantiles(clearance[prior, side]),
                        "at_gap_clearance_p05_p50_p95_m": quantiles(clearance[onset, side]),
                        "fraction_gap_onsets_prior_clearance_above_stable_p90": float(np.mean(
                            clearance[prior, side] > p90)),
                        "fraction_gap_onsets_prior_clearance_above_zero": float(np.mean(
                            clearance[prior, side] > 0.)),
                        "fraction_gap_onsets_at_gap_clearance_above_zero": float(np.mean(
                            clearance[onset, side] > 0.))}
    return {"kind": "saved_state_finite_cylinder_clearance_precursor_not_physics",
            "rollout_sha256": sha256(rollout), "model_sha256": sha256(model_file),
            "source_sha256": sha256(__file__), "core_time_s": [1.44, 4.0],
            "sides": values,
            "limitations": ["Analytic finite-cylinder/plane geometry from saved qpos, not a reproduced PhysX contact manifold.",
                            "A signed zero-clearance threshold does not determine positive MuJoCo contact force.",
                            "Retrospective classification is not a causal controller or held-out prediction test."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserving previous clearance audit")
    result = audit(args.rollout, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
