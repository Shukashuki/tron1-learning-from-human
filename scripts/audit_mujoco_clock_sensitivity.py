"""Evaluation-only MuJoCo physics-clock sweep of one frozen drift actor.

This is not a paired Isaac/MuJoCo assessment: changing the MuJoCo step size
changes its physical episode. The original 200 Hz episode and gates remain
untouched. Actor, motion, model, actuator/domain draw and 50 Hz action period
are pinned to the historical rollout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from assess_drift import THRESHOLDS, assess_episode, intervals, sideslip
import eval_tracking_mujoco as evaluator
from training.tron1_physics_clock import apply_physics_clock_to_contract


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def native_contact_failures(arrays, speed, beta):
    time = np.asarray(arrays["time_s"], float)
    dt = float(arrays["physics_dt_s"])
    if not np.allclose(np.diff(time), dt, rtol=0, atol=1e-9):
        raise ValueError("Native physical clock is not uniform")
    counts = np.asarray(arrays["wheel_ground_contact_count"])
    slip = np.asarray(arrays["wheel_lateral_contact_slip_m_s"])
    th = THRESHOLDS
    moving = ((speed >= th["drift_min_planar_speed_m_s"])
              & (np.abs(np.degrees(beta)) >= th["drift_min_abs_sideslip_deg"])
              & (np.cos(beta) > 0))
    support = (counts > 0).all(axis=1)
    sufficient_slip = np.abs(slip).mean(axis=1) >= th["drift_min_mean_abs_contact_slip_m_s"]
    support_failed = moving & ~support
    slip_only_failed = moving & support & ~sufficient_slip
    return {"moving_drift_samples": int(moving.sum()),
            "support_failure_samples_while_moving_drift": int(support_failed.sum()),
            "slip_only_failure_samples_while_supported_moving_drift": int(slip_only_failed.sum()),
            "support_failure_times_s": time[support_failed].tolist(),
            "slip_only_failure_times_s": time[slip_only_failed].tolist()}


def coarse_observation_diagnostic(arrays, baseline_arrays):
    """Describe 200 Hz subsampling only; never substitute it for native gating."""
    native_dt = float(arrays["physics_dt_s"])
    ratio = int(round(.005 / native_dt))
    if ratio not in (2, 4) or not np.isclose(native_dt * ratio, .005, rtol=0, atol=1e-12):
        raise ValueError("Expected a 400/800 Hz episode aligned to the original 200 Hz grid")
    baseline_time = baseline_arrays["time_s"]
    indices = np.arange(len(baseline_time)) * ratio
    if (len(indices) != len(arrays["time_s"][::ratio])
            or not np.allclose(arrays["time_s"][indices], baseline_time, rtol=0, atol=1e-9)):
        raise AssertionError("Refined episode does not align with the baseline clock")
    speed, beta = sideslip(arrays["root_quat_wxyz"], arrays["root_com_lin_vel_w"])
    th = THRESHOLDS
    moving = ((speed >= th["drift_min_planar_speed_m_s"])
              & (np.abs(np.degrees(beta)) >= th["drift_min_abs_sideslip_deg"])
              & (np.cos(beta) > 0))
    support = (arrays["wheel_ground_contact_count"] > 0).all(axis=1)
    slip = np.abs(arrays["wheel_lateral_contact_slip_m_s"]).mean(axis=1)
    qualifying = moving & support & (slip >= th["drift_min_mean_abs_contact_slip_m_s"])
    coarse_runs = intervals(arrays["time_s"][indices], qualifying[indices])
    action_error = np.abs(arrays["actions_clipped"] - baseline_arrays["actions_clipped"])
    root_error = arrays["root_pos"][indices] - baseline_arrays["root_pos"]
    return {
        "kind": "observation_aliasing_and_closed_loop_divergence_not_an_accepted_gate",
        "coarse_200hz_max_contiguous_drift_s": max((r["duration_s"] for r in coarse_runs), default=0.),
        "coarse_200hz_support_failure_samples_while_moving_drift": int(np.count_nonzero(moving[indices] & ~support[indices])),
        "native_support_failure_samples_while_moving_drift": int(np.count_nonzero(moving & ~support)),
        "mean_abs_50hz_action_difference_from_baseline": float(action_error.mean()),
        "max_abs_50hz_action_difference_from_baseline": float(action_error.max()),
        "aligned_root_position_rmse_m": float(np.sqrt(np.mean(np.sum(root_error**2, axis=1)))),
    }


def characterize(arrays, reference, report):
    assessment = assess_episode(arrays, reference, report=report)
    speed, beta = sideslip(arrays["root_quat_wxyz"], arrays["root_com_lin_vel_w"])
    contacts = native_contact_failures(arrays, speed, beta)
    return {
        "sample_dt_s": float(arrays["physics_dt_s"]),
        "physical_samples": int(len(arrays["time_s"])),
        "policy_samples": int(len(arrays["actions_raw"])),
        "mujo_only_verdict": assessment["verdict"],
        "failed_checks": assessment["failed_checks"],
        "sustained_drift_check_passed": assessment["checks"]["sustained_grounded_contact_patch_drift"]["passed"],
        "max_contiguous_grounded_drift_s": assessment["metrics"]["maximum_contiguous_drift_s"],
        **contacts,
        "full_two_wheel_support_fraction": assessment["metrics"]["full_two_wheel_support_fraction"],
        "reference_root_rmse_m": assessment["metrics"]["reference_tracking"]["base_position_rmse_m"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Preserving existing study: {args.output_dir}")
    prior = json.loads(args.historical_report.read_text())
    original_rollout = args.historical_report.parent / "rollout.npz"
    if (prior.get("engine") != "MuJoCo" or prior.get("completed_full_reference") is not True
            or prior.get("physics_steps") != 1152 or prior.get("policy_steps") != 288
            or prior.get("contract", {}).get("physics_dt") != .005
            or prior.get("contract", {}).get("decimation") != 4):
        raise ValueError("Expected the unchanged full-length 200 Hz / 50 Hz drift baseline")
    contract = evaluator.read_contract(prior["contract_source"])
    if contract != prior["contract"]:
        raise ValueError("Saved contract differs from historical evaluation report")
    hashes = {key: evaluator.sha256(prior[key]) for key in ("policy", "motion_file", "model", "contract_source")}
    for key, value in hashes.items():
        report_key = {"policy": "policy_sha256", "motion_file": "motion_sha256",
                      "model": "model_sha256", "contract_source": "contract_source_sha256"}[key]
        if prior[report_key] != value:
            raise ValueError(f"Historical {key} input changed")
    reference = read_npz(prior["motion_file"])
    baseline_arrays = read_npz(original_rollout)
    baseline = characterize(baseline_arrays, reference, prior)
    baseline.update(kind="historical_unchanged_200Hz", rollout_sha256=evaluator.sha256(original_rollout))
    args.output_dir.mkdir(parents=True)
    variants = []
    for dt, label in ((.0025, "400Hz"), (.00125, "800Hz")):
        variant_dir = args.output_dir / label
        variant_dir.mkdir()
        variant_contract = apply_physics_clock_to_contract(contract, dt, trained_contract=contract)
        differences = {key for key in set(contract) | set(variant_contract)
                       if contract.get(key) != variant_contract.get(key)}
        if differences != {"physics_dt", "decimation", "physics_clock_refinement"}:
            raise AssertionError(f"Unexpected contract differences: {differences}")
        contract_path = variant_dir / "clock_contract.json"
        contract_path.write_text(json.dumps(variant_contract, indent=2, allow_nan=False) + "\n")
        evaluated = evaluator.run_evaluation(
            evaluator.load_torchscript(prior["policy"]), prior["motion_file"], prior["model"],
            variant_dir / "mujoco", policy_path=prior["policy"],
            contact_threshold_n=5., no_visual_mesh=prior["no_visual_mesh"],
            contract_path=contract_path,
        )
        arrays = read_npz(variant_dir / "mujoco" / "rollout.npz")
        if (float(arrays["physics_dt_s"]) != dt or float(arrays["policy_dt_s"]) != .02
                or len(arrays["actions_raw"]) != baseline["policy_samples"]
                or abs(float(arrays["time_s"][-1]) - float(read_npz(original_rollout)["time_s"][-1])) > 1e-8):
            raise AssertionError("Clock variant changed the policy period or full episode duration")
        row = characterize(arrays, reference, evaluated)
        row["coarse_observation_diagnostic"] = coarse_observation_diagnostic(arrays, baseline_arrays)
        row.update(kind="MuJoCo_only_different_physical_episode",
                   contract_sha256=evaluator.sha256(contract_path),
                   rollout_sha256=evaluator.sha256(variant_dir / "mujoco" / "rollout.npz"),
                   report_sha256=evaluator.sha256(variant_dir / "mujoco" / "report.json"))
        variants.append(row)
    result = {
        "kind": "MuJoCo_only_clock_sensitivity_not_a_paired_sim2sim_assessment",
        "historical_report_sha256": evaluator.sha256(args.historical_report),
        "inputs_sha256": hashes,
        "diagnostic_source_sha256": evaluator.sha256(__file__),
        "unchanged_thresholds": dict(THRESHOLDS),
        "baseline": baseline, "variants": variants,
        "limitations": [
            "The actor is not retrained; policy actions are evaluated anew in changed MuJoCo dynamics.",
            "Isaac was not rerun at either refined clock, so these are not paired sim2sim verdicts.",
            "A reduced gap count can reflect a changed drift window and is not proof of better contact robustness.",
            "The original 200 Hz assessment remains authoritative; no gap is bridged or threshold changed.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"baseline_s": baseline["max_contiguous_grounded_drift_s"],
                      "variants": [{"dt": v["sample_dt_s"],
                                    "max_s": v["max_contiguous_grounded_drift_s"],
                                    "support_failures": v["support_failure_samples_while_moving_drift"]}
                                   for v in variants]}))


if __name__ == "__main__":
    main()
