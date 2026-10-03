"""Paired fixed-draw MuJoCo validation of baseline and DR jump actors.

This is a small, reproducible validation suite, not a population success-rate
estimate or an Isaac/MuJoCo same-actor transfer assessment. Nominal policy
contracts stay separate from the parameters actually applied to each episode.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assess_sim2sim import assess_episode, actuator_contract_comparison
from compare_tracking import read_json, read_npz, sha256
from eval_tracking_mujoco import load_torchscript, read_contract, run_evaluation
from training.tron1_domain_randomization import validate_draw

THRESHOLDS = {
    "min_height_gain_m": .20, "min_flight_s": .20,
    "min_tail_support_fraction": .90, "tail_seconds": .50,
    "max_tail_tilt_deg": 30., "contact_threshold_n": 5.,
}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def make_draws(joint_names, seed=20261004, random_draws=16):
    """One nominal draw, then independent per-joint PCG64 uniform parameters.

    These draws deliberately do not use the training reset sampler or its 20%
    nominal mixture. The actual values, not just the seed, are saved to disk.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(random_draws, bool) or not isinstance(random_draws, int) or random_draws < 1:
        raise ValueError("random_draws must be a positive integer")
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = []
    for index in range(random_draws + 1):
        parameters = validate_draw({
            "joint_names": list(joint_names),
            "torque_scale": [1.] * 8 if index == 0 else rng.uniform(.85, 1., size=8).tolist(),
            "velocity_scale": [1.] * 8 if index == 0 else rng.uniform(.85, 1., size=8).tolist(),
            "wheel_friction_nm": [0.] * 2 if index == 0 else rng.uniform(0., .3, size=2).tolist(),
            "friction_smoothing_rad_s": .5, "is_nominal": index == 0,
        })
        draws.append({"id": "nominal" if index == 0 else f"validation_{index:03d}",
                      "kind": "nominal" if index == 0 else "random_validation",
                      "domain_parameters": parameters,
                      "domain_parameters_sha256": canonical_hash(parameters)})
    return {"schema_version": 1, "seed": seed, "rng": "NumPy PCG64",
            "random_draw_count": random_draws, "nominal_draw_count": 1,
            "sampling": "independent uniform per-joint draws: eight torque scales, eight speed scales, two axle losses; no nominal mixture",
            "ranges": {"torque_scale": [.85, 1.], "velocity_scale": [.85, 1.],
                       "wheel_friction_nm": [0., .3]}, "draws": draws}


def verify_applied_parameters(report, expected):
    """Fail closed if requested parameters are missing, changed or resampled."""
    expected = validate_draw(expected)
    for label, container in (("report", report), ("contract", report.get("contract", {}))):
        if "domain_parameters" not in container:
            raise ValueError(f"Missing {label}.domain_parameters")
        if validate_draw(container["domain_parameters"]) != expected:
            raise ValueError(f"Applied {label}.domain_parameters differ from requested draw")
    if report.get("domain_parameters_resampled_during_episode") is not False:
        raise ValueError("Domain parameters must remain fixed throughout the episode")
    if report.get("actual_joint_velocity_hard_clipped") is not False:
        raise ValueError("Actual joint velocities must not be hard clipped")


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _metrics(assessment):
    flight = assessment.get("selected_flight") or {}
    return {"base_height_gain_m": assessment.get("base_height_gain_m"),
            "flight_s": flight.get("no_contact_duration_s"),
            "tail_two_wheel_support_fraction": assessment.get("tail_two_wheel_support_fraction"),
            "max_tail_tilt_deg": assessment.get("max_tail_tilt_deg")}


def summarize_pairs(pairs):
    """Keep nominal separate; count failures, never discard failed episodes."""
    result = {}
    for kind, key in (("nominal", "nominal"), ("random_validation", "random_only")):
        subset = [pair for pair in pairs if pair["kind"] == kind]
        outcomes = [pair["paired_outcome"] for pair in subset]
        result[key] = {
            "draw_count": len(subset),
            "baseline_pass_count": sum(pair["baseline"]["verdict"] == "pass" for pair in subset),
            "candidate_pass_count": sum(pair["candidate"]["verdict"] == "pass" for pair in subset),
            **{name: outcomes.count(name) for name in ("candidate_only_pass", "baseline_only_pass",
                                                      "both_pass", "both_fail")},
        }
    return result


def run_benchmark(baseline_policy, baseline_contract, candidate_policy, candidate_contract,
                  motion_file, model, output_dir, *, seed=20261004, random_draws=16):
    paths = {"baseline": (Path(baseline_policy), Path(baseline_contract)),
             "candidate": (Path(candidate_policy), Path(candidate_contract))}
    contracts = {name: read_contract(paths[name][1]) for name in paths}
    common = actuator_contract_comparison(contracts["baseline"], contracts["candidate"])
    if not common["matched"] or not common["dc_motor_involved"]:
        raise ValueError(f"Actors need the same nominal DC actuator contract: {common['mismatched_fields']}")
    names = contracts["baseline"].get("observed_joint_velocity_order")
    for name, contract in contracts.items():
        if (names != contract.get("observed_joint_velocity_order")
                or names != contract.get("leg_joint_names", []) + contract.get("wheel_joint_names", [])):
            raise ValueError("Both actors must have the same named eight-joint action order")
        if contract.get("actor_includes_observation_normalizer") is not True:
            raise ValueError(f"{name} actor must include its observation normalizer")
        if "domain_parameters" in contract and not validate_draw(contract["domain_parameters"])["is_nominal"]:
            raise ValueError("Pass nominal policy contracts, not an already randomized episode contract")
    draw_manifest = make_draws(names, seed, random_draws)
    motion_hash, model_hash = sha256(motion_file), sha256(model)
    actors = {}
    for name, (policy_path, contract_path) in paths.items():
        policy_hash = sha256(policy_path)
        if contracts[name].get("actor_sha256", policy_hash) != policy_hash:
            raise ValueError(f"{name} policy bytes differ from the explicit contract actor hash")
        if contracts[name].get("reference_file_sha256", motion_hash) != motion_hash:
            raise ValueError(f"{name} contract was exported for a different reference")
        actors[name] = {"policy_sha256": policy_hash, "nominal_contract_sha256": sha256(contract_path),
                        "nominal_contract": contracts[name]}
    output = Path(output_dir)
    # Even an existing empty directory is not silently reused.
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "draws.json", draw_manifest)
    (output / "parameters").mkdir()
    for draw in draw_manifest["draws"]:
        _write_json(output / "parameters" / (draw["id"] + ".json"), draw["domain_parameters"])
    _write_json(output / "inputs.json", {"actors": actors, "motion_sha256": motion_hash,
                                         "model_sha256": model_hash, "thresholds": THRESHOLDS})
    policies = {name: load_torchscript(paths[name][0]) for name in paths}
    pairs = []
    for draw in draw_manifest["draws"]:
        pair = copy.deepcopy(draw)
        for name in ("baseline", "candidate"):
            trial_dir = output / "trials" / draw["id"] / name
            print(f"[{draw['id']}] {name}", flush=True)
            try:
                run_evaluation(policies[name], motion_file, model, trial_dir,
                               policy_path=paths[name][0], contract_path=paths[name][1],
                               domain_parameters=copy.deepcopy(draw["domain_parameters"]),
                               contact_threshold_n=THRESHOLDS["contact_threshold_n"], no_visual_mesh=True)
                report = read_json(trial_dir / "report.json")
                verify_applied_parameters(report, draw["domain_parameters"])
                for field, expected in (("policy_sha256", actors[name]["policy_sha256"]),
                                        ("contract_source_sha256", actors[name]["nominal_contract_sha256"]),
                                        ("motion_sha256", motion_hash), ("model_sha256", model_hash)):
                    if report.get(field) != expected:
                        raise ValueError(f"Episode {field} does not match the benchmark input")
                assessment = assess_episode(read_npz(trial_dir / "rollout.npz"), report,
                                            "MuJoCo", dict(THRESHOLDS))
                if not assessment["complete_contact_evidence"]:
                    assessment["verdict"] = "fail"
                    assessment["fail_reasons"].append("complete_contact_evidence_required")
                evidence = {"report_sha256": sha256(trial_dir / "report.json"),
                            "rollout_sha256": sha256(trial_dir / "rollout.npz"),
                            "engine_version": report.get("engine_version"),
                            "actual_domain_parameters": report["domain_parameters"]}
            except (ValueError, KeyError, TypeError, OSError, RuntimeError, FloatingPointError) as exc:
                assessment = {"verdict": "fail", "fail_reasons": ["invalid_episode_evidence"],
                              "error": f"{type(exc).__name__}: {exc}", "complete_contact_evidence": False}
                evidence = {}
            trial_dir.mkdir(parents=True, exist_ok=True)
            _write_json(trial_dir / "assessment.json", assessment)
            pair[name] = {"verdict": assessment["verdict"], "fail_reasons": assessment["fail_reasons"],
                          "metrics": _metrics(assessment), "complete_contact_evidence": assessment["complete_contact_evidence"],
                          "trial_directory": str(trial_dir.relative_to(output)), **evidence}
        bp, cp = [pair[name]["verdict"] == "pass" for name in ("baseline", "candidate")]
        pair["paired_outcome"] = ("both_pass" if bp and cp else "both_fail" if not bp and not cp
                                  else "candidate_only_pass" if cp else "baseline_only_pass")
        pair["metric_delta_candidate_minus_baseline"] = {
            key: (pair["candidate"]["metrics"][key] - pair["baseline"]["metrics"][key]
                  if pair["candidate"]["metrics"][key] is not None and pair["baseline"]["metrics"][key] is not None
                  else None) for key in pair["baseline"]["metrics"]}
        pairs.append(pair)
    result = {
        "schema_version": 1, "status": "completed", "benchmark_kind": "paired_fixed_draw_validation",
        "engine": "MuJoCo", "seed": seed, "thresholds": dict(THRESHOLDS),
        "draw_manifest_sha256": sha256(output / "draws.json"), "actors": actors,
        "motion_sha256": motion_hash, "model_sha256": model_hash,
        "runner_source_sha256": sha256(__file__), "assessor_source_sha256": sha256(ROOT / "scripts/assess_sim2sim.py"),
        "summary": summarize_pairs(pairs), "pairs": pairs,
        "hardware_ready": False, "population_success_rate_estimate": False,
        "limitations": [
            "One initial reference-frame-zero state; only three simulated actuator parameter families vary, independently per joint.",
            "Nominal is reported separately from fixed random validation draws; no confidence or population-rate claim.",
            "Repeated use for checkpoint selection makes this a validation suite, not an independent final test set.",
            "Both actors use the same stored draws and original termination predicates; failures are retained.",
            "Nominal policy mappings stay fixed. Runtime scales apply to both DC saturation and effort and to no-load speed.",
            "Wheel drag is friction_nm*tanh(wheel_velocity/0.5), not a stiction model; net torque retains nominal +/-80/12 Nm guards.",
            "Simulation parameters are assumptions, not independently verified manufacturer ratings or hardware validation.",
            "This does not replace same-actor Isaac/MuJoCo transfer assessment; no nominal Isaac evidence is paired with randomized physics.",
        ],
    }
    _write_json(output / "report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-policy", "baseline-contract", "candidate-policy", "candidate-contract",
                 "motion-file", "model", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--random-draws", type=int, default=16)
    args = parser.parse_args(argv)
    result = run_benchmark(**vars(args))
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
