"""Strict, offline same-policy jump-transfer acceptance over recorded episodes.

This never runs a simulator or changes the original termination predicates.
Height, sustained flight, landing and stable tail support are separate gates;
the evaluator's older short-bounce success flag is deliberately not trusted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_tracking import load_episode, read_json, read_npz, sha256
from eval_tracking import contact_events


def _check(checks, name, passed, detail=None):
    checks[name] = {"passed": bool(passed)}
    if detail is not None:
        checks[name]["detail"] = detail


def _hash_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _audit_value(value):
    """Keep rejected nonfinite values visible without emitting invalid JSON."""
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_audit_value(item) for item in value]
    return value


def actuator_contract_comparison(exported, deployed):
    """Compare complete declared actuator/clock parameters, not policy bytes alone.

    Historical contracts omitted actuator_model and used ideal PD. Numeric
    parameters never receive defaults: two equally missing fields are not an
    agreement. The current shared DC profile is explicitly checked below.
    """
    models = [contract.get("actuator_model", "ideal_pd") for contract in (exported, deployed)]
    fields = {"actuator_model": {"isaac": models[0], "mujoco": models[1],
                                "matched": models[0] == models[1] and models[0] in ("ideal_pd", "dc_motor")}}
    common = ("leg_kp", "leg_kd", "leg_position_scale", "wheel_torque_scale_nm",
              "leg_torque_limit_nm", "wheel_torque_limit_nm", "physics_dt", "decimation", "policy_fps",
              "raw_action_clip")
    expected_dc = {"leg_saturation_effort_nm": 80., "wheel_saturation_effort_nm": 12.,
                   "leg_motor_velocity_limit_rad_s": 15., "wheel_motor_velocity_limit_rad_s": 100.,
                   "solver_joint_velocity_limit_rad_s": 1000.}
    is_dc = "dc_motor" in models
    for name in common + (tuple(expected_dc) if is_dc else ()):
        values = [contract.get(name) for contract in (exported, deployed)]
        try:
            arrays = [np.asarray(value) for value in values]
        except (ValueError, TypeError):
            arrays = [np.asarray(None), np.asarray(None)]
        shape = (2,) if name == "raw_action_clip" else ()
        valid = all(name in contract for contract in (exported, deployed))
        valid = valid and all(a.shape == shape and a.dtype.kind in "iuf" and np.isfinite(a).all() for a in arrays)
        if valid:
            if name == "raw_action_clip":
                valid = all(a[0] < a[1] for a in arrays)
            else:
                valid = all(float(a) >= 0 if name == "leg_kd" else float(a) > 0 for a in arrays)
                if name == "decimation":
                    valid = valid and all(float(a).is_integer() for a in arrays)
            valid = valid and np.allclose(arrays[0], arrays[1], rtol=1e-12, atol=0.)
            if name in expected_dc:
                valid = valid and all(np.isclose(a, expected_dc[name], rtol=1e-12, atol=0.) for a in arrays)
        fields[name] = {"isaac": _audit_value(values[0]), "mujoco": _audit_value(values[1]), "matched": bool(valid)}
        if name in expected_dc:
            fields[name]["required_shared_dc_profile_value"] = expected_dc[name]
    clock_fields = ("physics_dt", "decimation", "policy_fps")
    clock_valid = all(fields[name]["matched"] for name in clock_fields)
    if clock_valid:
        clock_valid = all(np.isclose(contract["physics_dt"] * contract["decimation"] * contract["policy_fps"],
                                    1., rtol=1e-9, atol=0.) for contract in (exported, deployed))
    fields["consistent_control_clock"] = {"matched": bool(clock_valid)}
    if is_dc:
        semantics = "no_load_speed_for_torque_speed_curve_not_hard_qvel_clip"
        values = [contract.get("motor_velocity_limit_semantics") for contract in (exported, deployed)]
        fields["motor_velocity_limit_semantics"] = {"isaac": values[0], "mujoco": values[1],
                                                    "matched": all(v == semantics for v in values)}
    mismatches = [name for name, field in fields.items() if not field["matched"]]
    return {"matched": not mismatches, "mismatched_fields": mismatches, "fields": fields,
            "dc_motor_involved": is_dc,
            "parameter_assumptions": {
                "isaac": exported.get("actuator_limits_provenance"),
                "mujoco": deployed.get("actuator_limits_provenance"),
                "note": "Shared simulation parameters, not independently verified manufacturer hardware ratings.",
            }, "hardware_ready": False}


def legacy_friction_audit_valid(isaac, exported):
    audit = isaac.get("legacy_joint_friction_audit")
    if not isinstance(audit, dict) or audit.get("all_environments_verified_zero") is not True:
        return False
    expected = exported.get("leg_joint_names", []) + exported.get("wheel_joint_names", [])
    names = audit.get("joint_names")
    if (not isinstance(names, list) or not all(isinstance(n, str) for n in names)
            or len(expected) != 8 or len(names) != 8 or len(set(names)) != 8):
        return False
    if len(names) != 8 or set(names) != set(expected):
        return False
    if "actual_joint_order" in exported and names != exported["actual_joint_order"]:
        return False
    after = np.asarray(audit.get("after_first_environment"))
    return bool(after.shape == (8,) and after.dtype.kind in "iuf" and np.isfinite(after).all()
                and np.all(after == 0.))


def check_provenance(isaac_dir, mujoco_dir, isaac, mujoco, actor_path=None):
    """Validate actual actor bytes, source-reference hashes and observation audit."""
    checks = {}
    exported = isaac.get("policy_export", {})
    contract = mujoco.get("contract", {})
    reference_hash = isaac.get("motion_file_sha256")
    _check(checks, "same_reference", _hash_string(reference_hash)
           and reference_hash == mujoco.get("motion_sha256")
           and reference_hash == exported.get("reference_file_sha256"))
    if actor_path is None:
        name = Path(exported.get("actor_file", "actor_normalized.pt"))
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Isaac actor_file must be relative to its evaluation directory")
        actor_path = isaac_dir / name
    actor_hash = sha256(actor_path)
    _check(checks, "same_actor", actor_hash == mujoco.get("policy_sha256"))
    if "actor_sha256" in exported:
        _check(checks, "isaac_export_hash", actor_hash == exported["actor_sha256"])
    _check(checks, "normalized_actor", exported.get("actor_includes_observation_normalizer") is True
           and mujoco.get("policy_kind") == "normalized_torchscript_actor")
    _check(checks, "matching_policy_dimensions", exported.get("actor_dim") == contract.get("actor_dim") == 51
           and exported.get("action_dim") == contract.get("action_dim") == 8)
    legs = exported.get("leg_joint_names", [])
    _check(checks, "matching_leg_action_order", len(legs) == len(set(legs)) == 6
           and legs == contract.get("leg_joint_names"))
    actuator = actuator_contract_comparison(exported, contract)
    _check(checks, "matching_actuator_contract", actuator["matched"],
           {"mismatched_fields": actuator["mismatched_fields"]})
    _check(checks, "simulation_parameter_assumptions_disclosed",
           all(c.get("hardware_ready") is False and isinstance(c.get("actuator_limits_provenance"), str)
               and bool(c["actuator_limits_provenance"].strip()) for c in (exported, contract)))
    if actuator["dc_motor_involved"]:
        _check(checks, "legacy_joint_friction_cleared_and_audited", legacy_friction_audit_valid(isaac, exported))
        _check(checks, "actual_joint_velocities_not_hard_clipped", mujoco.get("actual_joint_velocity_hard_clipped") is False)
    _check(checks, "matching_reference_length", isaac.get("reference_frames")
           == mujoco.get("reference_frames") and isinstance(isaac.get("reference_frames"), int)
           and isaac["reference_frames"] > 1)
    trajectory_hash = sha256(isaac_dir / "trajectory.npz")
    verification = mujoco.get("isaac_observation_validation") or {}
    maximum = verification.get("max_abs_error")
    verified = (verification.get("status") == "passed"
                and verification.get("observations_checked", 0) > 0
                and verification.get("trajectory_sha256") == trajectory_hash
                and isinstance(maximum, (float, int)) and np.isfinite(maximum) and maximum >= 0)
    _check(checks, "isaac_observation_reconstruction_verified", verified)
    return {"checks": checks, "actor_sha256": actor_hash, "reference_sha256": reference_hash,
            "actuator_contract": actuator,
            "isaac_trajectory_sha256": trajectory_hash,
            "mujoco_trajectory_sha256": sha256(mujoco_dir / "rollout.npz"),
            "isaac_report_sha256": sha256(isaac_dir / "report.json"),
            "mujoco_report_sha256": sha256(mujoco_dir / "report.json")}


def tail_support_fraction(times, supported, seconds):
    """Duration-weighted zero-order-hold support; never count unobserved time."""
    start, end = float(times[-1] - seconds), float(times[-1])
    if start < times[0] - 1e-9:
        return None
    widths = np.maximum(0., np.minimum(times[1:], end) - np.maximum(times[:-1], start))
    return float(np.dot(widths, np.asarray(supported[:-1], dtype=float)) / seconds)


def _select_series(values, arrays, engine):
    values = np.asarray(values)
    if engine == "Isaac":
        if values.ndim < 2 or values.shape[:2] != arrays["valid_mask"].shape:
            raise ValueError("Isaac contact evidence must have time/environment leading dimensions")
        return values[arrays["valid_mask"][:, 0], 0]
    return values


def assess_episode(arrays, report, engine, thresholds):
    """Assess environment zero only; repeated deterministic environments are not trials."""
    episode = load_episode(arrays, engine)
    times, root, quat = episode["time"], episode["root"], episode["quat"]
    if engine == "Isaac":
        candidates = [e for e in report.get("summary", {}).get("episodes", []) if e.get("environment") == 0]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one Isaac environment-zero episode report")
        outcome = candidates[0]
        contract = report.get("policy_export", {})
    else:
        outcome, contract = report, report.get("contract", {})
    checks = {}
    refs = _select_series(arrays["reference_frame"], arrays, engine)
    if refs.shape != times.shape or refs.dtype.kind not in "iu":
        raise ValueError("Reference indices must be a one-dimensional integer array")
    frames = report.get("reference_frames", 0)
    fps = contract.get("policy_fps", 0)
    if not isinstance(frames, int) or frames < 2 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Reference frame count and policy frequency must be positive")
    expected_duration = frames / fps
    _check(checks, "uniform_contact_sampling", np.allclose(np.diff(times), np.median(np.diff(times)),
                                                         rtol=1e-5, atol=1e-9))
    _check(checks, "frame_zero_start", outcome.get("start_reference_frame") == 0
           and refs[0] == 0 and abs(times[0]) <= 1e-9)
    _check(checks, "monotonic_reference_without_reset", bool(np.all(np.diff(refs) >= 0))
           and bool(np.all((refs >= 0) & (refs < frames))))
    _check(checks, "completed_full_reference", outcome.get("completed_full_reference") is True
           and refs[-1] == frames - 1 and outcome.get("final_reference_frame") == frames - 1)
    _check(checks, "full_recorded_duration", abs(float(times[-1]) - expected_duration) <= 1e-6,
           {"recorded_s": float(times[-1]), "expected_s": expected_duration})
    terms = outcome.get("termination_terms")
    _check(checks, "no_early_termination", isinstance(terms, list) and not any(t != "motion_end" for t in terms)
           and outcome.get("early_terminated", False) is False
           and (outcome.get("end_reason") == "timeout" if engine == "Isaac"
                else outcome.get("termination") == "motion_end"))
    if engine == "Isaac":
        safe = (report.get("reference_state_initialization_during_evaluation") is False
                and report.get("terminal_states_captured_before_auto_reset") is True
                and contract.get("root_prescribed_during_steps") is False
                and contract.get("motion_end_hidden_teleport") is False)
    else:
        safe = (report.get("physics_stepped") is True and report.get("root_state_writes") == 1
                and report.get("hidden_resets") == 0 and contract.get("root_prescribed_during_steps") is False
                and contract.get("motion_end_hidden_teleport") is False)
    _check(checks, "physics_only_episode_no_hidden_resets", safe)

    height_gain = float(root[:, 2].max() - root[0, 2])
    _check(checks, "minimum_base_height_gain", height_gain + 1e-9 >= thresholds["min_height_gain_m"])
    ground_key, net_key = "wheel_ground_contact_force_w_n", "wheel_contact_force_w_n"
    force_key = ground_key if ground_key in arrays else (net_key if engine == "Isaac" and net_key in arrays else None)
    force_source = ("ground_only" if force_key == ground_key else
                    "net_force_proxy_including_possible_self_contact" if force_key else "unavailable")
    forces = _select_series(arrays[force_key], arrays, engine) if force_key else np.full((len(times), 2, 3), np.nan)
    events = contact_events(times, forces, threshold_n=thresholds["contact_threshold_n"],
                            support_s=.1, flight_s=thresholds["min_flight_s"])
    candidates = [e for e in events["flight_intervals"]
                  if e["no_contact_duration_s"] + 1e-9 >= thresholds["min_flight_s"]
                  and e["landing_s"] is not None and not e["right_censored"]]
    selected = max(candidates, key=lambda e: e["no_contact_duration_s"], default=None)
    _check(checks, "sustained_flight_followed_by_landing", selected is not None)
    support_fraction = None
    if events["available"]:
        supported = np.all(np.linalg.norm(forces, axis=-1) >= thresholds["contact_threshold_n"], axis=-1)
        support_fraction = tail_support_fraction(times, supported, thresholds["tail_seconds"])
    _check(checks, "stable_two_wheel_tail_support", support_fraction is not None
           and support_fraction + 1e-9 >= thresholds["min_tail_support_fraction"])
    tilt = np.degrees(np.arccos(np.clip(1. - 2. * (quat[:, 1] ** 2 + quat[:, 2] ** 2), -1., 1.)))
    tail_start = times[-1] - thresholds["tail_seconds"]
    # Include the preceding bracket: a sparse sample just before the tail must
    # not hide an unsafe orientation over the beginning of that interval.
    first_tail = max(0, int(np.searchsorted(times, tail_start, side="right")) - 1)
    max_tail_tilt = float(tilt[first_tail:].max())
    _check(checks, "stable_tail_tilt", max_tail_tilt <= thresholds["max_tail_tilt_deg"] + 1e-9)

    nonwheel = {"status": "unavailable", "reason": "No non-wheel/ground contact telemetry recorded"}
    for key in ("nonwheel_ground_contact_count", "nonwheel_ground_contact_force_norm_n"):
        if key not in arrays:
            continue
        values = np.asarray(_select_series(arrays[key], arrays, engine), dtype=float)
        if values.shape != times.shape or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Non-wheel ground contact evidence must be finite nonnegative per-sample values")
        # Counts must exclude geometric proximity-only contacts; force norms
        # below numerical noise are not treated as an impact.
        tolerance = 0. if key.endswith("count") else 1e-6
        detected = bool(np.any(values > tolerance))
        nonwheel = {"status": "contact_detected" if detected else "confirmed_absent", "field": key,
                    "maximum": float(values.max()), "tolerance": tolerance}
        _check(checks, "no_nonwheel_ground_contact", not detected)
        break
    failures = [name for name, check in checks.items() if not check["passed"]]
    return {"verdict": "fail" if failures else "pass", "fail_reasons": failures, "checks": checks,
            "environment_index": 0, "recorded_duration_s": float(times[-1]),
            "base_height_gain_m": height_gain, "selected_flight": selected,
            "contact_force_source": force_source, "contact_events": events,
            "tail_two_wheel_support_fraction": support_fraction, "max_tail_tilt_deg": max_tail_tilt,
            "nonwheel_ground_contact": nonwheel,
            "complete_contact_evidence": force_source == "ground_only" and nonwheel["status"] != "unavailable"}


def assess_runs(isaac_dir, mujoco_dir, *, actor_path=None, min_height_gain=.20, min_flight=.20,
                min_tail_support=.90, tail_seconds=.50, max_tail_tilt_deg=30., contact_threshold_n=5.,
                require_complete_contact_evidence=False):
    """Return a serializable pass/fail report without writing any files."""
    thresholds = {"min_height_gain_m": min_height_gain, "min_flight_s": min_flight,
                  "min_tail_support_fraction": min_tail_support, "tail_seconds": tail_seconds,
                  "max_tail_tilt_deg": max_tail_tilt_deg, "contact_threshold_n": contact_threshold_n}
    if (not all(np.isfinite(v) and v > 0 for v in thresholds.values())
            or min_tail_support > 1 or max_tail_tilt_deg > 180):
        raise ValueError("Thresholds must be finite and positive; support <= 1 and tilt <= 180 degrees")
    isaac_dir, mujoco_dir = Path(isaac_dir), Path(mujoco_dir)
    isaac, mujoco = read_json(isaac_dir / "report.json"), read_json(mujoco_dir / "report.json")
    provenance = check_provenance(isaac_dir, mujoco_dir, isaac, mujoco, actor_path)
    results = {}
    for key, path, report, engine in (("isaac", isaac_dir / "trajectory.npz", isaac, "Isaac"),
                                      ("mujoco", mujoco_dir / "rollout.npz", mujoco, "MuJoCo")):
        try:
            results[key] = assess_episode(read_npz(path), report, engine, thresholds)
        except (ValueError, KeyError, TypeError) as exc:
            results[key] = {"verdict": "fail", "fail_reasons": ["invalid_episode_evidence"],
                            "error": f"{type(exc).__name__}: {exc}", "complete_contact_evidence": False}
    failures = ["provenance." + name for name, check in provenance["checks"].items() if not check["passed"]]
    failures += [engine + "." + name for engine, result in results.items() for name in result["fail_reasons"]]
    complete = all(result["complete_contact_evidence"] for result in results.values())
    if require_complete_contact_evidence and not complete:
        failures.append("complete_contact_evidence_required")
    return {"schema_version": 1, "verdict": "fail" if failures else "pass", "fail_reasons": failures,
            "thresholds": thresholds, "provenance": provenance, **results,
            "complete_contact_evidence": complete, "hardware_ready": False,
            "require_complete_contact_evidence": bool(require_complete_contact_evidence),
            "statistical_robustness_benchmark": False,
            "limitations": [
                "One deterministic frame-zero episode per engine; not a statistical success rate or hardware validation.",
                "The longest qualifying flight with a recorded landing is used; earlier short bounces do not count.",
                "Height is measured base-link rise from the initial pose, not whole-robot COM height.",
                "Isaac net-force telemetry is an explicitly labeled proxy when ground-only telemetry is absent.",
                "Unavailable non-wheel ground-contact telemetry is disclosed, never interpreted as confirmed absence.",
                "A pass with complete_contact_evidence=false is limited to the recorded contact channels.",
                "This assessor never relaxes termination predicates; full duration and original non-early terminal reports are required.",
                "Support is duration-weighted with held samples; landing detection inherits the sensor sample-rate limit.",
            ]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-dir", type=Path, required=True)
    parser.add_argument("--mujoco-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path; never overwritten")
    parser.add_argument("--actor-path", type=Path)
    parser.add_argument("--min-height-gain", type=float, default=.20)
    parser.add_argument("--min-flight", type=float, default=.20)
    parser.add_argument("--min-tail-support", type=float, default=.90)
    parser.add_argument("--tail-seconds", type=float, default=.50)
    parser.add_argument("--max-tail-tilt-deg", type=float, default=30.)
    parser.add_argument("--contact-threshold-n", type=float, default=5.)
    parser.add_argument("--require-complete-contact-evidence", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    report = assess_runs(args.isaac_dir, args.mujoco_dir, actor_path=args.actor_path,
                         min_height_gain=args.min_height_gain, min_flight=args.min_flight,
                         min_tail_support=args.min_tail_support, tail_seconds=args.tail_seconds,
                         max_tail_tilt_deg=args.max_tail_tilt_deg, contact_threshold_n=args.contact_threshold_n,
                         require_complete_contact_evidence=args.require_complete_contact_evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"verdict": report["verdict"], "fail_reasons": report["fail_reasons"]}))
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
