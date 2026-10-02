"""Compare a SAME-actor Isaac/MuJoCo tracking episode without re-simulation.

The engine-to-engine RMSE uses only the shared valid time interval of Isaac
environment zero and the single MuJoCo episode. Full-episode survival, height,
tilt and contact events remain separate: surviving is not equivalent to jumping.
Repeated identical Isaac environments do not form a robustness probability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def require(mapping, key, context):
    if key not in mapping:
        raise ValueError(f"Missing {context}.{key}")
    return mapping[key]


def validate_provenance(isaac_dir, mujoco_dir, isaac, mujoco):
    policy = require(isaac, "policy_export", "Isaac report")
    isaac_motion = require(isaac, "motion_file_sha256", "Isaac report")
    mujoco_motion = require(mujoco, "motion_sha256", "MuJoCo report")
    if not isaac_motion or isaac_motion != mujoco_motion:
        raise ValueError("Reference motion SHA256 mismatch")
    if require(policy, "reference_file_sha256", "Isaac policy contract") != isaac_motion:
        raise ValueError("Isaac policy contract/reference SHA256 mismatch")
    if require(policy, "actor_includes_observation_normalizer", "Isaac policy contract") is not True:
        raise ValueError("Isaac actor must include its trained observation normalizer")
    actor_name = Path(require(policy, "actor_file", "Isaac policy contract"))
    if actor_name.is_absolute() or ".." in actor_name.parts:
        raise ValueError("Actor file must reside within the supplied Isaac evaluation directory")
    actor_path = isaac_dir / actor_name
    if not actor_path.is_file():
        raise FileNotFoundError(f"Missing Isaac actor file: {actor_path}")
    actor_hash = sha256(actor_path)
    if actor_hash != require(mujoco, "policy_sha256", "MuJoCo report"):
        raise ValueError("Actor policy SHA256 mismatch")
    if require(mujoco, "policy_kind", "MuJoCo report") != "normalized_torchscript_actor":
        raise ValueError("MuJoCo output is not a deployed normalized TorchScript actor")
    other_contract = require(mujoco, "contract", "MuJoCo report")
    for contract, name in ((policy, "Isaac policy contract"), (other_contract, "MuJoCo policy contract")):
        if require(contract, "actor_dim", name) != 51 or require(contract, "action_dim", name) != 8:
            raise ValueError(f"{name} must declare 51 observations and 8 actions")
    leg_names = require(policy, "leg_joint_names", "Isaac policy contract")
    if len(leg_names) != 6 or len(set(leg_names)) != 6 or any(name.startswith("wheel_") for name in leg_names):
        raise ValueError("Exactly six unique leg joint names are required")
    if require(other_contract, "leg_joint_names", "MuJoCo policy contract") != leg_names:
        raise ValueError("Policy leg action order differs across evaluators")
    verification = require(mujoco, "isaac_observation_validation", "MuJoCo report")
    if not isinstance(verification, dict) or verification.get("status") != "passed":
        raise ValueError("51D Isaac observation reconstruction verification did not pass")
    if require(verification, "observations_checked", "Observation verification") < 1:
        raise ValueError("Observation verification checked no samples")
    trajectory_hash = sha256(isaac_dir / "trajectory.npz")
    if require(verification, "trajectory_sha256", "Observation verification") != trajectory_hash:
        raise ValueError("Observation verification trajectory SHA256 mismatch")
    maximum = float(require(verification, "max_abs_error", "Observation verification"))
    if not np.isfinite(maximum) or maximum < 0:
        raise ValueError("Observation verification has an invalid maximum error")
    return {"reference_sha256": isaac_motion, "actor_sha256": actor_hash,
            "isaac_actor_file": str(actor_path.resolve()),
            "isaac_trajectory_sha256": trajectory_hash,
            "mujoco_trajectory_sha256": sha256(mujoco_dir / "rollout.npz"),
            "isaac_report_sha256": sha256(isaac_dir / "report.json"),
            "mujoco_report_sha256": sha256(mujoco_dir / "report.json"),
            "actor_dim": 51, "action_dim": 8, "observation_verification": verification,
            "leg_joint_order": leg_names}


def load_episode(arrays, engine):
    times = np.asarray(require(arrays, "time_s", f"{engine} trajectory"), dtype=float)
    names = np.asarray(require(arrays, "joint_names", f"{engine} trajectory")).astype(str).tolist()
    if len(names) != 8 or len(set(names)) != 8:
        raise ValueError(f"{engine} must have eight unique named joints")
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"{engine} timestamps must be finite, strictly increasing and nonempty")
    observations = np.asarray(require(arrays, "action_observation", f"{engine} trajectory"))
    if engine == "Isaac":
        root = np.asarray(require(arrays, "root_position_m", "Isaac trajectory"))
        quat = np.asarray(require(arrays, "root_quaternion_wxyz", "Isaac trajectory"))
        joints = np.asarray(require(arrays, "joint_position_rad", "Isaac trajectory"))
        valid = np.asarray(require(arrays, "valid_mask", "Isaac trajectory"))
        if root.ndim != 3 or root.shape[0] != len(times) or root.shape[-1] != 3 or root.shape[1] < 1:
            raise ValueError("Isaac root trajectory must be [time, environment, 3]")
        environments = root.shape[1]
        if (quat.shape != (len(times), environments, 4) or joints.shape != (len(times), environments, 8)
                or valid.shape != (len(times), environments) or valid.dtype.kind != "b"
                or observations.shape != (len(times), environments, 51)):
            raise ValueError("Isaac valid-mask, state or 51D observation shapes are inconsistent")
        mask = valid[:, 0]
        prefix = np.flatnonzero(mask)
        if len(prefix) < 2 or not np.array_equal(prefix, np.arange(len(prefix))):
            raise ValueError("Isaac environment0 needs a contiguous valid first-episode prefix")
        root, quat, joints, times = root[mask, 0], quat[mask, 0], joints[mask, 0], times[mask]
        obs = observations[mask, 0]
    elif engine == "MuJoCo":
        root = np.asarray(require(arrays, "root_pos", "MuJoCo trajectory"))
        quat = np.asarray(require(arrays, "root_quat_wxyz", "MuJoCo trajectory"))
        joints = np.asarray(require(arrays, "joint_pos", "MuJoCo trajectory"))
        environments = 1
        if observations.ndim != 2 or observations.shape[1] != 51 or not len(observations):
            raise ValueError("MuJoCo requires a nonempty [policy time, 51] observation array")
        obs = observations
    else:
        raise ValueError(f"Unsupported engine {engine}")
    if root.shape != (len(times), 3) or quat.shape != (len(times), 4) or joints.shape != (len(times), 8):
        raise ValueError(f"{engine} state array shapes disagree")
    if not all(np.isfinite(value).all() for value in (root, quat, joints, obs)):
        raise ValueError(f"{engine} valid first-episode data must be finite")
    norms = np.linalg.norm(quat, axis=1)
    if np.any(norms < 1e-8) or not np.allclose(norms, 1., atol=1e-3):
        raise ValueError(f"{engine} root quaternions are not normalized")
    quat = quat / norms[:, None]
    return {"time": times, "root": root, "quat": quat, "joints": joints,
            "names": names, "environments": environments}


def common_window_rmse(isaac, mujoco, leg_names):
    """Exact time integral of squared piecewise-linear trajectory difference.

    This neither extrapolates beyond a terminal state nor overweights the
    denser MuJoCo telemetry. Position RMSE is Euclidean XYZ; leg RMSE averages
    six named joint errors. Wheel spin is deliberately excluded.
    """
    start = float(max(isaac["time"][0], mujoco["time"][0]))
    stop = float(min(isaac["time"][-1], mujoco["time"][-1]))
    if stop <= start:
        raise ValueError("No positive-duration common evaluation window")
    knots = np.unique(np.r_[start, stop, isaac["time"], mujoco["time"]])
    knots = knots[(knots >= start) & (knots <= stop)]

    def interpolate(episode, values):
        return np.column_stack([np.interp(knots, episode["time"], values[:, col])
                                for col in range(values.shape[1])])

    def leg_values(episode):
        try:
            return episode["joints"][:, [episode["names"].index(name) for name in leg_names]]
        except ValueError as exc:
            raise ValueError("Trajectory is missing a required named leg joint") from exc

    root_delta = interpolate(isaac, isaac["root"]) - interpolate(mujoco, mujoco["root"])
    joint_delta = interpolate(isaac, leg_values(isaac)) - interpolate(mujoco, leg_values(mujoco))
    widths = np.diff(knots)[:, None]

    def mean_square(delta):
        left, right = delta[:-1], delta[1:]
        return np.sum(widths * (left * left + left * right + right * right) / 3., axis=0) / (stop - start)

    position_mse, joint_mse = mean_square(root_delta), mean_square(joint_delta)
    return {"start_s": start, "end_s": stop, "duration_s": stop - start,
            "isaac_environment_index": 0, "interpolation_knots": len(knots),
            "alignment": "shared simulation clock; no phase shifting, DTW, spatial alignment, or extrapolation",
            "integration": "exact duration-weighted squared difference of linearly interpolated trajectories",
            "base_position_rmse_m": float(np.sqrt(position_mse.sum())),
            "base_z_rmse_m": float(np.sqrt(position_mse[2])),
            "base_axis_rmse_m": np.sqrt(position_mse).tolist(),
            "leg_joint_rmse_rad": float(np.sqrt(joint_mse.mean())),
            "leg_joint_rmse_rad_by_name": dict(zip(leg_names, np.sqrt(joint_mse).tolist())),
            "leg_joint_order": leg_names, "wheel_angles_compared": False,
            "initial_common_base_position_difference_m": float(np.linalg.norm(root_delta[0])),
            "initial_common_leg_joint_rmse_rad": float(np.sqrt(np.mean(joint_delta[0] ** 2)))}


def full_episode_metrics(episode, outcome, engine):
    root, quat = episode["root"], episode["quat"]
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * np.sum(quat[:, 1:3] ** 2, axis=1), -1, 1)))
    contact = require(outcome, "contact", f"{engine} episode report")
    if not isinstance(contact, dict) or require(contact, "available", f"{engine} contact") is not True:
        raise ValueError(f"{engine} contact event data is unavailable; cannot silently infer flight")
    for key in ("flight_intervals", "first_takeoff_s", "first_landing_s", "max_flight_duration_s"):
        require(contact, key, f"{engine} contact")
    full_clip = require(outcome, "completed_full_reference", f"{engine} episode report")
    landed = require(outcome, "jump_and_landing_detected", f"{engine} episode report")
    if not isinstance(full_clip, bool) or not isinstance(landed, bool):
        raise ValueError(f"{engine} full-clip and contact-jump outcomes must be booleans")
    return {"environment_index": 0, "recorded_samples": len(episode["time"]),
            "observed_start_s": float(episode["time"][0]), "observed_end_s": float(episode["time"][-1]),
            "completed_full_reference": full_clip,
            "termination": outcome.get("termination", outcome.get("end_reason")),
            "termination_terms": require(outcome, "termination_terms", f"{engine} episode report"),
            "base_link_initial_height_m": float(root[0, 2]),
            "base_link_peak_height_m": float(root[:, 2].max()),
            "base_link_height_gain_m": float(root[:, 2].max() - root[0, 2]),
            "max_base_tilt_deg": float(tilt.max()), "final_base_tilt_deg": float(tilt[-1]),
            "reference_height_rmse_m": require(outcome, "reference_height_rmse_m", f"{engine} episode report"),
            "contact_flight_detected": bool(contact["flight_intervals"]),
            "contact_landing_detected": contact["first_landing_s"] is not None,
            "jump_and_landing_detected": landed,
            "full_clip_with_detected_jump_and_landing": bool(full_clip and landed),
            "contact": contact}


def compare(isaac_dir, mujoco_dir, output_dir):
    isaac_dir, mujoco_dir, output_dir = (Path(path).resolve() for path in (isaac_dir, mujoco_dir, output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Preserving existing comparison results: {output_dir}")
    isaac = read_json(isaac_dir / "report.json")
    mujoco = read_json(mujoco_dir / "report.json")
    if require(isaac, "simulator", "Isaac report") != "IsaacLab/PhysX":
        raise ValueError("Expected an IsaacLab/PhysX evaluation report")
    if require(mujoco, "engine", "MuJoCo report") != "MuJoCo":
        raise ValueError("Expected a MuJoCo policy evaluation report")
    provenance = validate_provenance(isaac_dir, mujoco_dir, isaac, mujoco)
    isaac_episode = load_episode(read_npz(isaac_dir / "trajectory.npz"), "Isaac")
    mujoco_episode = load_episode(read_npz(mujoco_dir / "rollout.npz"), "MuJoCo")
    if set(isaac_episode["names"]) != set(mujoco_episode["names"]):
        raise ValueError("Isaac and MuJoCo do not contain the same joint names")
    outcomes = require(require(isaac, "summary", "Isaac report"), "episodes", "Isaac summary")
    zero = [item for item in outcomes if item.get("environment") == 0]
    if len(zero) != 1:
        raise ValueError("Expected exactly one Isaac environment0 episode outcome")
    common = common_window_rmse(isaac_episode, mujoco_episode, provenance["leg_joint_order"])
    result = {
        "schema_version": 1, "status": "compared", "method": "same normalized PPO actor sim2sim trajectory comparison",
        "isaac_directory": str(isaac_dir), "mujoco_directory": str(mujoco_dir),
        "provenance": provenance, "common_window": common,
        "isaac_env0": full_episode_metrics(isaac_episode, zero[0], "Isaac"),
        "mujoco": full_episode_metrics(mujoco_episode, mujoco, "MuJoCo"),
        "isaac_recorded_environments": isaac_episode["environments"],
        "comparison_environment_index": 0, "hardware_ready": False,
        "statistical_robustness_benchmark": False,
        "limitations": [
            "Common-window errors compare the two engines, not either engine against the reference.",
            "Comparison stops at the earlier episode endpoint; no frozen or reset tail is included in RMSE.",
            "Isaac environment0 is used. Repeated same-initial-state deterministic environments are not independent robustness trials or a statistical success probability.",
            "Full-reference survival, measured height gain, flight, landing and stable recovery are distinct outcomes; none alone proves a successful safe jump.",
            "Contact events retain each evaluator's original sampling rate and definition: Isaac wheel net forces may include self-contact; MuJoCo wheel-ground forces exclude it.",
            "Reference-height errors and full-episode extrema cover each run's own observed duration, unlike the shared-window sim2sim RMSE.",
            "Contact solvers and joint velocity limits remain simulator-specific; this is not hardware validation.",
        ],
    }
    # No input is changed; write only after all required checks succeed.
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-dir", type=Path, required=True)
    parser.add_argument("--mujoco-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = compare(args.isaac_dir, args.mujoco_dir, args.output_dir)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Tracking comparison failed: {exc}\n")
    common = report["common_window"]
    print(json.dumps({"status": report["status"], "common_duration_s": common["duration_s"],
                      "base_position_rmse_m": common["base_position_rmse_m"],
                      "base_z_rmse_m": common["base_z_rmse_m"],
                      "leg_joint_rmse_rad": common["leg_joint_rmse_rad"],
                      "isaac_full_clip": report["isaac_env0"]["completed_full_reference"],
                      "mujoco_full_clip": report["mujoco"]["completed_full_reference"],
                      "isaac_jump_and_landing": report["isaac_env0"]["jump_and_landing_detected"],
                      "mujoco_jump_and_landing": report["mujoco"]["jump_and_landing_detected"]}, indent=2))


if __name__ == "__main__":
    main()
