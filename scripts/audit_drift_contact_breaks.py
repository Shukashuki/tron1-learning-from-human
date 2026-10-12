"""Recorded-only gate decomposition; no simulation, actor, phase fit or gap bridging."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from assess_drift import sideslip, THRESHOLDS

ASSESSOR_SHA256 = "a2214c47d0f8b262fb8f5b70a709908ef1d7487fd2b1393becb8ad8dfc672fa5"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decompose(time, speed, beta, counts, slip, normal_forces):
    t, speed, beta, counts, slip, forces = map(np.asarray, (time, speed, beta, counts, slip, normal_forces))
    n = len(t)
    if (t.shape != (n,) or speed.shape != (n,) or beta.shape != (n,)
            or counts.shape != (n, 2) or slip.shape != (n, 2) or forces.shape != (n, 2)
            or not all(np.isfinite(x).all() for x in (t, speed, beta, counts, slip, forces))
            or not np.allclose(np.diff(t), .005, rtol=0, atol=1e-9)
            or np.any(counts < 0) or np.any(counts != np.floor(counts))):
        raise ValueError("Finite aligned 200Hz contact evidence required")
    th = THRESHOLDS
    moving = (speed >= th["drift_min_planar_speed_m_s"]) & (np.abs(np.degrees(beta)) >= th["drift_min_abs_sideslip_deg"]) & (np.cos(beta) > 0)
    support = (counts > 0).all(1)
    mean_slip = np.abs(slip).mean(1)
    sufficient_slip = mean_slip >= th["drift_min_mean_abs_contact_slip_m_s"]
    indices = np.flatnonzero(moving & ~(support & sufficient_slip))
    return {"moving_drift_samples": int(moving.sum()),
        "support_failure_samples_while_moving_drift": int((moving & ~support).sum()),
        "slip_only_failure_samples_while_supported_moving_drift": int((moving & support & ~sufficient_slip).sum()),
        "events": [{"sample_index": int(i), "time_s": float(t[i]), "planar_speed_m_s": float(speed[i]),
            "sideslip_deg": float(np.degrees(beta[i])), "bearing_contact_count": counts[i].tolist(),
            "archived_ground_normal_force_n": forces[i].tolist(),
            "mean_contact_patch_lateral_slip_m_s": float(mean_slip[i]),
            "support_failed": not bool(support[i]), "slip_failed": not bool(sufficient_slip[i])} for i in indices]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assessor = Path(__file__).with_name("assess_drift.py")
    if sha(assessor) != ASSESSOR_SHA256:
        raise ValueError("Original assessor source changed")
    if args.output.exists():
        raise FileExistsError("Preserve prior diagnostic")
    original = sha(args.rollout)
    with np.load(args.rollout, allow_pickle=False) as a:
        if float(a["physics_dt_s"]) != .005 or float(a["policy_dt_s"]) != .02:
            raise ValueError("Original 200/50Hz required")
        speed, beta = sideslip(a["root_quat_wxyz"], a["root_com_lin_vel_w"])
        result = decompose(a["time_s"], speed, beta, a["wheel_ground_contact_count"],
            a["wheel_lateral_contact_slip_m_s"], a["wheel_ground_contact_force_w_n"][:, :, 2])
    if sha(args.rollout) != original:
        raise ValueError("Raw evidence changed")
    result.update(schema_version=1, kind="recorded_MuJoCo_contact_break_decomposition",
        rollout_sha256=original, assessor_source_sha256=ASSESSOR_SHA256,
        diagnostic_source_sha256=sha(__file__), thresholds=dict(THRESHOLDS),
        simulation_steps=0, actor_inference_calls=0, gates_changed=False, contact_gaps_bridged=False,
        limitations=["Raw saved contact evidence, not a causal test or new episode.",
            "Loss of bearing contact does not by itself identify collision geometry or an empty raw manifold.",
            "Supported low-slip intervals are separately retained; continuous support alone does not guarantee drift."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("moving_drift_samples", "support_failure_samples_while_moving_drift", "slip_only_failure_samples_while_supported_moving_drift")}))


if __name__ == "__main__":
    main()
