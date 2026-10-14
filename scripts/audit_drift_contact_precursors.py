"""Read-only 200 Hz wheel-height precursors to MuJoCo contact losses.

This checks whether a future physical-rate feedback layer would have advance
information. Wheel-link height is a proxy, not cylinder-plane clearance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval_tracking_mujoco import sha256


def quantiles(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None
    return np.quantile(values, [.05, .5, .95]).tolist()


def analyze(arrays, *, start_s=1.44, end_s=4.):
    time = arrays["time_s"]
    z = arrays["wheel_pos"][:, :, 2]
    force = arrays["wheel_ground_contact_force_w_n"][:, :, 2]
    count = arrays["wheel_ground_contact_count"]
    if (z.shape != force.shape or count.shape != force.shape or z.shape[1] != 2
            or not np.allclose(np.diff(time), .005, rtol=0, atol=1e-9)):
        raise ValueError("Aligned 200 Hz two-wheel physical arrays required")
    # Backward difference only: a pre-gap statistic must not see the gap step.
    rate = np.zeros_like(z)
    rate[1:] = np.diff(z, axis=0) / np.diff(time)[:, None]
    core = (time >= start_s - 1e-8) & (time <= end_s + 1e-8)
    result = {}
    for side, wheel in (("L", 0), ("R", 1)):
        supported = (count[:, wheel] > 0) & (force[:, wheel] >= 5.)
        onset = np.flatnonzero(core & ~supported & np.r_[True, supported[:-1]])
        missed = np.flatnonzero(core & ~supported)
        prior = onset[onset >= 4] - 1
        stable = np.flatnonzero(core & supported)
        p90_z = float(np.quantile(z[stable, wheel], .9)) if len(stable) else None
        p90_vz = float(np.quantile(rate[stable, wheel], .9)) if len(stable) else None
        result[side] = {
            "unsupported_samples": int(len(missed)), "gap_onsets": int(len(onset)),
            "gap_onset_times_s": time[onset].tolist(),
            "stable_wheel_link_height_p05_p50_p95_m": quantiles(z[stable, wheel]),
            "stable_wheel_link_vz_p05_p50_p95_m_s": quantiles(rate[stable, wheel]),
            "one_step_before_onset_height_p05_p50_p95_m": quantiles(z[prior, wheel]),
            "one_step_before_onset_vz_p05_p50_p95_m_s": quantiles(rate[prior, wheel]),
            "one_step_before_onset_normal_force_p05_p50_p95_n": quantiles(force[prior, wheel]),
            "fraction_onsets_with_prior_height_above_stable_p90": (
                float(np.mean(z[prior, wheel] > p90_z)) if len(prior) else None),
            "fraction_onsets_with_prior_vz_above_stable_p90": (
                float(np.mean(rate[prior, wheel] > p90_vz)) if len(prior) else None),
            "unsupported_with_nonzero_bearing_count": int(np.sum(count[missed, wheel] > 0)),
        }
    return {"core_time_s": [start_s, end_s], "sides": result,
            "limitations": ["Read-only descriptive audit; no controller intervention or causal claim.",
                            "Wheel-link center height is not signed cylinder-plane clearance.",
                            "A predictor must be validated on held-out episodes before deployment."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserving previous contact precursor audit")
    with np.load(args.rollout, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    result = analyze(arrays)
    result["rollout_sha256"] = sha256(args.rollout)
    result["audit_source_sha256"] = sha256(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
