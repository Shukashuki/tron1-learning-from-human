"""Causal finite-cylinder clearance feedback around the MuJoCo WF drift actor.

The 50 Hz actor, physical model, DC motor, reference, and frozen 200 Hz gate
remain unchanged. Only an explicit bounded 200 Hz knee-target correction is
explored; this is not paired Isaac/MuJoCo policy success.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_drift_one_step_action_sensitivity import signed_clearance
from eval_drift_predictive_knee_mujoco import evaluate
from eval_tracking_mujoco import load_torchscript, sha256


class PredictiveClearanceFeedback:
    def __init__(self, *, left_threshold_m, right_threshold_m,
                 lookahead_s, gain, max_bias_rad, max_change_rad_per_step):
        values = [left_threshold_m, right_threshold_m, lookahead_s,
                  gain, max_bias_rad, max_change_rad_per_step]
        if (not np.isfinite(values).all()
                or not all(-.005 <= value <= .005 for value in values[:2])
                or not 0. <= lookahead_s <= .03 or not 0. <= gain <= 3.
                or not 0. <= max_bias_rad <= .05
                or not 0. <= max_change_rad_per_step <= .05):
            raise ValueError("Invalid bounded finite-cylinder clearance feedback settings")
        self.thresholds = np.array(values[:2], dtype=float)
        self.lookahead_s = float(lookahead_s)
        self.gain = float(gain)
        self.max_bias = float(max_bias_rad)
        self.max_change = float(max_change_rad_per_step)
        self.env = None
        self.bias = np.zeros(2)
        self.previous_clearance = None

    def attach(self, env):
        model = env.model
        margins = [float(model.geom_margin[model.geom(name).id])
                   for name in ("floor", "wheel_L_collision", "wheel_R_collision")]
        if not np.allclose(margins, [.001, .001, .001], atol=1e-12, rtol=0):
            raise ValueError(f"Expected audited 1 mm plane/wheel margins, got {margins}")
        self.env = env

    def reset(self):
        self.bias[:] = 0.
        self.previous_clearance = None

    def modify(self, base_action, ignored_wheel_link_height):
        if self.env is None:
            raise RuntimeError("Controller must be attached to its live MuJoCo environment")
        action = np.asarray(base_action, dtype=float)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError("Expected finite eight-action actor output")
        current = np.array([signed_clearance(self.env.measured, side) for side in range(2)])
        rate = (np.zeros(2) if self.previous_clearance is None
                else (current - self.previous_clearance) / .005)
        predicted = current + self.lookahead_s * np.maximum(rate, 0.)
        desired = np.clip(self.gain * np.maximum(predicted - self.thresholds, 0.) / .245,
                          0., self.max_bias)
        self.bias += np.clip(desired - self.bias, -self.max_change, self.max_change)
        effective = action.copy()
        effective[4] -= self.bias[0]
        effective[5] += self.bias[1]
        self.previous_clearance = current.copy()
        return effective, {"signed_finite_cylinder_clearance_m": current,
                           "clearance_rate_backward_m_s": rate,
                           "predicted_feedback_signal_m": predicted,
                           "knee_bias_magnitude_rad": self.bias.copy()}

    def config(self):
        return {"signal_kind": "finite_cylinder_plane_signed_clearance_m",
                "left_threshold_m": float(self.thresholds[0]),
                "right_threshold_m": float(self.thresholds[1]),
                "lookahead_s": self.lookahead_s, "gain": self.gain,
                "max_bias_rad": self.max_bias,
                "max_change_rad_per_step": self.max_change,
                "wheel_height_per_knee_rad_assumption_m": .245,
                "compiled_floor_and_wheel_contact_margin_m": .001,
                "controller_module_source_sha256": sha256(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("policy", "motion-file", "model", "contract", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--left-threshold-m", type=float, required=True)
    parser.add_argument("--right-threshold-m", type=float, required=True)
    parser.add_argument("--lookahead-s", type=float, required=True)
    parser.add_argument("--gain", type=float, required=True)
    parser.add_argument("--max-bias-rad", type=float, required=True)
    parser.add_argument("--max-change-rad-per-step", type=float, required=True)
    args = parser.parse_args()
    controller = PredictiveClearanceFeedback(left_threshold_m=args.left_threshold_m,
                                             right_threshold_m=args.right_threshold_m,
                                             lookahead_s=args.lookahead_s, gain=args.gain,
                                             max_bias_rad=args.max_bias_rad,
                                             max_change_rad_per_step=args.max_change_rad_per_step)
    result = evaluate(load_torchscript(args.policy), args.motion_file, args.model,
                      args.contract, args.output_dir, controller, actor_file=args.policy)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
