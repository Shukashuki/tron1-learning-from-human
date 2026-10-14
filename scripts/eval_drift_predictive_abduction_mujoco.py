"""MuJoCo-only 200 Hz wheel-clearance-to-abduction drift feedback probe.

The frozen 50 Hz actor remains unchanged. Both-side positive abduction action
signs come from exact local counterfactuals, not a global physical guarantee.
This is not a paired Isaac/MuJoCo policy-transfer result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from audit_drift_one_step_action_sensitivity import signed_clearance
from drift_mujoco_training_env import MujocoDriftEnv
from eval_tracking_mujoco import (action_torques, load_torchscript, sha256,
                                  training_termination_terms)


class PredictiveAbductionFeedback:
    def __init__(self, *, threshold_m, lookahead_s, gain, max_bias_rad,
                 max_change_rad_per_step):
        values = (threshold_m, lookahead_s, gain, max_bias_rad, max_change_rad_per_step)
        if (not np.isfinite(values).all() or not 0. <= threshold_m <= .004
                or not 0. <= lookahead_s <= .03 or not 0. <= gain <= 3.
                or not 0. <= max_bias_rad <= .08
                or not 0. <= max_change_rad_per_step <= .08):
            raise ValueError("Invalid bounded abduction-feedback settings")
        self.threshold = float(threshold_m)
        self.lookahead = float(lookahead_s)
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

    def modify(self, base_action):
        if self.env is None:
            raise RuntimeError("Attach controller to the live MuJoCo environment")
        action = np.asarray(base_action, dtype=float)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError("Expected finite eight-action actor output")
        clearance = np.array([signed_clearance(self.env.measured, side) for side in range(2)])
        rate = (np.zeros(2) if self.previous_clearance is None
                else (clearance - self.previous_clearance) / .005)
        predicted = clearance + self.lookahead * np.maximum(rate, 0.)
        # Local rescued windows give ~0.005 m clearance reduction per +1 rad
        # of same-side abduction action; this is only a response scale.
        desired = np.clip(self.gain * np.maximum(predicted - self.threshold, 0.) / .005,
                          0., self.max_bias)
        self.bias += np.clip(desired - self.bias, -self.max_change, self.max_change)
        effective = action.copy()
        effective[:2] += self.bias
        self.previous_clearance = clearance.copy()
        return effective, {"signed_clearance_m": clearance,
                           "clearance_rate_backward_m_s": rate,
                           "predicted_clearance_m": predicted,
                           "abduction_bias_rad": self.bias.copy()}

    def config(self):
        return {"signal_kind": "finite_cylinder_plane_signed_clearance_m",
                "threshold_m": self.threshold, "lookahead_s": self.lookahead,
                "gain": self.gain, "max_bias_rad": self.max_bias,
                "max_change_rad_per_step": self.max_change,
                "local_clearance_reduction_per_positive_abduction_action_m_per_rad": .005,
                "compiled_floor_and_wheel_contact_margin_m": .001,
                "controller_source_sha256": sha256(__file__)}


def evaluate(policy, motion, model, contract, output, controller, *, actor_file):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserving previous controller rollout: {output}")
    env = MujocoDriftEnv(motion, model, contract)
    controller.attach(env)
    controller.reset()
    physical, decisions = [], []

    def record(torque, base_action, effective_action, feedback):
        ground = env.measured.ground_contact_slip()
        physical.append({"time_s": float(env.data.time),
                         "root_pos": env.state["root_pos"].copy(),
                         "root_com_lin_vel_w": env.state["root_com_lin_vel_w"].copy(),
                         "root_quat_wxyz": env.state["root_quat_wxyz"].copy(),
                         "wheel_pos": env.state["wheel_pos"].copy(),
                         "qpos": env.data.qpos.copy(),
                         "joint_torque_nm": torque.copy(),
                         "wheel_ground_contact_count": ground["wheel_ground_contact_count"].copy(),
                         "wheel_ground_contact_force_w_n": ground["wheel_ground_contact_force_w_n"].copy(),
                         "wheel_lateral_contact_slip_m_s": ground["wheel_lateral_contact_slip_m_s"].copy(),
                         "base_action": base_action.copy(),
                         "effective_action": effective_action.copy(),
                         "abduction_bias_rad": feedback["abduction_bias_rad"].copy(),
                         "predicted_clearance_m": feedback["predicted_clearance_m"].copy()})

    blank = np.zeros(8)
    record(blank, blank, blank, {"abduction_bias_rad": np.zeros(2),
                                "predicted_clearance_m": np.zeros(2)})
    termination, terms = "motion_end", []
    for frame in range(env.reference.frames):
        target = env.reference.sample(frame)
        observation = env.observation()
        raw = np.asarray(policy(observation), dtype=float).reshape(-1)
        if raw.shape != (8,) or not np.isfinite(raw).all():
            raise ValueError("Invalid frozen actor output")
        base = np.clip(raw, *env.contract["raw_action_clip"])
        decisions.append({"action_observation": observation.copy(),
                          "actions_raw": raw.copy(), "actions_clipped": base.copy()})
        for _ in range(env.contract["decimation"]):
            effective, feedback = controller.modify(base)
            effective = np.clip(effective, *env.contract["raw_action_clip"])
            torque = action_torques(effective, env.state["joint_pos"], env.state["joint_vel"],
                                    env.reference.q0, env.contract)
            env.data.ctrl[env.measured.aids] = torque
            mujoco.mj_step(env.model, env.data)
            mujoco.mj_forward(env.model, env.data)
            env.state = env.measured.measure()
            record(torque, base, effective, feedback)
        env.previous_action = base.copy()
        env.frame += 1
        terms = training_termination_terms(env.state, target)
        if terms:
            termination = "early_termination"
            break
    arrays = {name: np.asarray([row[name] for row in physical]) for name in physical[0]}
    arrays.update({name: np.asarray([row[name] for row in decisions]) for name in decisions[0]})
    report = {"kind": "evaluation_only_predictive_abduction_feedback_not_paired_sim2sim",
              "engine": "MuJoCo", "engine_version": mujoco.__version__,
              "termination": termination, "termination_terms": terms,
              "completed_full_reference": termination == "motion_end",
              "policy_steps": len(decisions), "physics_steps": len(physical) - 1,
              "policy_sha256": sha256(actor_file), "motion_sha256": sha256(motion),
              "model_sha256": sha256(model), "contract_sha256": sha256(contract),
              "controller": controller.config(),
              "max_abs_abduction_bias_rad": np.max(np.abs(arrays["abduction_bias_rad"]), axis=0).tolist(),
              "root_state_writes": 1, "hidden_resets": 0,
              "limitations": ["MuJoCo-only feedback feasibility; no identical Isaac physical-rate implementation yet.",
                              "The frozen actor observation remains unchanged; corrected substep targets are an inner controller.",
                              "No contact gaps are bridged or ignored by the frozen assessor."]}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "rollout.npz", **arrays)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("policy", "motion-file", "model", "contract", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threshold-m", type=float, required=True)
    parser.add_argument("--lookahead-s", type=float, required=True)
    parser.add_argument("--gain", type=float, required=True)
    parser.add_argument("--max-bias-rad", type=float, required=True)
    parser.add_argument("--max-change-rad-per-step", type=float, required=True)
    args = parser.parse_args()
    controller = PredictiveAbductionFeedback(threshold_m=args.threshold_m,
        lookahead_s=args.lookahead_s, gain=args.gain, max_bias_rad=args.max_bias_rad,
        max_change_rad_per_step=args.max_change_rad_per_step)
    result = evaluate(load_torchscript(args.policy), args.motion_file, args.model,
                      args.contract, args.output_dir, controller, actor_file=args.policy)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
