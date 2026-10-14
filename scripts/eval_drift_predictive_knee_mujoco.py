"""Evaluation-only 200 Hz predictive wheel-height knee feedback for WF drift.

The normalized 50 Hz actor, observation, reference, actuator envelope, model,
friction and frozen drift assessor are unchanged. A causal, bounded physical-
rate knee target correction is added before the original DC-motor/PD law. This
is a MuJoCo feasibility test, not sim2sim until the *same* outer actor and
inner feedback law are independently implemented and tested in Isaac.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from drift_mujoco_training_env import MujocoDriftEnv
from eval_tracking_mujoco import (action_torques, load_torchscript, sha256,
                                  training_termination_terms)


class PredictiveKneeFeedback:
    def __init__(self, *, left_threshold_m, right_threshold_m,
                 lookahead_s, gain, max_bias_rad, max_change_rad_per_step):
        values = [left_threshold_m, right_threshold_m, lookahead_s,
                  gain, max_bias_rad, max_change_rad_per_step]
        if (not np.isfinite(values).all() or not .12 <= left_threshold_m <= .14
                or not .12 <= right_threshold_m <= .14
                or not 0. <= lookahead_s <= .03 or not 0. <= gain <= 3.
                or not 0. <= max_bias_rad <= .05
                or not 0. <= max_change_rad_per_step <= .05):
            raise ValueError("Invalid bounded predictive knee feedback settings")
        self.thresholds = np.array([left_threshold_m, right_threshold_m])
        self.lookahead_s = float(lookahead_s)
        self.gain = float(gain)
        self.max_bias = float(max_bias_rad)
        self.max_change = float(max_change_rad_per_step)
        self.bias = np.zeros(2)
        self.previous_height = None

    def reset(self):
        self.bias[:] = 0.
        self.previous_height = None

    def modify(self, base_action, wheel_link_height_m):
        action = np.asarray(base_action, dtype=float)
        z = np.asarray(wheel_link_height_m, dtype=float)
        if action.shape != (8,) or z.shape != (2,) or not np.isfinite(action).all() or not np.isfinite(z).all():
            raise ValueError("Expected finite eight actions and two wheel heights")
        velocity = (np.zeros(2) if self.previous_height is None
                    else (z - self.previous_height) / .005)
        predicted = z + self.lookahead_s * np.maximum(velocity, 0.)
        # |d wheel-link z / d own knee q| is about 0.24-0.25 m/rad in the
        # audited turn core. Left negative/right positive action lowers wheels.
        desired = np.clip(self.gain * np.maximum(predicted - self.thresholds, 0.) / .245,
                          0., self.max_bias)
        self.bias += np.clip(desired - self.bias, -self.max_change, self.max_change)
        effective = action.copy()
        effective[4] -= self.bias[0]
        effective[5] += self.bias[1]
        self.previous_height = z.copy()
        return effective, {"wheel_link_height_m": z.copy(),
                           "wheel_link_vz_backward_m_s": velocity.copy(),
                           "predicted_feedback_signal_m": predicted.copy(),
                           "knee_bias_magnitude_rad": self.bias.copy()}

    def config(self):
        return {"signal_kind": "wheel_link_height_m",
                "left_threshold_m": float(self.thresholds[0]),
                "right_threshold_m": float(self.thresholds[1]),
                "lookahead_s": self.lookahead_s, "gain": self.gain,
                "max_bias_rad": self.max_bias,
                "max_change_rad_per_step": self.max_change,
                "wheel_height_per_knee_rad_assumption_m": .245}


def evaluate(policy, motion, model, contract, output, controller, *, actor_file):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserving previous controller rollout: {output}")
    env = MujocoDriftEnv(motion, model, contract)
    if hasattr(controller, "attach"):
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
                         "knee_bias_magnitude_rad": feedback["knee_bias_magnitude_rad"].copy(),
                         "predicted_feedback_signal_m": feedback["predicted_feedback_signal_m"].copy()})

    blank = np.zeros(8)
    record(blank, blank, blank, {"knee_bias_magnitude_rad": np.zeros(2),
                                "predicted_feedback_signal_m": env.state["wheel_pos"][:, 2]})
    termination = "motion_end"
    terms = []
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
            effective, feedback = controller.modify(base, env.state["wheel_pos"][:, 2])
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
    report = {"kind": "evaluation_only_predictive_knee_feedback_not_paired_sim2sim",
              "engine": "MuJoCo", "engine_version": mujoco.__version__,
              "termination": termination, "termination_terms": terms,
              "completed_full_reference": termination == "motion_end",
              "policy_steps": len(decisions), "physics_steps": len(physical) - 1,
              "policy_sha256": sha256(actor_file), "motion_sha256": sha256(motion),
              "model_sha256": sha256(model), "contract_sha256": sha256(contract),
              "controller": controller.config(), "controller_source_sha256": sha256(__file__),
              "max_abs_knee_bias_rad": np.max(np.abs(arrays["knee_bias_magnitude_rad"]), axis=0).tolist(),
              "root_state_writes": 1, "hidden_resets": 0,
              "limitations": ["MuJoCo-only feedback feasibility; identical physical-rate law is not yet implemented in Isaac.",
                              "The original actor observation remains unchanged; corrected substep targets are an inner controller.",
                              "No contact gaps are bridged or ignored by the frozen assessor."]}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "rollout.npz", **arrays)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


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
    controller = PredictiveKneeFeedback(left_threshold_m=args.left_threshold_m,
                                        right_threshold_m=args.right_threshold_m,
                                        lookahead_s=args.lookahead_s, gain=args.gain,
                                        max_bias_rad=args.max_bias_rad,
                                        max_change_rad_per_step=args.max_change_rad_per_step)
    report = evaluate(load_torchscript(args.policy), args.motion_file, args.model,
                      args.contract, args.output_dir, controller, actor_file=args.policy)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
