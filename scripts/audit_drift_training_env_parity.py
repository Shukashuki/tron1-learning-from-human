"""Require exact 50/200 Hz rollout parity before any MuJoCo policy training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from drift_mujoco_training_env import MujocoDriftEnv
from eval_tracking_mujoco import load_torchscript, sha256


def replay(env, policy, baseline):
    with np.load(baseline, allow_pickle=False) as archive:
        saved = {name: archive[name].copy() for name in archive.files}
    initial = env.reset()
    np.testing.assert_allclose(initial, saved["action_observation"][0], rtol=0, atol=1e-7)
    max_qpos = max_torque = max_force = max_observation = 0.
    exact_counts = exact_actions = 0
    steps = 0
    for frame in range(len(saved["actions_clipped"])):
        observation = env.observation()
        max_observation = max(max_observation, float(np.max(np.abs(
            observation - saved["action_observation"][frame]))))
        raw = policy(observation)
        action = np.clip(raw, *env.contract["raw_action_clip"])
        exact_actions += int(np.array_equal(action, saved["actions_clipped"][frame]))
        _, done, info = env.step(raw)
        for substep, sample in enumerate(info["substeps"], 1):
            index = frame * env.contract["decimation"] + substep
            max_qpos = max(max_qpos, float(np.max(np.abs(sample["qpos"] - saved["qpos"][index]))))
            max_torque = max(max_torque, float(np.max(np.abs(
                sample["joint_torque_nm"] - saved["joint_torque_nm"][index]))))
            max_force = max(max_force, float(np.max(np.abs(
                sample["wheel_ground_contact_force_w_n"]
                - saved["wheel_ground_contact_force_w_n"][index]))))
            exact_counts += int(np.array_equal(sample["wheel_ground_contact_count"],
                                               saved["wheel_ground_contact_count"][index]))
            if not np.isclose(sample["time_s"], saved["time_s"][index], rtol=0, atol=1e-10):
                raise AssertionError(f"Physical time mismatch at sample {index}")
            steps += 1
        if done and frame + 1 != len(saved["actions_clipped"]):
            raise AssertionError("Environment terminated earlier than the evaluator")
    if (max_qpos > 1e-10 or max_torque > 1e-10 or max_force > 1e-7
            or max_observation > 1e-7 or exact_counts != steps
            or exact_actions != len(saved["actions_clipped"])):
        raise AssertionError("Training environment does not reproduce frozen evaluator rollout")
    return {"policy_steps": len(saved["actions_clipped"]), "physical_steps": steps,
            "max_qpos_abs_error": max_qpos, "max_torque_abs_error_nm": max_torque,
            "max_ground_force_abs_error_n": max_force,
            "max_actor_observation_abs_error": max_observation,
            "exact_action_count": exact_actions, "exact_contact_count_samples": exact_counts,
            "baseline_rollout_sha256": sha256(baseline)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--baseline-rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserving existing parity audit")
    env = MujocoDriftEnv(args.motion_file, args.model, args.contract)
    result = replay(env, load_torchscript(args.policy), args.baseline_rollout)
    result.update(kind="full_rollout_mujoco_training_environment_parity",
                  policy_sha256=sha256(args.policy), motion_sha256=sha256(args.motion_file),
                  model_sha256=sha256(args.model), contract_sha256=sha256(args.contract),
                  environment_source_sha256=sha256(Path(__file__).with_name("drift_mujoco_training_env.py")),
                  audit_source_sha256=sha256(__file__),
                  limitations=["This proves baseline evaluator parity, not that a newly trained actor transfers to Isaac."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
