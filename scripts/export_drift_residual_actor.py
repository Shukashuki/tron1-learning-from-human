"""Export a deterministic residual actor and prove MuJoCo deployment parity.

Checkpoint weights are loaded with the exact frozen parent actor. The exported
TorchScript takes the same 51D observation and returns the same eight raw
actions as the training evaluator. A full physical replay against the saved
checkpoint evaluation must match before this actor can be sent to Isaac.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from eval_tracking_mujoco import load_torchscript, run_evaluation, sha256
from train_drift_mujoco_residual import ResidualActorCritic


class DeployResidualActor(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy.eval()

    def forward(self, observation):
        base, residual, _ = self.policy(observation)
        return base + residual


def export_checkpoint(parent, checkpoint, output):
    if output.exists():
        raise FileExistsError(f"Preserving existing actor: {output}")
    archive = torch.load(checkpoint, map_location="cpu", weights_only=False)
    manifest = archive["manifest"]
    if manifest["policy_sha256"] != sha256(parent):
        raise ValueError("Checkpoint parent actor differs from supplied parent")
    base = torch.jit.load(str(parent), map_location="cpu").eval()
    policy = ResidualActorCritic(base).cpu().eval()
    policy.load_state_dict(archive["state_dict"], strict=True)
    wrapper = DeployResidualActor(policy)
    example = torch.zeros((1, 51), dtype=torch.float32)
    traced = torch.jit.trace(wrapper, example, check_trace=True).eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output))
    return manifest, int(archive["update"])


def verify(parent, checkpoint, actor, baseline_rollout, motion_file, model, contract, output_dir):
    manifest, update = export_checkpoint(parent, checkpoint, actor)
    loaded = load_torchscript(actor)
    with np.load(baseline_rollout, allow_pickle=False) as archive:
        saved = {name: archive[name].copy() for name in archive.files}
    # Test every actual observation, not only the dummy trace input.
    original = torch.jit.load(str(parent), map_location="cpu").eval()
    trained = ResidualActorCritic(original).cpu().eval()
    trained.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"])
    with torch.no_grad():
        obs = torch.from_numpy(saved["action_observation"].astype(np.float32))
        base, mean, _ = trained(obs)
        expected = (base + mean).numpy()
    actor_error = float(np.max(np.abs(expected - np.stack([loaded(row) for row in obs.numpy()]))))
    if actor_error > 1e-6:
        raise AssertionError(f"Exported actor differs on real observations: {actor_error}")
    report = run_evaluation(loaded, motion_file, model, output_dir, policy_path=actor,
                            contract_path=contract, no_visual_mesh=True)
    with np.load(output_dir / "rollout.npz", allow_pickle=False) as archive:
        actual = {name: archive[name].copy() for name in archive.files}
    if len(actual["qpos"]) != len(saved["qpos"]):
        raise AssertionError("Exported actor episode length differs")
    qpos_error = float(np.max(np.abs(actual["qpos"] - saved["qpos"])))
    torque_error = float(np.max(np.abs(actual["joint_torque_nm"] - saved["joint_torque_nm"])))
    count_match = bool(np.array_equal(actual["wheel_ground_contact_count"],
                                      saved["wheel_ground_contact_count"]))
    if qpos_error > 1e-7 or torque_error > 1e-7 or not count_match:
        raise AssertionError("Exported actor full MuJoCo replay differs from checkpoint evaluation")
    return {"kind": "residual_actor_export_full_mujoco_deployment_parity",
            "checkpoint_update": update, "checkpoint_sha256": sha256(checkpoint),
            "parent_actor_sha256": sha256(parent), "exported_actor_sha256": sha256(actor),
            "baseline_rollout_sha256": sha256(baseline_rollout),
            "exported_rollout_sha256": sha256(output_dir / "rollout.npz"),
            "actor_max_abs_error_on_saved_observations": actor_error,
            "full_rollout_max_qpos_abs_error": qpos_error,
            "full_rollout_max_torque_abs_error_nm": torque_error,
            "all_physical_contact_counts_match": count_match,
            "full_reference_completed": report["completed_full_reference"],
            "source_manifest": manifest,
            "limitations": ["MuJoCo deployment parity only; same exported actor still requires independent Isaac evaluation."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "checkpoint", "baseline-rollout", "motion-file", "model", "contract", "actor-output", "rollout-output-dir", "audit-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.audit_output.exists():
        parser.error("Preserving existing export audit")
    result = verify(args.parent, args.checkpoint, args.actor_output,
                    args.baseline_rollout, args.motion_file, args.model,
                    args.contract, args.rollout_output_dir)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in (
        "exported_actor_sha256", "actor_max_abs_error_on_saved_observations",
        "full_rollout_max_qpos_abs_error", "all_physical_contact_counts_match")}, indent=2))


if __name__ == "__main__":
    main()
