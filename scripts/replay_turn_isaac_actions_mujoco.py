"""Diagnostic: replay Isaac's recorded actor actions open-loop in MuJoCo.

This intentionally does NOT run the policy in MuJoCo, so it cannot demonstrate
sim2sim transfer. It separates action-feedback divergence from fixed-action
physics response without modifying the reference, controller or contact law.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_tracking_mujoco as evaluator


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def applied_actions(archive):
    raw = np.asarray(archive["actions_raw"])
    used = np.asarray(archive["action_was_applied"])
    frames = np.asarray(archive["reference_frame_for_action"])
    if (raw.ndim != 3 or raw.shape[1:] != (1, 8) or
            used.shape != (len(raw), 1) or used.dtype.kind != "b" or
            frames.shape != (len(raw), 1) or len(raw) < 2 or
            used[0, 0] or not used[1:, 0].all() or
            not np.array_equal(frames[1:, 0], np.arange(len(raw) - 1)) or
            not np.isfinite(raw).all()):
        raise ValueError("Isaac actions are not one complete, contiguous first episode")
    return raw[1:, 0].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-trajectory", type=Path, required=True)
    parser.add_argument("--isaac-report", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Preserving prior diagnostic rollout")
    with np.load(args.isaac_trajectory, allow_pickle=False) as archive:
        actions = applied_actions(archive)
    report = json.loads(args.isaac_report.read_text())
    if (report.get("motion_file_sha256") != digest(args.motion_file) or
            report.get("policy_export") != json.loads(args.contract.read_text())):
        raise ValueError("Isaac's exported motion or deployment contract differs from replay inputs")
    contract = evaluator.read_contract(args.contract)
    reference = evaluator.MotionReference(args.motion_file, contract)
    if len(actions) != reference.frames:
        raise ValueError("Isaac action count differs from the full reference")
    calls = 0

    def replay(observation):
        nonlocal calls
        if calls >= len(actions) or not np.isfinite(observation).all():
            raise ValueError("Open-loop action replay exceeded the source or saw invalid state")
        action = actions[calls]
        calls += 1
        return action

    result = evaluator.run_evaluation(
        replay, args.motion_file, args.model, args.output_dir,
        policy_path=None, contact_threshold_n=5., no_visual_mesh=True,
        contract_path=args.contract,
    )
    with np.load(args.output_dir / "rollout.npz", allow_pickle=False) as archive:
        actual = archive["actions_raw"]
    if calls != len(actual) or not np.array_equal(actual, actions[:calls]):
        raise RuntimeError("MuJoCo replayed actions differ from the Isaac action prefix")
    summary = {
        "kind": "open_loop_isaac_action_replay_not_policy_transfer",
        "isaac_trajectory_sha256": digest(args.isaac_trajectory),
        "isaac_report_sha256": digest(args.isaac_report),
        "contract_sha256": digest(args.contract),
        "motion_sha256": digest(args.motion_file),
        "model_sha256": digest(args.model),
        "replayed_action_count": calls,
        "source_action_count": len(actions),
        "mujoco_rollout_sha256": digest(args.output_dir / "rollout.npz"),
        "mujoco_report_sha256": digest(args.output_dir / "report.json"),
        "termination": result["termination"],
        "no_actor_calls_in_mujoco": True,
        "actions_equal_source_prefix": True,
    }
    (args.output_dir / "open_loop_diagnostic.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
