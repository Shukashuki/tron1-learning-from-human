"""Create a matched saved MuJoCo contract for the one-factor leg-Kd diagnostic."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_tracking_mujoco import validate_contract
from training.tron1_drift import load_task_profile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def variant(saved, profile, reference_hash):
    if (saved.get("reference_file_sha256") != reference_hash or
            saved.get("actor_file") != "actor_normalized.pt" or
            saved.get("leg_kd") != 10.0 or
            "leg_kd" in saved.get("task_profile", {})):
        raise ValueError("Expected original worst-substep actor/reference and nominal Kd=10")
    baseline = copy.deepcopy(saved["task_profile"])
    changed = copy.deepcopy(profile)
    if changed.get("leg_kd") != 13.0:
        raise ValueError("Only the predeclared Kd=13 diagnostic is supported")
    del changed["leg_kd"]
    changed["description"] = baseline["description"]
    if changed != baseline:
        raise ValueError("Task profile changed beyond leg damping and metadata")
    out = copy.deepcopy(saved)
    # Isaac's audited deployment override retains the training description.
    # The descriptive diagnostic text in the input file is not a physics change.
    out["task_profile"] = copy.deepcopy(baseline)
    out["task_profile"]["leg_kd"] = 13.0
    out["leg_kd"] = 13.0
    validate_contract(out)
    restored = copy.deepcopy(out)
    restored["task_profile"] = baseline
    restored["leg_kd"] = 10.0
    if restored != saved:
        raise ValueError("Deployment contract changed beyond declared leg damping")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-contract", type=Path, required=True)
    parser.add_argument("--task-profile", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserving previous variant contract")
    saved = json.loads(args.saved_contract.read_text())
    profile = load_task_profile(args.task_profile)
    out = variant(saved, profile, digest(args.motion_file))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"kind": "evaluation_only_matched_leg_kd_13_variant",
                      "base_contract_sha256": digest(args.saved_contract),
                      "variant_contract_sha256": digest(args.output),
                      "actor_sha256_unchanged": digest(args.actor),
                      "motion_sha256_unchanged": digest(args.motion_file),
                      "task_profile_sha256": digest(args.task_profile)}, indent=2))


if __name__ == "__main__":
    main()
