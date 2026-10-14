"""Create an evaluation-only MuJoCo contact-response variant of a saved actor contract.

The actor, reference, controller, friction, clocks, and all acceptance gates
stay fixed. This is a contact-law sensitivity diagnostic, not retraining.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.tron1_drift import validate_task_profile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def variant(saved, actor_hash, reference_hash, *, contact_timeconst_s=.04):
    out = copy.deepcopy(saved)
    if (out.get("task_profile", {}).get("mujoco_contact_solref") is not None
            or out.get("reference_file_sha256") != reference_hash
            or out.get("actor_file") != "actor_normalized.pt"):
        raise ValueError("Expected the original generic drift-turn actor/reference contract")
    if not isinstance(actor_hash, str) or len(actor_hash) != 64:
        raise ValueError("Actor digest is required")
    out["task_profile"]["mujoco_contact_solref"] = [contact_timeconst_s, 2.0]
    validate_task_profile(out["task_profile"])
    restored = copy.deepcopy(out)
    del restored["task_profile"]["mujoco_contact_solref"]
    if restored != saved:
        raise ValueError("Contact response must be the only contract change")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-contract", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-timeconst-s", type=float, default=.04,
                        help="Evaluation-only MuJoCo contact time constant, with damping ratio fixed at 2")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserving prior contact variant")
    saved = json.loads(args.saved_contract.read_text())
    actor_hash, reference_hash = digest(args.actor), digest(args.motion_file)
    changed = variant(saved, actor_hash, reference_hash,
                      contact_timeconst_s=args.contact_timeconst_s)
    with args.output.open("x") as stream:
        json.dump(changed, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "evaluation_only_contact_response_variant",
                      "base_contract_sha256": digest(args.saved_contract),
                      "variant_contract_sha256": digest(args.output),
                      "actor_sha256_unchanged": actor_hash,
                      "reference_sha256_unchanged": reference_hash,
                      "only_change": f"task_profile.mujoco_contact_solref absent → {[args.contact_timeconst_s, 2.0]}"},
                     indent=2))


if __name__ == "__main__":
    main()
