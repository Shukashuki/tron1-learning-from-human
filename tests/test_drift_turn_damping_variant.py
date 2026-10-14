"""The Kd diagnostic changes the same leg damping in both contracts only."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_drift_turn_damping_variant import variant
from training.tron1_drift import validate_task_profile


class DampingVariantTests(unittest.TestCase):
    def test_profile_changes_only_kd_and_description(self):
        baseline = json.loads((ROOT / "config/drift_turn_worst_load_task.json").read_text())
        changed = json.loads((ROOT / "config/drift_turn_leg_damping13_diag.json").read_text())
        validate_task_profile(changed)
        self.assertEqual(changed["leg_kd"], 13.0)
        changed.pop("leg_kd")
        changed["description"] = baseline["description"]
        self.assertEqual(changed, baseline)

    def test_saved_actor_contract_changes_only_matched_leg_kd(self):
        saved = json.loads((ROOT / "outputs/drift-drl-turn-worst-load-20261014/eval_isaac_final/policy_contract.json").read_text())
        profile = json.loads((ROOT / "config/drift_turn_leg_damping13_diag.json").read_text())
        original = json.loads(json.dumps(saved))
        out = variant(saved, profile, saved["reference_file_sha256"])
        self.assertEqual(saved, original)
        self.assertEqual(out["leg_kd"], 13.0)
        self.assertEqual(out["task_profile"]["leg_kd"], 13.0)
        out["leg_kd"] = 10.0
        out["task_profile"] = original["task_profile"]
        self.assertEqual(out, original)


if __name__ == "__main__":
    unittest.main()
