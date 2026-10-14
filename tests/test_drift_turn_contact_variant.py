"""The contact-response sensitivity contract may change exactly one field."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_drift_turn_contact_variant import variant


class ContactVariantTests(unittest.TestCase):
    def setUp(self):
        self.saved = {"task_profile": json.loads((ROOT / "config/car_drift_task.json").read_text()),
                      "reference_file_sha256": "a" * 64, "actor_file": "actor_normalized.pt",
                      "physics_dt": .005, "decimation": 4}

    def test_only_historical_mujoco_response_is_added(self):
        original = json.loads(json.dumps(self.saved))
        out = variant(self.saved, "b" * 64, "a" * 64)
        self.assertEqual(self.saved, original)
        self.assertEqual(out["task_profile"]["mujoco_contact_solref"], [0.04, 2.0])
        del out["task_profile"]["mujoco_contact_solref"]
        self.assertEqual(out, self.saved)

    def test_rejects_mismatched_reference_and_existing_override(self):
        with self.assertRaisesRegex(ValueError, "generic drift-turn"):
            variant(self.saved, "b" * 64, "c" * 64)
        self.saved["task_profile"]["mujoco_contact_solref"] = [0.04, 2.0]
        with self.assertRaisesRegex(ValueError, "generic drift-turn"):
            variant(self.saved, "b" * 64, "a" * 64)


if __name__ == "__main__":
    unittest.main()
