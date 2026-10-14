"""The contact-robustness candidate changes training restitution only."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.tron1_drift import validate_task_profile, robot_restitution_event_settings


class RestitutionDrTests(unittest.TestCase):
    def setUp(self):
        self.original = json.loads((ROOT / "config/drift_turn_worst_load_task.json").read_text())
        self.candidate = json.loads((ROOT / "config/drift_turn_restitution_dr_task.json").read_text())

    def test_only_training_restitution_and_description_change(self):
        validate_task_profile(self.candidate)
        changed = copy.deepcopy(self.candidate)
        self.assertEqual(changed.pop("training_robot_restitution_range"), [0., .3])
        changed["description"] = self.original["description"]
        self.assertEqual(changed, self.original)

    def test_training_draw_and_nominal_evaluation_are_distinct(self):
        self.assertEqual(robot_restitution_event_settings(self.candidate, eval_mode=False),
                         {"restitution_range": (0., .3), "num_buckets": 32})
        self.assertEqual(robot_restitution_event_settings(self.candidate, eval_mode=True),
                         {"restitution_range": (0., 0.), "num_buckets": 1})
        self.assertIsNone(robot_restitution_event_settings(self.original, eval_mode=False))

    def test_rejects_malformed_or_nonzero_nominal_training_range(self):
        for bad in (None, (0., .3), [0.], [0., .3, .4], [.01, .3], [0., .04],
                    [0., .51], [0., True], [0., float("nan")]):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "restitution range"):
                validate_task_profile({**self.candidate, "training_robot_restitution_range": bad})


if __name__ == "__main__":
    unittest.main()
