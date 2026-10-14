"""One-factor new-task load-reserve profile preflight, without Isaac imports."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from training.tron1_drift import validate_task_profile

ROOT = Path(__file__).resolve().parents[1]


class DriftTurnLoadReserveProfileTests(unittest.TestCase):
    def test_exact_baseline_except_one_supported_reward_and_description(self):
        baseline = json.loads((ROOT / "config/car_drift_task.json").read_text())
        candidate = json.loads((ROOT / "config/drift_turn_load_reserve_task.json").read_text())
        checked = validate_task_profile(candidate)
        self.assertEqual(checked["wheel_ground_load_reserve"],
                         {"weight": -2.0, "scale_fraction": 0.2})
        self.assertEqual(set(candidate) - set(baseline), {"wheel_ground_load_reserve"})
        self.assertEqual(set(baseline) - set(candidate), set())
        for key in baseline:
            if key != "description":
                self.assertEqual(candidate[key], baseline[key], key)


if __name__ == "__main__":
    unittest.main()
