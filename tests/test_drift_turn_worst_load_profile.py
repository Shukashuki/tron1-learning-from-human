"""The turn's four-substep load objective uses the already verified sampler."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from training.tron1_drift import validate_task_profile

ROOT = Path(__file__).resolve().parents[1]


class DriftTurnWorstLoadProfileTests(unittest.TestCase):
    def test_candidate_changes_only_declared_load_weight_and_sampling(self):
        baseline = json.loads((ROOT / "config/drift_turn_load_reserve_task.json").read_text())
        candidate = json.loads((ROOT / "config/drift_turn_worst_load_task.json").read_text())
        checked = validate_task_profile(candidate)
        self.assertEqual(checked["wheel_ground_load_reserve"], {"weight": -1.0, "scale_fraction": .20})
        self.assertEqual(checked["wheel_ground_load_reserve_sampling"], "all_substeps_max_cost")
        self.assertEqual(set(candidate) - set(baseline), {"wheel_ground_load_reserve_sampling"})
        for key in baseline:
            if key == "wheel_ground_load_reserve":
                self.assertEqual(candidate[key]["scale_fraction"], baseline[key]["scale_fraction"])
            elif key != "description":
                self.assertEqual(candidate[key], baseline[key], key)

    def test_rejects_unsupported_weights_and_competing_substep_owner(self):
        candidate = json.loads((ROOT / "config/drift_turn_worst_load_task.json").read_text())
        candidate["wheel_ground_load_reserve"]["weight"] = -2.0
        with self.assertRaisesRegex(ValueError, "frozen -1/.20"):
            validate_task_profile(candidate)
        candidate["wheel_ground_load_reserve"]["weight"] = -1.0
        candidate["wheel_vertical_velocity_weight"] = -1.0
        candidate["wheel_vertical_velocity_std_m_s"] = .02
        candidate["wheel_vertical_velocity_sampling"] = "all_substeps_mean"
        with self.assertRaisesRegex(ValueError, "second substep owner"):
            validate_task_profile(candidate)


if __name__ == "__main__":
    unittest.main()
