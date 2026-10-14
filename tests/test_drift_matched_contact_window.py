"""Pure reductions for four-step matched-state contact diagnostics."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_drift_matched_contact_window import HORIZON, summarize


def window(support, errors, start=1.44):
    return {
        "start_time_s": start,
        "initial_wheel_position_max_abs_error_m": 1e-7,
        "substeps": [
            {"isaac_both_wheels_supported": True,
             "mujoco_both_wheels_supported": supported,
             "isaac_wheel_z_m": [.127, .127],
             "mujoco_wheel_z_m": [.127 + error, .127 - error],
             "root_position_error_m": error}
            for supported, error in zip(support, errors)
        ],
    }


class MatchedContactWindowSummaryTests(unittest.TestCase):
    def test_support_reduced_per_step_and_uninterrupted_window(self):
        rows = [window([True] * HORIZON, [.001] * HORIZON),
                window([True, False, True, True], [.002] * HORIZON, start=1.46)]
        result = summarize(rows)
        self.assertEqual(result["policy_windows"], 2)
        self.assertEqual(result["native_steps"], 8)
        self.assertEqual(result["isaac_all_four_steps_supported_window_fraction"], 1.)
        self.assertEqual(result["mujoco_two_wheel_support_fraction"], 7 / 8)
        self.assertEqual(result["mujoco_all_four_steps_supported_window_fraction"], .5)
        self.assertEqual(result["mujoco_two_wheel_support_fraction_by_substep"], [1., .5, 1., 1.])
        self.assertAlmostEqual(result["wheel_z_rmse_m_by_substep_and_side"][0][0],
                               (2.5e-6)**.5)

    def test_empty_windows_rejected(self):
        with self.assertRaisesRegex(ValueError, "No matched policy windows"):
            summarize([])


if __name__ == "__main__":
    unittest.main()
