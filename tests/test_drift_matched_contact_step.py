"""Pure reductions for matched-state contact diagnostics."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_drift_matched_contact_step import summarize


def row(*, mj_force, mj_count, initial_error=1e-7):
    return {"initial_wheel_position_max_abs_error_m": initial_error,
            "initial_twist_max_abs_error": 1e-14,
            "initial_mujoco_signed_wheel_clearance_m": [0., 0.],
            "initial_mujoco_bearing_count": [1, 1],
            "isaac_next_ground_normal_force_n": [50., 50.],
            "mujoco_next_ground_normal_force_n": mj_force,
            "isaac_next_bearing_count": [1, 1],
            "mujoco_next_bearing_count": mj_count,
            "isaac_next_wheel_z_m": [.127, .127],
            "mujoco_next_wheel_z_m": [.128, .126]}


class MatchedContactSummaryTests(unittest.TestCase):
    def test_support_and_position_are_reduced_per_same_step(self):
        result = summarize([row(mj_force=[20., 0.], mj_count=[1, 0]),
                            row(mj_force=[20., 20.], mj_count=[1, 1])])
        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["isaac_both_wheels_supported_fraction"], 1.)
        self.assertEqual(result["mujoco_both_wheels_supported_fraction_from_same_isaac_state_and_torque"], .5)
        self.assertEqual(result["fraction_isaac_supported_but_mujoco_not"], .5)
        self.assertAlmostEqual(result["wheel_center_next_step_z_rmse_m"][0], .001)

    def test_bad_initial_alignment_is_not_summarized(self):
        with self.assertRaisesRegex(ValueError, "Initial cross-engine state mismatch"):
            summarize([row(mj_force=[20., 20.], mj_count=[1, 1], initial_error=.003)])


if __name__ == "__main__":
    unittest.main()
