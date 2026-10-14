"""Analytic finite-cylinder clearance should respect axial width and tilt."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_turn_wheel_clearance import cylinder_clearance


class TurnWheelClearanceTests(unittest.TestCase):
    def test_horizontal_axle_bottom_is_radius_below_center(self):
        self.assertAlmostEqual(cylinder_clearance(np.array([0., 0., .127]),
                               np.array([0., 1., 0.]), np.zeros(3),
                               np.array([0., 0., 1.]), .127, .025), 0.)

    def test_axial_tilt_increases_support_with_actual_halfwidth(self):
        axis = np.array([0., np.sqrt(.75), .5])
        clearance = cylinder_clearance(np.array([0., 0., .127]), axis,
                                       np.zeros(3), np.array([0., 0., 1.]), .127, .025)
        self.assertAlmostEqual(clearance, .127 - .127 * np.sqrt(.75) - .0125)


if __name__ == "__main__":
    unittest.main()
