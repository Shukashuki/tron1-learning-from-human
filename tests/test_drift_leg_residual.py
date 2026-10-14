"""Check symmetric raw-action residual parameterization."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from search_drift_leg_residual import candidate_name, mirrored_leg_bias


class DriftLegResidualTests(unittest.TestCase):
    def test_mirrored_offset_leaves_abduction_and_wheels_unchanged(self):
        bias = mirrored_leg_bias(.04, -.03)
        self.assertEqual(bias[[0, 1, 6, 7]].tolist(), [0.] * 4)
        self.assertAlmostEqual(bias[2], -bias[3])
        self.assertAlmostEqual(bias[4], -bias[5])
        self.assertAlmostEqual(bias[2], .04)
        self.assertAlmostEqual(bias[4], -.03)

    def test_nonfinite_or_large_offsets_rejected(self):
        for value in (float("nan"), float("inf"), .151):
            with self.assertRaises(ValueError):
                mirrored_leg_bias(value, 0.)

    def test_names_distinguish_signs(self):
        self.assertNotEqual(candidate_name(.04, 0.), candidate_name(-.04, 0.))


if __name__ == "__main__":
    unittest.main()
