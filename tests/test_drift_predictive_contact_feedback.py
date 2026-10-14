"""Causal bounded inner-controller action mapping tests."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_drift_predictive_knee_mujoco import PredictiveKneeFeedback
from eval_drift_predictive_clearance_mujoco import PredictiveClearanceFeedback


class PredictiveFeedbackTests(unittest.TestCase):
    def test_height_feedback_lowers_each_wheel_with_bounded_knee_change(self):
        controller = PredictiveKneeFeedback(left_threshold_m=.128,
                                             right_threshold_m=.128,
                                             lookahead_s=.01, gain=1.,
                                             max_bias_rad=.02,
                                             max_change_rad_per_step=.003)
        actor = np.zeros(8)
        action, info = controller.modify(actor, [.130, .130])
        np.testing.assert_allclose(action[[4, 5]], [-.003, .003])
        np.testing.assert_array_equal(action[[0, 1, 2, 3, 6, 7]], np.zeros(6))
        np.testing.assert_allclose(info["knee_bias_magnitude_rad"], [.003, .003])
        action, _ = controller.modify(actor, [.131, .131])
        np.testing.assert_allclose(action[[4, 5]], [-.006, .006])

    def test_zero_gain_is_inert_even_when_wheels_are_high(self):
        controller = PredictiveKneeFeedback(left_threshold_m=.128,
                                             right_threshold_m=.128,
                                             lookahead_s=.02, gain=0.,
                                             max_bias_rad=.02,
                                             max_change_rad_per_step=.003)
        actor = np.arange(8, dtype=float) / 20.
        action, _ = controller.modify(actor, [.140, .140])
        np.testing.assert_array_equal(action, actor)

    def test_clearance_feedback_uses_live_geometry_with_bounded_action(self):
        controller = PredictiveClearanceFeedback(left_threshold_m=.001,
                                                 right_threshold_m=.001,
                                                 lookahead_s=.01, gain=1.,
                                                 max_bias_rad=.02,
                                                 max_change_rad_per_step=.003)
        controller.env = SimpleNamespace(measured=object())
        with patch("eval_drift_predictive_clearance_mujoco.signed_clearance",
                   side_effect=[.002, .002]):
            action, info = controller.modify(np.zeros(8), [999., 999.])
        np.testing.assert_allclose(action[[4, 5]], [-.003, .003])
        np.testing.assert_allclose(info["signed_finite_cylinder_clearance_m"], [.002, .002])


if __name__ == "__main__":
    unittest.main()
