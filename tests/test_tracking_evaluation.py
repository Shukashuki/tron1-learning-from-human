"""Offline pytest/unittest regression checks for physical rollout reporting.

These tests import only NumPy helpers; Isaac Sim, Torch and RSL-RL are not
required. They do not replace the remote simulator/API smoke evaluation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_tracking.py"
SPEC = importlib.util.spec_from_file_location("tron1_eval_tracking_test_module", SCRIPT)
EVALUATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATION)


class ContactEventTests(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(30) * 0.02
        self.forces = np.zeros((30, 2, 3))
        self.forces[:10, :, 2] = 100.0
        self.forces[15:, :, 2] = 100.0

    def test_true_runs_are_half_open(self):
        self.assertEqual(EVALUATION.true_runs([False, True, True, False, True]), [(1, 3), (4, 5)])
        self.assertEqual(EVALUATION.true_runs([]), [])
        self.assertEqual(EVALUATION.true_runs([True, True]), [(0, 2)])

    def test_support_flight_landing_times(self):
        report = EVALUATION.contact_events(self.times, self.forces)
        self.assertTrue(report["available"])
        self.assertAlmostEqual(report["first_takeoff_s"], 0.20)
        self.assertAlmostEqual(report["first_landing_s"], 0.30)
        self.assertAlmostEqual(report["max_flight_duration_s"], 0.10)
        self.assertFalse(report["flight_intervals"][0]["right_censored"])

    def test_reset_with_no_initial_contact_is_not_a_takeoff(self):
        forces = np.zeros_like(self.forces)
        forces[10:, :, 2] = 100.0
        report = EVALUATION.contact_events(self.times, forces)
        self.assertIsNone(report["first_takeoff_s"])
        self.assertEqual(report["flight_intervals"], [])

    def test_unlanded_flight_is_right_censored(self):
        forces = np.zeros_like(self.forces)
        forces[:10, :, 2] = 100.0
        report = EVALUATION.contact_events(self.times, forces)
        self.assertIsNone(report["first_landing_s"])
        self.assertTrue(report["flight_intervals"][0]["right_censored"])
        # The observed lower bound ends at the last sample, not one timestep
        # beyond it, because no further physics was observed.
        self.assertAlmostEqual(report["max_flight_duration_s"], 0.38)

    def test_missing_force_sensor_is_explicitly_unavailable(self):
        report = EVALUATION.contact_events(self.times, np.full_like(self.forces, np.nan))
        self.assertFalse(report["available"])
        self.assertIsNone(report["max_flight_duration_s"])

    def test_short_contact_dropout_is_not_a_flight(self):
        forces = self.forces.copy()
        forces[10:15, :, 2] = 100.0
        forces[10:12, :, 2] = 0.0  # only a 20 ms observed span
        self.assertIsNone(EVALUATION.contact_events(self.times, forces)["first_takeoff_s"])

    def test_one_wheel_still_touching_is_not_two_wheel_flight(self):
        forces = self.forces.copy()
        forces[10:15, 0, 2] = 100.0
        self.assertIsNone(EVALUATION.contact_events(self.times, forces)["first_takeoff_s"])


class EpisodeMetricsTests(unittest.TestCase):
    def setUp(self):
        self.samples = 30
        self.times = np.arange(self.samples) * 0.02
        root = np.zeros((self.samples, 2, 3))
        root[:, :, 2] = 0.9
        root[10:16, :, 2] = 1.05
        quat = np.zeros((self.samples, 2, 4))
        quat[:, :, 0] = 1.0
        valid = np.ones((self.samples, 2), dtype=bool)
        valid[21:, 1] = False
        ref_ids = np.broadcast_to(np.arange(self.samples)[:, None], (self.samples, 2)).copy()
        # An invalid second episode can have arbitrary reset/replay state; it
        # must never affect first-episode metrics.
        root[21:, 1] = np.nan
        quat[21:, 1] = np.nan
        ref_ids[21:, 1] = -1
        forces = np.zeros((self.samples, 2, 2, 3))
        forces[:10, :, :, 2] = 100.0
        forces[15:, :, :, 2] = 100.0
        self.arrays = {
            "policy_dt_s": np.array(0.02), "valid_mask": valid, "time_s": self.times,
            "root_position_m": root, "root_quaternion_wxyz": quat,
            "reference_frame": ref_ids, "wheel_contact_force_w_n": forces,
        }
        self.reference = {"body_pos_w": np.zeros((self.samples, 1, 3))}
        self.reference["body_pos_w"][:, 0, 2] = root[:, 0, 2]
        self.outcomes = [
            {"timed_out": True, "early_terminated": False, "termination_terms": ["motion_end"],
             "end_reason": "timeout"},
            {"timed_out": False, "early_terminated": True, "termination_terms": ["anchor_ori"],
             "end_reason": "early_termination"},
        ]

    def report(self):
        return EVALUATION.summarize_episodes(self.arrays, self.outcomes, self.reference, 5.0)

    def test_complete_and_early_terminated_episodes_are_separate(self):
        report = self.report()
        self.assertEqual(report["full_reference_survival_rate"], 0.5)
        self.assertEqual(report["early_termination_rate"], 0.5)
        self.assertTrue(report["episodes"][0]["completed_full_reference"])
        self.assertFalse(report["episodes"][1]["completed_full_reference"])

    def test_mask_excludes_every_post_reset_state(self):
        baseline = self.report()
        self.arrays["root_position_m"][21:, 1, 2] = 999.0
        self.arrays["root_quaternion_wxyz"][21:, 1] = [0.0, 1.0, 0.0, 0.0]
        changed = self.report()
        self.assertEqual(baseline, changed)
        self.assertEqual(changed["episodes"][1]["recorded_samples"], 21)
        self.assertAlmostEqual(changed["episodes"][1]["recorded_duration_s"], 0.4)

    def test_measured_height_gain_and_reference_rmse(self):
        report = self.report()
        self.assertAlmostEqual(report["mean_base_link_height_gain_m"], 0.15)
        self.assertEqual(report["mean_reference_height_rmse_m"], 0.0)
        self.assertEqual(report["maximum_base_tilt_deg"], 0.0)

    def test_reference_peak_does_not_become_actual_jump_height(self):
        self.arrays["root_position_m"][:, 0, 2] = 0.9
        report = self.report()["episodes"][0]
        self.assertEqual(report["base_link_height_gain_m"], 0.0)
        self.assertAlmostEqual(report["reference_base_link_height_gain_m"], 0.15)
        self.assertGreater(report["reference_height_rmse_m"], 0.0)
        # Flight-like force absence alone is not enough: require a measured
        # root rise too, while still not claiming dynamic stability.
        self.assertFalse(report["jump_and_landing_detected"])

    def test_landed_jump_detection_does_not_override_early_termination(self):
        report = self.report()
        self.assertTrue(report["episodes"][1]["jump_and_landing_detected"])
        self.assertFalse(report["episodes"][1]["full_clip_with_detected_jump_and_landing"])

    def test_timeout_before_clip_end_is_not_full_reference_completion(self):
        self.outcomes[1] = {"timed_out": True, "early_terminated": False,
                            "termination_terms": ["time_out"], "end_reason": "timeout"}
        report = self.report()
        self.assertFalse(report["episodes"][1]["completed_full_reference"])
        self.assertEqual(report["full_reference_survival_rate"], 0.5)

    def test_tilt_uses_wxyz_and_normalizes_quaternion(self):
        half = np.radians(30.0) / 2.0
        self.arrays["root_quaternion_wxyz"][:, 0] = 2.0 * np.array([np.cos(half), 0.0, np.sin(half), 0.0])
        self.assertAlmostEqual(self.report()["maximum_base_tilt_deg"], 30.0)


if __name__ == "__main__":
    unittest.main()
