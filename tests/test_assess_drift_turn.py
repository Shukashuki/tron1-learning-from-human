"""The new turn gate must not splice separate fast/sliding/contact episodes."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from assess_drift_turn import assess, completed_from_report, physical_episode


def fixture():
    time = np.arange(601) * .005
    course = np.deg2rad(60) * time / time[-1]
    velocity = np.c_[np.cos(course), np.sin(course), np.zeros(len(time))]
    position = np.c_[np.cumsum(velocity[:, :2] * .005, axis=0), np.full(len(time), .75)]
    yaw = course - np.deg2rad(12)
    quat = Rotation.from_euler("z", yaw[:, None]).as_quat()[:, [3, 0, 1, 2]]
    rollout = {"time_s": time, "root_pos": position,
               "root_com_lin_vel_w": velocity, "root_quat_wxyz": quat,
               "wheel_ground_contact_force_w_n": np.tile([[0., 0., 20.]] * 2, (len(time), 1, 1)),
               "wheel_ground_contact_count": np.ones((len(time), 2), dtype=int),
               "wheel_lateral_contact_slip_m_s": np.full((len(time), 2), .1)}
    reference = {"time_s": time, "source_frame_numbers": np.linspace(1553, 1584, len(time)),
                 "source_phase": np.zeros(len(time), dtype=np.int8),
                 "root_target_position_m": position.copy()}
    return rollout, reference


class TurnAssessmentTests(unittest.TestCase):
    def test_one_continuous_fast_sliding_supported_turn_passes(self):
        rollout, reference = fixture()
        result = assess(rollout, reference)
        self.assertEqual(result["status"], "PASS")
        self.assertAlmostEqual(result["largest_qualifying_course_turn_deg"], 60.)

    def test_speed_drop_splits_the_turn_and_prevents_cherry_picking(self):
        rollout, reference = fixture()
        rollout["root_com_lin_vel_w"][250:350, :2] *= .5
        result = assess(rollout, reference)
        self.assertEqual(result["status"], "FAIL")
        self.assertLess(result["largest_qualifying_course_turn_deg"], 45.)

    def test_single_native_contact_gap_is_not_bridged(self):
        rollout, reference = fixture()
        rollout["wheel_ground_contact_count"][300, 1] = 0
        result = assess(rollout, reference)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(len(result["qualifying_windows"]), 2)

    def test_unrelated_off_path_turn_does_not_pass(self):
        rollout, reference = fixture()
        rollout["root_pos"][:, 1] += .4
        result = assess(rollout, reference)
        self.assertEqual(result["status"], "FAIL")
        self.assertGreater(result["core_path_rmse_m"], .15)

    def test_patch_slip_and_body_slip_are_independent_requirements(self):
        rollout, reference = fixture()
        rollout["wheel_lateral_contact_slip_m_s"][:] = 0
        self.assertEqual(assess(rollout, reference)["status"], "FAIL")
        rollout, reference = fixture()
        rollout["root_quat_wxyz"] = Rotation.from_euler(
            "z", (np.deg2rad(60) * rollout["time_s"] / 3.)[:, None]).as_quat()[:, [3, 0, 1, 2]]
        self.assertEqual(assess(rollout, reference)["status"], "FAIL")

    def test_isaac_first_episode_maps_to_the_same_native_gate(self):
        rollout, reference = fixture()
        isaac = {"time_s": rollout["time_s"],
                 "root_position_m": rollout["root_pos"][:, None],
                 "root_linear_velocity_w_m_s": rollout["root_com_lin_vel_w"][:, None],
                 "root_quaternion_wxyz": rollout["root_quat_wxyz"][:, None],
                 "wheel_ground_contact_force_w_n": rollout["wheel_ground_contact_force_w_n"][:, None],
                 "wheel_ground_contact_count": rollout["wheel_ground_contact_count"][:, None],
                 "wheel_lateral_contact_slip_m_s": rollout["wheel_lateral_contact_slip_m_s"][:, None],
                 "valid_mask": np.ones((len(rollout["time_s"]), 1), dtype=bool)}
        converted = physical_episode(isaac, "isaac")
        self.assertEqual(assess(converted, reference)["status"], "PASS")
        isaac["valid_mask"][300, 0] = False
        with self.assertRaisesRegex(ValueError, "contiguous"):
            physical_episode(isaac, "isaac")

    def test_full_episode_completion_is_independent_of_turn_window(self):
        self.assertFalse(completed_from_report({"completed_full_reference": False}, "mujoco"))
        self.assertTrue(completed_from_report(
            {"summary": {"episodes": [{"completed_full_reference": True}]}}, "isaac"))
        with self.assertRaisesRegex(ValueError, "completion verdict"):
            completed_from_report({"summary": {"episodes": []}}, "isaac")


if __name__ == "__main__":
    unittest.main()
