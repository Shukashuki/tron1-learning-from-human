"""Pure reward and PPO-return checks for the MuJoCo residual scaffold."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from train_drift_mujoco_residual import gae
from training.drift_mujoco_residual_reward import reward_components
from training.drift_mujoco_streak import DriftStreakTracker


def physical_step(force, slip=.1):
    quat = Rotation.from_euler("z", -.2).as_quat()
    return {"time_s": 1.5, "root_quat_wxyz": quat[[3, 0, 1, 2]],
            "root_com_lin_vel_w": np.array([1., 0., 0.]),
            "wheel_ground_contact_count": np.array([int(force > 0)] * 2),
            "wheel_ground_contact_force_w_n": np.array([[0., 0., force]] * 2),
            "wheel_lateral_contact_slip_m_s": np.array([slip, slip])}


class ResidualRewardTests(unittest.TestCase):
    def test_support_and_slide_reward_only_in_core(self):
        quat = physical_step(10.)["root_quat_wxyz"]
        target = {"root_pos": np.array([0., 0., .5]),
                  "root_com_lin_vel_w": np.array([1., 0., 0.]),
                  "root_quat_wxyz": quat}
        state = dict(target)
        supported = reward_components(target, state, [physical_step(10.) for _ in range(4)],
                                      np.zeros(8))
        airborne = reward_components(target, state, [physical_step(0.) for _ in range(4)],
                                     np.zeros(8))
        self.assertGreater(supported["reward"], airborne["reward"])
        self.assertEqual(supported["support_fraction"], 1.)
        self.assertEqual(airborne["support_fraction"], 0.)
        self.assertEqual(airborne["joint_qualifying_soft_fraction"], 0.)

    def test_wrong_substep_count_rejected(self):
        with self.assertRaisesRegex(ValueError, "four native steps"):
            reward_components({}, {}, [physical_step(10.)], np.zeros(8))


class AdvantageTests(unittest.TestCase):
    def test_terminal_episode_boundary_is_not_bootstrapped(self):
        advantage, returns = gae([1., 2.], [0., 0.], [True, True], gamma=1., lam=1.)
        np.testing.assert_array_equal(advantage, [1., 2.])
        np.testing.assert_array_equal(returns, [1., 2.])

    def test_nonterminal_reward_carries_to_next_step(self):
        advantage, _ = gae([1., 2.], [0., 0.], [False, True], gamma=1., lam=1.)
        np.testing.assert_array_equal(advantage, [3., 2.])


class StreakTests(unittest.TestCase):
    def test_native_contact_gap_closes_window_without_bridging(self):
        tracker = DriftStreakTracker()
        for index in range(4):
            sample = physical_step(10.)
            angle = .05 * index
            sample["time_s"] = 1.44 + .005*index
            sample["root_com_lin_vel_w"] = np.array([np.cos(angle), np.sin(angle), 0.])
            self.assertEqual(tracker.observe(sample), 0.)
        gap = physical_step(0.)
        gap["time_s"] = 1.46
        gain = tracker.observe(gap)
        self.assertGreater(gain, 0.)
        self.assertAlmostEqual(tracker.longest_qualifying_duration_s, .015)
        self.assertAlmostEqual(tracker.largest_qualifying_turn_deg,
                               np.degrees(.15))
        later = physical_step(10.)
        later["time_s"] = 1.465
        self.assertEqual(tracker.observe(later), 0.)
        self.assertEqual(tracker.finish(), 0.)


if __name__ == "__main__":
    unittest.main()
