"""Physics-counter recording, reset exclusion and held policy context."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_tracking.py"
SPEC = importlib.util.spec_from_file_location("tron1_physics_recording", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PhysicsTrajectoryRecorder = MODULE.PhysicsTrajectoryRecorder


def snapshot(value=0., num_envs=2, frame=0):
    quaternion = np.zeros((num_envs, 4))
    quaternion[:, 0] = 1.
    return {
        "root_position_m": np.full((num_envs, 3), value, dtype=float),
        "root_quaternion_wxyz": quaternion,
        "root_linear_velocity_w_m_s": np.full((num_envs, 3), value, dtype=float),
        "root_angular_velocity_w_rad_s": np.zeros((num_envs, 3)),
        "joint_position_rad": np.full((num_envs, 8), value, dtype=float),
        "joint_velocity_rad_s": np.zeros((num_envs, 8)),
        "wheel_ground_contact_count": np.ones((num_envs, 2), dtype=int),
        "wheel_lateral_contact_slip_m_s": np.zeros((num_envs, 2)),
        "reference_frame": np.full(num_envs, frame, dtype=int),
    }


class PhysicsCaptureTests(unittest.TestCase):
    def setUp(self):
        self.recorder = PhysicsTrajectoryRecorder(2, .005, 100, snapshot(), np.zeros((2, 51)))

    def begin(self, valid=None, value=1., frame=0):
        valid = np.ones(2, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        self.recorder.begin_step(valid, np.full((2, 51), value), np.full((2, 8), value),
                                 np.full((2, 8), value / 10), np.full(2, frame, dtype=int))

    def collect(self, first, last, frame=0):
        for counter in range(first, last + 1):
            self.recorder.capture(counter, snapshot(counter, frame=frame))

    def arrays(self):
        return self.recorder.arrays([f"joint_{i}" for i in range(8)], ["base_Link"],
                                   ["wheel_L_Link", "wheel_R_Link"], .02)

    def test_one_initial_sample_and_every_counter_increment(self):
        self.assertFalse(self.recorder.should_capture(101))
        self.begin()
        self.collect(101, 104)
        self.recorder.finish_step(np.zeros(2, dtype=bool), 4)
        arrays = self.arrays()
        np.testing.assert_allclose(arrays["time_s"], [0., .005, .010, .015, .020])
        np.testing.assert_array_equal(arrays["simulation_step_counter"], [100, 101, 102, 103, 104])
        self.assertEqual(arrays["qpos"].shape, (5, 2, 15))
        self.assertEqual(arrays["joint_names"].shape, (8,))
        np.testing.assert_allclose(arrays["qpos"][:, :, :3], arrays["root_position_m"])
        np.testing.assert_allclose(arrays["action_observation"][1:], 1.)
        np.testing.assert_allclose(arrays["actions_clipped"][1:], .1)
        np.testing.assert_array_equal(arrays["action_was_applied"][0], False)
        np.testing.assert_array_equal(arrays["action_was_applied"][1:], True)

    def test_same_counter_reset_cannot_replace_pre_reset_terminal(self):
        self.begin()
        self.collect(101, 104)
        # A scene update after reset reuses counter104. Its pose must not enter
        # the trajectory, even while wrapped.step has not yet returned.
        self.assertFalse(self.recorder.capture(104, snapshot(999., frame=0)))
        self.recorder.finish_step(np.array([True, False]), 4)
        arrays = self.arrays()
        np.testing.assert_allclose(arrays["root_position_m"][-1], 104.)
        np.testing.assert_array_equal(arrays["terminal_sample"][-1], [True, False])
        self.assertFalse(self.recorder.should_capture(105))

    def test_completed_environment_has_contiguous_valid_prefix_only(self):
        self.begin()
        self.collect(101, 104)
        self.recorder.finish_step(np.array([True, False]), 4)
        self.begin([False, True], value=2., frame=1)
        self.collect(105, 108, frame=1)
        self.recorder.finish_step(np.array([False, True]), 4)
        arrays = self.arrays()
        np.testing.assert_array_equal(arrays["valid_mask"][:, 0], [True] * 5 + [False] * 4)
        np.testing.assert_array_equal(arrays["valid_mask"][:, 1], True)
        self.assertTrue(np.isnan(arrays["root_position_m"][5:, 0]).all())
        np.testing.assert_array_equal(arrays["reference_frame"][5:, 0], -1)
        np.testing.assert_array_equal(arrays["terminal_sample"][:, 0], [False] * 4 + [True] + [False] * 4)
        np.testing.assert_array_equal(arrays["terminal_sample"][-1], [False, True])
        with self.assertRaisesRegex(ValueError, "become valid"):
            self.begin([True, True])

    def test_counter_gap_and_missed_substep_fail_closed(self):
        self.begin()
        with self.assertRaisesRegex(RuntimeError, "skipped"):
            self.recorder.capture(102, snapshot())
        self.collect(101, 103)
        with self.assertRaisesRegex(RuntimeError, "missed"):
            self.recorder.finish_step(np.zeros(2, dtype=bool), 4)

    def test_held_policy_context_changes_only_between_action_intervals(self):
        self.begin(value=1., frame=0)
        self.collect(101, 104, frame=0)
        self.recorder.finish_step(np.zeros(2, dtype=bool), 4)
        self.begin(value=2., frame=1)
        self.collect(105, 108, frame=1)
        self.recorder.finish_step(np.zeros(2, dtype=bool), 4)
        arrays = self.arrays()
        np.testing.assert_allclose(arrays["action_observation"][1:5], 1.)
        np.testing.assert_allclose(arrays["action_observation"][5:], 2.)
        np.testing.assert_array_equal(arrays["reference_frame"][:5], 0)
        np.testing.assert_array_equal(arrays["reference_frame"][5:], 1)
        self.assertIn("held across decimation", str(arrays["action_observation_kind"]))

    def test_capture_copies_inputs_without_mutating_policy_records(self):
        initial = snapshot()
        observation = np.zeros((2, 51))
        recorder = PhysicsTrajectoryRecorder(2, .005, 0, initial, observation)
        observation[:] = 99.
        initial["root_position_m"][:] = 99.
        arrays = recorder.arrays([f"joint_{i}" for i in range(8)], ["base_Link"], [], .02)
        np.testing.assert_allclose(arrays["action_observation"], 0.)
        np.testing.assert_allclose(arrays["root_position_m"], 0.)


if __name__ == "__main__":
    unittest.main()
