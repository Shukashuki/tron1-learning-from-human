"""Bounded shared-controller tests; no Isaac launch and no physics success claim."""
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sim2sim_common import (JOINT_NAMES, LEGS, WHEELS, ReferenceMotion,
                            SharedController, load_config, stop_reason)

REFERENCE = ROOT / "outputs/mink-cmu-16_03/robot_reference.npz"


def neutral_state():
    return {"joint_pos": np.zeros(8), "root_pos": np.array([0., 0., 1.]),
            "root_quat_wxyz": np.array([1., 0., 0., 0.]),
            "axle_pos": np.array([0., 0., .127]),
            "body_com": np.array([0., 0., .7])}


def constant_reference(root_height=1.):
    def sample(time_s):
        return {"joint_pos": np.zeros(8), "joint_vel": np.zeros(8),
                "root_pos": np.array([0., 0., root_height]),
                "axle_pos": np.array([0., 0., .127]), "axle_vel": np.zeros(3),
                "source_time_s": time_s, "phase": "motion"}
    return types.SimpleNamespace(sample=sample)


class SharedControllerTests(unittest.TestCase):
    def test_pose_difference_rates_and_leg_pd_sign(self):
        config = load_config()
        control = SharedController(constant_reference(), config)
        state = neutral_state()
        first = control.compute(0., state)
        np.testing.assert_allclose(first["joint_vel"], 0.)
        state["joint_pos"][LEGS] = .01
        dt = config["physics_dt"]
        second = control.compute(dt, state)
        np.testing.assert_allclose(second["joint_vel"][LEGS], .01 / dt)
        expected = -config["leg_kp"] * .01 - config["leg_kd"] * .01 / dt
        np.testing.assert_allclose(second["raw_torque"][LEGS], expected)
        with self.assertRaises(ValueError):
            control.compute(dt, state)

    def test_explicit_joint_torque_limits_and_no_state_mutation(self):
        config = load_config()
        control = SharedController(constant_reference(), config)
        state = neutral_state()
        state["joint_pos"][LEGS] = 10.
        state["body_com"][0] = 1.
        saved = {key: value.copy() for key, value in state.items()}
        output = control.compute(0., state)
        np.testing.assert_allclose(output["torque"][LEGS], -config["leg_torque_limit_nm"])
        np.testing.assert_allclose(abs(output["torque"][WHEELS]), config["wheel_torque_limit_nm"])
        for key in saved:
            np.testing.assert_array_equal(state[key], saved[key])

    def test_root_height_target_is_diagnostic_not_a_forcing_command(self):
        config = load_config()
        low = SharedController(constant_reference(.1), config)
        high = SharedController(constant_reference(10.), config)
        a, b = low.compute(0., neutral_state()), high.compute(0., neutral_state())
        self.assertNotEqual(a["root_ref"][2], b["root_ref"][2])
        np.testing.assert_array_equal(a["torque"], b["torque"])
        self.assertEqual(set(a), {"torque", "raw_torque", "joint_vel", "joint_ref", "root_ref",
                                  "balance_state", "base_tilt_rad", "source_time_s", "phase"})

    def test_stop_thresholds_are_reported_not_recovery_or_pose_corrections(self):
        config = load_config()
        state = neutral_state()
        output = SharedController(constant_reference(), config).compute(0., state)
        self.assertIsNone(stop_reason(state, output, config))
        state["root_pos"][2] = config["fall_root_height_m"] - .01
        self.assertEqual(stop_reason(state, output, config), "fall_low_base")
        state["root_pos"][2] = 1.
        output["base_tilt_rad"] = np.deg2rad(config["fall_tilt_deg"] + 1)
        self.assertEqual(stop_reason(state, output, config), "fall_excessive_tilt")


@unittest.skipUnless(REFERENCE.exists(), "Local reference not present")
class ReferenceSamplingTests(unittest.TestCase):
    def test_motion_timing_standing_and_end_clamping(self):
        config = load_config()
        motion = ReferenceMotion(REFERENCE, config=config)
        standing = ReferenceMotion(REFERENCE, case="standing", config=config)
        self.assertEqual(motion.sample(0.)["phase"], "settle")
        np.testing.assert_allclose(motion.sample(0.)["joint_vel"], 0.)
        start = motion.sample(config["settle_seconds"])
        self.assertEqual(start["phase"], "motion")
        self.assertEqual(start["source_time_s"], 0.)
        end = motion.sample(motion.duration + 10.)
        self.assertEqual(end["phase"], "recovery")
        self.assertEqual(end["source_time_s"], motion.clip_duration)
        np.testing.assert_allclose(end["joint_pos"], motion.joints[-1])
        np.testing.assert_allclose(end["joint_vel"], 0.)
        for time_s in (0., 2., 20.):
            target = standing.sample(time_s)
            self.assertEqual(target["phase"], "standing")
            np.testing.assert_array_equal(target["joint_pos"], motion.joints[0])
            np.testing.assert_array_equal(target["joint_vel"], np.zeros(8))
        self.assertAlmostEqual(motion.initial_qpos[2] - motion.root_pos[0, 2], config["initial_clearance_m"])

    def test_input_joint_order_is_resolved_by_name_and_outputs_are_canonical(self):
        with np.load(REFERENCE, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        expected = ReferenceMotion(REFERENCE)
        permutation = np.array([7, 3, 2, 5, 0, 6, 1, 4])
        arrays["joint_names"] = arrays["joint_names"][permutation]
        arrays["joint_positions_rad"] = arrays["joint_positions_rad"][:, permutation]
        with tempfile.TemporaryDirectory(prefix="tron1-sim2sim-test-") as directory:
            path = Path(directory) / "shuffled.npz"
            np.savez_compressed(path, **arrays)
            shuffled = ReferenceMotion(path)
        self.assertEqual(shuffled.joint_names, JOINT_NAMES)
        np.testing.assert_array_equal(shuffled.joints, expected.joints)
        for path in (ROOT / "outputs/sim2sim-cmu-16_03").glob("*/rollout.npz"):
            with np.load(path, allow_pickle=False) as rollout:
                self.assertEqual(tuple(rollout["joint_names"]), JOINT_NAMES, str(path))


if __name__ == "__main__":
    unittest.main()
