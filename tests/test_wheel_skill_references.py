import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_wheel_skill_references import adapt_intent, load_human, prepare_reference, solve_reference


def human_fixture():
    frames = 241
    times = np.arange(frames) / 120
    positions = np.zeros((frames, 1, 3))
    positions[:, 0, 0] = 2 * times - .5 * times ** 2
    positions[:, 0, 2] = 1 - .4 * np.sin(np.pi * times / 2) ** 2
    a = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    quats = Rotation.from_matrix(np.repeat(a[None], frames, axis=0)).as_quat()[:, [3, 0, 1, 2]]
    return {"fps": np.array(120.), "time_s": times, "frame_numbers": np.arange(1, frames + 1),
            "joint_names": np.array(["root"]), "joint_positions_world_m": positions,
            "joint_rotations_world_wxyz": quats[:, None], "metadata_json": np.array("{}")}


class WheelIntentTests(unittest.TestCase):
    def test_rolling_preserves_forward_intent_not_human_steps(self):
        human = human_fixture()
        before = {key: value.copy() for key, value in human.items()}
        intent = adapt_intent(human, "rolling_stop", rolling_height_scale=0)
        root = intent["root_position"]
        self.assertGreater(root[-1, 0], .5)
        np.testing.assert_array_equal(root[:, 1], 0)
        self.assertLess(abs(root[-1, 0] - root[-2, 0]), 1e-6)
        self.assertEqual(intent["time_s"][-1], 4.)
        for key in human:
            np.testing.assert_array_equal(human[key], before[key])

    def test_crouch_explicit_depth_static_root_but_not_claiming_feet_fixed(self):
        intent = adapt_intent(human_fixture(), "crouch", first_frame=1, last_frame=241)
        root = intent["root_position"]
        np.testing.assert_array_equal(root[:, :2], 0)
        self.assertAlmostEqual(root[:, 2].min(), .71)
        self.assertAlmostEqual(root[0, 2], .89)
        self.assertAlmostEqual(root[-1, 2], .89)
        self.assertIn("wheel centers may roll", intent["metadata"]["stationary_crouch"])

    def test_bad_parameters_and_missing_crop_rejected(self):
        for options in ({"time_scale": 0}, {"pitch_scale": float("nan")},
                        {"smoothing_source_s": -1}, {"first_frame": 0},
                        {"first_frame": 10, "last_frame": 10}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                adapt_intent(human_fixture(), "rolling_stop", **options)

    def test_load_rejects_fps_or_bad_quaternion(self):
        for key, value in (("fps", np.array(0.)), ("joint_rotations_world_wxyz", np.zeros((241, 1, 4)))):
            with tempfile.TemporaryDirectory() as directory:
                human = human_fixture()
                human[key] = value
                path = Path(directory) / "human.npz"
                np.savez(path, **human)
                with self.assertRaises(ValueError):
                    load_human(path)


@unittest.skipUnless(importlib.util.find_spec("mujoco") and importlib.util.find_spec("mink"), "MuJoCo/Mink required for robot FK")
class WheelRobotTests(unittest.TestCase):
    def test_rolling_ik_moves_both_wheels_and_accounts_for_axle_motion(self):
        intent = adapt_intent(human_fixture(), "rolling_stop", rolling_height_scale=.03)
        arrays, report = solve_reference(intent)
        sites = arrays["site_positions_world_m"][:, -2:]
        self.assertTrue(np.all(sites[-1, :, 0] - sites[0, :, 0] > .5))
        np.testing.assert_allclose(sites[:, :, 2], .127, atol=1e-7)
        # Trapezoid-integrated spin is differentiated again by the exporter;
        # the one-sided endpoints carry O(dt) error. This is a millimeter/s
        # geometric tolerance, not a contact-force or physical slip claim.
        self.assertLess(report["max_rolling_contact_tangent_velocity_residual_m_s"], .005)
        self.assertEqual(report["max_joint_limit_violation_rad"], 0)
        self.assertLess(report["max_ground_penetration_m"], 1e-7)
        self.assertLess(report["max_self_penetration_m"], 1e-7)
        # General contact Jacobian spin is not naively root-X/r: leg axle
        # rotation and translation contribute during height changes.
        names = arrays["joint_names"].tolist()
        spin = arrays["joint_velocities_rad_s"][:, names.index("wheel_L_Joint")]
        naive = np.gradient(arrays["qpos"][:, 0], arrays["time_s"]) / .127
        self.assertGreater(np.max(np.abs(spin - naive)), .01)

    def test_crouch_bounded_ik_and_export_hold_masks_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "human.npz"
            np.savez(source, **human_fixture())
            original = source.read_bytes()
            output = Path(directory) / "reference"
            summary = prepare_reference(source, output, "crouch", first_frame=1, last_frame=241)
            self.assertFalse(summary["physics_validated"])
            self.assertFalse(summary["policy_trained"])
            self.assertEqual(summary["motion_frames"], 251)
            self.assertEqual(summary["sample_span_s"], 5.)
            self.assertEqual(source.read_bytes(), original)
            with np.load(output / "motion.npz", allow_pickle=False) as motion:
                self.assertFalse(motion["joint_tracking_mask"][-2:].any())
                np.testing.assert_array_equal(motion["qpos_mujoco"][-50:], np.repeat(motion["qpos_mujoco"][-1:], 50, axis=0))
            meta = json.loads((output / "motion.json").read_text())
            self.assertLess(meta["independent_urdf_fk_max_position_error_m"], 1e-9)
            self.assertIn("Jacobian", meta["wheel_spin_reference"])
            with self.assertRaises(FileExistsError):
                prepare_reference(source, output, "crouch", first_frame=1, last_frame=241)

    def test_unreachable_root_fails_without_artifact(self):
        intent = adapt_intent(human_fixture(), "rolling_stop", root_height=2.)
        with self.assertRaisesRegex(ValueError, "Infeasible"):
            solve_reference(intent)


if __name__ == "__main__":
    unittest.main()
