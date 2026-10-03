"""Mink mapping unit tests plus read-only checks of locally generated results."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("mink", "mujoco"))
if AVAILABLE:
    import mujoco
    from retarget_mink import (anchored_targets, calibrated_root_rotations,
                              finite_difference_qvel, root_rotating_targets)


@unittest.skipUnless(AVAILABLE, "Optional Mink environment not installed")
class RetargetMappingTests(unittest.TestCase):
    def test_rotating_offsets_equal_original_when_root_does_not_rotate(self):
        root = np.array([[0., 0., 1.], [.1, .2, 1.3], [.3, .1, 1.05]])
        relative = np.array([[0., .1, -.9], [0., -.1, -.9]])
        human = root[:, None] + relative
        human[1, 0, 0] += .03
        initial = np.array([[.01, .15, .127], [.01, -.15, .127]])
        actual = root_rotating_targets(human, root, initial, [0, 0, .9], .7,
                                       np.tile([1, 0, 0, 0], (3, 1)))
        np.testing.assert_allclose(actual, anchored_targets(human, initial, .7), atol=1e-14)

    def test_rigid_yaw_rotates_complete_robot_landmark_offsets(self):
        rotation = Rotation.from_euler("z", [[0.], [45.], [90.]], degrees=True)
        root = np.array([[0., 0., 1.], [.1, .2, 1.3], [.3, .1, 1.05]])
        human_relative = np.array([[.04, .1, -.9], [.04, -.1, -.9]])
        human = root[:, None] + np.einsum("fij,kj->fki", rotation.as_matrix(), human_relative)
        robot_root = np.array([0., 0., .9])
        initial = np.array([[.01, .15, .127], [.01, -.15, .127]])
        actual = root_rotating_targets(human, root, initial, robot_root, .7,
                                       rotation.as_quat()[:, [3, 0, 1, 2]])
        expected = (anchored_targets(root, robot_root, .7)[:, None]
                    + np.einsum("fij,kj->fki", rotation.as_matrix(), initial - robot_root))
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        np.testing.assert_allclose(actual[0], initial, atol=1e-14)
        np.testing.assert_allclose(actual[1, :, 2] - actual[0, :, 2], .21, atol=1e-14)
        self.assertGreater(np.max(np.abs(actual - anchored_targets(human, initial, .7))), .01)

    def test_rotating_offsets_reject_nonfinite_and_malformed_inputs(self):
        human = np.zeros((2, 2, 3))
        root = np.zeros((2, 3))
        initial = np.zeros((2, 3))
        quats = np.tile([1., 0., 0., 0.], (2, 1))
        for scale in (0., -1., float("nan"), float("inf")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                root_rotating_targets(human, root, initial, np.zeros(3), scale, quats)
        with self.assertRaises(ValueError):
            root_rotating_targets(human, root[:1], initial, np.zeros(3), 1., quats)
        with self.assertRaises(ValueError):
            root_rotating_targets(human, root, initial, np.zeros(3), 1., quats * 2)
        human[1, 1, 0] = float("nan")
        with self.assertRaises(ValueError):
            root_rotating_targets(human, root, initial, np.zeros(3), 1., quats)

    def test_anchoring_retains_scaled_global_vertical_motion(self):
        source = np.array([[[1., 2., .05], [2., 3., .06]],
                           [[1.1, 2.2, .55], [2.1, 3.2, .56]],
                           [[1.2, 2.3, .10], [2.2, 3.3, .11]]])
        initial = np.array([[0., .15, .127], [0., -.15, .127]])
        target = anchored_targets(source, initial, .72)
        np.testing.assert_allclose(target[0], initial)
        np.testing.assert_allclose(np.diff(target, axis=0), .72 * np.diff(source, axis=0))
        self.assertAlmostEqual(target[1, 0, 2] - target[0, 0, 2], .36)
        self.assertGreater(target[-1, 0, 2], target[0, 0, 2])

    def test_asf_basis_and_initial_mounting_are_removed(self):
        a = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        mounting = Rotation.from_euler("ZYX", [45., 20., -7.], degrees=True).as_matrix()
        exported = Rotation.from_matrix(mounting @ a).as_quat()[[3, 0, 1, 2]]
        result, heading = calibrated_root_rotations(np.tile(exported, (3, 1)))
        np.testing.assert_allclose(result, np.tile([1, 0, 0, 0], (3, 1)), atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(heading), 1.)
        np.testing.assert_allclose(heading[2], [0, 0, 1], atol=1e-14)

    def test_yaw_motion_is_not_removed_framewise(self):
        a = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        frames = Rotation.from_euler("z", [[40.], [50.], [60.]], degrees=True).as_matrix() @ a
        exported = Rotation.from_matrix(frames).as_quat()[:, [3, 0, 1, 2]]
        result, _ = calibrated_root_rotations(exported)
        actual = Rotation.from_quat(result[:, [1, 2, 3, 0]]).as_euler("ZYX", degrees=True)
        np.testing.assert_allclose(actual[:, 0], [0., 10., 20.], atol=1e-12)

    def test_configuration_velocity_uses_mujoco_tangent_space(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
        times = np.arange(5) / 120
        positions = np.tile(model.qpos0, (5, 1))
        expected = np.array([.5, -.1, .2, 0., .3, 0.])
        for frame in range(1, 5):
            mujoco.mj_integratePos(model, positions[frame], expected, times[frame])
        sampled, intervals = finite_difference_qvel(model, positions, times)
        np.testing.assert_allclose(intervals, np.tile(expected, (4, 1)), atol=1e-12)
        np.testing.assert_allclose(sampled, np.tile(expected, (5, 1)), atol=1e-12)


ARTIFACT = ROOT / "outputs/mink-cmu-16_03/robot_reference.npz"
MODEL = ROOT / "assets/robots/WF_TRON1A/mujoco/robot.xml"


@unittest.skipUnless(AVAILABLE and ARTIFACT.exists() and MODEL.exists(), "Local IK output/model not present")
class MinkResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(ARTIFACT, allow_pickle=False) as archive:
            cls.result = {name: archive[name] for name in archive.files}
        cls.report = json.loads((ARTIFACT.parent / "report.json").read_text())
        cls.model = mujoco.MjModel.from_xml_path(str(MODEL))

    def test_shape_finiteness_and_quaternion_norms(self):
        r = self.result
        self.assertEqual(r["qpos"].shape, (410, 15))
        self.assertEqual(r["qvel"].shape, (410, 14))
        for name, values in r.items():
            if values.dtype.kind in "fiu":
                self.assertTrue(np.isfinite(values).all(), name)
        np.testing.assert_allclose(np.linalg.norm(r["qpos"][:, 3:7], axis=1), 1., atol=1e-12)
        np.testing.assert_allclose(np.diff(r["time_s"]), 1 / 120, atol=1e-12)

    def test_limits_and_actual_interframe_speed(self):
        r = self.result
        mask = r["joint_limited"]
        q = r["joint_positions_rad"][:, mask]
        self.assertTrue((q >= r["joint_ranges_rad"][mask, 0] - 1e-8).all())
        self.assertTrue((q <= r["joint_ranges_rad"][mask, 1] + 1e-8).all())
        speed = np.diff(r["joint_positions_rad"], axis=0) / np.diff(r["time_s"])[:, None]
        self.assertLessEqual(abs(speed).max(), self.report["settings"]["preview_joint_speed_limit_rad_s"] + 1e-6)

    def test_prescribed_root_and_frozen_wheels_are_honest(self):
        r = self.result
        np.testing.assert_allclose(r["qpos"][:, :3], r["root_target_position_m"], atol=1e-9)
        dots = np.abs((r["qpos"][:, 3:7] * r["root_target_quat_wxyz"]).sum(axis=1))
        np.testing.assert_allclose(dots, 1., atol=1e-12)
        wheels = [i for i, name in enumerate(r["joint_names"]) if str(name).startswith("wheel_")]
        np.testing.assert_allclose(r["joint_positions_rad"][:, wheels], 0., atol=1e-12)
        for flag in ("physics_validated", "policy_trained", "hardware_ready"):
            self.assertIs(self.report[flag], False)
        self.assertIs(self.report["root_prescribed"], True)

    def test_targets_recompute_from_original_motion(self):
        r = self.result
        with np.load(ROOT / "outputs/cmu-16_03/human_reference.npz", allow_pickle=False) as source:
            names = list(source["joint_names"])
            human = source["joint_positions_world_m"][:, [names.index("ltibia"), names.index("rtibia")]]
        transformed_delta = (human - human[:1]) @ np.array(self.report["initial_heading_alignment"]).T
        expected = np.array(self.report["initial_wheel_position_m"])[None] + self.report["uniform_human_scale"] * transformed_delta
        np.testing.assert_allclose(r["target_positions_world_m"], expected, atol=1e-12)
        self.assertAlmostEqual(self.report["uniform_human_scale"], .7221145082725878, places=9)

    def test_saved_sites_match_full_clip_forward_kinematics(self):
        r = self.result
        data = mujoco.MjData(self.model)
        sites = [self.model.site(str(name)).id for name in r["site_names"]]
        for frame, q in enumerate(r["qpos"]):
            data.qpos[:] = q
            mujoco.mj_forward(self.model, data)
            np.testing.assert_allclose(data.site_xpos[sites], r["site_positions_world_m"][frame], atol=1e-11)

    def test_error_report_matches_actual_position_residual(self):
        r = self.result
        names = list(r["site_names"])
        selected = [names.index(name) for name in r["target_names"]]
        errors = np.linalg.norm(r["site_positions_world_m"][:, selected] - r["target_positions_world_m"], axis=-1)
        np.testing.assert_allclose(errors, r["target_error_m"], atol=1e-12)
        self.assertAlmostEqual(np.sqrt(np.mean(errors ** 2)), self.report["all_wheel_errors"]["rmse_m"])
        self.assertAlmostEqual(errors.max(), self.report["all_wheel_errors"]["max_m"])
        self.assertAlmostEqual(r["ground_penetration_m"].max(), self.report["max_ground_penetration_m"])


if __name__ == "__main__":
    unittest.main()
