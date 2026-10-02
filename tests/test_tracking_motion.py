"""Offline reference contract: named reordering, SO(3), world heights and FK."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
AVAILABLE = importlib.util.find_spec("mujoco") is not None
if AVAILABLE:
    import mujoco
    from export_tracking_motion import (
        DEFAULT_MODEL, DEFAULT_URDF, ISAAC_BODY_NAMES, ISAAC_JOINT_NAMES,
        UrdfKinematics, angular_velocity_world, continuous_quaternions,
        convert_motion, resample_qpos,
    )
ARTIFACT = ROOT / "outputs/gmr-cmu-16_03/robot_reference.npz"


class RuntimeNameMappingTests(unittest.TestCase):
    """Run the real task's NumPy-compatible name mapping without Isaac/Torch."""
    @classmethod
    def setUpClass(cls):
        path = ROOT / "training/tron1_tracking.py"
        if not path.exists():
            raise unittest.SkipTest("Training task is not present")
        tree = ast.parse(path.read_text())
        functions = [copy.deepcopy(node) for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("reference_name_permutation", "reorder_motion_reference")]
        namespace = {"np": np}
        exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(path), "exec"), namespace)
        cls.permutation = staticmethod(namespace["reference_name_permutation"])
        cls.reorder = staticmethod(namespace["reorder_motion_reference"])

    def test_every_body_feature_and_joint_feature_follows_names_not_axis_position(self):
        reference_bodies = ["base_Link", "abad_L_Link", "hip_L_Link", "knee_L_Link", "wheel_L_Link",
                            "abad_R_Link", "hip_R_Link", "knee_R_Link", "wheel_R_Link", "limx_imu"]
        native_bodies = ["base_Link", "abad_L_Link", "abad_R_Link", "limx_imu", "hip_L_Link",
                         "hip_R_Link", "knee_L_Link", "knee_R_Link", "wheel_L_Link", "wheel_R_Link"]
        reference_joints = [f"j{i}" for i in range(8)]
        native_joints = reference_joints[::-1]
        body_order = [0, 1, 5, 9, 2, 6, 3, 7, 4, 8]
        arrays = {
            "joint_pos": np.arange(3 * 8).reshape(3, 8),
            "joint_vel": np.arange(3 * 8).reshape(3, 8) + 100,
            "_body_pos_w": np.arange(3 * 10 * 3).reshape(3, 10, 3) + 200,
            "_body_quat_w": np.arange(3 * 10 * 4).reshape(3, 10, 4) + 300,
            "_body_lin_vel_w": np.arange(3 * 10 * 3).reshape(3, 10, 3) + 400,
            "_body_ang_vel_w": np.arange(3 * 10 * 3).reshape(3, 10, 3) + 500,
        }
        original = {name: values.copy() for name, values in arrays.items()}
        motion = SimpleNamespace(**arrays, _body_indexes=np.array([0, 8, 9]))
        mapping = self.reorder(motion, reference_joints, reference_bodies, native_joints, native_bodies)
        self.assertEqual(mapping["runtime_body_to_reference_indices"], body_order)
        self.assertEqual(mapping["runtime_joint_to_reference_indices"], list(range(7, -1, -1)))
        json.dumps(mapping)  # This object is safe to embed in the run manifest.
        for name, source in original.items():
            indices = body_order if name.startswith("_body") else list(range(7, -1, -1))
            np.testing.assert_array_equal(getattr(motion, name), source[:, indices])
            np.testing.assert_array_equal(arrays[name], source)  # source arrays preserved
        np.testing.assert_array_equal(motion._body_indexes, [0, 8, 9])
        # Existing upstream subset selection now picks both actual wheels.
        np.testing.assert_array_equal(motion._body_pos_w[:, motion._body_indexes],
                                      original["_body_pos_w"][:, [0, 4, 8]])

    def test_missing_extra_duplicate_and_malformed_names_fail_fast(self):
        invalid = [(["a", "b"], ["a", "c"]), (["a", "b"], ["a"]),
                   (["a"], ["a", "b"]), (["a", "a"], ["a", "b"]),
                   (["a", "b"], ["a", "a"]), ([["a", "b"]], ["a", "b"]),
                   ([1, 2], ["1", "2"]), (["", "a"], ["", "a"])]
        for reference, native in invalid:
            with self.subTest(reference=reference, native=native), self.assertRaises(ValueError):
                self.permutation(reference, native, "test")

    def test_bad_feature_shape_fails_before_mutating_any_tensor(self):
        motion = SimpleNamespace(joint_pos=np.array([[1, 2]]), joint_vel=np.array([[3, 4]]),
                                 _body_pos_w=np.zeros((1, 2, 3)), _body_quat_w=np.zeros((1, 2, 4)),
                                 _body_lin_vel_w=np.zeros((1, 2, 3)), _body_ang_vel_w=np.zeros((1, 2, 4)))
        with self.assertRaises(ValueError):
            self.reorder(motion, ["j0", "j1"], ["b0", "b1"], ["j1", "j0"], ["b1", "b0"])
        np.testing.assert_array_equal(motion.joint_pos, [[1, 2]])


class CommandLifecycleTests(unittest.TestCase):
    """Execute the actual task's lifecycle methods with tiny NumPy stubs.

    This checks phase/reset logic without importing Isaac or installing Torch.
    Runtime Isaac smoke tests must separately validate FK and manager wiring.
    """
    @classmethod
    def setUpClass(cls):
        task_path = ROOT / "training/tron1_tracking.py"
        if not task_path.exists():
            raise unittest.SkipTest("Training task is not present")
        tree = ast.parse(task_path.read_text())
        task = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TronMotionCommand")
        methods = [copy.deepcopy(node) for node in task.body
                   if isinstance(node, ast.FunctionDef) and node.name in ("_resample_command", "_update_command")]
        if len(methods) != 2:
            raise AssertionError("Expected explicit phase-preserving resampling override")

        class Zeroable(np.ndarray):
            def zero_(self):
                self.fill(0)

        class Base:
            def __init__(self):
                self._env = SimpleNamespace(episode_length_buf=np.array([0, 3]))
                self.time_steps = np.array([7, 3])
                self.motion = SimpleNamespace(time_step_total=221)
                self.cfg = SimpleNamespace(adaptive_alpha=.1)
                self.bin_failed_count = np.zeros(4)
                self._current_bin_failed = np.zeros(4).view(Zeroable)
                self.refreshed_phases = []
                self.resets = []

            def _resample_command(self, env_ids):
                self.resets.append(list(env_ids))
                self.time_steps[env_ids] = 0

            def _refresh_relative_targets(self):
                self.refreshed_phases.append(self.time_steps.copy())

        module = ast.Module(body=[ast.ClassDef(name="Probe", bases=[ast.Name(id="Base", ctx=ast.Load())],
                                              keywords=[], body=methods, decorator_list=[])], type_ignores=[])
        namespace = {"Base": Base, "torch": SimpleNamespace(clamp=lambda values, max: np.minimum(values, max))}
        exec(compile(ast.fix_missing_locations(module), str(task_path), "exec"), namespace)
        cls.Probe = namespace["Probe"]

    def test_explicit_reset_refreshes_without_advancing(self):
        command = self.Probe()
        command._resample_command(np.array([0]))
        np.testing.assert_array_equal(command.time_steps, [0, 3])
        np.testing.assert_array_equal(command.refreshed_phases[-1], [0, 3])
        self.assertEqual(command.resets, [[0]])

    def test_auto_reset_frame_is_not_skipped_but_live_env_advances(self):
        command = self.Probe()
        command._resample_command(np.array([0]))
        command._update_command()
        np.testing.assert_array_equal(command.time_steps, [0, 4])
        command._env.episode_length_buf[:] = [1, 4]
        command._update_command()
        np.testing.assert_array_equal(command.time_steps, [1, 5])

    def test_clip_end_clamps_without_resampling_or_teleport(self):
        command = self.Probe()
        command.time_steps[:] = [219, 220]
        command._env.episode_length_buf[:] = [220, 221]
        command._update_command()
        np.testing.assert_array_equal(command.time_steps, [220, 220])
        self.assertEqual(command.resets, [])

    def test_empty_resample_does_not_refresh_stale_targets(self):
        command = self.Probe()
        command._resample_command(np.array([], dtype=int))
        self.assertEqual(command.refreshed_phases, [])


@unittest.skipUnless(AVAILABLE, "Optional MuJoCo environment not installed")
class ResampleMotionTests(unittest.TestCase):
    def test_world_height_and_joint_names_are_not_rebased(self):
        times = np.arange(121) / 120
        qpos = np.zeros((121, 15))
        qpos[:, 0], qpos[:, 2], qpos[:, 3] = 2 + times, 1.3 + times, 1
        qpos[:, 7:] = times[:, None] * np.arange(1, 9)
        t, st, q, info = resample_qpos(times, qpos, 50)
        np.testing.assert_allclose(t, np.arange(51) / 50)
        np.testing.assert_allclose(st, t)
        np.testing.assert_allclose(q[:, 2], 1.3 + t)
        np.testing.assert_allclose(q[:, 7:], t[:, None] * np.arange(1, 9))
        self.assertIs(info["framewise_ground_alignment"], False)

    def test_slerp_preserves_world_rotation_and_sign_continuity(self):
        times = np.arange(121) / 120
        rot = Rotation.from_rotvec(times[:, None] * np.array([.4, -.3, .2]))
        qpos = np.zeros((121, 15))
        qpos[:, 3:7] = rot.as_quat()[:, [3, 0, 1, 2]]
        qpos[::2, 3:7] *= -1
        t, _, q, _ = resample_qpos(times, qpos, 50)
        expected = Rotation.from_rotvec(t[:, None] * np.array([.4, -.3, .2]))
        actual = Rotation.from_quat(q[:, [4, 5, 6, 3]])
        np.testing.assert_allclose((expected.inv() * actual).magnitude(), 0, atol=1e-14)
        self.assertTrue(np.all(np.sum(q[1:, 3:7] * q[:-1, 3:7], axis=1) >= 0))

    def test_world_angular_velocity_is_not_body_angular_velocity(self):
        dt = .02
        times = np.arange(51) * dt
        omega = np.array([.7, -.2, .1])
        initial = Rotation.from_euler("xyz", [.4, .5, -.3])
        rot = Rotation.from_rotvec(times[:, None] * omega) * initial
        q = rot.as_quat()[:, [3, 0, 1, 2]]
        np.testing.assert_allclose(angular_velocity_world(q, dt),
                                   np.broadcast_to(omega, (len(q), 3)), atol=1e-13)

    def test_optional_holds_preserve_clip_and_report_truncated_subframe(self):
        times = np.arange(410) / 120
        qpos = np.zeros((410, 15))
        qpos[:, 2], qpos[:, 3] = 1 + times, 1
        t, st, q, info = resample_qpos(times, qpos, 50, .2, .3)
        self.assertEqual(len(t), 171 + 10 + 15)
        np.testing.assert_allclose(q[:11, 2], 1)
        np.testing.assert_allclose(q[-16:, 2], 4.4)
        self.assertAlmostEqual(info["omitted_subframe_tail_s"], 1 / 120)
        self.assertTrue(np.all(np.diff(st) >= 0))

    def test_invalid_inputs_are_rejected(self):
        q = np.zeros((3, 15)); q[:, 3] = 1
        for times, fps, pre in (([0, 0, 1], 50, 0), ([0, 1, 2], 0, 0), ([0, 1, 2], 50, -1)):
            with self.assertRaises(ValueError):
                resample_qpos(np.array(times), q, fps, pre)
        with self.assertRaises(ValueError):
            continuous_quaternions(np.zeros((2, 4)))


@unittest.skipUnless(AVAILABLE and ARTIFACT.exists() and DEFAULT_MODEL.exists() and DEFAULT_URDF.exists(),
                     "Local GMR reference and official WF model not present")
class TrackingReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result, cls.report = convert_motion(ARTIFACT)
        cls.model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL))
        cls.urdf = UrdfKinematics(DEFAULT_URDF)

    def test_upstream_shapes_float32_and_name_order(self):
        r = self.result
        self.assertEqual(r["fps"].tolist(), [50.])
        self.assertEqual(tuple(r["joint_names"]), ISAAC_JOINT_NAMES)
        self.assertEqual(tuple(r["body_names"]), ISAAC_BODY_NAMES)
        self.assertEqual(r["joint_pos"].shape, (171, 8))
        for key, width in (("body_pos_w", 3), ("body_quat_w", 4),
                           ("body_lin_vel_w", 3), ("body_ang_vel_w", 3)):
            self.assertEqual(r[key].shape, (171, 10, width))
            self.assertEqual(r[key].dtype, np.float32)
            self.assertTrue(np.isfinite(r[key]).all())
        np.testing.assert_allclose(np.linalg.norm(r["body_quat_w"], axis=-1), 1., atol=2e-7)

    def test_named_joint_mapping_and_full_height_are_preserved(self):
        r = self.result
        for index, name in enumerate(ISAAC_JOINT_NAMES):
            qid = int(self.model.joint(name).qposadr[0])
            np.testing.assert_allclose(r["joint_pos"][:, index], r["qpos_mujoco"][:, qid], atol=6e-8)
        np.testing.assert_allclose(r["body_pos_w"][:, 0], r["qpos_mujoco"][:, :3], atol=6e-8)
        self.assertGreater(self.report["root_rise_from_initial_m"], .3)
        self.assertLess(self.report["independent_urdf_fk_max_position_error_m"], 1e-12)
        self.assertLess(self.report["independent_urdf_fk_max_rotation_matrix_error"], 1e-12)

    def test_fixed_imu_pose_follows_root_but_com_velocities_differ(self):
        r = self.result
        imu = ISAAC_BODY_NAMES.index("limx_imu")
        np.testing.assert_array_equal(r["body_pos_w"][:, 0], r["body_pos_w"][:, imu])
        np.testing.assert_array_equal(r["body_quat_w"][:, 0], r["body_quat_w"][:, imu])
        np.testing.assert_allclose(r["body_lin_vel_w"][:, imu], r["qvel_mujoco"][:, :3], atol=1e-7)
        self.assertGreater(np.max(abs(r["body_lin_vel_w"][:, 0] - r["body_lin_vel_w"][:, imu])), .01)

    def test_com_jacobian_velocity_matches_infinitesimal_fk_difference(self):
        """Independent check catches local/world angular and COM/link mixups."""
        r, model = self.result, self.model
        epsilon = 1e-6
        joint_names = tuple(model.joint(i).name for i in range(1, model.njnt))
        qids = [int(model.joint(name).qposadr[0]) for name in joint_names]
        for frame in (0, 30, 78, 90, 120, 170):
            q, v = r["qpos_mujoco"][frame], r["qvel_mujoco"][frame]
            plus, minus = q.copy(), q.copy()
            mujoco.mj_integratePos(model, plus, v, epsilon)
            mujoco.mj_integratePos(model, minus, v, -epsilon)
            poses = [self.urdf.forward(state[:3], state[3:7],
                                       {name: state[index] for name, index in zip(joint_names, qids)})
                     for state in (plus, minus)]
            for index, name in enumerate(ISAAC_BODY_NAMES):
                p_plus, r_plus = poses[0][name]
                p_minus, r_minus = poses[1][name]
                velocity = ((p_plus + r_plus @ self.urdf.com[name]) -
                            (p_minus + r_minus @ self.urdf.com[name])) / (2 * epsilon)
                omega = Rotation.from_matrix(r_plus @ r_minus.T).as_rotvec() / (2 * epsilon)
                np.testing.assert_allclose(velocity, r["body_lin_vel_w"][frame, index], atol=3e-7)
                np.testing.assert_allclose(omega, r["body_ang_vel_w"][frame, index], atol=3e-7)

    def test_wheel_placeholders_are_explicitly_excluded(self):
        r = self.result
        np.testing.assert_array_equal(r["joint_tracking_mask"], [1, 1, 1, 1, 1, 1, 0, 0])
        for key in ("body_orientation_tracking_mask", "body_angular_velocity_tracking_mask"):
            for index, name in enumerate(ISAAC_BODY_NAMES):
                self.assertEqual(bool(r[key][index]), not name.startswith("wheel_"))
        for flag in ("wheel_spin_observed", "wheel_orientation_is_training_target", "physics_validated", "policy_trained"):
            self.assertIs(self.report[flag], False)

    def test_single_uniform_floor_offset_changes_neither_shape_nor_velocity(self):
        shifted, info = convert_motion(ARTIFACT, z_offset=.0066)
        delta = shifted["body_pos_w"] - self.result["body_pos_w"]
        np.testing.assert_allclose(delta[..., :2], 0, atol=1e-7)
        np.testing.assert_allclose(delta[..., 2], .0066, atol=1.3e-7)
        np.testing.assert_allclose(shifted["qpos_mujoco"][:, 2] - self.result["qpos_mujoco"][:, 2], .0066, atol=1e-15)
        for key in ("joint_pos", "joint_vel", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            np.testing.assert_allclose(shifted[key], self.result[key], atol=1e-7)
        self.assertEqual(info["uniform_z_offset_m"], .0066)
        self.assertAlmostEqual(info["root_rise_from_initial_m"], self.report["root_rise_from_initial_m"])

    def test_serializes_without_pickle(self):
        with tempfile.TemporaryDirectory(prefix="tron1-motion-test-") as directory:
            path = Path(directory) / "test.npz"
            np.savez_compressed(path, **self.result)
            with np.load(path, allow_pickle=False) as archive:
                for key, value in self.result.items():
                    np.testing.assert_array_equal(archive[key], value)

    def test_incompatible_isaac_order_is_rejected(self):
        with self.assertRaises(ValueError):
            convert_motion(ARTIFACT, joint_names=ISAAC_JOINT_NAMES[::-1][:-1])
        with self.assertRaises(ValueError):
            convert_motion(ARTIFACT, body_names=ISAAC_BODY_NAMES[::-1])


if __name__ == "__main__":
    unittest.main()
