"""Actual-contact slip geometry and PhysX-buffer decoding without Isaac."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.tron1_contact_slip import (
    aggregate_ground_contacts, configure_nonwheel_ground_sensors,
    configure_wheel_ground_sensors, contact_point_lateral_slip_m_s,
    sample_nonwheel_ground_force_norm, sample_raw_isaac_ground_contacts,
)


class SlipGeometryTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[0., 0., -.127]])
        self.normals = np.array([[0., 0., 1.]])
        self.center = np.zeros(3)
        self.velocity = np.zeros(3)
        self.angular = np.zeros(3)
        self.quaternion = np.array([1., 0., 0., 0.])

    def calculate(self, **overrides):
        arguments = dict(points_w=self.points, normals_w=self.normals,
                         body_com_position_w=self.center, body_com_linear_velocity_w=self.velocity,
                         body_angular_velocity_w=self.angular, body_link_quaternion_wxyz=self.quaternion)
        arguments.update(overrides)
        return contact_point_lateral_slip_m_s(**arguments)

    def test_lateral_translation_is_actual_point_slip(self):
        self.velocity[:] = [1.2, -.35, .1]
        np.testing.assert_allclose(self.calculate(), [-.35])

    def test_pure_rolling_and_spinning_have_zero_lateral_slip(self):
        self.angular[:] = [0., 10., 0.]
        self.velocity[:] = [1.27, 0., 0.]
        np.testing.assert_allclose(self.calculate(), [0.], atol=1e-12)
        self.velocity[:] = 0.
        np.testing.assert_allclose(self.calculate(), [0.], atol=1e-12)

    def test_roll_rotation_contributes_at_contact_not_center(self):
        self.angular[:] = [2., 0., 0.]
        np.testing.assert_allclose(self.calculate(), [.254])

    def test_com_offset_must_pair_with_com_velocity(self):
        self.center[:] = [.2, 0., 0.]
        self.velocity[:] = [0., .4, 0.]
        self.angular[:] = [0., 0., 2.]
        np.testing.assert_allclose(self.calculate(), [0.], atol=1e-12)
        # A link-origin position paired with COM velocity invents 0.4 m/s slip.
        np.testing.assert_allclose(self.calculate(body_com_position_w=np.zeros(3)), [.4])

    def test_world_rotation_invariance_and_axle_projection(self):
        rotation = Rotation.from_euler("xyz", [.4, -.2, 1.1])
        matrix = rotation.as_matrix()
        quaternion = rotation.as_quat()[[3, 0, 1, 2]]
        self.velocity[:] = [.8, .3, -.1]
        self.angular[:] = [.2, -.4, .1]
        baseline = self.calculate()
        actual = self.calculate(points_w=self.points @ matrix.T, normals_w=self.normals @ matrix.T,
                                body_com_position_w=matrix @ self.center,
                                body_com_linear_velocity_w=matrix @ self.velocity,
                                body_angular_velocity_w=matrix @ self.angular,
                                body_link_quaternion_wxyz=quaternion)
        np.testing.assert_allclose(actual, baseline, atol=1e-12)

    def test_normal_sign_and_magnitude_do_not_change_slip(self):
        self.velocity[:] = [0., .3, 0.]
        np.testing.assert_allclose(self.calculate(normals_w=-2 * self.normals), [.3])

    def test_ground_relative_velocity_is_subtracted(self):
        self.velocity[:] = [0., .35, 0.]
        np.testing.assert_allclose(self.calculate(ground_point_velocity_w=[0., .1, 0.]), [.25])
        np.testing.assert_allclose(self.calculate(ground_point_velocity_w=[[0., .1, 0.]]), [.25])

    def test_degenerate_axle_rejected(self):
        with self.assertRaisesRegex(ValueError, "undefined"):
            self.calculate(normals_w=np.array([[0., 1., 0.]]))

    def test_nonfinite_and_mismatched_shapes_rejected(self):
        for overrides in ({"points_w": [[np.nan, 0., 0.]]},
                          {"normals_w": np.zeros((2, 3))},
                          {"body_com_position_w": [0., 0.]},
                          {"body_link_quaternion_wxyz": np.zeros(4)},
                          {"ground_point_velocity_w": np.zeros((2, 3))}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.calculate(**overrides)

    def test_no_contact_and_below_threshold_have_zero_count(self):
        arguments = (self.center, self.velocity, self.angular, self.quaternion)
        count, force, slip = aggregate_ground_contacts(np.empty((0, 3)), np.empty((0, 3)), np.empty(0), *arguments)
        self.assertEqual((count, slip), (0, 0.))
        np.testing.assert_array_equal(force, [0., 0., 0.])
        count, force, slip = aggregate_ground_contacts(self.points, self.normals, np.array([5.]), *arguments)
        self.assertEqual((count, slip), (0, 0.))
        np.testing.assert_array_equal(force, [0., 0., 5.])

    def test_force_weighted_rms_preserves_opposite_contact_slip(self):
        points = np.array([[.1, 0., -.127], [-.2, 0., -.127]])
        normals = np.repeat(self.normals, 2, axis=0)
        count, force, slip = aggregate_ground_contacts(
            points, normals, np.array([10., 30.]), self.center,
            self.velocity, [0., 0., 2.], self.quaternion,
        )
        self.assertEqual(count, 2)
        np.testing.assert_array_equal(force, [0., 0., 40.])
        self.assertAlmostEqual(slip, np.sqrt((10 * .2**2 + 30 * .4**2) / 40))

    def test_force_and_threshold_validation(self):
        for forces, threshold in (([-1.], 5.), ([np.nan], 5.), ([10.], -1.), ([10.], np.nan)):
            with self.subTest(forces=forces, threshold=threshold), self.assertRaises(ValueError):
                aggregate_ground_contacts(self.points, self.normals, forces, self.center,
                                          self.velocity, self.angular, self.quaternion, threshold)


def fake_isaac_env():
    names = ["base_Link", "wheel_L_Link", "wheel_R_Link"]
    center = np.zeros((2, 3, 3))
    center[:, :, 2] = .127
    velocity = np.zeros_like(center)
    velocity[:, 1:, 1] = [[.2, -.3], [.4, -.5]]
    quaternion = np.zeros((2, 3, 4))
    quaternion[:, :, 0] = 1.
    data = SimpleNamespace(body_com_pos_w=center, body_com_lin_vel_w=velocity,
                           body_com_ang_vel_w=np.zeros_like(center), body_link_quat_w=quaternion)
    sensors = {}
    for wheel, side in enumerate(("L", "R")):
        # Two filters: deliberately use nontrivial offsets and leave first slots unused.
        force = np.array([[0.], [10. + wheel], [20. + wheel], [30. + wheel]])
        points = np.zeros((4, 3))
        normals = np.repeat([[0., 0., 1.]], 4, axis=0)
        counts = np.array([[1, 1], [1, 0]], dtype=int)
        starts = np.array([[1, 3], [2, 0]], dtype=int)
        buffers = (force, points, normals, np.zeros((4, 1)), counts, starts)
        view = SimpleNamespace(filter_count=2, get_contact_data=lambda dt, buffers=buffers: buffers)
        sensors[f"wheel_{side}_ground"] = SimpleNamespace(
            num_bodies=1, body_names=[f"wheel_{side}_Link"],
            cfg=SimpleNamespace(filter_prim_paths_expr=["/actual/ground/a", "/actual/ground/b"]),
            contact_physx_view=view,
        )

    class Scene(dict):
        pass

    scene = Scene(robot=SimpleNamespace(body_names=names, data=data))
    scene.sensors = sensors
    return SimpleNamespace(num_envs=2, scene=scene, cfg=SimpleNamespace(sim=SimpleNamespace(dt=.005)))


class IsaacBufferTests(unittest.TestCase):
    def test_raw_contact_ranges_and_per_wheel_body_mapping(self):
        result = sample_raw_isaac_ground_contacts(fake_isaac_env())
        np.testing.assert_array_equal(result["wheel_ground_contact_count"], [[2, 2], [1, 1]])
        np.testing.assert_allclose(result["wheel_ground_contact_force_w_n"][:, :, 2], [[40., 42.], [20., 21.]])
        np.testing.assert_allclose(result["wheel_lateral_contact_slip_m_s"], [[.2, .3], [.4, .5]])

    def test_unmatched_filter_and_multiple_sensor_bodies_rejected(self):
        env = fake_isaac_env()
        env.scene.sensors["wheel_L_ground"].cfg.filter_prim_paths_expr = []
        with self.assertRaisesRegex(ValueError, "no ground filter"):
            sample_raw_isaac_ground_contacts(env)
        env = fake_isaac_env()
        env.scene.sensors["wheel_L_ground"].num_bodies = 2
        with self.assertRaisesRegex(ValueError, "select only"):
            sample_raw_isaac_ground_contacts(env)

    def test_out_of_bounds_contact_buffer_rejected(self):
        env = fake_isaac_env()
        view = env.scene.sensors["wheel_L_ground"].contact_physx_view
        original = view.get_contact_data(dt=.005)
        broken = (*original[:4], np.array([[1, 100], [1, 0]]), original[5])
        view.get_contact_data = lambda dt: broken
        with self.assertRaisesRegex(ValueError, "buffer overrun"):
            sample_raw_isaac_ground_contacts(env)

    def test_configuration_requires_ground_paths_and_separate_sensors(self):
        fake_cfg = lambda **kwargs: SimpleNamespace(**kwargs)
        fake_module = SimpleNamespace(ContactSensorCfg=fake_cfg)
        with patch.dict(sys.modules, {"isaaclab.sensors": fake_module}):
            scene = SimpleNamespace()
            configure_wheel_ground_sensors(scene, ["/ground/actualCollider"])
            self.assertEqual(scene.wheel_L_ground.prim_path, "{ENV_REGEX_NS}/Robot/wheel_L_Link")
            self.assertEqual(scene.wheel_R_ground.prim_path, "{ENV_REGEX_NS}/Robot/wheel_R_Link")
            self.assertTrue(scene.wheel_L_ground.track_contact_points)
            self.assertEqual(scene.wheel_L_ground.filter_prim_paths_expr, ["/ground/actualCollider"])
            with self.assertRaisesRegex(ValueError, "overwrite"):
                configure_wheel_ground_sensors(scene, ["/ground/actualCollider"])
            with self.assertRaisesRegex(ValueError, "absolute ground"):
                configure_wheel_ground_sensors(SimpleNamespace(), [])

    def test_nonwheel_raw_ground_force_excludes_unused_buffer_entries(self):
        env = fake_isaac_env()
        sensor = env.scene.sensors["wheel_L_ground"]
        sensor.body_names = ["base_Link"]
        env.scene.sensors["base_ground"] = sensor
        # Selected actual ground entries are 10,30 in env0 and20 in env1.
        result = sample_nonwheel_ground_force_norm(env, ["base_ground"])
        np.testing.assert_array_equal(result["nonwheel_ground_contact_count"], [2, 1])
        np.testing.assert_allclose(result["nonwheel_ground_contact_force_norm_n"], [40., 20.])
        result = sample_nonwheel_ground_force_norm(env, ["base_ground"], force_threshold_n=25.)
        np.testing.assert_array_equal(result["nonwheel_ground_contact_count"], [1, 0])
        np.testing.assert_allclose(result["nonwheel_ground_contact_force_norm_n"], [40., 20.])

    def test_nonwheel_sampling_rejects_wheels_and_duplicate_sensors(self):
        env = fake_isaac_env()
        with self.assertRaisesRegex(ValueError, "nonwheel body"):
            sample_nonwheel_ground_force_norm(env, ["wheel_L_ground"])
        with self.assertRaisesRegex(ValueError, "unique"):
            sample_nonwheel_ground_force_norm(env, ["wheel_L_ground", "wheel_L_ground"])

    def test_nonwheel_configuration_has_one_sensor_per_body(self):
        fake_cfg = lambda **kwargs: SimpleNamespace(**kwargs)
        fake_module = SimpleNamespace(ContactSensorCfg=fake_cfg)
        with patch.dict(sys.modules, {"isaaclab.sensors": fake_module}):
            scene = SimpleNamespace()
            names = configure_nonwheel_ground_sensors(scene, ["/actual/ground"], ["base_Link", "hip_L_Link"])
            self.assertEqual(names, ("nonwheel_ground_base_Link", "nonwheel_ground_hip_L_Link"))
            self.assertEqual(scene.nonwheel_ground_base_Link.prim_path, "{ENV_REGEX_NS}/Robot/base_Link")
            self.assertEqual(scene.nonwheel_ground_hip_L_Link.prim_path, "{ENV_REGEX_NS}/Robot/hip_L_Link")
            self.assertEqual(scene.nonwheel_ground_hip_L_Link.filter_prim_paths_expr, ["/actual/ground"])
            with self.assertRaisesRegex(ValueError, "nonwheel body"):
                configure_nonwheel_ground_sensors(SimpleNamespace(), ["/actual/ground"], ["wheel_L_Link"])


if __name__ == "__main__":
    unittest.main()
