"""Torch-free deployment-contract tests, plus optional local-asset integration."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_tracking_mujoco import (
    DEFAULT_MODEL, MotionReference, MujocoState, action_torques, build_observation,
    configure_model, read_contract, run_evaluation, training_termination_terms,
    load_model, verify_isaac_observations,
)


def quat(matrix):
    return Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]


def state_and_target():
    state = {"joint_pos": np.arange(8.) / 10, "joint_vel": np.arange(8.) + 1,
             "root_pos": np.array([1., 2., 3.]), "root_quat_wxyz": np.array([1., 0, 0, 0]),
             "root_com_lin_vel_w": np.array([.4, .5, .6]), "root_ang_vel_w": np.array([.7, .8, .9]),
             "wheel_pos": np.array([[1., 2., .127], [1., 1., .127]])}
    target = {"joint_pos": np.arange(8.) / 3, "joint_vel": np.arange(8.) / 2,
              "root_pos": np.array([1.1, 2.2, 3.3]), "root_quat_wxyz": np.array([1., 0, 0, 0]),
              "phase": .7, "wheel_pos": state["wheel_pos"].copy()}
    return state, target


class ObservationTests(unittest.TestCase):
    def test_contract_without_isaac_or_torch(self):
        c = read_contract()
        self.assertEqual(sum(size for _, size in c["actor_terms"]), 51)
        self.assertEqual(c["observed_joint_velocity_order"], c["leg_joint_names"] + c["wheel_joint_names"])
        self.assertAlmostEqual(c["physics_dt"] * c["decimation"], 1 / c["policy_fps"])

    def test_observation_slices_and_no_wheel_angle(self):
        state, target = state_and_target()
        q0, previous = np.arange(8.) / 20, np.arange(8.) / 10
        obs = build_observation(state, target, q0, previous)
        self.assertEqual(obs.shape, (51,))
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_allclose(obs[:6], target["joint_pos"][:6])
        np.testing.assert_allclose(obs[6:12], target["joint_vel"][:6])
        np.testing.assert_allclose(obs[12:14], [np.sin(.7), np.cos(.7)])
        np.testing.assert_allclose(obs[14:17], [.1, .2, .3])
        np.testing.assert_allclose(obs[17:23], [1, 0, 0, 1, 0, 0])
        np.testing.assert_allclose(obs[23:29], [.4, .5, .6, .7, .8, .9])
        np.testing.assert_allclose(obs[29:35], state["joint_pos"][:6] - q0[:6])
        np.testing.assert_allclose(obs[35:43], state["joint_vel"])
        np.testing.assert_allclose(obs[43:], previous)
        state["joint_pos"][6:] += 1000
        np.testing.assert_array_equal(build_observation(state, target, q0, previous), obs)

    def test_anchor_transform_and_row_major_columns(self):
        state, target = state_and_target()
        r = Rotation.from_euler("xyz", [.4, -.6, .8]).as_matrix()
        ref = Rotation.from_euler("xyz", [-.2, .3, -.7]).as_matrix()
        state["root_quat_wxyz"], target["root_quat_wxyz"] = quat(r), quat(ref)
        obs = build_observation(state, target, np.zeros(8), np.zeros(8))
        np.testing.assert_allclose(obs[14:17], r.T @ (target["root_pos"] - state["root_pos"]), atol=1e-7)
        np.testing.assert_allclose(obs[17:23], (r.T @ ref)[:, :2].reshape(-1), atol=1e-7)
        self.assertFalse(np.allclose(obs[17:23], (r.T @ ref)[:, :2].T.reshape(-1)))
        np.testing.assert_allclose(obs[23:26], r.T @ state["root_com_lin_vel_w"], atol=1e-7)
        np.testing.assert_allclose(obs[26:29], r.T @ state["root_ang_vel_w"], atol=1e-7)

    def test_action_q0_offset_pd_and_clip(self):
        c = read_contract()
        q0 = np.arange(8.) * .02
        action = np.array([2, -2, .1, -.1, 0, .05, 3, -4])
        q = q0.copy()
        dq = np.array([0, 0, 1, -1, 2, -2, 100, -100])
        actual = action_torques(action, q, dq, q0, c)
        np.testing.assert_allclose(actual, [80, -80, 40, -40, -20, 45, 12, -12])
        # Desired leg velocity is ZERO, not reference joint velocity.
        np.testing.assert_allclose(action_torques(np.zeros(8), q0, np.ones(8), q0, c), [-10] * 6 + [0, 0])

    def test_upstream_termination_orientation_is_not_radians(self):
        state, target = state_and_target()
        state["root_pos"] = target["root_pos"].copy()
        state["root_quat_wxyz"] = quat(Rotation.from_euler("x", .8).as_matrix())
        self.assertNotIn("anchor_ori", training_termination_terms(state, target))
        state["root_quat_wxyz"] = quat(Rotation.from_euler("x", 1.2).as_matrix())
        self.assertIn("anchor_ori", training_termination_terms(state, target))
        state["root_pos"][2] -= .351
        state["wheel_pos"][0, 2] -= .251
        self.assertEqual(set(training_termination_terms(state, target)), {"anchor_pos", "anchor_ori", "ee_body_pos"})


MOTION = ROOT / "outputs/tracking-cmu-16_03/motion.npz"


@unittest.skipUnless(MOTION.is_file() and DEFAULT_MODEL.is_file() and importlib.util.find_spec("mujoco"),
                     "Optional integration needs prepared local motion and USD-matched MuJoCo model")
class PreparedAssetTests(unittest.TestCase):
    def setUp(self):
        import mujoco
        self.mj = mujoco
        self.contract = read_contract()
        self.reference = MotionReference(MOTION, self.contract)
        self.model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL))
        configure_model(self.model, self.contract)
        self.data = mujoco.MjData(self.model)
        self.measured = MujocoState(self.model, self.data, self.reference)

    def test_reference_name_reorder_not_positional(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = {key: value.copy() for key, value in self.reference.archive.items()}
            order = [7, 3, 0, 6, 1, 5, 2, 4]
            for key in ("joint_pos", "joint_vel"):
                archive[key] = archive[key][:, order]
            archive["joint_names"] = archive["joint_names"][order]
            path = Path(tmp) / "permuted.npz"
            np.savez(path, **archive)
            permuted = MotionReference(path, self.contract)
            for frame in (0, 70, self.reference.frames - 1):
                for key in ("joint_pos", "joint_vel"):
                    np.testing.assert_array_equal(permuted.sample(frame)[key], self.reference.sample(frame)[key])

    def test_initial_com_velocity_and_observation(self):
        state = self.measured.initialize()
        target = self.reference.sample(0)
        np.testing.assert_allclose(state["root_pos"], target["root_pos"], atol=1e-12)
        np.testing.assert_allclose(state["root_com_lin_vel_w"], target["root_com_lin_vel_w"], atol=1e-10)
        np.testing.assert_allclose(state["root_ang_vel_w"], target["root_ang_vel_w"], atol=1e-10)
        np.testing.assert_allclose(state["joint_pos"], target["joint_pos"], atol=1e-12)
        np.testing.assert_allclose(state["joint_vel"], target["joint_vel"], atol=1e-12)
        obs = build_observation(state, target, self.reference.q0, np.zeros(8))
        np.testing.assert_allclose(obs[14:17], 0, atol=1e-9)
        np.testing.assert_allclose(obs[17:23], [1, 0, 0, 1, 0, 0], atol=1e-7)
        np.testing.assert_allclose(obs[29:35], 0, atol=1e-9)

    def test_actual_dynamics_fake_policy_no_torch(self):
        inputs = []
        def policy(obs):
            inputs.append(obs.copy())
            return np.ones(8) * .01
        with tempfile.TemporaryDirectory() as tmp:
            report = run_evaluation(policy, MOTION, DEFAULT_MODEL, Path(tmp) / "run", max_policy_steps=2)
            self.assertEqual(report["termination"], "evaluation_step_limit")
            self.assertEqual(report["root_state_writes"], 1)
            self.assertEqual(report["physics_steps"], 8)
            self.assertEqual(report["policy_steps"], 2)
            self.assertAlmostEqual(report["recorded_duration_s"], .04)
            np.testing.assert_allclose(inputs[0][43:], 0)
            np.testing.assert_allclose(inputs[1][43:], .01)
            with np.load(Path(tmp) / "run/rollout.npz", allow_pickle=False) as result:
                self.assertEqual(result["root_pos"].shape, (9, 3))
                self.assertEqual(result["action_observation"].shape, (2, 51))
                self.assertFalse(np.allclose(result["root_pos"][0], result["root_pos"][-1]))
                self.assertTrue(all(np.isfinite(result[name]).all() for name in result.files
                                    if result[name].dtype.kind in "fiu"))
            self.assertEqual(json.loads((Path(tmp) / "run/report.json").read_text())["policy_kind"], "injected_test_callable")

    def test_no_visual_mesh_preserves_mass_geometry_and_rollout(self):
        stripped, removed = load_model(DEFAULT_MODEL, no_visual_mesh=True)
        self.assertEqual(removed, 9)
        self.assertEqual(stripped.nmesh, 0)
        self.assertEqual(stripped.ngeom, self.model.ngeom - 9)
        for attribute in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "jnt_axis", "jnt_range"):
            np.testing.assert_allclose(getattr(stripped, attribute), getattr(self.model, attribute), atol=1e-12)
        for name in ("base_collision", "wheel_L_collision", "wheel_R_collision", "floor"):
            np.testing.assert_allclose(stripped.geom(name).size, self.model.geom(name).size)
            np.testing.assert_allclose(stripped.geom(name).pos, self.model.geom(name).pos)
        with tempfile.TemporaryDirectory() as tmp:
            report = run_evaluation(lambda obs: np.zeros(8), MOTION, DEFAULT_MODEL, Path(tmp) / "run",
                                    max_policy_steps=2, no_visual_mesh=True)
            self.assertEqual(report["removed_visual_geom_count"], 9)
            self.assertEqual(report["physics_steps"], 8)

    def test_mesh_removal_rejects_collision_capable_visual(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = ET.parse(DEFAULT_MODEL)
            tree.getroot().find("default/default[@class='visual']/geom").set("contype", "1")
            path = Path(tmp) / "bad.xml"
            tree.write(path)
            with self.assertRaisesRegex(ValueError, "collision-capable"):
                load_model(path, no_visual_mesh=True)

    def test_isaac_actor_reconstruction_uses_pre_action_state(self):
        first = self.measured.initialize()
        states = [{key: value.copy() for key, value in first.items()} for _ in range(3)]
        states[1]["root_pos"] += [.01, -.02, .03]
        states[1]["joint_pos"][:6] += .01
        states[2]["root_pos"] += [.04, -.03, .02]
        actions = np.array([np.zeros(8), np.ones(8) * .01, np.ones(8) * -.02])
        frames = [0, 0, 1]
        observations = [build_observation(states[max(0, i - 1)], self.reference.sample(frames[i]),
                                         self.reference.q0, np.zeros(8) if i == 0 else actions[i - 1])
                        for i in range(3)]
        mapping = {"root_position_m": "root_pos", "root_quaternion_wxyz": "root_quat_wxyz",
                   "root_linear_velocity_w_m_s": "root_com_lin_vel_w", "root_angular_velocity_w_rad_s": "root_ang_vel_w",
                   "joint_position_rad": "joint_pos", "joint_velocity_rad_s": "joint_vel"}
        record = {name: np.array([state[field] for state in states])[:, None] for name, field in mapping.items()}
        record.update(joint_names=np.array(self.reference.action_names), valid_mask=np.ones((3, 1), bool),
                      action_was_applied=np.array([[False], [True], [True]]), actions_clipped=actions[:, None],
                      reference_frame_for_action=np.array(frames)[:, None], action_observation=np.array(observations)[:, None])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "isaac.npz"
            np.savez(path, **record)
            result = verify_isaac_observations(path, MOTION)
            self.assertEqual(result["observations_checked"], 3)
            self.assertEqual(result["max_abs_error"], 0.)
            record["action_observation"][2, 0, 43] += .1
            np.savez(path, **record)
            with self.assertRaisesRegex(AssertionError, "dim=43"):
                verify_isaac_observations(path, MOTION)


if __name__ == "__main__":
    unittest.main()
