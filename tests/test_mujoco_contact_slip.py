"""True contact-point MuJoCo recording against articulated Jacobians."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_tracking_mujoco import (
    DEFAULT_MODEL, MotionReference, MujocoState, configure_model, load_model,
    read_contract, run_evaluation,
)

MOTION = ROOT / "outputs/tracking-cmu-16_03/motion.npz"


@unittest.skipUnless(MOTION.is_file() and DEFAULT_MODEL.is_file() and importlib.util.find_spec("mujoco"),
                     "Optional contact integration needs prepared motion and MuJoCo model")
class ActualGroundContactTests(unittest.TestCase):
    def setUp(self):
        import mujoco
        self.mj = mujoco
        self.contract = read_contract()
        self.reference = MotionReference(MOTION, self.contract)
        self.model, _ = load_model(DEFAULT_MODEL, no_visual_mesh=True)
        configure_model(self.model, self.contract)
        self.data = mujoco.MjData(self.model)
        self.measured = MujocoState(self.model, self.data, self.reference)
        self.measured.initialize()
        # Diagnostic reset only, to ensure force-bearing wheel contacts before
        # checking the measurement without needing a trained balancing actor.
        self.data.qpos[2] -= .01
        self.data.qvel[:] = 0.
        self.mj.mj_forward(self.model, self.data)

    def test_real_contacts_measure_lateral_translation(self):
        root_vid = self.measured.root_vid
        self.data.qvel[root_vid + 1] = .3
        self.mj.mj_forward(self.model, self.data)
        result = self.measured.ground_contact_slip()
        self.assertTrue(np.all(result["wheel_ground_contact_count"] > 0))
        self.assertTrue(np.all(result["wheel_ground_contact_force_w_n"][:, 2] > 5.))
        np.testing.assert_allclose(result["wheel_lateral_contact_slip_m_s"], [.3, .3], atol=1e-10)

    def test_rotation_transport_matches_direct_contact_point_jacobian(self):
        self.data.qvel[self.measured.root_vid:self.measured.root_vid + 6] = [.4, -.2, .1, .7, .3, -.6]
        self.data.qvel[self.measured.vids] = [.2, -.3, .1, -.2, .4, .3, 15., -12.]
        self.mj.mj_forward(self.model, self.data)
        measured = self.measured.ground_contact_slip()
        for wheel, body in enumerate(self.measured.wheels):
            sum_force, sum_squared_speed, count = 0., 0., 0
            axle = self.data.xmat[body].reshape(3, 3)[:, 1]
            for index, contact in enumerate(self.data.contact):
                bodies = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
                if 0 not in bodies or body not in bodies:
                    continue
                wrench = np.zeros(6)
                self.mj.mj_contactForce(self.model, self.data, index, wrench)
                if wrench[0] <= 5.:
                    continue
                normal = contact.frame[:3]
                tangent = axle - normal * np.dot(axle, normal)
                tangent /= np.linalg.norm(tangent)
                jacp, jacr = np.empty((3, self.model.nv)), np.empty((3, self.model.nv))
                self.mj.mj_jac(self.model, self.data, jacp, jacr, contact.pos, int(body))
                speed = np.dot(jacp @ self.data.qvel, tangent)
                sum_force += wrench[0]
                sum_squared_speed += wrench[0] * speed * speed
                count += 1
            self.assertGreater(count, 0)
            self.assertEqual(measured["wheel_ground_contact_count"][wheel], count)
            self.assertAlmostEqual(measured["wheel_lateral_contact_slip_m_s"][wheel],
                                   np.sqrt(sum_squared_speed / sum_force), places=11)

    def test_wheel_spin_does_not_create_lateral_slip(self):
        self.data.qvel[self.measured.vids[6:]] = [30., -30.]
        self.mj.mj_forward(self.model, self.data)
        measured = self.measured.ground_contact_slip()
        self.assertTrue(np.all(measured["wheel_ground_contact_count"] > 0))
        np.testing.assert_allclose(measured["wheel_lateral_contact_slip_m_s"], 0., atol=1e-10)

    def test_airborne_and_below_threshold_do_not_claim_ground_slip(self):
        measured = self.measured.ground_contact_slip(force_threshold_n=1e8)
        np.testing.assert_array_equal(measured["wheel_ground_contact_count"], 0)
        np.testing.assert_array_equal(measured["wheel_lateral_contact_slip_m_s"], 0.)
        self.assertGreater(measured["wheel_ground_contact_force_w_n"][:, 2].min(), 0.)
        self.data.qpos[2] += 2.
        self.data.qvel[self.measured.root_vid + 1] = .3
        self.mj.mj_forward(self.model, self.data)
        airborne = self.measured.ground_contact_slip()
        for value in airborne.values():
            np.testing.assert_array_equal(value, 0.)

    def test_per_physics_records_and_report_identify_true_contacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "evaluation"
            report = run_evaluation(lambda obs: np.zeros(8), MOTION, DEFAULT_MODEL, output,
                                    max_policy_steps=2, no_visual_mesh=True)
            self.assertEqual(report["wheel_lateral_contact_slip_evidence_kind"], "actual_ground_contact_points")
            self.assertEqual(report["contact_slip_measurement"]["active_point_normal_force_threshold_n"], 5.)
            self.assertEqual(report["contact_slip_helper_sha256"],
                             hashlib.sha256((ROOT / "training/tron1_contact_slip.py").read_bytes()).hexdigest())
            with np.load(output / "rollout.npz", allow_pickle=False) as result:
                self.assertEqual(result["wheel_lateral_contact_slip_m_s"].shape, (9, 2))
                self.assertEqual(result["wheel_ground_contact_count"].shape, (9, 2))
                self.assertEqual(result["wheel_ground_contact_force_w_n"].shape, (9, 2, 3))
                self.assertEqual(result["wheel_ground_total_contact_force_w_n"].shape, (9, 2, 3))
                self.assertEqual(result["nonwheel_ground_contact_count"].shape, (9,))
                self.assertTrue(np.isfinite(result["wheel_lateral_contact_slip_m_s"]).all())
                self.assertTrue(np.all(result["wheel_lateral_contact_slip_m_s"] >= 0))
                no_contact = result["wheel_ground_contact_count"] == 0
                np.testing.assert_array_equal(result["wheel_lateral_contact_slip_m_s"][no_contact], 0.)


if __name__ == "__main__":
    unittest.main()
