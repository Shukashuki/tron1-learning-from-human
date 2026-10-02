"""Run with python -m unittest discover -s tests -p 'test_balance_lqr.py'."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from balance_lqr import WIPParameters, design_lqr, linear_model, nonlinear_rhs


class BalanceLQRTests(unittest.TestCase):
    def setUp(self) -> None:
        # Representative rigid WIP parameters; these are not TRON1 identification.
        self.parameters = WIPParameters(
            body_mass=18.0, wheel_mass_total=3.0, wheel_inertia_total=0.04,
            radius=0.166, length=0.7, body_inertia=1.2,
        )
        self.dt = 1 / 120
        self.weights = (4.0, 2.0, 160.0, 8.0)
        self.input_weight = 0.05

    def test_linear_model_matches_nonlinear_finite_differences(self) -> None:
        A, B = linear_model(self.parameters)
        epsilon = 1e-6
        zero = np.zeros(4)
        A_numeric = np.column_stack([
            (nonlinear_rhs(epsilon * axis, 0.0, self.parameters)
             - nonlinear_rhs(-epsilon * axis, 0.0, self.parameters)) / (2 * epsilon)
            for axis in np.eye(4)
        ])
        B_numeric = (
            nonlinear_rhs(zero, epsilon, self.parameters)
            - nonlinear_rhs(zero, -epsilon, self.parameters)
        )[:, None] / (2 * epsilon)
        np.testing.assert_allclose(A, A_numeric, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(B, B_numeric, rtol=1e-8, atol=1e-9)

    def test_controllability_has_rank_four(self) -> None:
        A, B = linear_model(self.parameters)
        controllability = np.column_stack([B, A @ B, A @ A @ B, A @ A @ A @ B])
        self.assertEqual(np.linalg.matrix_rank(controllability), 4)

    def test_discrete_closed_loop_is_stable(self) -> None:
        design = design_lqr(self.parameters, self.dt, self.weights, self.input_weight)
        self.assertEqual(design.K.shape, (1, 4))
        self.assertLess(np.max(np.abs(design.closed_loop_poles)), 1.0)
        # Exact ZOH has the semigroup property even with a held constant input.
        half = design_lqr(self.parameters, self.dt / 2, self.weights, self.input_weight)
        np.testing.assert_allclose(design.Ad, half.Ad @ half.Ad, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(design.Bd, half.Ad @ half.Bd + half.Bd, rtol=1e-12, atol=1e-13)

    def test_forward_torque_accelerates_axle_and_reacts_on_body(self) -> None:
        derivative = nonlinear_rhs(np.zeros(4), 1.0, self.parameters)
        self.assertGreater(derivative[1], 0.0)
        self.assertLess(derivative[3], 0.0)
        p = self.parameters
        # Newton/Euler balance independently checks both spin inertia and -tau.
        self.assertAlmostEqual(
            p.effective_mass * derivative[1] + p.coupling * derivative[3], 1 / p.radius,
        )
        self.assertAlmostEqual(
            p.coupling * derivative[1] + p.inertia_about_axle * derivative[3], -1.0,
        )

    def test_nonlinear_five_degree_disturbances_return_upright(self) -> None:
        design = design_lqr(self.parameters, self.dt, self.weights, self.input_weight)
        for angle in (-5.0, 5.0):
            with self.subTest(initial_degrees=angle):
                state = np.array([0.0, 0.0, np.deg2rad(angle), 0.0])
                maximum_lean = abs(state[2])
                for _ in range(round(12 / self.dt)):
                    # Digital control: torque is held across all RK4 sub-stages.
                    torque = float((-design.K @ state).item())
                    k1 = nonlinear_rhs(state, torque, self.parameters)
                    k2 = nonlinear_rhs(state + 0.5 * self.dt * k1, torque, self.parameters)
                    k3 = nonlinear_rhs(state + 0.5 * self.dt * k2, torque, self.parameters)
                    k4 = nonlinear_rhs(state + self.dt * k3, torque, self.parameters)
                    state += self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                    maximum_lean = max(maximum_lean, abs(state[2]))
                self.assertTrue(np.isfinite(state).all())
                self.assertLess(maximum_lean, np.deg2rad(8))
                self.assertLess(abs(state[0]), 0.01)
                self.assertLess(abs(state[1]), 0.01)
                self.assertLess(abs(state[2]), np.deg2rad(0.1))
                self.assertLess(abs(state[3]), np.deg2rad(0.1))

    def test_invalid_sample_period_rejected(self) -> None:
        for dt in (0.0, -0.01, float("nan"), float("inf")):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                design_lqr(self.parameters, dt, self.weights, self.input_weight)


if __name__ == "__main__":
    unittest.main()
