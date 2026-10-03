"""Torch-free DR deployment tests: independent formulas and optional MuJoCo.

The oracle below deliberately does not call the production DC bounds helper.
No checkpoint, actor, remote job, or prepared asset is modified by these tests.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_tracking_mujoco as deployment

JOINT_NAMES = [
    "abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint",
    "knee_L_Joint", "knee_R_Joint", "wheel_L_Joint", "wheel_R_Joint",
]
MOTION = ROOT / "outputs/tracking-cmu-16_03/motion.npz"


def draw(*, torque=None, speed=None, friction=None):
    torque = [1.] * 8 if torque is None else list(torque)
    speed = [1.] * 8 if speed is None else list(speed)
    friction = [0., 0.] if friction is None else list(friction)
    return {
        "joint_names": JOINT_NAMES.copy(), "torque_scale": torque,
        "velocity_scale": speed, "wheel_friction_nm": friction,
        "friction_smoothing_rad_s": .5,
        "is_nominal": torque == [1.] * 8 and speed == [1.] * 8 and friction == [0., 0.],
    }


def randomized_contract(parameters):
    contract = deepcopy(deployment.read_contract())
    contract["domain_parameters"] = deepcopy(parameters)
    return contract


def independent_expected(action, q, dq, q0, contract, parameters):
    action = np.clip(action, *contract["raw_action_clip"])
    requested = np.r_[
        contract["leg_kp"] * (q0[:6] + contract["leg_position_scale"] * action[:6] - q[:6])
        - contract["leg_kd"] * dq[:6],
        contract["wheel_torque_scale_nm"] * action[6:],
    ]
    effort0 = np.array([contract["leg_torque_limit_nm"]] * 6 + [contract["wheel_torque_limit_nm"]] * 2)
    effort = effort0 * parameters["torque_scale"]
    stall = np.array([contract["leg_saturation_effort_nm"]] * 6
                     + [contract["wheel_saturation_effort_nm"]] * 2) * parameters["torque_scale"]
    speed = np.array([contract["leg_motor_velocity_limit_rad_s"]] * 6
                     + [contract["wheel_motor_velocity_limit_rad_s"]] * 2) * parameters["velocity_scale"]
    electrical_velocity = np.clip(dq, -speed * (1 + effort / stall), speed * (1 + effort / stall))
    lower = np.maximum(stall * (-1 - electrical_velocity / speed), -effort)
    upper = np.minimum(stall * (1 - electrical_velocity / speed), effort)
    motor = np.clip(requested, lower, upper)
    motor[6:] -= np.asarray(parameters["wheel_friction_nm"]) * np.tanh(dq[6:] / .5)
    return np.clip(motor, -effort0, effort0)


class DomainRandomizationDeploymentTests(unittest.TestCase):
    def test_nominal_draw_is_bitwise_identical_to_unrandomized_dc(self):
        original = deployment.read_contract()
        explicit = randomized_contract(draw())
        rng = np.random.default_rng(941)
        for _ in range(20):
            action = rng.uniform(-2, 2, 8)
            q, q0 = rng.normal(size=(2, 8))
            dq = rng.uniform(-3, 3, 8) * np.array([15.] * 6 + [100.] * 2)
            np.testing.assert_array_equal(
                deployment.action_torques(action, q, dq, q0, explicit),
                deployment.action_torques(action, q, dq, q0, original),
            )

    def test_torque_scale_is_per_joint_and_scales_stall_and_effort(self):
        factors = np.linspace(.85, 1., 8)
        contract = randomized_contract(draw(torque=factors))
        zero = np.zeros(8)
        actual = deployment.action_torques(np.ones(8), zero, zero, zero, contract)
        np.testing.assert_allclose(actual, np.array([80.] * 6 + [12.] * 2) * factors)
        dq = np.array([7.5] * 6 + [50.] * 2)
        actual = deployment.action_torques(np.ones(8), zero, dq, zero, contract)
        np.testing.assert_allclose(actual, np.array([40.] * 6 + [6.] * 2) * factors)

    def test_velocity_scale_is_independent_and_preserves_stall_torque(self):
        factors = np.linspace(1., .85, 8)
        contract = randomized_contract(draw(speed=factors))
        zero = np.zeros(8)
        effort = np.array([80.] * 6 + [12.] * 2)
        np.testing.assert_allclose(deployment.action_torques(np.ones(8), zero, zero, zero, contract), effort)
        dq = np.array([7.5] * 6 + [50.] * 2)
        actual = deployment.action_torques(np.ones(8), zero, dq, zero, contract)
        np.testing.assert_allclose(actual, effort * (1 - .5 / factors))

    def test_three_axes_match_independent_four_quadrant_oracle(self):
        parameters = draw(torque=np.linspace(.85, 1., 8), speed=np.linspace(1., .85, 8), friction=[.3, .13])
        contract = randomized_contract(parameters)
        rng = np.random.default_rng(37)
        for _ in range(30):
            action = rng.uniform(-2, 2, 8)
            q, q0 = rng.normal(scale=.3, size=(2, 8))
            dq = rng.uniform(-4, 4, 8) * np.array([15.] * 6 + [100.] * 2)
            actual = deployment.action_torques(action, q, dq, q0, contract)
            np.testing.assert_allclose(actual, independent_expected(action, q, dq, q0, contract, parameters), atol=1e-12)

    def test_axle_friction_is_wheel_only_smooth_and_dissipative(self):
        zero = np.zeros(8)
        nominal = deployment.read_contract()
        contract = randomized_contract(draw(friction=[.3, .12]))
        for speed in (0., .01, .5, 5., 25.):
            dq = np.r_[np.zeros(6), speed, -speed]
            base = deployment.action_torques(zero, zero, dq, zero, nominal)
            actual = deployment.action_torques(zero, zero, dq, zero, contract)
            np.testing.assert_array_equal(actual[:6], base[:6])
            friction = actual[6:] - base[6:]
            np.testing.assert_allclose(friction, -np.array([.3, .12]) * np.tanh(dq[6:] / .5))
            self.assertTrue(np.all(friction * dq[6:] <= 0))
            if speed:
                self.assertLess(friction[0], 0)
                self.assertGreater(friction[1], 0)

    def test_friction_follows_motor_clipping_and_nominal_net_guard(self):
        zero = np.zeros(8)
        dq = np.r_[np.zeros(6), 150., -150.]
        action = np.r_[np.zeros(6), -1., 1.]
        # Motor braking plus axle friction must not exceed the nominal guard.
        full = randomized_contract(draw(friction=[.3, .3]))
        np.testing.assert_allclose(deployment.action_torques(action, zero, dq, zero, full)[6:], [-12., 12.])
        # Passive friction is NOT clipped back to the reduced motor ceiling.
        reduced = randomized_contract(draw(torque=[.85] * 8, friction=[.3, .3]))
        np.testing.assert_allclose(deployment.action_torques(action, zero, dq, zero, reduced)[6:], [-10.5, 10.5])

    def test_inputs_and_nested_contract_are_not_mutated(self):
        contract = randomized_contract(draw(torque=[.91] * 8, speed=[.88] * 8, friction=[.2, .3]))
        before = deepcopy(contract)
        arrays = [np.arange(8.) / 10, np.arange(8.) / 100, np.linspace(-180., 180., 8), np.zeros(8)]
        copies = [value.copy() for value in arrays]
        for value in arrays:
            value.flags.writeable = False
        deployment.action_torques(*arrays, contract)
        self.assertEqual(contract, before)
        for value, prior in zip(arrays, copies):
            np.testing.assert_array_equal(value, prior)

    def test_named_permutation_is_canonicalized_without_mutating_input(self):
        parameters = draw(torque=np.linspace(.85, 1., 8), speed=np.linspace(1., .85, 8), friction=[.3, .12])
        permutation = [7, 2, 4, 1, 6, 0, 5, 3]
        shuffled = deepcopy(parameters)
        for key in ("joint_names", "torque_scale", "velocity_scale"):
            shuffled[key] = [parameters[key][index] for index in permutation]
        original = deepcopy(shuffled)
        reordered = randomized_contract(shuffled)
        canonical = randomized_contract(parameters)
        dq = np.array([3., -6., 7.5, -9., 11., -12., 50., -60.])
        action = np.array([1., -1.] * 4)
        np.testing.assert_allclose(
            deployment.action_torques(action, np.zeros(8), dq, np.zeros(8), reordered),
            deployment.action_torques(action, np.zeros(8), dq, np.zeros(8), canonical),
        )
        self.assertEqual(reordered["domain_parameters"], original)

    def test_is_nominal_metadata_is_recomputed_from_parameters(self):
        parameters = draw(friction=[.1, .2])
        parameters["is_nominal"] = True
        contract = randomized_contract(parameters)
        normalized = deployment.validate_contract(contract)
        self.assertIs(normalized["domain_parameters"]["is_nominal"], False)
        self.assertIs(contract["domain_parameters"]["is_nominal"], True)
        parameters = draw()
        parameters["is_nominal"] = False
        normalized = deployment.validate_contract(randomized_contract(parameters))
        self.assertIs(normalized["domain_parameters"]["is_nominal"], True)

    def test_randomization_rejects_legacy_pd_even_for_nominal_draw(self):
        for legacy in ("ideal_pd", None):
            contract = randomized_contract(draw())
            if legacy is None:
                contract.pop("actuator_model")
            else:
                contract["actuator_model"] = legacy
            with self.subTest(legacy=legacy), self.assertRaises(ValueError):
                deployment.action_torques(*(np.zeros(8) for _ in range(4)), contract)

    def test_invalid_names_shapes_bounds_nonfinite_and_schema_fail(self):
        cases = []
        cases.extend([("joint_names", ["unknown_Joint"] + JOINT_NAMES[1:]), ("joint_names", JOINT_NAMES[:-1]),
                      ("joint_names", [JOINT_NAMES[0]] * 8), ("torque_scale", [1.] * 7),
                      ("velocity_scale", [1.] * 9), ("wheel_friction_nm", [0.]),
                      ("friction_smoothing_rad_s", 0.), ("friction_smoothing_rad_s", -1.),
                      ("friction_smoothing_rad_s", np.nan), ("friction_smoothing_rad_s", np.inf)])
        for key, count in (("torque_scale", 8), ("velocity_scale", 8), ("wheel_friction_nm", 2)):
            for bad in (np.nan, np.inf, -np.inf):
                values = ([1.] if count == 8 else [0.]) * count
                values[0] = bad
                cases.append((key, values))
        for key in ("torque_scale", "velocity_scale"):
            for bad in (.009, 1.001, -1., 0.):
                cases.append((key, [bad] + [1.] * 7))
        cases.extend([("wheel_friction_nm", [-.001, 0.]), ("wheel_friction_nm", [12.001, 0.])])
        for key, value in cases:
            parameters = draw()
            parameters[key] = value
            if key != "is_nominal":
                parameters["is_nominal"] = False
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                deployment.action_torques(*(np.zeros(8) for _ in range(4)), randomized_contract(parameters))
        for change in ("missing", "missing_names"):
            parameters = draw(friction=[.1, .1])
            if change == "missing":
                del parameters["velocity_scale"]
            else:
                del parameters["joint_names"]
            with self.subTest(change=change), self.assertRaises(ValueError):
                deployment.action_torques(*(np.zeros(8) for _ in range(4)), randomized_contract(parameters))

    def test_rollout_does_not_assign_positions_or_velocities(self):
        tree = ast.parse(Path(deployment.__file__).read_text())
        rollout = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_evaluation")
        for node in ast.walk(rollout):
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else [])
            for target in targets:
                for component in ast.walk(target):
                    if isinstance(component, ast.Attribute):
                        self.assertNotIn(component.attr, ("qpos", "qvel"), "Rollout may not prescribe physical state")


@unittest.skipUnless(MOTION.is_file() and deployment.DEFAULT_MODEL.is_file() and importlib.util.find_spec("mujoco"),
                     "Optional integration requires local motion and USD-matched MuJoCo model")
class DomainRandomizationRolloutTests(unittest.TestCase):
    def test_explicit_draw_two_steps_records_actual_physics_and_metadata(self):
        parameters = draw(torque=np.linspace(.85, .98, 8), speed=np.linspace(.99, .86, 8), friction=[.12, .3])
        untouched = deepcopy(parameters)
        inputs = []

        def policy(observation):
            inputs.append(observation.copy())
            return np.full(8, .01)

        with tempfile.TemporaryDirectory(prefix="tron1-dr-deployment-test-") as temp:
            temp = Path(temp)
            contract_path = temp / "manifest.json"
            contract_path.write_text(json.dumps({"contract": deployment.read_contract()}))
            report = deployment.run_evaluation(
                policy, MOTION, deployment.DEFAULT_MODEL, temp / "run",
                max_policy_steps=2, no_visual_mesh=True, contract_path=contract_path,
                domain_parameters=parameters,
            )
            self.assertEqual(parameters, untouched)
            self.assertEqual(report["domain_parameters"], parameters)
            self.assertFalse(report["domain_parameters_resampled_during_episode"])
            self.assertEqual(report["contract"]["domain_parameters"], parameters)
            self.assertEqual(report["contract_source"], str(contract_path.resolve()))
            self.assertEqual(report["contract_source_sha256"], deployment.sha256(contract_path))
            self.assertEqual(report["termination"], "evaluation_step_limit")
            self.assertEqual(report["policy_steps"], 2)
            self.assertEqual(report["physics_steps"], 8)
            self.assertAlmostEqual(report["recorded_duration_s"], .04)
            self.assertEqual(report["root_state_writes"], 1)
            self.assertEqual(report["hidden_resets"], 0)
            self.assertFalse(report["actual_joint_velocity_hard_clipped"])
            np.testing.assert_array_equal(inputs[0][43:], 0)
            np.testing.assert_allclose(inputs[1][43:], .01)
            saved = json.loads((temp / "run/report.json").read_text())
            self.assertEqual(saved["domain_parameters"], parameters)
            with np.load(temp / "run/rollout.npz", allow_pickle=False) as result:
                self.assertEqual(result["root_pos"].shape, (9, 3))
                self.assertEqual(result["action_observation"].shape, (2, 51))
                self.assertFalse(np.allclose(result["root_pos"][0], result["root_pos"][-1]))
                for key in result.files:
                    if result[key].dtype.kind in "fiu":
                        self.assertTrue(np.isfinite(result[key]).all(), key)


if __name__ == "__main__":
    unittest.main()
