"""NumPy/AST-only regression tests for the shared Isaac/MuJoCo DC model."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_tracking_mujoco import (
    TASK_SOURCE, action_torques, dc_motor_torque_bounds, read_contract, resolve_contract_path, validate_contract,
)


class DCMotorEnvelopeTests(unittest.TestCase):
    def test_positive_and_negative_four_quadrants(self):
        dq = np.array([-15., -7.5, 0., 7.5, 15.])
        original = dq.copy()
        lower, upper = dc_motor_torque_bounds(dq, 80., 80., 15.)
        np.testing.assert_array_equal(lower, [0., -40., -80., -80., -80.])
        np.testing.assert_array_equal(upper, [80., 80., 80., 40., 0.])
        np.testing.assert_array_equal(dq, original)

    def test_overspeed_brakes_without_mutating_joint_velocity(self):
        dq = np.array([-300., -200., -150., 150., 200., 300.])
        lower, upper = dc_motor_torque_bounds(dq, 12., 12., 100.)
        np.testing.assert_array_equal(lower, [12., 12., 6., -12., -12., -12.])
        np.testing.assert_array_equal(upper, [12., 12., 12., -6., -12., -12.])
        # Even zero requested torque is electrical braking above no-load speed.
        np.testing.assert_array_equal(np.clip(np.zeros(6), lower, upper), [12., 12., 6., -6., -12., -12.])
        np.testing.assert_array_equal(dq, [-300., -200., -150., 150., 200., 300.])

    def test_distinct_stall_and_continuous_effort_matches_official_curve(self):
        dq = np.array([-45., -22.5, -15., 0., 15., 22.5, 45.])
        lower, upper = dc_motor_torque_bounds(dq, 80., 40., 15.)
        np.testing.assert_array_equal(lower, [40., 40., 0., -40., -40., -40., -40.])
        np.testing.assert_array_equal(upper, [40., 40., 40., 40., 0., -40., -40.])

    def test_motor_groups_broadcast_and_effort_ceilings(self):
        speed = np.array([15.] * 6 + [100.] * 2)
        effort = np.array([80.] * 6 + [12.] * 2)
        rng = np.random.default_rng(4)
        dq = rng.uniform(-3., 3., (30, 8)) * speed
        lower, upper = dc_motor_torque_bounds(dq, effort, effort, speed)
        self.assertTrue(np.all(lower <= upper))
        self.assertTrue(np.all(lower >= -effort))
        self.assertTrue(np.all(upper <= effort))
        reversed_lower, reversed_upper = dc_motor_torque_bounds(-dq, effort, effort, speed)
        np.testing.assert_allclose(reversed_lower, -upper)
        np.testing.assert_allclose(reversed_upper, -lower)

    def test_invalid_and_nonfinite_motor_inputs_fail(self):
        for index in range(4):
            for bad in (np.nan, np.inf, -np.inf):
                inputs = [1., 80., 80., 15.]
                inputs[index] = bad
                with self.subTest(index=index, bad=bad), self.assertRaises(ValueError):
                    dc_motor_torque_bounds(*inputs)
        for index in (1, 2, 3):
            for bad in (0., -1.):
                inputs = [1., 80., 80., 15.]
                inputs[index] = bad
                with self.subTest(index=index, bad=bad), self.assertRaises(ValueError):
                    dc_motor_torque_bounds(*inputs)


class DCMotorContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = read_contract()

    def test_current_source_and_both_actuator_groups_are_dc(self):
        c = self.contract
        self.assertEqual(c["actuator_model"], "dc_motor")
        self.assertEqual(c["leg_motor_velocity_limit_rad_s"], 15.)
        self.assertEqual(c["wheel_motor_velocity_limit_rad_s"], 100.)
        self.assertEqual(c["solver_joint_velocity_limit_rad_s"], 1000.)
        self.assertEqual(c["rigid_body_max_angular_speed_rad_s"], 100.)
        tree = ast.parse(TASK_SOURCE.read_text())
        motors = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == "DCMotorCfg"]
        self.assertEqual(len(motors), 2)
        for node, group in zip(motors, ("leg", "wheel")):
            values = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords
                      if isinstance(kw.value, ast.Constant)}
            self.assertEqual(values["velocity_limit_sim"], 1000.)
            self.assertEqual(values["velocity_limit"], c[f"{group}_motor_velocity_limit_rad_s"])
            self.assertEqual(values["saturation_effort"], c[f"{group}_saturation_effort_nm"])
            self.assertEqual(values["effort_limit"], c[f"{group}_torque_limit_nm"])
            self.assertEqual(values["effort_limit_sim"], c[f"{group}_torque_limit_nm"])

    def test_raw_pd_then_dc_clipping_and_full_braking(self):
        c = self.contract
        dq = np.array([7.5, -7.5, 15., -15., 22.5, -22.5, 100., -100.])
        action = np.array([1., -1., 1., -1., 1., -1., 1., -1.])
        result = action_torques(action, np.zeros(8), dq, np.zeros(8), c)
        np.testing.assert_allclose(result, [40., -40., 0., 0., -40., 40., 0., 0.])
        # Braking remains available at the full continuous torque ceiling.
        result = action_torques(-action, np.zeros(8), dq, np.zeros(8), c)
        np.testing.assert_allclose(result, [-80., 80., -80., 80., -80., 80., -12., 12.])

    def test_legacy_missing_model_keeps_old_ideal_pd(self):
        c = dict(self.contract)
        c.pop("actuator_model")
        dq = np.array([0., 0., 1., -1., 2., -2., 100., -100.])
        action = np.array([2., -2., .1, -.1, 0., .05, 3., -4.])
        np.testing.assert_allclose(action_torques(action, np.zeros(8), dq, np.zeros(8), c),
                                   [80., -80., 40., -40., -20., 45., 12., -12.])
        c["actuator_model"] = "ideal_pd"
        np.testing.assert_allclose(action_torques(np.ones(8), np.zeros(8), dq, np.zeros(8), c)[6:], [12., 12.])

    def test_json_contract_and_manifest_roundtrip_and_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, payload in (("policy_contract.json", self.contract), ("manifest.json", {"contract": self.contract})):
                path = Path(tmp) / name
                path.write_text(json.dumps(payload))
                self.assertEqual(read_contract(path), self.contract)
            legacy = dict(self.contract)
            legacy.pop("actuator_model")
            path.write_text(json.dumps({"contract": legacy}))
            self.assertNotIn("actuator_model", read_contract(path))

    def test_saved_adjacent_contract_wins_over_new_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            actor = Path(tmp) / "actor_normalized.pt"
            self.assertEqual(resolve_contract_path(), TASK_SOURCE)
            with self.assertWarnsRegex(UserWarning, "CURRENT task physics"):
                self.assertEqual(resolve_contract_path(policy_path=actor), TASK_SOURCE)
            saved = Path(tmp) / "policy_contract.json"
            saved.write_text(json.dumps(self.contract))
            self.assertEqual(resolve_contract_path(policy_path=actor), saved)
            explicit = Path(tmp) / "other.json"
            self.assertEqual(resolve_contract_path(explicit, actor), explicit)

    def test_unknown_and_incomplete_models_and_nonfinite_contracts_fail(self):
        for changes in ({"actuator_model": "mystery"}, {"leg_saturation_effort_nm": None},
                        {"wheel_motor_velocity_limit_rad_s": 0.}, {"leg_kp": np.inf},
                        {"leg_kd": np.nan}, {"solver_joint_velocity_limit_rad_s": -1.},
                        {"raw_action_clip": [np.nan, 1.]}, {"raw_action_clip": [1., -1.]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_contract({**self.contract, **changes})
        with self.assertRaises(ValueError):
            validate_contract({})
        with self.assertRaises(ValueError):
            validate_contract([])

    def test_nonfinite_action_and_state_are_rejected(self):
        for index in range(4):
            vectors = [np.zeros(8) for _ in range(4)]
            vectors[index][0] = np.nan
            with self.subTest(index=index), self.assertRaises(FloatingPointError):
                action_torques(*vectors, self.contract)


if __name__ == "__main__":
    unittest.main()
