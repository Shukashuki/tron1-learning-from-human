"""Pure NumPy and fake-Torch tests; real Isaac smoke runs are separate."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.tron1_domain_randomization import (
    JOINT_NAMES, apply_draw_to_contract, default_profile, domain_randomization_audit,
    expanded_motor_parameters, make_actuator_class, nominal_draws,
    reset_domain_randomization, sample_draws, single_draw, validate_draw, validate_profile,
)


class ProfileTests(unittest.TestCase):
    def test_default_json_and_named_nominal_roundtrip(self):
        profile = validate_profile(json.loads(json.dumps(default_profile(12))))
        self.assertEqual(profile["torque_scale_range"], [.85, 1.])
        self.assertEqual(profile["velocity_scale_range"], [.85, 1.])
        self.assertEqual(profile["wheel_friction_nm_range"], [0., .3])
        draw = single_draw(nominal_draws(1), 0)
        self.assertTrue(draw["is_nominal"])
        self.assertEqual(validate_draw(json.loads(json.dumps(draw))), draw)

    def test_deterministic_seed_env_episode_and_subset_order(self):
        profile = {**default_profile(23), "nominal_probability": 0.}
        batch = sample_draws(profile, [5, 2, 9], [0, 3, 8])
        permuted = sample_draws(profile, [9, 5, 2], [8, 0, 3])
        for left, right in ((0, 1), (1, 2), (2, 0)):
            self.assertEqual(single_draw(batch, left), single_draw(permuted, right))
        isolated = sample_draws(profile, [2], [3])
        self.assertEqual(single_draw(batch, 1), single_draw(isolated, 0))
        different = sample_draws(profile, [2], [4])
        self.assertNotEqual(single_draw(isolated, 0), single_draw(different, 0))

    def test_per_joint_bounds_and_whole_environment_nominal_mixture(self):
        batch = sample_draws(default_profile(91), np.arange(1000), np.zeros(1000, dtype=int))
        self.assertTrue(150 < np.count_nonzero(batch["nominal_mask"]) < 250)
        for row in np.flatnonzero(batch["nominal_mask"]):
            self.assertTrue(single_draw(batch, row)["is_nominal"])
        nonnominal = ~batch["nominal_mask"]
        for key in ("torque_scale", "velocity_scale"):
            self.assertTrue(np.all(batch[key] >= .85))
            self.assertTrue(np.all(batch[key] <= 1.))
            self.assertTrue(np.any(batch[key][nonnominal, 0] != batch[key][nonnominal, 1]))
        self.assertTrue(np.all(batch["wheel_friction_nm"] >= 0))
        self.assertTrue(np.all(batch["wheel_friction_nm"] <= .3))

    def test_nominal_probability_one_and_minimal_fixed_profile(self):
        profile = {**default_profile(), "nominal_probability": 1.}
        draw = single_draw(sample_draws(profile, [0], [0]), 0)
        self.assertTrue(draw["is_nominal"])
        draw["torque_scale"][2] = .9
        draw["is_nominal"] = True  # User metadata cannot falsify nominal status.
        fixed = validate_profile({"mode": "fixed", "fixed_draw": draw})
        result = sample_draws(fixed, [0, 7], [0, 35])
        self.assertFalse(result["nominal_mask"].any())
        self.assertEqual(single_draw(result, 0), single_draw(result, 1))

    def test_name_permutation_and_copy_contract_without_changing_nominals(self):
        draw = single_draw(nominal_draws(1), 0)
        draw["torque_scale"] = np.linspace(.85, 1., 8).tolist()
        expected = validate_draw(draw)
        permutation = [6, 2, 7, 1, 4, 0, 3, 5]
        for key in ("joint_names", "torque_scale", "velocity_scale"):
            draw[key] = [draw[key][i] for i in permutation]
        self.assertEqual(validate_draw(draw), expected)
        contract = {"actuator_model": "dc_motor", "leg_torque_limit_nm": 80.,
                    "wheel_torque_limit_nm": 12., "leg_motor_velocity_limit_rad_s": 15.}
        merged = apply_draw_to_contract(contract, draw)
        self.assertNotIn("domain_parameters", contract)
        self.assertEqual(merged["leg_torque_limit_nm"], 80.)
        self.assertEqual(merged["domain_parameters"], expected)
        merged["domain_parameters"]["torque_scale"][0] = .01
        self.assertEqual(expected["torque_scale"][0], .85)

    def test_invalid_profiles_names_shapes_ranges_and_finite_values(self):
        changes = [{"seed": -1}, {"seed": True}, {"seed": 2**32}, {"mode": "surprise"},
                   {"torque_scale_range": [1., .9]}, {"velocity_scale_range": [.85, 1.01]},
                   {"wheel_friction_nm_range": [-.1, .3]}, {"nominal_probability": np.nan},
                   {"friction_smoothing_rad_s": 0.}, {"mode": "fixed"}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_profile({**default_profile(), **change})
        nominal = single_draw(nominal_draws(1), 0)
        cases = [{**nominal, "joint_names": JOINT_NAMES[:-1]},
                 {**nominal, "torque_scale": [1.] * 7},
                 {**nominal, "velocity_scale": [np.inf] * 8},
                 {**nominal, "wheel_friction_nm": [np.nan, 0.]},
                 {key: value for key, value in nominal.items() if key != "joint_names"}]
        for draw in cases:
            with self.subTest(draw=draw), self.assertRaises(ValueError):
                validate_draw(draw)
        for ids, counts in (([1, 1], [0, 0]), ([-1], [0]), ([0], [-1]), ([0.5], [0]), ([0, 1], [0])):
            with self.subTest(ids=ids, counts=counts), self.assertRaises(ValueError):
                sample_draws(default_profile(), ids, counts)

    def test_expanded_motor_cache_scales_both_torque_limits(self):
        draw = single_draw(nominal_draws(1), 0)
        draw["torque_scale"] = [.85] * 8
        draw["velocity_scale"] = [.9] * 8
        params = expanded_motor_parameters(draw)
        np.testing.assert_allclose(params["effort_limit_nm"], [68.] * 6 + [10.2] * 2)
        np.testing.assert_allclose(params["saturation_effort_nm"], params["effort_limit_nm"])
        np.testing.assert_allclose(params["velocity_at_effort_limit_rad_s"], [27.] * 6 + [180.] * 2)
        np.testing.assert_allclose(params["net_effort_guard_nm"], [80.] * 6 + [12.] * 2)


class Tensor(np.ndarray):
    def clone(self):
        return self.copy()

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


def tensor(value, dtype=None, device=None):
    return np.asarray(value, dtype=dtype).view(Tensor)


class FakeDCMotor:
    """Only the inspected electrical clip; adapter code itself runs unchanged."""
    def __init__(self, cfg, joint_names, num_envs, device="cpu"):
        self.cfg, self.joint_names = cfg, joint_names
        shape = (num_envs, len(joint_names))
        self.effort_limit = tensor(np.full(shape, cfg.effort_limit))
        self.effort_limit_sim = tensor(np.full(shape, cfg.effort_limit_sim))
        self.velocity_limit = tensor(np.full(shape, cfg.velocity_limit))
        self._saturation_effort = cfg.saturation_effort
        self._vel_at_effort_lim = self.velocity_limit * (1 + self.effort_limit / self._saturation_effort)
        self.applied_effort = tensor(np.zeros(shape))

    def compute(self, action, joint_pos, joint_vel):
        self._joint_vel = np.clip(joint_vel, -self._vel_at_effort_lim, self._vel_at_effort_lim)
        upper = np.minimum(self._saturation_effort * (1 - self._joint_vel / self.velocity_limit), self.effort_limit)
        lower = np.maximum(self._saturation_effort * (-1 - self._joint_vel / self.velocity_limit), -self.effort_limit)
        self.applied_effort = tensor(np.clip(action.joint_efforts, lower, upper))
        action.joint_efforts = self.applied_effort
        return action


FAKE_TORCH = types.SimpleNamespace(
    as_tensor=tensor, long=np.int64,
    full_like=lambda values, fill: tensor(np.full_like(values, fill)),
    zeros_like=lambda values: tensor(np.zeros_like(values)),
    tanh=np.tanh, clamp=lambda values, min, max: tensor(np.clip(values, min, max)))


class IsaacAdapterTests(unittest.TestCase):
    def setUp(self):
        fake_actuators = types.ModuleType("isaaclab.actuators")
        fake_actuators.DCMotor = FakeDCMotor
        self.modules = patch.dict(sys.modules, {"torch": FAKE_TORCH, "isaaclab.actuators": fake_actuators})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        cls = make_actuator_class()
        leg_cfg = types.SimpleNamespace(saturation_effort=80., effort_limit=80., effort_limit_sim=80., velocity_limit=15.)
        wheel_cfg = types.SimpleNamespace(saturation_effort=12., effort_limit=12., effort_limit_sim=12., velocity_limit=100.)
        # Deliberately scrambled order within each actuator group.
        legs = cls(leg_cfg, [JOINT_NAMES[i] for i in [4, 0, 3, 1, 5, 2]], 4)
        wheels = cls(wheel_cfg, [JOINT_NAMES[7], JOINT_NAMES[6]], 4)
        robot = types.SimpleNamespace(device="cpu", actuators={"legs": legs, "wheels": wheels})
        self.env = types.SimpleNamespace(num_envs=4, scene={"robot": robot})

    def test_subset_reset_name_mapping_noncompounding_cache_and_audit(self):
        profile = {**default_profile(123), "nominal_probability": 0.}
        reset_domain_randomization(self.env, [0, 1, 2, 3], profile)
        audit = domain_randomization_audit(self.env)
        self.assertTrue(audit["all_sampled_parameters_readback_verified"])
        self.assertEqual(audit["reset_counts"], [0, 0, 0, 0])
        previous = copy.deepcopy(self.env.tron1_domain_randomization["draw_arrays"])
        reset_domain_randomization(self.env, tensor([3, 1], dtype=np.int64), profile)
        self.assertEqual(domain_randomization_audit(self.env)["reset_counts"], [0, 1, 0, 1])
        current = self.env.tron1_domain_randomization["draw_arrays"]
        for key in ("torque_scale", "velocity_scale", "wheel_friction_nm"):
            np.testing.assert_array_equal(current[key][[0, 2]], previous[key][[0, 2]])
        expected = sample_draws(profile, [3, 1], [1, 1])
        self.assertEqual(single_draw(current, 3), single_draw(expected, 0))
        self.assertEqual(single_draw(current, 1), single_draw(expected, 1))
        legs = self.env.scene["robot"].actuators["legs"]
        name_index = JOINT_NAMES.index(legs.joint_names[0])
        self.assertAlmostEqual(legs.effort_limit[3, 0], 80 * expected["torque_scale"][0, name_index])
        reset_domain_randomization(self.env, [], profile)
        self.assertEqual(domain_randomization_audit(self.env)["reset_counts"], [0, 1, 0, 1])

    def test_fixed_repeated_reset_and_corruption_detected_across_all_envs(self):
        draw = single_draw(nominal_draws(1), 0)
        draw["torque_scale"] = [.9] * 8
        draw["velocity_scale"] = [.95] * 8
        draw["wheel_friction_nm"] = [.1, .3]
        profile = {"mode": "fixed", "fixed_draw": draw}
        for _ in range(3):
            reset_domain_randomization(self.env, slice(None), profile)
        audit = domain_randomization_audit(self.env)
        self.assertEqual(audit["reset_counts"], [2] * 4)
        np.testing.assert_allclose(audit["actual_first16"]["effort_limit_nm"], [[72.] * 6 + [10.8] * 2] * 4)
        wheels = self.env.scene["robot"].actuators["wheels"]
        # Corrupt an environment other than env0: audit must still reject it.
        wheels.axle_friction_nm[3, 0] += .05
        with self.assertRaisesRegex(RuntimeError, "axle_friction"):
            domain_randomization_audit(self.env)

    def test_audit_rejects_stale_cache(self):
        reset_domain_randomization(self.env, [0, 1, 2, 3], default_profile())
        self.env.scene["robot"].actuators["legs"]._vel_at_effort_lim[2, 1] += 1.
        with self.assertRaisesRegex(RuntimeError, "cache is stale"):
            domain_randomization_audit(self.env)

    def test_loss_opposes_actual_velocity_and_net_guard_is_fixed(self):
        wheel = self.env.scene["robot"].actuators["wheels"]
        wheel.axle_friction_nm[:] = .3
        wheel.friction_smoothing_rad_s = .5
        dq = tensor([[1., -1.]] * 4)
        output = wheel.compute(types.SimpleNamespace(joint_efforts=tensor(np.zeros((4, 2)))), tensor(np.zeros((4, 2))), dq)
        np.testing.assert_allclose(output.joint_efforts, -0.3 * np.tanh(dq / .5))
        requested = tensor([[-12., 12.]] * 4)
        output = wheel.compute(types.SimpleNamespace(joint_efforts=requested), tensor(np.zeros((4, 2))), dq)
        np.testing.assert_array_equal(output.joint_efforts, requested)
        np.testing.assert_array_equal(dq, [[1., -1.]] * 4)

    def test_zero_friction_nominal_adapter_matches_parent_exactly(self):
        wheel = self.env.scene["robot"].actuators["wheels"]
        dq = tensor([[0., 10.], [-50., 150.], [-250., 250.], [-100., 100.]])
        effort = tensor([[12., 5.], [-7., 0.], [0., 0.], [12., -12.]])
        baseline = FakeDCMotor(wheel.cfg, wheel.joint_names, 4)
        expected = baseline.compute(types.SimpleNamespace(joint_efforts=effort.copy()), None, dq).joint_efforts
        actual = wheel.compute(types.SimpleNamespace(joint_efforts=effort.copy()), None, dq).joint_efforts
        np.testing.assert_array_equal(actual, expected)

    def test_friction_uses_actual_not_electrically_clipped_velocity(self):
        wheel = self.env.scene["robot"].actuators["wheels"]
        wheel.axle_friction_nm[:] = .3
        wheel.friction_smoothing_rad_s = 200.
        dq = tensor([[250., -250.]] * 4)
        wheel.compute(types.SimpleNamespace(joint_efforts=tensor(np.zeros((4, 2)))), None, dq)
        np.testing.assert_array_equal(wheel._joint_vel, [[200., -200.]] * 4)
        np.testing.assert_allclose(wheel.axle_friction_effort, .3 * np.tanh(dq / 200.))
        np.testing.assert_array_equal(dq, [[250., -250.]] * 4)


if __name__ == "__main__":
    unittest.main()
