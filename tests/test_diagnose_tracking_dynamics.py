"""Pure NumPy tests for bounded dynamics diagnostics; no simulator needed."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diagnose_tracking_dynamics.py"
SPEC = importlib.util.spec_from_file_location("tracking_dynamics_test_module", SCRIPT)
DIAG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAG)
NAMES = ["abad_L_Joint", "wheel_R_Joint", "hip_L_Joint", "wheel_L_Joint"]


def options(**changes):
    values = dict(torque_nm=1.0, extra_torque_nm=4.0, root_raise_m=2.0,
                  steps_per_case=1, zero_legacy_friction=False, legacy_wheel_friction=None)
    values.update(changes)
    return argparse.Namespace(**values)


class ForceCasesTests(unittest.TestCase):
    def test_baseline_and_signed_basis_for_two_levels(self):
        cases = DIAG.force_cases(NAMES, 1.0, 4.0)
        self.assertEqual(len(cases), 1 + 4 * len(NAMES))
        self.assertEqual(cases[0][0], "gravity_only")
        np.testing.assert_array_equal(cases[0][1], np.zeros(len(NAMES)))
        for block, magnitude in enumerate((1.0, 4.0)):
            for joint in range(len(NAMES)):
                positive = cases[1 + block * 2 * len(NAMES) + 2 * joint][1]
                negative = cases[2 + block * 2 * len(NAMES) + 2 * joint][1]
                self.assertEqual(np.count_nonzero(positive), 1)
                self.assertEqual(positive[joint], magnitude)
                np.testing.assert_array_equal(negative, -positive)

    def test_single_level_count_and_distinct_storage(self):
        cases = DIAG.force_cases(NAMES, 1.0)
        self.assertEqual(len(cases), 1 + 2 * len(NAMES))
        cases[1][1][0] = 999.0
        self.assertEqual(cases[0][1][0], 0.0)
        self.assertEqual(cases[2][1][0], -1.0)

    def test_invalid_names_and_torques_are_rejected(self):
        for names in ([], ["same", "same"]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                DIAG.force_cases(names, 1.0)
        for value in (0.0, -1.0, 10.1, np.nan, np.inf, -np.inf):
            with self.subTest(torque=value), self.assertRaises(ValueError):
                DIAG.force_cases(NAMES, value)
        with self.assertRaises(ValueError):
            DIAG.force_cases(NAMES, 1.0, 1.0)


class LegacyOverrideTests(unittest.TestCase):
    def setUp(self):
        self.before = np.array([[0.2, 0.0, 0.3, 0.0], [0.4, 0.1, 0.5, 0.1]], dtype=np.float32)

    def test_named_wheel_override_preserves_nonwheel_values(self):
        after = DIAG.override_legacy_friction(self.before, NAMES, wheel_value=0.01)
        np.testing.assert_array_equal(after[:, [0, 2]], self.before[:, [0, 2]])
        np.testing.assert_allclose(after[:, [1, 3]], 0.01)
        self.assertEqual(self.before[0, 1], 0.0)
        self.assertFalse(np.shares_memory(after, self.before))

    def test_explicit_zero_wheels_is_not_none_default(self):
        original = self.before.copy()
        after = DIAG.override_legacy_friction(original, NAMES, wheel_value=0.0)
        np.testing.assert_array_equal(after[:, [1, 3]], 0.0)
        np.testing.assert_array_equal(after[:, [0, 2]], original[:, [0, 2]])
        np.testing.assert_array_equal(DIAG.override_legacy_friction(original, NAMES), original)

    def test_zero_all_and_mutual_exclusion(self):
        np.testing.assert_array_equal(DIAG.override_legacy_friction(self.before, NAMES, zero_all=True), 0.0)
        with self.assertRaises(ValueError):
            DIAG.override_legacy_friction(self.before, NAMES, wheel_value=0.01, zero_all=True)

    def test_invalid_matrix_and_coefficient_rejected(self):
        for values in (np.zeros(4), np.zeros((1, 3)), np.full((1, 4), np.nan)):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                DIAG.override_legacy_friction(values, NAMES, wheel_value=0.01)
        for value in (-0.01, 1.01, np.inf, np.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DIAG.override_legacy_friction(self.before, NAMES, wheel_value=value)
        with self.assertRaises(ValueError):
            DIAG.override_legacy_friction(self.before, ["a", "b", "c", "d"], wheel_value=0.01)


class BoundsAndMetadataTests(unittest.TestCase):
    def test_single_step_and_multistep_physics_counts(self):
        single = DIAG.step_metadata(33, 1, 0.005)
        multi = DIAG.step_metadata(33, 20, 0.005)
        self.assertEqual(single["physics_steps"], 33)
        self.assertEqual(multi["physics_steps"], 660)
        self.assertEqual(multi["case_count"], 33)
        self.assertEqual(multi["steps_per_case"], 20)
        self.assertAlmostEqual(multi["force_interval_s"], 0.1)
        self.assertAlmostEqual(multi["total_integrated_time_s"], 3.3)

    def test_metadata_invalid_bounds_and_nonfinite_dt(self):
        for steps in (0, 21, 1.5, True):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                DIAG.step_metadata(33, steps, 0.005)
        for dt in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                DIAG.step_metadata(33, 1, dt)

    def test_valid_cli_values_and_boundary_values(self):
        DIAG.validate_options(options())
        DIAG.validate_options(options(torque_nm=10.0, steps_per_case=20, legacy_wheel_friction=1.0))
        DIAG.validate_options(options(legacy_wheel_friction=0.0, root_raise_m=-0.005))
        DIAG.validate_options(options(max_body_angular_speed_rad_s=100.0))

    def test_invalid_and_nonfinite_cli_values(self):
        changes = [dict(torque_nm=0.0), dict(extra_torque_nm=-1.0), dict(torque_nm=10.01),
                   dict(torque_nm=np.nan), dict(extra_torque_nm=np.inf), dict(root_raise_m=np.inf),
                   dict(root_raise_m=np.nan), dict(steps_per_case=0), dict(steps_per_case=21),
                   dict(steps_per_case=1.5), dict(steps_per_case=True), dict(legacy_wheel_friction=np.nan),
                   dict(legacy_wheel_friction=-0.01), dict(legacy_wheel_friction=1.01),
                   dict(extra_torque_nm=1.0), dict(zero_legacy_friction=True, legacy_wheel_friction=0.0)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                DIAG.validate_options(options(**change))
        for value in (0.0, -1.0, 1000.1, np.nan, np.inf):
            with self.subTest(angular_speed=value), self.assertRaises(ValueError):
                DIAG.validate_options(options(max_body_angular_speed_rad_s=value))

    def test_argparse_mutual_exclusion_before_simulator_import(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--engine", "mujoco", "--output-dir", "unused-output",
             "--legacy-wheel-friction", "0.01", "--zero-legacy-friction"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("not allowed with argument", completed.stderr)
        self.assertNotIn("ModuleNotFoundError", completed.stderr)


if __name__ == "__main__":
    unittest.main()
