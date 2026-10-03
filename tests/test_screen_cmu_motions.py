"""Synthetic NumPy/SciPy screening tests; no Mink, Torch, network, or robot."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import screen_cmu_motions as screening


def synthetic():
    names = np.array(["root", "ltibia", "rtibia", "lfoot", "ltoes", "rfoot", "rtoes"])
    count, fps = 7, 10.
    p = np.zeros((count, len(names), 3))
    p[:, :, 0] = np.arange(count)[:, None] * .1
    p[:, 0, 2] = [1., 1., 1.1, 1.3, 1.2, 1., 1.]
    foot = np.array([0., 0., .1, .2, .1, 0., 0.])
    p[:, 1:3, 2] = foot[:, None] + .1
    p[:, 3:, 2] = foot[:, None]
    p[:, [1, 3, 4], 1] = .1
    p[:, [2, 5, 6], 1] = -.1
    q = Rotation.from_matrix(screening.ASF_TO_WORLD).as_quat()[[3, 0, 1, 2]]
    return {"fps": np.asarray(fps), "time_s": np.arange(count) / fps,
            "frame_numbers": np.arange(10, 10 + count), "joint_names": names,
            "parent_indices": np.array([-1, 0, 0, 1, 3, 2, 5]),
            "joint_positions_world_m": p, "joint_rotations_world_wxyz": np.tile(q, (count, len(names), 1)),
            "metadata_json": np.asarray(json.dumps({"output_up_axis": "+Z", "root_is_com": False,
                "source": {"dataset": "synthetic", "amc_sha256": "a" * 64},
                "floor_alignment": {"note": "initial standing is assumed, not measured"}}))}


class ScreeningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tron1-cmu-screen-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def save(self, arrays=None, name="human_motion.npz"):
        path = self.root / name
        np.savez_compressed(path, **(synthetic() if arrays is None else arrays))
        return path

    def test_exact_measured_metrics_and_no_mutation(self):
        path = self.save()
        original_hash = screening.sha256(path)
        motion = screening.load_human_motion(path)
        positions = motion["joint_positions_world_m"].copy()
        times = motion["time_s"].copy()
        report = screening.analyze_motion(motion, "synthetic")
        self.assertEqual(report["frames"], 7)
        self.assertEqual(report["fps"], 10.)
        self.assertAlmostEqual(report["timestamp_span_s"], .6)
        np.testing.assert_allclose(report["pelvis"]["displacement_xyz_m"], [.6, 0, 0])
        self.assertAlmostEqual(report["pelvis"]["height_excursion_m"], .3)
        self.assertAlmostEqual(report["pelvis"]["max_rise_from_initial_m"], .3)
        self.assertFalse(report["pelvis"]["is_whole_body_com"])
        intervals = report["kinematic_both_feet_clear"]["intervals"]
        self.assertEqual(len(intervals), 1)
        self.assertEqual((intervals[0]["start_sample"], intervals[0]["last_clear_sample"]), (2, 4))
        self.assertAlmostEqual(intervals[0]["observed_clear_span_s"], .2)
        self.assertFalse(report["kinematic_both_feet_clear"]["measured_contact"])
        self.assertEqual(report["feet"]["left"]["nodes"], ["lfoot", "ltoes"])
        for key in ("robot_retargeted", "physics_validated", "policy_trained", "motion_modified", "retimed", "scaled", "clipped"):
            self.assertFalse(report[key], key)
        np.testing.assert_array_equal(motion["joint_positions_world_m"], positions)
        np.testing.assert_array_equal(motion["time_s"], times)
        self.assertEqual(screening.sha256(path), original_hash)

    def test_both_feet_required_not_one_foot_or_ankle_height(self):
        arrays = synthetic()
        arrays["joint_positions_world_m"][:, 5:, 2] = 0.
        report = screening.analyze_motion(screening.load_human_motion(self.save(arrays)), "one-foot")
        self.assertEqual(report["kinematic_both_feet_clear"]["intervals"], [])
        self.assertGreater(report["feet"]["left"]["height_min_max_m"][1], .05)

    def test_end_baseline_and_penetration_warnings_do_not_reground(self):
        arrays = synthetic()
        arrays["joint_positions_world_m"][-3:, 3:, 2] -= .12
        original = arrays["joint_positions_world_m"].copy()
        motion = screening.load_human_motion(self.save(arrays))
        report = screening.analyze_motion(motion, "different-support")
        text = " ".join(report["warnings"])
        self.assertIn("end/start", text)
        self.assertIn("below the initial floor", text)
        self.assertAlmostEqual(report["feet"]["right"]["height_min_max_m"][0], -.12)
        np.testing.assert_array_equal(motion["joint_positions_world_m"], original)

    def test_gaps_split_intervals_without_retiming(self):
        times = np.array([0., .1, .4, .5])
        frames = np.array([1, 2, 5, 6])
        intervals = screening.clear_intervals(np.ones(4, dtype=bool), times, frames, .05)
        self.assertEqual(len(intervals), 2)
        self.assertAlmostEqual(sum(item["observed_clear_span_s"] for item in intervals), .2)
        self.assertTrue(all(item["left_censored"] and item["right_censored"] for item in intervals))

    def test_heading_matches_existing_mink_calibration_without_importing_mink(self):
        source = ast.parse((ROOT / "scripts/retarget_mink.py").read_text())
        function = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                        and node.name == "calibrated_root_rotations")
        namespace = {"np": np, "Rotation": Rotation}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "calibration-only", "exec"), namespace)
        human = Rotation.from_euler("ZYX", [[170., 5., 3.], [180., 8., -3.], [195., 2., 4.]], degrees=True).as_matrix()
        raw = Rotation.from_matrix(human @ screening.ASF_TO_WORLD).as_quat()[:, [3, 0, 1, 2]]
        calibrated, _ = namespace["calibrated_root_rotations"](raw)
        matrix = Rotation.from_quat(calibrated[:, [1, 2, 3, 0]]).as_matrix()
        expected = np.unwrap(np.arctan2(matrix[:, 1, 0], matrix[:, 0, 0]))
        np.testing.assert_allclose(screening.calibrated_heading(raw), expected, atol=1e-14)
        self.assertGreater(np.degrees(expected[-1]), 20.)

    def test_missing_rotations_is_explicit_and_vertical_heading_unavailable(self):
        arrays = synthetic()
        del arrays["joint_rotations_world_wxyz"]
        report = screening.analyze_motion(screening.load_human_motion(self.save(arrays)), "no-quats")
        self.assertIsNone(report["calibrated_root_heading"]["change_deg"])
        human = Rotation.from_euler("y", [[0.], [90.]], degrees=True).as_matrix()
        quats = Rotation.from_matrix(human @ screening.ASF_TO_WORLD).as_quat()[:, [3, 0, 1, 2]]
        self.assertIsNone(screening.calibrated_heading(quats))

    def test_malformed_source_is_rejected(self):
        cases = []
        bad = synthetic(); bad["time_s"][2] = bad["time_s"][1]; cases.append(bad)
        bad = synthetic(); bad["time_s"] *= 2; cases.append(bad)
        bad = synthetic(); bad["joint_positions_world_m"][0, 0, 0] = np.nan; cases.append(bad)
        bad = synthetic(); bad["joint_names"][2] = "ltibia"; cases.append(bad)
        bad = synthetic(); bad["parent_indices"][1] = 1; cases.append(bad)
        bad = synthetic(); bad["joint_rotations_world_wxyz"][0, 0] = 0; cases.append(bad)
        bad = synthetic(); bad["fps"] = np.array(-1.); cases.append(bad)
        for index, arrays in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                screening.load_human_motion(self.save(arrays, f"bad-{index}.npz"))

    def test_rejects_reused_output_duplicate_ids_and_invalid_thresholds(self):
        path = self.save()
        output = self.root / "occupied"
        output.mkdir()
        (output / "keep.txt").write_text("unchanged")
        with self.assertRaises(FileExistsError):
            screening.screen_motions([("A", path)], output)
        self.assertEqual((output / "keep.txt").read_text(), "unchanged")
        with self.assertRaises(ValueError):
            screening.screen_motions([("A", path), ("A", path)], self.root / "unused")
        motion = screening.load_human_motion(path)
        for args in ({"clearance_m": 0.}, {"initial_window_s": np.nan}, {"minimum_clear_span_s": -1.}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                screening.analyze_motion(motion, "bad", **args)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Optional overview test requires Matplotlib")
    def test_screen_outputs_png_json_and_provenance(self):
        path = self.save()
        output = self.root / "screen"
        source_hash = screening.sha256(path)
        report = screening.screen_motions([("demo", path)], output)
        self.assertEqual({p.name for p in output.iterdir()}, {"summary.json", "overview.png"})
        self.assertEqual((output / "overview.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(json.loads((output / "summary.json").read_text()), report)
        self.assertEqual(report["clips"][0]["input"]["npz_sha256"], source_hash)
        self.assertEqual(report["clips"][0]["source_metadata"]["source"]["amc_sha256"], "a" * 64)
        self.assertEqual(screening.sha256(path), source_hash)
        self.assertFalse(report["source_arrays_modified"])


if __name__ == "__main__":
    unittest.main()
