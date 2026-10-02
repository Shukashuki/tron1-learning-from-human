"""Checks for actual-upstream GMR integration and locally produced WF motion."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("mink", "mujoco"))
if AVAILABLE:
    import mujoco
    from retarget_gmr import gmr_frame_data, load_upstream

CONFIG = ROOT / "config/gmr_retarget_wf.json"
MODEL = ROOT / "assets/robots/WF_TRON1A/mujoco/robot.xml"
UPSTREAM = ROOT / "third_party/GMR/general_motion_retargeting/motion_retarget.py"
ARTIFACT = ROOT / "outputs/gmr-cmu-16_03/robot_reference.npz"
BASELINE = ROOT / "outputs/mink-cmu-16_03/robot_reference.npz"


@unittest.skipUnless(AVAILABLE, "Optional Mink environment not installed")
class GMRAdapterTests(unittest.TestCase):
    def test_uniform_upstream_scale_preserves_anchored_targets(self):
        scale = .7221145082725878
        pelvis = np.array([.1, .2, 1.2])
        wheels = np.array([[.2, .3, .5], [.2, -.3, .5]])
        frame = gmr_frame_data(pelvis, wheels, np.array([1., 0., 0., 0.]), scale)
        # This is the uniform special case of upstream's root-relative scaling.
        root = frame["Pelvis"][0]
        np.testing.assert_allclose(root * scale, pelvis)
        for index, name in enumerate(("LeftAnkle", "RightAnkle")):
            expected = (frame[name][0] - root) * scale + root * scale
            np.testing.assert_allclose(expected, wheels[index], atol=1e-14)

    @unittest.skipUnless(UPSTREAM.exists() and MODEL.exists(), "Local GMR/model not present")
    def test_core_is_pinned_unmodified_and_api_routes_limits_by_keyword(self):
        settings = json.loads(CONFIG.read_text())["_adapter"]
        cls, _ = load_upstream(settings, MODEL, CONFIG)
        self.assertEqual(Path(sys.modules[cls.__module__].__file__), UPSTREAM)
        self.assertEqual(hashlib.sha256(UPSTREAM.read_bytes()).hexdigest(),
                         settings["upstream_core_sha256"])
        shim = cls.retarget.__globals__["mink"]
        limits = [object()]
        with patch("mink.solve_ik", return_value=np.zeros(1)) as solve:
            shim.solve_ik("configuration", "tasks", .01, "daqp", .5, limits)
        self.assertEqual(solve.call_args.args, ())
        self.assertIs(solve.call_args.kwargs["limits"], limits)
        self.assertIs(solve.call_args.kwargs["safety_break"], True)
        self.assertEqual(solve.call_args.kwargs["primal_tol"], 1e-10)

    def test_config_has_two_distinct_stages_and_no_human_knee_or_ankle_rotation(self):
        config = json.loads(CONFIG.read_text())
        self.assertIs(config["use_ik_match_table1"], True)
        self.assertIs(config["use_ik_match_table2"], True)
        self.assertNotEqual(config["ik_match_table1"], config["ik_match_table2"])
        self.assertIs(config["_adapter"]["offset_to_ground"], False)
        for stage in ("ik_match_table1", "ik_match_table2"):
            self.assertEqual(set(config[stage]), {"base_Link", "wheel_L_Link", "wheel_R_Link"})
            for name in ("wheel_L_Link", "wheel_R_Link"):
                self.assertEqual(config[stage][name][2], 0.)


@unittest.skipUnless(AVAILABLE and ARTIFACT.exists() and BASELINE.exists() and MODEL.exists(),
                     "Local GMR result/model not present")
class GMRResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(ARTIFACT, allow_pickle=False) as archive:
            cls.result = {key: archive[key] for key in archive.files}
        with np.load(BASELINE, allow_pickle=False) as archive:
            cls.baseline = {key: archive[key] for key in archive.files}
        cls.report = json.loads((ARTIFACT.parent / "report.json").read_text())
        cls.model = mujoco.MjModel.from_xml_path(str(MODEL))

    def test_shapes_finiteness_and_normalized_quaternions(self):
        r = self.result
        self.assertEqual(r["qpos"].shape, (410, 15))
        self.assertEqual(r["qvel"].shape, (410, 14))
        for key, values in r.items():
            if values.dtype.kind in "fiu":
                self.assertTrue(np.isfinite(values).all(), key)
        np.testing.assert_allclose(np.linalg.norm(r["qpos"][:, 3:7], axis=1), 1., atol=1e-12)

    def test_exact_target_and_time_matching_for_fair_comparison(self):
        for key in ("target_positions_world_m", "root_target_position_m", "root_target_quat_wxyz",
                    "time_s", "fps", "joint_names", "human_positions_world_m"):
            np.testing.assert_array_equal(self.result[key], self.baseline[key], err_msg=key)

    def test_joint_limits_and_frozen_wheels(self):
        r = self.result
        mask = r["joint_limited"]
        q = r["joint_positions_rad"][:, mask]
        self.assertTrue((q >= r["joint_ranges_rad"][mask, 0] - 1e-8).all())
        self.assertTrue((q <= r["joint_ranges_rad"][mask, 1] + 1e-8).all())
        wheels = [i for i, n in enumerate(r["joint_names"]) if str(n).startswith("wheel_")]
        np.testing.assert_allclose(r["joint_positions_rad"][:, wheels], 0., atol=1e-8)

    def test_actual_free_root_and_kinematic_only_flags(self):
        r = self.result
        self.assertIs(self.report["root_prescribed"], False)
        self.assertGreater(abs(r["qpos"][:, :3] - r["root_target_position_m"]).max(), 1e-6)
        for flag in ("physics_validated", "policy_trained", "hardware_ready", "upstream_core_modified"):
            self.assertIs(self.report[flag], False)
        self.assertEqual(self.report["upstream_retarget_calls"], len(r["time_s"]))

    def test_sites_match_full_clip_forward_kinematics(self):
        r = self.result
        data = mujoco.MjData(self.model)
        sites = [self.model.site(str(name)).id for name in r["site_names"]]
        for frame, qpos in enumerate(r["qpos"]):
            data.qpos[:] = qpos
            mujoco.mj_forward(self.model, data)
            np.testing.assert_allclose(data.site_xpos[sites], r["site_positions_world_m"][frame], atol=1e-11)

    def test_errors_and_actual_speed_are_reported(self):
        r = self.result
        sites = [list(r["site_names"]).index(name) for name in r["target_names"]]
        error = np.linalg.norm(r["site_positions_world_m"][:, sites] - r["target_positions_world_m"], axis=-1)
        np.testing.assert_allclose(error, r["target_error_m"], atol=1e-12)
        self.assertAlmostEqual(np.sqrt(np.mean(error ** 2)), self.report["all_wheel_errors"]["rmse_m"])
        self.assertAlmostEqual(r["ground_penetration_m"].max(), self.report["max_ground_penetration_m"])
        speed = np.diff(r["joint_positions_rad"], axis=0) / np.diff(r["time_s"])[:, None]
        self.assertAlmostEqual(abs(speed).max(), self.report["max_interval_joint_speed_rad_s"])


if __name__ == "__main__":
    unittest.main()
