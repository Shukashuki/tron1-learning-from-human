"""CMU reader/FK regression tests; real-data tests are opt-in, never download."""

import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cmu_motion import (SOURCE_TO_WORLD, _fixed_rotation, forward_kinematics,
                        load_motion, read_amc, read_asf, save_motion)


ASF = """:version 1.10
:units
mass 1.0
length 0.0254
angle deg
:root
order TZ TX TY RZ RX RY
axis XYZ
position 1 2 3
orientation 0 0 90
:bonedata
begin
id 1
name lfoot
direction 0 0 1.000001
length 2
axis 0 0 90 XYZ
dof rx ry rz
limits (-180 180)
       (-180 180)
       (-180 180)
end
begin
id 2
name ltoes
direction 1 0 0
length 0.5
axis 10 20 30 ZXY
dof rx
limits (-5 5)
end
:hierarchy
begin
root lfoot
lfoot ltoes
end
"""

AMC = """:FULLY-SPECIFIED
:DEGREES
10
root 10 20 30 15 25 35
lfoot 90 0 0
ltoes 10
11
root 11 21 32 15 25 35
lfoot 100 10 -20
ltoes 20
13
root 12 22 34 15 25 35
lfoot 80 -10 20
ltoes -5
14
root 13 23 30 15 25 35
lfoot 95 5 10
ltoes 0
"""


def rx(degrees):
    c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(degrees):
    c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(degrees):
    c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class CMUMotionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="cmu-motion-test-")
        self.addCleanup(self.tmp.cleanup)
        self.asf = Path(self.tmp.name) / "subject.asf"
        self.amc = Path(self.tmp.name) / "clip.amc"
        self.asf.write_text(ASF)
        self.amc.write_text(AMC)

    def test_extrinsic_xyz_rotation_order(self):
        actual = _fixed_rotation("XYZ", np.array([20, 30, 40]), degrees=True)
        np.testing.assert_allclose(actual, rz(40) @ ry(30) @ rx(20), atol=1e-15)
        self.assertGreater(np.max(np.abs(actual - rx(20) @ ry(30) @ rz(40))), 0.1)

    def test_root_order_static_orientation_and_axis_corrected_bones(self):
        skeleton = read_asf(self.asf)
        numbers, channels, degrees = read_amc(self.amc, skeleton)
        p, r = forward_kinematics(skeleton, channels, degrees)
        np.testing.assert_array_equal(numbers, [10, 11, 13, 14])
        np.testing.assert_array_equal(skeleton.parent_indices, [-1, 0, 1])
        root_r = rz(90) @ ry(35) @ rx(25) @ rz(15)
        foot_r = root_r @ (rz(90) @ rx(90) @ rz(90).T)
        toe_basis = ry(20) @ rx(10) @ rz(30)
        toe_r = foot_r @ (toe_basis @ rx(10) @ toe_basis.T)
        np.testing.assert_allclose(p[0, 0], [21, 32, 13], atol=1e-14)
        np.testing.assert_allclose(r[0], [root_r, foot_r, toe_r], atol=1e-14)
        np.testing.assert_allclose(p[0, 1], p[0, 0] + foot_r @ [0, 0, 2], atol=1e-14)
        np.testing.assert_allclose(p[0, 2], p[0, 1] + toe_r @ [0.5, 0, 0], atol=1e-14)
        # The authored 10 degrees must not be clipped to the ASF limit of 5.
        self.assertEqual(channels["ltoes"][0, 0], 10)

    def test_single_axis_one_and_many_frames(self):
        for values in (np.array([[30.0]]), np.array([[30.0], [60.0], [90.0]])):
            r = _fixed_rotation("x", values, degrees=True)
            self.assertEqual(r.shape, (len(values), 3, 3))
            for actual, angle in zip(r, values[:, 0]):
                np.testing.assert_allclose(actual, rx(angle), atol=1e-15)

    def test_constant_alignment_preserves_root_trajectory(self):
        motion = load_motion(self.asf, self.amc, fps=4)
        source = motion["source_root_translation_m"]
        root = motion["joint_positions_world_m"][:, 0]
        np.testing.assert_allclose(np.diff(root, axis=0), np.diff(source, axis=0) @ SOURCE_TO_WORLD.T, atol=1e-14)
        np.testing.assert_allclose(root[0, :2], 0, atol=1e-14)
        np.testing.assert_allclose(motion["time_s"], [0, 0.25, 0.75, 1])
        np.testing.assert_allclose(np.diff(root[:, 2]), [2, 2, -4], atol=1e-14)
        self.assertFalse(motion["metadata"]["root_is_com"])

    def test_floor_is_not_reset_for_airborne_frames(self):
        self.amc.write_text(AMC.replace("lfoot 100 10 -20", "lfoot 90 0 0")
                            .replace("lfoot 80 -10 20", "lfoot 90 0 0")
                            .replace("lfoot 95 5 10", "lfoot 90 0 0")
                            .replace("ltoes 20", "ltoes 10")
                            .replace("ltoes -5", "ltoes 10")
                            .replace("ltoes 0", "ltoes 10"))
        motion = load_motion(self.asf, self.amc, fps=4)
        floor_markers = motion["joint_positions_world_m"][:, 1:, 2].min(axis=1)
        np.testing.assert_allclose(floor_markers, [0, 2, 4, 0], atol=1e-14)

    def test_bone_lengths_rotations_and_finite_values(self):
        motion = load_motion(self.asf, self.amc, fps=4)
        positions = motion["joint_positions_world_m"]
        parents = motion["parent_indices"]
        lengths = np.linalg.norm(positions[:, 1:] - positions[:, parents[1:]], axis=-1)
        np.testing.assert_allclose(lengths, np.tile([2, 0.5], (4, 1)), atol=1e-14)
        self.assertLess(motion["metadata"]["max_bone_length_error_m"], 1e-14)
        quats = motion["joint_rotations_world_wxyz"]
        np.testing.assert_allclose(np.linalg.norm(quats, axis=-1), 1, atol=1e-15)
        matrices = Rotation.from_quat(quats[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix()
        np.testing.assert_allclose(np.linalg.det(matrices), 1, atol=2e-15)
        self.assertEqual(np.linalg.det(SOURCE_TO_WORLD), 1)
        skeleton = read_asf(self.asf)
        _, channels, degrees = read_amc(self.amc, skeleton)
        _, source_rotations = forward_kinematics(skeleton, channels, degrees)
        np.testing.assert_allclose(matrices.reshape(4, 3, 3, 3), SOURCE_TO_WORLD @ source_rotations, atol=2e-15)
        for name, value in motion.items():
            if isinstance(value, np.ndarray) and value.dtype.kind not in "US":
                self.assertTrue(np.isfinite(value).all(), name)

    def test_non_pickled_exports_and_provenance(self):
        motion = load_motion(self.asf, self.amc, fps=4)
        paths = save_motion(motion, Path(self.tmp.name) / "exports")
        with np.load(paths["npz"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["joint_positions_world_m"], motion["joint_positions_world_m"])
            metadata = json.loads(str(saved["metadata_json"]))
            self.assertEqual(len(metadata["source"]["asf_sha256"]), 64)
        self.assertEqual(json.loads(paths["metadata"].read_text()), motion["metadata"])
        with paths["csv"].open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4 * 3)
        self.assertEqual(rows[0]["joint_name"], "root")

    def test_missing_fully_specified_channel_is_rejected(self):
        self.amc.write_text(AMC.replace("ltoes 20\n", ""))
        with self.assertRaisesRegex(ValueError, "missing channels"):
            load_motion(self.asf, self.amc)

    def test_sparse_motion_holds_previous_values(self):
        self.amc.write_text(AMC.replace(":FULLY-SPECIFIED\n", "").replace("ltoes 20\n", ""))
        _, channels, _ = read_amc(self.amc, read_asf(self.asf))
        np.testing.assert_array_equal(channels["ltoes"][:, 0], [10, 10, -5, 0])

    def test_radian_amc_matches_degree_amc(self):
        skeleton = read_asf(self.asf)
        _, channels, _ = read_amc(self.amc, skeleton)
        radians = {name: values.copy() for name, values in channels.items()}
        rotation_indices = [i for i, name in enumerate(skeleton.root_order) if name.startswith("r")]
        radians["root"][:, rotation_indices] = np.deg2rad(radians["root"][:, rotation_indices])
        for name in radians.keys() - {"root"}:
            radians[name] = np.deg2rad(radians[name])
        expected = forward_kinematics(skeleton, channels, True)
        actual = forward_kinematics(skeleton, radians, False)
        for first, second in zip(expected, actual):
            np.testing.assert_allclose(first, second, atol=1e-14)

    def test_invalid_fps_and_nonfinite_motion_are_rejected(self):
        for value in (0, -1, np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "fps"):
                load_motion(self.asf, self.amc, fps=value)
        self.amc.write_text(AMC.replace("ltoes 20", "ltoes nan"))
        with self.assertRaisesRegex(ValueError, "Invalid AMC"):
            load_motion(self.asf, self.amc)


@unittest.skipUnless(os.environ.get("CMU_TEST_DATA_DIR"), "Set CMU_TEST_DATA_DIR to a local 16.asf / 16_03.amc directory")
class CMURealClipTests(unittest.TestCase):
    def test_16_03_against_independently_computed_endpoints(self):
        data = Path(os.environ["CMU_TEST_DATA_DIR"])
        motion = load_motion(data / "16.asf", data / "16_03.amc")
        self.assertEqual(motion["joint_positions_world_m"].shape, (410, 31, 3))
        self.assertEqual(motion["metadata"]["source"]["amc_sha256"], "2f60815066e68bd2f70a576400434e53b43af35f8a009f1342e9367ab93c03f9")
        names = motion["joint_names"].tolist()
        p = motion["joint_positions_world_m"]
        floor = motion["metadata"]["floor_alignment"]["estimated_floor_before_shift_m"]
        # Independent McCann-convention FK, before the constant floor shift.
        expected_source_y = {"root": 1.0063141333, "ltibia": 0.084597999,
                             "rtibia": 0.08571694, "lfoot": 0.0421472,
                             "ltoes": 0.0348796, "rfoot": 0.0321966,
                             "rtoes": 0.0185792, "head": 1.6468186}
        for name, expected in expected_source_y.items():
            self.assertAlmostEqual(p[0, names.index(name), 2] + floor, expected, delta=1e-6, msg=name)
        feet = [names.index(name) for name in ("lfoot", "ltoes", "rfoot", "rtoes")]
        clearance = p[:, feet, 2].min(axis=1)
        self.assertAlmostEqual(clearance.min(), 0, delta=1e-12)
        self.assertEqual(np.argmax(clearance), 216)
        self.assertAlmostEqual(clearance.max(), 0.4275099111, delta=1e-6)
        np.testing.assert_array_equal(np.flatnonzero(clearance > 0.06), np.arange(185, 247))
        self.assertLess(motion["metadata"]["max_bone_length_error_m"], 1e-12)
        self.assertGreater(motion["metadata"]["floor_alignment"]["last_window_lowest_foot_residual"]["median_m"], 0.05)


if __name__ == "__main__":
    unittest.main()
