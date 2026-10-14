"""Offline invariants for the separately pinned drift_drl turn reference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_drift_drl_turn_reference as turn


def car_csv():
    # CARLA's left-handed Y and local lateral velocity become right-handed -Y.
    rows = [",".join(turn.COLUMNS)]
    for index in range(3):
        rows.append(f"{index},{.2 * index},0,10,2,{np.degrees(np.arctan2(2, 10))},0,0,1,0,0")
    return ("\n".join(rows) + "\n").encode()


class DriftDrlReferenceTests(unittest.TestCase):
    def test_decoder_converts_local_velocity_handedness_and_estimates_clock(self):
        source = turn.decode_csv(car_csv())
        np.testing.assert_allclose(source["position_xy_m"], [[0, 0], [1, -.2], [2, -.4]])
        np.testing.assert_allclose(source["velocity_xy_m_s"], [[10, -2]] * 3)
        np.testing.assert_allclose(source["time_s_estimated"], [0, .1, .2])
        np.testing.assert_allclose(source["beta_rad"], -np.arctan2(2, 10))
        np.testing.assert_array_equal(source["unique_pose_indices"], [0, 1, 2])
        self.assertEqual(source["max_clock_displacement_residual_m"], 0.)

    def test_decoder_rejects_bad_schema_and_inconsistent_slip(self):
        with self.assertRaisesRegex(ValueError, "schema"):
            turn.decode_csv(car_csv().replace(b"local_vy", b"world_vy"))
        with self.assertRaisesRegex(ValueError, "slip angle"):
            turn.decode_csv(car_csv().replace(str(np.degrees(np.arctan2(2, 10))).encode(), b"0", 1))
        with self.assertRaisesRegex(ValueError, "hand-brake"):
            turn.decode_csv(car_csv().replace(b",1,0,0\n", b",1,False,0\n"))

    def test_prepare_rejects_unpinned_payload_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "car.csv"
            csv_path.write_bytes(car_csv())
            config = json.loads(turn.CONFIG.read_text())
            config["sha256"] = hashlib.sha256(b"different source").hexdigest()
            config_path = root / "source.json"
            config_path.write_text(json.dumps(config))
            output = root / "output"
            with self.assertRaisesRegex(ValueError, "pinned SHA-256"):
                turn.prepare(config_path, output, raw_csv=csv_path)
            self.assertFalse(output.exists())

    def test_exported_reference_turn_gate_checks_speed_slip_and_turn(self):
        times = np.arange(6) * .02
        course = np.deg2rad(np.linspace(0, 60, len(times)))
        xy = np.cumsum(.02 * np.c_[np.cos(course), np.sin(course)], axis=0)
        arrays = {"time_s": times, "root_target_position_m": np.c_[xy, np.full(6, .75)],
                  "source_frame_numbers": np.arange(6), "source_phase": np.zeros(6),
                  "root_yaw_rad": course - np.deg2rad(12)}
        config = {"drift_row_start_inclusive": 0, "drift_row_end_inclusive": 5}
        evidence = turn.reference_turn_evidence(arrays, config)
        self.assertGreater(evidence["course_turn_deg"], 45.)
        self.assertGreater(evidence["minimum_speed_m_s"], .8)
        self.assertGreater(evidence["minimum_abs_sideslip_deg"], 8.)
        arrays["root_yaw_rad"] = course
        with self.assertRaisesRegex(ValueError, "lost the high-speed"):
            turn.reference_turn_evidence(arrays, config)


if __name__ == "__main__":
    unittest.main()
