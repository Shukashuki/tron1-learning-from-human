"""The open-loop replay must not silently misalign the Isaac action clock."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from replay_turn_isaac_actions_mujoco import applied_actions


class ActionReplayTests(unittest.TestCase):
    def test_skips_initial_snapshot_and_requires_contiguous_frames(self):
        archive = {"actions_raw": np.arange(32, dtype=np.float32).reshape(4, 1, 8),
                   "action_was_applied": np.array([[False], [True], [True], [True]]),
                   "reference_frame_for_action": np.array([[0], [0], [1], [2]])}
        np.testing.assert_array_equal(applied_actions(archive), archive["actions_raw"][1:, 0])
        archive["reference_frame_for_action"][2, 0] = 2
        with self.assertRaisesRegex(ValueError, "contiguous"):
            applied_actions(archive)


if __name__ == "__main__":
    unittest.main()
