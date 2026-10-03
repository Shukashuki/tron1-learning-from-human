"""CPU-only checkpoint export checks; no Isaac imports or runtime required."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    import export_tracking_checkpoint as exporter

TEMPLATE = ROOT / "outputs/remote-beyondmimic/eval-pilot-final/actor_normalized.pt"
CHECKPOINT = ROOT / "outputs/remote-beyondmimic/pilot-512x1000/model_final.pt"
OBSERVATIONS = ROOT / "outputs/remote-beyondmimic/eval-pilot-final/trajectory.npz"


@unittest.skipUnless(TORCH_AVAILABLE and TEMPLATE.exists() and CHECKPOINT.exists(), "Local Torch/template/checkpoint required")
class CheckpointExporterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        cls.template = torch.jit.load(str(TEMPLATE), map_location="cpu")

    def test_all_actor_and_normalizer_buffers_map_exactly(self):
        state, names = exporter.map_checkpoint_state(self.checkpoint["model_state_dict"], self.template.state_dict())
        self.assertEqual(len(state), 12)
        self.assertEqual(len(names), 12)
        self.assertEqual(names["actor_obs_normalizer._std"], "normalizer._std")
        self.assertEqual(names["actor_obs_normalizer.count"], "normalizer.count")
        for name, value in self.template.state_dict().items():
            self.assertTrue(torch.equal(state[name], value), name)

    def test_missing_or_extra_policy_state_rejected(self):
        for name in ("actor.2.weight", "actor_obs_normalizer._mean", "actor_obs_normalizer._std", "actor_obs_normalizer.count"):
            state = dict(self.checkpoint["model_state_dict"])
            del state[name]
            with self.subTest(name=name), self.assertRaises(ValueError):
                exporter.map_checkpoint_state(state, self.template.state_dict())
        state = dict(self.checkpoint["model_state_dict"])
        state["actor.99.bias"] = torch.zeros(1)
        with self.assertRaises(ValueError):
            exporter.map_checkpoint_state(state, self.template.state_dict())

    def test_shape_dtype_and_nonfinite_tensors_rejected(self):
        for bad in (torch.zeros(7), torch.zeros(8, dtype=torch.float64), torch.full((8,), float("nan"))):
            state = dict(self.checkpoint["model_state_dict"])
            state["actor.6.bias"] = bad
            with self.subTest(shape=tuple(bad.shape), dtype=bad.dtype), self.assertRaises(ValueError):
                exporter.map_checkpoint_state(state, self.template.state_dict())

    def test_inconsistent_or_uninitialized_normalizer_rejected(self):
        for key, tensor in (("actor_obs_normalizer._var", -torch.ones(1, 51)),
                            ("actor_obs_normalizer._std", torch.zeros(1, 51)),
                            ("actor_obs_normalizer.count", torch.tensor(0))):
            state = dict(self.checkpoint["model_state_dict"])
            state[key] = tensor
            with self.subTest(key=key), self.assertRaises(ValueError):
                exporter.map_checkpoint_state(state, self.template.state_dict())

    @unittest.skipUnless(OBSERVATIONS.exists(), "Stored Isaac observations required")
    def test_original_checkpoint_matches_template_for_every_stored_observation(self):
        with tempfile.TemporaryDirectory(prefix="tron1-checkpoint-export-test-") as directory:
            output = Path(directory) / "new-export"
            with patch.object(exporter.torch, "load", wraps=torch.load) as loader:
                report = exporter.export_checkpoint(CHECKPOINT, output, TEMPLATE, [OBSERVATIONS], True)
            self.assertIs(loader.call_args.kwargs["weights_only"], True)
            check = report["stored_observation_validations"][0]
            self.assertEqual(check["observations_checked"], 3552)
            self.assertLessEqual(check["max_abs_error_vs_original_template"], 1e-6)
            self.assertLessEqual(check["max_abs_error_vs_checkpoint_eager"], 1e-6)
            self.assertEqual(report["normalizer_epsilon"], .01)
            self.assertTrue((output / "actor_normalized.pt").exists())
            self.assertTrue((output / "manifest.json").exists())
            self.assertEqual(report["checkpoint_sha256"], exporter.sha256(CHECKPOINT))

    def test_refuses_existing_directory_and_wrong_epsilon(self):
        with tempfile.TemporaryDirectory(prefix="tron1-checkpoint-refuse-test-") as directory:
            with self.assertRaises(FileExistsError):
                exporter.export_checkpoint(CHECKPOINT, Path(directory), TEMPLATE)
            with self.assertRaises(ValueError):
                exporter.export_checkpoint(CHECKPOINT, Path(directory) / "new", TEMPLATE, expected_eps=1e-5)
            self.assertFalse((Path(directory) / "new").exists())

    def test_changed_checkpoint_replaces_weights_and_normalization(self):
        with tempfile.TemporaryDirectory(prefix="tron1-checkpoint-change-test-") as directory:
            path = Path(directory)
            checkpoint = {"model_state_dict": {key: value.clone() for key, value in self.checkpoint["model_state_dict"].items()}, "iter": 1234}
            checkpoint["model_state_dict"]["actor.6.bias"] += .25
            checkpoint["model_state_dict"]["actor_obs_normalizer._mean"] += .5
            torch.save(checkpoint, path / "changed.pt")
            report = exporter.export_checkpoint(path / "changed.pt", path / "export", TEMPLATE)
            self.assertEqual(report["checkpoint_iteration"], 1234)
            restored = torch.jit.load(str(path / "export/actor_normalized.pt"))
            self.assertTrue(torch.equal(restored.state_dict()["actor.6.bias"], checkpoint["model_state_dict"]["actor.6.bias"]))
            self.assertTrue(torch.equal(restored.state_dict()["normalizer._mean"], checkpoint["model_state_dict"]["actor_obs_normalizer._mean"]))


if __name__ == "__main__":
    unittest.main()
