"""Export a TRON1 RSL-RL checkpoint using the verified normalized JIT template.

No Isaac dependency or arbitrary checkpoint unpickling. The template fixes the
51->256->128->128->8 ELU architecture and the normalization computation. All
actor weights AND actor observation-normalizer buffers are replaced strictly.
The actor remains unclipped: deployment must clamp actions to [-1, 1].
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "outputs/remote-beyondmimic/eval-pilot-final/actor_normalized.pt"
EXPECTED_EPS = 0.01
ACTOR_SHAPES = {
    "actor.0.weight": (256, 51), "actor.0.bias": (256,),
    "actor.2.weight": (128, 256), "actor.2.bias": (128,),
    "actor.4.weight": (128, 128), "actor.4.bias": (128,),
    "actor.6.weight": (8, 128), "actor.6.bias": (8,),
}
NORMALIZER_SHAPES = {"normalizer._mean": (1, 51), "normalizer._var": (1, 51),
                     "normalizer._std": (1, 51), "normalizer.count": ()}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def map_checkpoint_state(model_state, template_state):
    """Validate exact policy keys/shapes/dtypes and return a strict JIT state."""
    expected = {**ACTOR_SHAPES, **NORMALIZER_SHAPES}
    if set(template_state) != set(expected):
        raise ValueError("Template does not have the inspected complete actor/normalizer state")
    mapped = {}
    mapping = {}
    for name, tensor in model_state.items():
        if name.startswith("actor."):
            destination = name
        elif name.startswith("actor_obs_normalizer."):
            destination = "normalizer." + name.removeprefix("actor_obs_normalizer.")
        else:
            continue  # critic, optimizer and exploration log_std are not inference actor state.
        if destination in mapped:
            raise ValueError(f"Duplicate mapped state: {destination}")
        mapped[destination] = tensor
        mapping[name] = destination
    missing, extra = sorted(set(expected) - set(mapped)), sorted(set(mapped) - set(expected))
    if missing or extra:
        raise ValueError(f"Incomplete or incompatible actor/normalizer state: missing={missing}, extra={extra}")
    for name, shape in expected.items():
        tensor, template_tensor = mapped[name], template_state[name]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tuple(template_tensor.shape) != shape:
            raise ValueError(f"Invalid tensor shape for {name}; expected {shape}")
        expected_dtype = torch.int64 if name == "normalizer.count" else torch.float32
        if tensor.dtype != expected_dtype or template_tensor.dtype != expected_dtype:
            raise ValueError(f"Unexpected dtype for {name}; expected {expected_dtype}")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Nonfinite checkpoint tensor: {name}")
        mapped[name] = tensor.detach().cpu().clone()
    if mapped["normalizer.count"].item() <= 0:
        raise ValueError("Observation normalizer has not accumulated any samples")
    if (mapped["normalizer._var"] < 0).any() or (mapped["normalizer._std"] < 0).any():
        raise ValueError("Normalizer variance/std must be nonnegative")
    if not torch.allclose(mapped["normalizer._std"], torch.sqrt(mapped["normalizer._var"]), rtol=1e-5, atol=1e-6):
        raise ValueError("Normalizer _std does not match sqrt(_var)")
    return mapped, mapping


def eager_forward(observations, state, eps):
    """Independent inference from raw checkpoint tensors, including normalization."""
    x = (observations - state["normalizer._mean"]) / (state["normalizer._std"] + eps)
    for layer in (0, 2, 4, 6):
        x = F.linear(x, state[f"actor.{layer}.weight"], state[f"actor.{layer}.bias"])
        if layer != 6:
            x = F.elu(x)
    return x


def validate_outputs(module, state, eps, observations, *, comparison_template=None, atol=1e-6):
    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim < 2 or observations.shape[-1] != 51 or not np.isfinite(observations).all():
        raise ValueError("Validation observations must be finite with last dimension 51")
    observations = observations.reshape(-1, 51)
    if not len(observations):
        raise ValueError("No validation observations")
    max_error, max_template_error = 0.0, 0.0
    with torch.inference_mode():
        for begin in range(0, len(observations), 512):
            batch = torch.from_numpy(observations[begin:begin + 512].copy())
            actual = module(batch)
            expected = eager_forward(batch, state, eps)
            if actual.shape != (len(batch), 8) or not torch.isfinite(actual).all().item():
                raise ValueError("Exported actor returned malformed/nonfinite actions")
            max_error = max(max_error, float(torch.max(torch.abs(actual - expected)).item()))
            if comparison_template is not None:
                max_template_error = max(max_template_error, float(torch.max(torch.abs(actual - comparison_template(batch))).item()))
    if max_error > atol or (comparison_template is not None and max_template_error > atol):
        raise ValueError(f"Export output validation failed: eager={max_error}, template={max_template_error}, tolerance={atol}")
    return {"observations_checked": len(observations), "max_abs_error_vs_checkpoint_eager": max_error,
            "max_abs_error_vs_original_template": max_template_error if comparison_template is not None else None,
            "absolute_tolerance": atol}


def export_checkpoint(checkpoint_path, output_dir, template_path=DEFAULT_TEMPLATE,
                      observation_paths=(), compare_template=False, expected_eps=EXPECTED_EPS):
    checkpoint_path, template_path, output_dir = map(Path, (checkpoint_path, template_path, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse or overwrite an existing output directory: {output_dir}")
    if not np.isfinite(expected_eps) or expected_eps <= 0:
        raise ValueError("Expected normalization epsilon must be finite and positive")
    checkpoint_hash, template_hash = sha256(checkpoint_path), sha256(template_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("Expected RSL-RL checkpoint containing model_state_dict")
    module = torch.jit.load(str(template_path), map_location="cpu").eval()
    if not hasattr(module, "normalizer") or not hasattr(module.normalizer, "eps"):
        raise ValueError("Template must include the observation normalizer with an explicit epsilon")
    eps = float(module.normalizer.eps)
    if not np.isclose(eps, expected_eps, rtol=0, atol=1e-12):
        raise ValueError(f"Template normalizer epsilon {eps} != expected {expected_eps}")
    state, mapping = map_checkpoint_state(checkpoint["model_state_dict"], module.state_dict())
    module.load_state_dict(state, strict=True)
    original_template = torch.jit.load(str(template_path), map_location="cpu").eval() if compare_template else None
    probe = np.random.default_rng(20261003).standard_normal((32, 51)).astype(np.float32)
    probes = validate_outputs(module, state, eps, probe, comparison_template=original_template)
    validations = []
    for path in observation_paths:
        path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            if "action_observation" not in archive:
                raise ValueError(f"Missing action_observation field: {path}")
            validation = validate_outputs(module, state, eps, archive["action_observation"],
                                          comparison_template=original_template)
        validations.append({"source": str(path.resolve()), "sha256": sha256(path), **validation})
    output_dir.mkdir(parents=True, exist_ok=False)
    actor_path = output_dir / "actor_normalized.pt"
    torch.jit.save(module, str(actor_path))
    restored = torch.jit.load(str(actor_path), map_location="cpu").eval()
    roundtrip = validate_outputs(restored, state, eps, probe)
    for name, value in state.items():
        if not torch.equal(restored.state_dict()[name], value):
            raise ValueError(f"Round-trip saved tensor differs: {name}")
    if sha256(checkpoint_path) != checkpoint_hash or sha256(template_path) != template_hash:
        raise ValueError("Input checkpoint or template changed during export")
    manifest = {
        "schema_version": 1, "status": "exported_and_validated", "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_iteration": int(checkpoint["iter"]) if "iter" in checkpoint else None,
        "template": str(template_path.resolve()), "template_sha256": template_hash,
        "actor": str(actor_path.resolve()), "actor_sha256": sha256(actor_path),
        "exporter_sha256": sha256(__file__), "torch_version": torch.__version__,
        "checkpoint_weights_only": True, "state_dict_strict": True,
        "mapped_state_keys": mapping, "all_normalizer_buffers_replaced": True,
        "normalizer_epsilon": eps, "normalizer_sample_count": int(state["normalizer.count"].item()),
        "normalizer_formula": "(obs - mean) / (std + eps)", "observation_dim": 51, "action_dim": 8,
        "actor_includes_observation_normalizer": True, "actor_output_is_unclipped": True,
        "deployment_action_clip": [-1.0, 1.0], "physics_or_success_claim": False,
        "synthetic_probe_validation": probes, "stored_observation_validations": validations,
        "serialized_actor_roundtrip_validation": roundtrip,
        "original_checkpoint_and_template_unchanged": True,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observations", type=Path, action="append", default=[],
                        help="NPZ action_observation arrays to verify; may repeat")
    parser.add_argument("--compare-template", action="store_true",
                        help="Also require output equality to original template (original checkpoint audit only)")
    parser.add_argument("--expected-normalizer-eps", type=float, default=EXPECTED_EPS)
    args = parser.parse_args()
    report = export_checkpoint(args.checkpoint, args.output_dir, args.template, args.observations,
                               args.compare_template, args.expected_normalizer_eps)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
