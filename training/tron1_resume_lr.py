"""Opt-in scalar learning-rate synchronization after a completed runner.load.

This pure-stdlib helper does not load checkpoints or import Torch/Isaac. The
caller must pin the supported RSL source and invoke it after optimizer restore,
before learn(). Only the PPO algorithm's learning_rate attribute is assigned;
optimizer groups/moments, policy/normalizers, counters, config, RNG and environment
are not modified. No claim about optimization or sim2sim success follows.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real


AUDIT_SCHEMA_VERSION = 1
AUDIT_SCOPE = "adaptive_ppo_learning_rate_scalar_from_loaded_optimizer"


def _positive_real(value, label):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a non-boolean real scalar")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be representable as a finite scalar") from error
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and strictly positive")
    return number


def _attribute(value, name, label):
    try:
        return getattr(value, name)
    except AttributeError as error:
        raise ValueError(f"Missing {label}") from error


def restore_resume_learning_rate(algorithm, *, expected_scalar_before=None,
                                 expected_loaded_lr=None):
    """Copy uniform restored optimizer LR to the adaptive PPO scalar, once.

    Every validation completes before the assignment. Optional expected values
    are exact, positive finite non-boolean real scalars, not tolerance checks.
    The prospective single-factor caller supplies .001 and 1e-5 explicitly.
    Distributed/multi-GPU learning is unsupported; CUDA contexts on other GPUs
    are not evidence that the PPO algorithm itself is distributed.
    """
    if _attribute(algorithm, "schedule", "algorithm.schedule") != "adaptive":
        raise ValueError("Only the unchanged adaptive learning-rate schedule is supported")
    if _attribute(algorithm, "is_multi_gpu", "algorithm.is_multi_gpu") is not False:
        raise ValueError("Only explicitly single-process, non-multi-GPU PPO is supported")
    before_value = _attribute(algorithm, "learning_rate", "algorithm.learning_rate")
    before = _positive_real(before_value, "algorithm.learning_rate")
    expected_before = (None if expected_scalar_before is None else
                       _positive_real(expected_scalar_before, "expected_scalar_before"))
    expected_loaded = (None if expected_loaded_lr is None else
                       _positive_real(expected_loaded_lr, "expected_loaded_lr"))
    if expected_before is not None and before_value != expected_scalar_before:
        raise ValueError("Algorithm scalar differs from expected_scalar_before")
    optimizer = _attribute(algorithm, "optimizer", "algorithm.optimizer")
    groups = _attribute(optimizer, "param_groups", "optimizer.param_groups")
    if (not isinstance(groups, Sequence) or isinstance(groups, (str, bytes, bytearray))
            or len(groups) == 0):
        raise ValueError("optimizer.param_groups must be a nonempty sequence of mappings")
    rates, raw_rates = [], []
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise ValueError(f"optimizer.param_groups[{index}] must be a mapping with lr")
        try:
            raw_rate = group["lr"]
        except KeyError as error:
            raise ValueError(f"optimizer.param_groups[{index}] must be a mapping with lr") from error
        rates.append(_positive_real(raw_rate, f"optimizer.param_groups[{index}].lr"))
        raw_rates.append(raw_rate)
    loaded = rates[0]
    if any(rate != raw_rates[0] for rate in raw_rates[1:]):
        raise ValueError("Restored optimizer learning rates must be exactly uniform")
    if expected_loaded is not None and raw_rates[0] != expected_loaded_lr:
        raise ValueError("Restored optimizer learning rate differs from expected_loaded_lr")

    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "scope": AUDIT_SCOPE,
        "status": "restored_from_loaded_optimizer",
        "scalar_before": before,
        "loaded_group_learning_rates": rates,
        "scalar_after": loaded,
        "touched_field": "algorithm.learning_rate",
        "optimizer_group_learning_rates_unchanged": True,
        "adaptive_schedule_preserved": True,
        "expected_scalar_before": expected_before,
        "expected_loaded_lr": expected_loaded,
        "multi_gpu": False,
    }
    algorithm.learning_rate = raw_rates[0]
    return audit
