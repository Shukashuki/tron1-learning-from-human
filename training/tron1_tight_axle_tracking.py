"""Trial019: the existing directed axle objective, with std fixed at .2 rad.

Reuse the source-pinned original angle geometry, but do not modify its .4-rad
API or monkeypatch it. LAST-state sampling, directed local+Y targets and wheel
spin invariance are unchanged. This reward does not certify ground contact.
Torch/Isaac are not imported and no reward is computed on module import.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from training import tron1_wheel_axle_tracking as original

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_SOURCE_SHA256 = "aa3dac2bdae03f029d2c4bb174e79007f9b7927b5eaa04d35e50eefcff7de256"
SETTINGS = {"weight": 1.0, "std": .2}
BODY_NAMES = original.BODY_NAMES
WHEEL_BODY_NAMES = original.WHEEL_BODY_NAMES
TERM_NAME = original.TERM_NAME


def _guard():
    if hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest() != ORIGINAL_SOURCE_SHA256:
        raise RuntimeError("Trial019 original axle geometry source changed")


def _std(value):
    value = original._number(value, "std")
    original._require(value == SETTINGS["std"], "Trial019 std must remain .2 rad")
    return value


def wheel_axle_orientation_reward(reference_quat, actual_quat, std=.2):
    """Exactly exp(-mean_two(original_directed_axis_angle²)/.2²)."""
    _guard()
    std = _std(std)
    error = original.wheel_axle_angle_error(reference_quat, actual_quat)
    return (-error.square().mean(dim=-1) / (std * std)).exp()


def wheel_axle_orientation_tracking(env, command_name="motion", std=.2):
    """Same named LAST command/state tensors; no collector/reset/RNG changes."""
    _guard()
    _std(std)
    original._require(type(command_name) is str and command_name == "motion",
                      "Only the existing motion command is supported")
    command = env.command_manager.get_term(command_name)
    body_names = command.cfg.body_names
    original._require(type(body_names) in (list, tuple)
                      and all(type(k) is str and k for k in body_names)
                      and len(body_names) == len(original.BODY_NAMES)
                      and len(set(body_names)) == len(original.BODY_NAMES)
                      and set(body_names) == set(original.BODY_NAMES),
                      "Expected exactly the nine unique tracking body names")
    original._require(not isinstance(env.num_envs, bool)
                      and isinstance(env.num_envs, original.Integral) and env.num_envs > 0,
                      "num_envs must be a positive integer")
    reference, actual = command.body_quat_relative_w, command.robot_body_quat_w
    expected_shape = (int(env.num_envs), len(body_names), 4)
    original._require(tuple(reference.shape) == expected_shape and tuple(actual.shape) == expected_shape,
                      "Command quaternion body/environment dimensions do not match names")
    indices = [body_names.index(k) for k in original.WHEEL_BODY_NAMES]
    return wheel_axle_orientation_reward(reference[:, indices], actual[:, indices], std)
