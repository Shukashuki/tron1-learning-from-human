"""Opt-in directed wheel-AXLE tracking, not full wheel-phase tracking.

Targets use the existing command's yaw-reanchored relative quaternions. Local
+Y is the named wheel joint axis: arbitrary local-Y spin and quaternion sign
do not change it. This adds emphasis to a direction already constrained by
parent-knee orientation, not a new teacher or contact/force guarantee.

Torch and Isaac imports are lazy. The reward samples the existing LAST state
at the 50 Hz reward clock; no collector, history, reset, RNG or actor is added.
GPU finite checks can synchronize the host; runtime/GPU behavior is unproven
until separately tested with the actual environment.
"""
from __future__ import annotations

import importlib
import math
from numbers import Integral, Real


WHEEL_BODY_NAMES = ("wheel_L_Link", "wheel_R_Link")
NON_WHEEL_BODY_NAMES = ("base_Link", "abad_L_Link", "abad_R_Link", "hip_L_Link", "hip_R_Link",
                       "knee_L_Link", "knee_R_Link")
BODY_NAMES = NON_WHEEL_BODY_NAMES + WHEEL_BODY_NAMES
TERM_NAME = "drift_wheel_axle_orientation"
SETTINGS = {"weight": 1.0, "std": .4}


def _require(condition, message):
    if not condition:
        raise ValueError("Wheel axle tracking: " + message)


def _number(value, label):
    _require(not isinstance(value, bool) and isinstance(value, Real), label + " must be a nonboolean number")
    value = float(value)
    _require(math.isfinite(value), label + " must be finite")
    return value


def validate_wheel_axle_settings(settings):
    """Only the declared new term: weight 1, angular scale .4 rad."""
    _require(type(settings) is dict and set(settings) == set(SETTINGS), "Exact weight/std settings required")
    checked = {k: _number(settings[k], k) for k in SETTINGS}
    _require(checked == SETTINGS, "Only explicit weight 1.0 and std .4 are supported")
    return checked


def _std(value):
    value = _number(value, "std")
    _require(value == SETTINGS["std"], "std must remain .4 rad")
    return value


def _unit_quaternion_axis(torch, value):
    _require(torch.isfinite(value).all().item(), "Quaternion values must be finite")
    length = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    _require((torch.isfinite(length) & (length > 0)).all().item(), "Quaternion norm must be finite and nonzero")
    quaternion = value / length  # out-of-place: never modify command/state tensors
    w, x, y, z = quaternion.unbind(dim=-1)
    axis = torch.stack((2 * (x * y - w * z), 1 - 2 * (x.square() + z.square()),
                        2 * (y * z + w * x)), dim=-1)
    return axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True)


def wheel_axle_angle_error(reference_quat, actual_quat):
    """Directed angles [N,2] in radians from finite wxyz tensors [N,2,4].

    atan2(norm(cross), dot) handles matching/opposite axes without an acos
    endpoint tolerance. Axle sign is directed: a 180-degree reversal is NOT a
    match. Quaternion normalization is local/out-of-place, not a state write.
    """
    torch = importlib.import_module("torch")
    for label, value in (("reference", reference_quat), ("actual", actual_quat)):
        _require(isinstance(value, torch.Tensor) and torch.is_floating_point(value), label + " must be a floating tensor")
        _require(value.ndim == 3 and value.shape[0] > 0 and tuple(value.shape[1:]) == (2, 4),
                 label + " quaternion shape must be [N,2,4]")
    _require(reference_quat.shape == actual_quat.shape and reference_quat.dtype == actual_quat.dtype
             and reference_quat.device == actual_quat.device, "Quaternion shape/dtype/device must match")
    reference = _unit_quaternion_axis(torch, reference_quat)
    actual = _unit_quaternion_axis(torch, actual_quat)
    cross = torch.linalg.vector_norm(torch.linalg.cross(reference, actual, dim=-1), dim=-1)
    dot = (reference * actual).sum(dim=-1).clamp(-1., 1.)
    return torch.atan2(cross, dot)


def wheel_axle_orientation_reward(reference_quat, actual_quat, std=.4):
    """Positive exp(-mean_two(directed_axis_angle²)/std²), shape [N]."""
    std = _std(std)
    error = wheel_axle_angle_error(reference_quat, actual_quat)
    return (-error.square().mean(dim=-1) / (std * std)).exp()


def wheel_axle_orientation_tracking(env, command_name="motion", std=.4):
    """Read only existing named command tensors; no native body-ID assumption."""
    _std(std)
    _require(type(command_name) is str and command_name == "motion", "Only the existing motion command is supported")
    command = env.command_manager.get_term(command_name)
    body_names = command.cfg.body_names
    _require(type(body_names) in (list, tuple) and all(type(k) is str and k for k in body_names)
             and len(body_names) == len(BODY_NAMES) and len(set(body_names)) == len(BODY_NAMES)
             and set(body_names) == set(BODY_NAMES), "Expected exactly the nine unique tracking body names")
    _require(not isinstance(env.num_envs, bool) and isinstance(env.num_envs, Integral) and env.num_envs > 0,
             "num_envs must be a positive integer")
    reference, actual = command.body_quat_relative_w, command.robot_body_quat_w
    expected_shape = (int(env.num_envs), len(body_names), 4)
    _require(tuple(reference.shape) == expected_shape and tuple(actual.shape) == expected_shape,
             "Command quaternion body/environment dimensions do not match names")
    # Exactly LEFT,RIGHT in the command's tracking subset, not native IDs 8,9.
    indices = [body_names.index(k) for k in WHEEL_BODY_NAMES]
    return wheel_axle_orientation_reward(reference[:, indices], actual[:, indices], std)


def configure_wheel_axle_tracking(cfg, settings):
    """After the original dense factory, add ONLY one new RewardTermCfg.

    Baseline profile validation/configuration belongs to the caller. Historical
    full wheel-quaternion exclusion stays unchanged; do not turn the existing
    wheel_orientation_tracking contract flag on for this axis-only objective.
    """
    checked = validate_wheel_axle_settings(settings)
    rewards = cfg.rewards
    _require(not hasattr(rewards, TERM_NAME), "Refusing to overwrite/reinstall an existing reward term")
    orientation = rewards.motion_body_ori
    _require(type(orientation.params) is dict and orientation.params.get("body_names") == list(NON_WHEEL_BODY_NAMES),
             "Original full-quaternion objective must still exclude BOTH wheels")
    _require(orientation.params.get("command_name") == "motion" and _number(orientation.params.get("std"), "original std") == .4
             and _number(orientation.weight, "original orientation weight") == 1.,
             "Original body-orientation weight/std/command must remain unchanged")
    managers = importlib.import_module("isaaclab.managers")
    term = managers.RewardTermCfg(func=wheel_axle_orientation_tracking, weight=checked["weight"],
                                 params={"command_name": "motion", "std": checked["std"]})
    setattr(rewards, TERM_NAME, term)
    return {"term_name": TERM_NAME, "settings": checked, "sampling": "last_physics_substep_at_50Hz_reward_clock",
            "target": "existing command.body_quat_relative_w, directed local +Y, ordered left/right",
            "wheel_axial_phase_tracked": False, "original_body_orientation_term_modified": False,
            "new_teacher_or_observation": False, "continuous_contact_certified": False}
