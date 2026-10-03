"""Portable, hidden motor-domain parameters and a lazy Isaac adapter.

Importing this module requires only NumPy. Domain parameters are constant during
an episode. Numerical body/joint guards and legacy PhysX friction stay fixed.
The explicit axle loss is applied AFTER the DC electrical envelope, then the
net effort is guarded at the original 80/12 Nm ceiling in both simulators.
This final guard is a simulation convention, not a pure electrical motor model.
"""
from __future__ import annotations

import copy
import json

import numpy as np

JOINT_NAMES = ["abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint",
               "knee_L_Joint", "knee_R_Joint", "wheel_L_Joint", "wheel_R_Joint"]
WHEEL_NAMES = ["wheel_L_Joint", "wheel_R_Joint"]


def default_profile(seed=42):
    return {"schema_version": 1, "mode": "random", "seed": seed,
            "torque_scale_range": [0.85, 1.0], "velocity_scale_range": [0.85, 1.0],
            "wheel_friction_nm_range": [0.0, 0.3], "nominal_probability": 0.2,
            "friction_smoothing_rad_s": 0.5}


def _finite_vector(value, size, label, minimum, maximum):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite {size}-vector")
    if np.any(result < minimum) or np.any(result > maximum):
        raise ValueError(f"{label} must be within [{minimum}, {maximum}]")
    return result


def validate_draw(draw):
    """Normalize one named draw to independent JSON values in actor order."""
    if not isinstance(draw, dict):
        raise ValueError("A domain draw must be a dictionary")
    if "joint_names" not in draw:
        raise ValueError("Domain draw requires explicit joint_names for safe mapping")
    names = list(draw["joint_names"])
    if len(names) != 8 or len(set(names)) != 8 or set(names) != set(JOINT_NAMES):
        raise ValueError("Domain draw needs exactly the eight named TRON1 joints")
    order = [names.index(name) for name in JOINT_NAMES]
    torque = _finite_vector(draw.get("torque_scale"), 8, "torque_scale", 0.01, 1.0)[order]
    velocity = _finite_vector(draw.get("velocity_scale"), 8, "velocity_scale", 0.01, 1.0)[order]
    friction = _finite_vector(draw.get("wheel_friction_nm"), 2, "wheel_friction_nm", 0.0, 12.0)
    smoothing = draw.get("friction_smoothing_rad_s", 0.5)
    if isinstance(smoothing, bool) or not isinstance(smoothing, (int, float)) or not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("friction_smoothing_rad_s must be finite and positive")
    return {"joint_names": list(JOINT_NAMES), "torque_scale": torque.tolist(),
            "velocity_scale": velocity.tolist(), "wheel_friction_nm": friction.tolist(),
            "friction_smoothing_rad_s": float(smoothing),
            "is_nominal": bool(np.all(torque == 1) and np.all(velocity == 1) and np.all(friction == 0))}


def validate_profile(profile):
    if not isinstance(profile, dict):
        raise ValueError("Domain randomization profile must be a dictionary")
    normalized = default_profile()
    normalized.update(copy.deepcopy(profile))
    if normalized.get("schema_version") != 1 or normalized.get("mode") not in ("random", "fixed"):
        raise ValueError("Expected domain schema_version=1 and mode random or fixed")
    seed = normalized["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Domain seed must be an integer in [0, 2**32)")
    for key, lower, upper in (("torque_scale_range", 0.01, 1.),
                              ("velocity_scale_range", 0.01, 1.),
                              ("wheel_friction_nm_range", 0., 12.)):
        bounds = _finite_vector(normalized[key], 2, key, lower, upper)
        if bounds[0] > bounds[1]:
            raise ValueError(f"{key} lower bound exceeds upper bound")
        normalized[key] = bounds.tolist()
    probability = normalized["nominal_probability"]
    if not isinstance(probability, (int, float)) or not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("nominal_probability must lie in [0, 1]")
    smoothing = normalized["friction_smoothing_rad_s"]
    if not isinstance(smoothing, (int, float)) or not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("friction_smoothing_rad_s must be finite and positive")
    if normalized["mode"] == "fixed":
        if "fixed_draw" not in normalized:
            raise ValueError("Fixed mode requires fixed_draw")
        supplied = dict(normalized["fixed_draw"])
        supplied.setdefault("friction_smoothing_rad_s", smoothing)
        normalized["fixed_draw"] = validate_draw(supplied)
        normalized["friction_smoothing_rad_s"] = normalized["fixed_draw"]["friction_smoothing_rad_s"]
    elif "fixed_draw" in normalized:
        raise ValueError("Random profile cannot contain a fixed_draw")
    # Reject non-JSON/nonfinite metadata as well as invalid numeric parameters.
    json.dumps(normalized, allow_nan=False)
    return normalized


def nominal_draws(num_envs):
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 0:
        raise ValueError("num_envs must be a nonnegative integer")
    return {"joint_names": list(JOINT_NAMES), "torque_scale": np.ones((num_envs, 8)),
            "velocity_scale": np.ones((num_envs, 8)), "wheel_friction_nm": np.zeros((num_envs, 2)),
            "friction_smoothing_rad_s": 0.5, "nominal_mask": np.ones(num_envs, dtype=bool)}


def _integer_vector(values, label):
    array = np.asarray(values)
    if array.ndim == 1 and array.size == 0:
        return np.empty(0, dtype=np.int64)
    if array.ndim != 1 or array.dtype.kind not in "iu" or np.any(array < 0):
        raise ValueError(f"{label} must be a nonnegative integer vector")
    return array.astype(np.int64)


def sample_draws(profile, env_ids, reset_counts):
    """Stateless per-env sampling; subset order and reset batching cannot leak.

    Each generator is keyed by (profile seed, environment ID, reset count).
    reset_count=0 denotes the first episode. The nominal mixture is per whole
    environment, while non-nominal parameter draws are independent per joint.
    """
    profile = validate_profile(profile)
    ids = _integer_vector(env_ids, "env_ids")
    counts = _integer_vector(reset_counts, "reset_counts")
    if ids.shape != counts.shape or len(set(ids.tolist())) != len(ids):
        raise ValueError("Unique env_ids and reset_counts must have matching shapes")
    result = nominal_draws(len(ids))
    result.update(env_ids=ids.copy(), reset_counts=counts.copy(),
                  friction_smoothing_rad_s=profile["friction_smoothing_rad_s"])
    for row, (env_id, count) in enumerate(zip(ids, counts)):
        if profile["mode"] == "fixed":
            draw = profile["fixed_draw"]
            for key in ("torque_scale", "velocity_scale", "wheel_friction_nm"):
                result[key][row] = draw[key]
            result["nominal_mask"][row] = draw["is_nominal"]
            continue
        rng = np.random.default_rng(np.random.SeedSequence([profile["seed"], int(env_id), int(count)]))
        if rng.random() < profile["nominal_probability"]:
            continue
        result["nominal_mask"][row] = False
        for key, range_key, size in (("torque_scale", "torque_scale_range", 8),
                                     ("velocity_scale", "velocity_scale_range", 8),
                                     ("wheel_friction_nm", "wheel_friction_nm_range", 2)):
            result[key][row] = rng.uniform(*profile[range_key], size=size)
    return result


def single_draw(batch, index):
    return validate_draw({"joint_names": batch["joint_names"],
                          **{key: np.asarray(batch[key])[index].tolist()
                             for key in ("torque_scale", "velocity_scale", "wheel_friction_nm")},
                          "friction_smoothing_rad_s": batch["friction_smoothing_rad_s"]})


def apply_draw_to_contract(contract, draw):
    """Preserve nominal scalar fields; deployment explicitly applies this draw."""
    if contract.get("actuator_model") != "dc_motor":
        raise ValueError("Motor-domain parameters require a DC motor contract")
    result = copy.deepcopy(contract)
    result["domain_parameters"] = validate_draw(draw)
    return result


def expanded_motor_parameters(draw, effort=(80., 12.), stall=(80., 12.), speed=(15., 100.)):
    """Pure reference expansion, including the DC overspeed cache invariant."""
    draw = validate_draw(draw)
    expand = lambda values: np.r_[np.full(6, values[0]), np.full(2, values[1])]
    effort_values = expand(effort) * draw["torque_scale"]
    stall_values = expand(stall) * draw["torque_scale"]
    speed_values = expand(speed) * draw["velocity_scale"]
    return {"effort_limit_nm": effort_values, "saturation_effort_nm": stall_values,
            "velocity_limit_rad_s": speed_values,
            "velocity_at_effort_limit_rad_s": speed_values * (1 + effort_values / stall_values),
            "net_effort_guard_nm": expand(effort)}


def make_actuator_class():
    """Construct the Isaac-only subclass after AppLauncher has started."""
    import torch
    from isaaclab.actuators import DCMotor

    class DomainRandomizedDCMotor(DCMotor):
        def __init__(self, cfg, *args, **kwargs):
            super().__init__(cfg, *args, **kwargs)
            self._saturation_effort = torch.full_like(self.effort_limit, float(cfg.saturation_effort))
            self.nominal_effort_limit = self.effort_limit.clone()
            self.nominal_saturation_effort = self._saturation_effort.clone()
            self.nominal_velocity_limit = self.velocity_limit.clone()
            self.net_effort_guard = self.effort_limit_sim.clone()
            self.axle_friction_nm = torch.zeros_like(self.effort_limit)
            self.friction_smoothing_rad_s = 0.5
            self.motor_effort = torch.zeros_like(self.effort_limit)
            self.axle_friction_effort = torch.zeros_like(self.effort_limit)

        def compute(self, control_action, joint_pos, joint_vel):
            action = super().compute(control_action, joint_pos, joint_vel)
            self.motor_effort[:] = self.applied_effort
            # joint_vel is the actual simulator velocity. DCMotor._joint_vel
            # was clipped only for its envelope and must NOT drive this loss.
            self.axle_friction_effort[:] = self.axle_friction_nm * torch.tanh(
                joint_vel / self.friction_smoothing_rad_s)
            self.applied_effort = torch.clamp(
                self.motor_effort - self.axle_friction_effort,
                min=-self.net_effort_guard, max=self.net_effort_guard)
            action.joint_efforts = self.applied_effort
            return action

    return DomainRandomizedDCMotor


def reset_domain_randomization(env, env_ids, profile):
    """Isaac reset event: update only selected rows before command RSI reset."""
    import torch
    profile = validate_profile(profile)
    robot = env.scene["robot"]
    if env_ids is None or isinstance(env_ids, slice):
        selected = np.arange(env.num_envs, dtype=np.int64)[slice(None) if env_ids is None else env_ids]
    elif hasattr(env_ids, "detach"):
        selected = _integer_vector(env_ids.detach().cpu().numpy(), "env_ids")
    else:
        selected = _integer_vector(env_ids, "env_ids")
    if len(set(selected.tolist())) != len(selected) or np.any(selected < 0) or np.any(selected >= env.num_envs):
        raise ValueError("Reset environment indices must be unique and in range")
    if len(selected) == 0:
        return
    state = getattr(env, "tron1_domain_randomization", None)
    if state is None:
        state = {"profile": profile, "reset_counts": np.full(env.num_envs, -1, dtype=np.int64),
                 "draw_arrays": nominal_draws(env.num_envs)}
        state["draw_arrays"]["friction_smoothing_rad_s"] = profile["friction_smoothing_rad_s"]
        env.tron1_domain_randomization = state
    elif state["profile"] != profile:
        raise ValueError("Changing domain profile during an environment lifetime is unsupported")
    counts = state["reset_counts"][selected] + 1
    sampled = sample_draws(profile, selected, counts)
    ids = torch.as_tensor(selected, dtype=torch.long, device=robot.device)
    covered = []
    for actuator in robot.actuators.values():
        names = list(actuator.joint_names)
        covered.extend(names)
        columns = [JOINT_NAMES.index(name) for name in names]
        if not hasattr(actuator, "nominal_saturation_effort"):
            raise TypeError("Domain reset requires DomainRandomizedDCMotor actuators")
        to_tensor = lambda values: torch.as_tensor(values, dtype=actuator.effort_limit.dtype, device=robot.device)
        torque_scale = to_tensor(sampled["torque_scale"][:, columns])
        speed_scale = to_tensor(sampled["velocity_scale"][:, columns])
        actuator.effort_limit[ids] = actuator.nominal_effort_limit[ids] * torque_scale
        actuator._saturation_effort[ids] = actuator.nominal_saturation_effort[ids] * torque_scale
        actuator.velocity_limit[ids] = actuator.nominal_velocity_limit[ids] * speed_scale
        actuator._vel_at_effort_lim[ids] = actuator.velocity_limit[ids] * (
            1.0 + actuator.effort_limit[ids] / actuator._saturation_effort[ids])
        friction = np.zeros((len(selected), len(names)))
        for local, name in enumerate(names):
            if name in WHEEL_NAMES:
                friction[:, local] = sampled["wheel_friction_nm"][:, WHEEL_NAMES.index(name)]
        actuator.axle_friction_nm[ids] = to_tensor(friction)
        actuator.friction_smoothing_rad_s = profile["friction_smoothing_rad_s"]
    if len(covered) != 8 or set(covered) != set(JOINT_NAMES):
        raise ValueError("Actuator groups must cover the eight named joints exactly once")
    state["reset_counts"][selected] = counts
    for key in ("torque_scale", "velocity_scale", "wheel_friction_nm", "nominal_mask"):
        state["draw_arrays"][key][selected] = sampled[key]


def domain_randomization_audit(env):
    """Read back actual actuator tensors; useful after reset, before rollout."""
    state = getattr(env, "tron1_domain_randomization", None)
    if state is None:
        return {"enabled": False, "mode": "disabled"}
    robot = env.scene["robot"]
    actual = {key: np.zeros((env.num_envs, 8)) for key in (
        "effort_limit_nm", "saturation_effort_nm", "velocity_limit_rad_s",
        "velocity_at_effort_limit_rad_s", "axle_friction_nm", "net_effort_guard_nm",
        "nominal_effort_limit_nm", "nominal_saturation_effort_nm", "nominal_velocity_limit_rad_s")}
    attrs = ("effort_limit", "_saturation_effort", "velocity_limit", "_vel_at_effort_lim",
             "axle_friction_nm", "net_effort_guard", "nominal_effort_limit",
             "nominal_saturation_effort", "nominal_velocity_limit")
    for actuator in robot.actuators.values():
        columns = [JOINT_NAMES.index(name) for name in actuator.joint_names]
        for key, attribute in zip(actual, attrs):
            actual[key][:, columns] = getattr(actuator, attribute).detach().cpu().numpy()
    expected_cache = actual["velocity_limit_rad_s"] * (1 + actual["effort_limit_nm"] / actual["saturation_effort_nm"])
    if not all(np.isfinite(value).all() for value in actual.values()) or not np.allclose(
            actual["velocity_at_effort_limit_rad_s"], expected_cache, rtol=1e-6, atol=1e-6):
        raise RuntimeError("Actuator domain parameters are nonfinite or DC overspeed cache is stale")
    draws = state["draw_arrays"]
    expected = {
        "effort_limit_nm": actual["nominal_effort_limit_nm"] * draws["torque_scale"],
        "saturation_effort_nm": actual["nominal_saturation_effort_nm"] * draws["torque_scale"],
        "velocity_limit_rad_s": actual["nominal_velocity_limit_rad_s"] * draws["velocity_scale"],
        "axle_friction_nm": np.c_[np.zeros((env.num_envs, 6)), draws["wheel_friction_nm"]],
        "net_effort_guard_nm": actual["nominal_effort_limit_nm"],
    }
    for key, values in expected.items():
        if not np.allclose(actual[key], values, rtol=1e-6, atol=1e-6):
            raise RuntimeError(f"Actual actuator {key} does not match the sampled domain parameters")
    return {"enabled": True, "mode": state["profile"]["mode"], "profile": copy.deepcopy(state["profile"]),
            "profile_source": state["profile"].get("source", "inline_profile"),
            "seed": state["profile"]["seed"], "joint_names": list(JOINT_NAMES),
            "reset_counts": state["reset_counts"].tolist(),
            "nominal_environment_count": int(np.count_nonzero(state["draw_arrays"]["nominal_mask"])),
            "first16_domain_parameters": [single_draw(state["draw_arrays"], row) for row in range(min(16, env.num_envs))],
            "actual_parameter_min_max": {key: [float(value.min()), float(value.max())] for key, value in actual.items()},
            "actual_first16": {key: value[:16].tolist() for key, value in actual.items()},
            "dc_overspeed_cache_verified": True, "all_sampled_parameters_readback_verified": True,
            "parameters_hidden_from_actor": True,
            "net_effort_guard_semantics": "axle_loss_after_dc_motor_then_fixed_nominal_net_torque_clip"}
