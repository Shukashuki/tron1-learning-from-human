"""Opt-in, GPU-local all-substep wheel-link vertical-velocity costs.

No Isaac or Torch import occurs until explicit installation. Install after any
scene-update observer, then close before that observer restores its methods.
This samples refreshed link-origin velocities, not COM aliases or contact-point
separation. It does not change actions, observations, physics, or resets.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import struct
from numbers import Integral, Real
from types import MethodType

MODE = "all_substeps_mean"
WHEEL_BODY_NAMES = ("wheel_L_Link", "wheel_R_Link")
_ENV_MARKER = "_tron1_substep_wheel_velocity_accumulator"
_SCENE_MARKER = "_tron1_substep_wheel_velocity_owner"
_MISSING = object()


def _integer(value, label, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a finite positive scalar")
    return float(value)


def _same(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


def _native_clock(sim):
    index = _integer(getattr(sim, "current_time_step_index", None), "sim.current_time_step_index")
    time = getattr(sim, "current_time", None)
    if isinstance(time, bool) or not isinstance(time, Real) or not math.isfinite(time) or time < 0:
        raise ValueError("sim.current_time must be a finite nonnegative scalar")
    return index, float(time)


def _float32(value):
    # PhysX callback step_size can be float32 even when config dt is a Python
    # double. This is explicit rounding, not a permissive clock-fit tolerance.
    try:
        result = struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError as error:
        raise ValueError("physics_dt is not representable as positive float32") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError("physics_dt is not representable as positive float32")
    return result


def _clock(env):
    decimation = _integer(env.cfg.decimation, "cfg.decimation", minimum=1)
    physics_dt = _positive(env.physics_dt, "env.physics_dt")
    simulator_dt = _positive(env.sim.get_physics_dt(), "sim.get_physics_dt()")
    config_dt = _positive(env.cfg.sim.dt, "cfg.sim.dt")
    step_dt = _positive(env.step_dt, "env.step_dt")
    if not _same(physics_dt, simulator_dt) or not _same(physics_dt, config_dt):
        raise ValueError("Simulator, environment and configured physics timesteps differ")
    if not _same(step_dt, decimation * physics_dt):
        raise ValueError("env.step_dt must equal cfg.decimation * actual physics_dt")
    return decimation, physics_dt, step_dt


def _restore_attribute(owner, name, previous):
    if previous is _MISSING:
        delattr(owner, name)
    else:
        setattr(owner, name, previous)


class WheelVerticalVelocityAccumulator:
    """Current-control-window statistic; never a policy observation/history.

    Use :func:`install_wheel_vertical_velocity_accumulator`, not this constructor
    directly. Only ``audit`` returns host metadata; costs remain on their device.
    """

    def __init__(self, env, wheel_body_ids=None):
        if hasattr(env, _ENV_MARKER) or hasattr(env.scene, _SCENE_MARKER):
            raise RuntimeError("Substep accumulator is already installed or a conflicting owner exists")
        self._env = env
        self._scene = env.scene
        self._torch = importlib.import_module("torch")
        self._robot = env.scene["robot"]
        self._decimation, self._physics_dt, self._step_dt = _clock(env)
        self._native_dt_rounding = abs(_float32(self._physics_dt) - self._physics_dt)
        _native_clock(env.sim)  # required actual-step clock, not just an env counter
        self._num_envs = _integer(env.num_envs, "env.num_envs", minimum=1)
        if wheel_body_ids is None:
            ids, names = self._robot.find_bodies(list(WHEEL_BODY_NAMES), preserve_order=True)
            if tuple(names) != WHEEL_BODY_NAMES:
                raise ValueError("Expected exactly the named left/right wheel links in order")
        else:
            ids = wheel_body_ids
        try:
            self._wheel_ids = tuple(_integer(value, "wheel body ID") for value in ids)
        except TypeError as error:
            raise ValueError("wheel_body_ids must contain exactly two distinct integer IDs") from error
        if len(self._wheel_ids) != 2 or len(set(self._wheel_ids)) != 2:
            raise ValueError("wheel_body_ids must contain exactly two distinct integer IDs")
        velocity = self._robot.data.body_link_lin_vel_w
        self._device, self._dtype = velocity.device, velocity.dtype
        self._body_count = velocity.shape[1] if len(velocity.shape) == 3 else -1
        self._validate_velocity(velocity)
        self._sum_sq = self._torch.zeros(self._num_envs, device=self._device, dtype=self._dtype)
        self._original_step = env.step
        self._original_update = env.scene.update
        if not callable(self._original_step) or not callable(self._original_update):
            raise TypeError("Environment step and scene update must be callable")
        self._previous_step_attribute = vars(env).get("step", _MISSING)
        self._previous_update_attribute = vars(env.scene).get("update", _MISSING)
        self._inside_step = self._sampling_open = self._inside_update = False
        self._closed = False
        self._count = self._total_samples = 0
        self._steps_started = self._steps_completed = self._steps_failed = self._cost_reads = 0
        self._start_counter = self._last_counter = None
        self._start_native_index = self._last_native_index = None
        self._start_native_time = self._last_native_time = None
        self._last_step_completed = False
        fingerprint = {"schema_version": 1, "mode": MODE, "wheel_body_ids": self._wheel_ids,
                       "decimation": self._decimation, "physics_dt_s": self._physics_dt,
                       "policy_dt_s": self._step_dt, "num_envs": self._num_envs}
        self._fingerprint = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()

        def step_wrapper(owner, *args, **kwargs):
            if owner is not self._env:
                raise RuntimeError("Substep step wrapper called with another environment")
            return self._step(*args, **kwargs)

        def update_wrapper(owner, *args, **kwargs):
            if owner is not self._scene:
                raise RuntimeError("Substep update wrapper called with another scene")
            return self._update(*args, **kwargs)

        self._step_wrapper = MethodType(step_wrapper, env)
        self._update_wrapper = MethodType(update_wrapper, env.scene)
        try:
            env.step = self._step_wrapper
            env.scene.update = self._update_wrapper
            setattr(env, _ENV_MARKER, self)
            setattr(env.scene, _SCENE_MARKER, self)
        except BaseException:
            if getattr(env, "step", None) is self._step_wrapper:
                _restore_attribute(env, "step", self._previous_step_attribute)
            if getattr(env.scene, "update", None) is self._update_wrapper:
                _restore_attribute(env.scene, "update", self._previous_update_attribute)
            for owner, marker in ((env, _ENV_MARKER), (env.scene, _SCENE_MARKER)):
                if getattr(owner, marker, None) is self:
                    delattr(owner, marker)
            raise

    def _validate_velocity(self, velocity):
        if tuple(velocity.shape) != (self._num_envs, self._body_count, 3) or self._body_count < 2:
            raise ValueError("Wheel-link velocity must have shape [num_envs, num_bodies, 3]")
        if max(self._wheel_ids) >= self._body_count:
            raise ValueError("Wheel body ID exceeds available body links")
        if not self._torch.is_floating_point(velocity):
            raise ValueError("Wheel-link velocity must be floating point")
        if velocity.device != self._torch.device(self._env.device) or velocity.device != self._device:
            raise ValueError("Wheel-link velocity must remain on the environment device")
        if velocity.dtype != self._dtype:
            raise ValueError("Wheel-link velocity dtype changed")

    def _verify_attachment(self):
        if self._closed:
            raise RuntimeError("Substep accumulator is closed")
        if (getattr(self._env, _ENV_MARKER, None) is not self or
            getattr(self._scene, _SCENE_MARKER, None) is not self or
            self._env.step is not self._step_wrapper or self._scene.update is not self._update_wrapper):
            raise RuntimeError("Substep wrapper ownership conflict; install last and remove first")

    def _counter(self):
        return _integer(self._env._sim_step_counter, "env._sim_step_counter")

    def _native_time_matches(self, current, previous, count):
        # Only nominal-vs-float32 callback-dt rounding and double accumulation/
        # subtraction noise are tolerated. Actual native step count stays exact.
        tolerance = count * self._native_dt_rounding + 8. * math.ulp(max(abs(current), abs(previous), self._physics_dt))
        return abs((current - previous) - count * self._physics_dt) <= tolerance

    def _step(self, *args, **kwargs):
        self._verify_attachment()
        if self._inside_step:
            raise RuntimeError("Reentrant environment step is not supported")
        clock = _clock(self._env)
        if clock[0] != self._decimation or any(not _same(a, b) for a, b in zip(clock[1:], (self._physics_dt, self._step_dt))):
            raise ValueError("Physics clock changed after accumulator installation")
        if _integer(self._env.num_envs, "env.num_envs", minimum=1) != self._num_envs:
            raise ValueError("Environment count changed after accumulator installation")
        self._start_counter = self._last_counter = self._counter()
        self._start_native_index, self._start_native_time = _native_clock(self._env.sim)
        self._last_native_index, self._last_native_time = self._start_native_index, self._start_native_time
        self._sum_sq.zero_()
        self._count = 0
        self._last_step_completed = False
        self._steps_started += 1
        self._inside_step = self._sampling_open = True
        try:
            result = self._original_step(*args, **kwargs)
            self._verify_attachment()
            self._require_complete()
            self._steps_completed += 1
            self._last_step_completed = True
            return result
        except BaseException:
            self._steps_failed += 1
            raise
        finally:
            self._inside_step = self._sampling_open = False

    def _update(self, *args, **kwargs):
        self._verify_attachment()
        if self._inside_update:
            raise RuntimeError("Reentrant scene update is not supported")
        self._inside_update = True
        try:
            # The original method advances lazy articulation timestamps; all
            # prior observers are chained exactly once, with identical arguments.
            result = self._original_update(*args, **kwargs)
            if not self._inside_step:
                return result
            counter = self._counter()
            native_index, native_time = _native_clock(self._env.sim)
            if counter == self._last_counter:
                if native_index != self._last_native_index or native_time != self._last_native_time:
                    raise RuntimeError("Native physics advanced without a new environment counter")
                return result  # reset/forward refresh: no physics increment
            if not self._sampling_open:
                raise RuntimeError("Unexpected physical substep after the completed reward window")
            if counter != self._last_counter + 1:
                raise RuntimeError("Nonconsecutive simulator counter in substep sampling")
            if native_index != self._last_native_index + 1 or not self._native_time_matches(native_time, self._last_native_time, 1):
                raise RuntimeError("Native physics must advance exactly one step and physics_dt per sample")
            if args and "dt" in kwargs:
                raise ValueError("scene.update received dt both positionally and by keyword")
            dt = args[0] if args else kwargs.get("dt", _MISSING)
            if dt is _MISSING or not _same(_positive(dt, "scene.update dt"), self._physics_dt):
                raise ValueError("Each sampled scene.update must use the actual physics_dt")
            velocity = self._robot.data.body_link_lin_vel_w
            self._validate_velocity(velocity)
            self._sum_sq.add_(velocity[:, list(self._wheel_ids), 2].square().sum(dim=-1))
            self._last_counter = counter
            self._last_native_index, self._last_native_time = native_index, native_time
            self._count += 1
            self._total_samples += 1
            if self._count > self._decimation:
                raise RuntimeError("Too many physical samples in one control step")
            if self._count == self._decimation:
                self._sampling_open = False
            return result
        finally:
            self._inside_update = False

    def _require_complete(self):
        native_index, native_time = _native_clock(self._env.sim)
        if (not self._inside_step or self._count != self._decimation or self._sampling_open or
            self._last_counter != self._start_counter + self._decimation or
            self._counter() != self._last_counter or
            native_index != self._start_native_index + self._decimation or
            native_index != self._last_native_index or native_time != self._last_native_time or
            not self._native_time_matches(native_time, self._start_native_time, self._decimation)):
            raise RuntimeError("Reward requires exactly all completed physical substeps, before reset")

    def cost(self, std_m_s):
        """Return mean-of-squared link vz, summed over wheels, divided by std².

        Positive dimensionless cost; the reward configuration supplies a negative
        weight. RewardManager's step_dt factor supplies physical time integration.
        """
        self._verify_attachment()
        std = _positive(std_m_s, "std_m_s")
        if not .005 <= std <= .1:
            raise ValueError("std_m_s must lie in the existing drift range [.005, .1]")
        self._require_complete()
        self._cost_reads += 1
        return self._sum_sq / (self._decimation * std * std)

    def audit(self):
        """Only scalar configuration/counter metadata; never copies GPU values."""
        return {"schema_version": 1, "mode": MODE, "fingerprint_sha256": self._fingerprint,
                "installed": not self._closed, "inside_step": self._inside_step,
                "sampling_open": self._sampling_open, "num_envs": self._num_envs,
                "wheel_body_ids": list(self._wheel_ids), "device": str(self._device), "dtype": str(self._dtype),
                "decimation": self._decimation, "physics_dt_s": self._physics_dt, "policy_dt_s": self._step_dt,
                "steps_started": self._steps_started, "steps_completed": self._steps_completed,
                "steps_failed": self._steps_failed, "samples_last_control_step": self._count,
                "samples_total": self._total_samples, "last_start_counter": self._start_counter,
                "last_sample_counter": self._last_counter, "last_step_completed": self._last_step_completed,
                "native_clock_required": True, "native_start_step_index": self._start_native_index,
                "native_last_step_index": self._last_native_index, "native_start_time_s": self._start_native_time,
                "native_last_time_s": self._last_native_time,
                "native_time_float32_dt_rounding_bound_s": self._native_dt_rounding,
                "native_time_double_rounding_allowance_ulps": 8,
                "cost_reads": self._cost_reads}

    def close(self):
        """Restore prior bound methods in LIFO order, without stepping/resetting."""
        if self._closed:
            return self.audit()
        self._verify_attachment()
        if self._inside_step or self._inside_update:
            raise RuntimeError("Cannot remove a substep accumulator during a step/update")
        _restore_attribute(self._env, "step", self._previous_step_attribute)
        _restore_attribute(self._scene, "update", self._previous_update_attribute)
        delattr(self._env, _ENV_MARKER)
        delattr(self._scene, _SCENE_MARKER)
        self._closed = True
        return self.audit()


def install_wheel_vertical_velocity_accumulator(env, *, wheel_body_ids=None):
    """Explicit opt-in; install after an existing physical-rate recorder wrapper."""
    return WheelVerticalVelocityAccumulator(env, wheel_body_ids=wheel_body_ids)


def substep_wheel_vertical_velocity_penalty(env, std_m_s):
    """Reward-manager entrypoint; no installation or last-substep fallback."""
    handler = getattr(env, _ENV_MARKER, None)
    if not isinstance(handler, WheelVerticalVelocityAccumulator):
        raise RuntimeError("All-substep reward requested without its explicit accumulator installation")
    return handler.cost(std_m_s)
