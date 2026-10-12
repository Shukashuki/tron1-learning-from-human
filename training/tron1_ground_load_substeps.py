"""Explicit mean-of-four *costs* for the unchanged wheel-ground load reserve.

This is a new opt-in consumer of the frozen reserve cache and the frozen
single-owner substep lifecycle.  Its inherited wheel-vz collection is unused:
it never consumes the inherited velocity cost or changes the velocity reward.
Reading filtered force tensors adds work (including existing host finite-check
synchronizations), not physics, actor, reward-manager, reset or RNG calls.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from training import tron1_ground_load_reserve as reserve
from training.tron1_substep_rewards import WheelVerticalVelocityAccumulator

MODE = "all_substeps_mean_cost"
ACCUMULATOR_ATTRIBUTE = "_tron1_ground_load_substep_accumulator"
_SCENE_MARKER = "_tron1_ground_load_substep_owner"
SETTINGS = {"weight": -1.0, "scale_fraction": .20}
RESERVE_DEPENDENCY_SHA256 = "761eccede3c0d90d744a4d9c45c9671b318426ab9bc6f9762ee7cacb3c14496d"
LIFECYCLE_DEPENDENCY_SHA256 = "aeec31437226984a326a263109aaadebca004a6ef333f25a1d3377688aa276a5"


def _source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class GroundLoadSubstepAccumulator(WheelVerticalVelocityAccumulator):
    """One current-window owner; exactly one force-cost read per control step."""

    def __init__(self, env):
        if hasattr(env, ACCUMULATOR_ATTRIBUTE) or hasattr(env.scene, _SCENE_MARKER):
            raise RuntimeError("Ground-load accumulator already exists or has a conflicting owner")
        if (_source_hash(reserve.__file__) != RESERVE_DEPENDENCY_SHA256 or
                _source_hash(Path(__file__).with_name("tron1_substep_rewards.py")) != LIFECYCLE_DEPENDENCY_SHA256):
            raise RuntimeError("Ground-load accumulator requires its exact frozen dependencies")
        self._force_initializing = True
        self._force_cache = None
        self._force_count = self._force_samples_total = self._force_cost_reads = 0
        self._force_cost_read_in_window = False
        self._force_sum = None
        super().__init__(env)
        try:
            if self._decimation != 4 or self._physics_dt != .005 or self._step_dt != .02:
                raise ValueError("Ground-load sampling requires exactly four 5ms substeps at 50Hz")
            if self._dtype != self._torch.float32:
                raise ValueError("Ground-load sampling requires float32 tensors")
            device = self._torch.device(self._device)
            if device.type != "cuda" or device.index is None:
                raise ValueError("Ground-load sampling requires an explicit CUDA ordinal")
            self._force_sum = self._torch.zeros(self._num_envs, device=self._device, dtype=self._dtype)
            setattr(env, ACCUMULATOR_ATTRIBUTE, self)
            setattr(env.scene, _SCENE_MARKER, self)
        except BaseException:
            for owner, marker in ((env, ACCUMULATOR_ATTRIBUTE), (env.scene, _SCENE_MARKER)):
                if getattr(owner, marker, None) is self:
                    delattr(owner, marker)
            super().close()
            raise
        self._force_initializing = False

    def _verify_attachment(self):
        super()._verify_attachment()
        if not self._force_initializing and (
                getattr(self._env, ACCUMULATOR_ATTRIBUTE, None) is not self or
                getattr(self._scene, _SCENE_MARKER, None) is not self):
            raise RuntimeError("Ground-load wrapper ownership conflict; install last and remove first")

    def _bind_cache(self):
        cache = getattr(self._env, reserve.CACHE_ATTRIBUTE, None)
        if type(cache) is not reserve._ReserveCache:
            raise RuntimeError("Ground-load sampling requires the exact explicitly initialized reserve cache")
        if self._force_cache is not None and cache is not self._force_cache:
            raise RuntimeError("Ground-load reserve cache changed after binding")
        if reserve.validate_load_reserve_settings(cache.settings) != SETTINGS:
            raise RuntimeError("Ground-load sampling preserves only reserve weight -1 and scale .20")
        if (type(cache.num_envs) is not int or cache.num_envs != self._num_envs or
                cache.device != self._torch.device(self._env.device) or
                cache.robot is not self._env.scene["robot"] or
                len(cache.sensors) != 2 or
                any(self._env.scene.sensors.get(name) is not sensor
                    for name, sensor in zip(reserve.SENSOR_NAMES, cache.sensors))):
            raise RuntimeError("Ground-load cache does not belong to this live environment")
        audit = cache.audit
        if (not isinstance(audit, dict) or type(audit.get("schema_version")) is not int or
                audit.get("schema_version") != 1 or
                audit.get("settings") != SETTINGS or audit.get("sampling") != "last_physics_substep" or
                audit.get("module_sha256") != RESERVE_DEPENDENCY_SHA256 or
                type(audit.get("num_envs")) is not int or audit.get("num_envs") != self._num_envs or
                audit.get("cuda_device") != str(self._device) or
                audit.get("raw_normal_forces_thresholded") is not False or
                audit.get("physics_dt_s") != .005 or audit.get("policy_dt_s") != .02 or
                audit.get("decimation") != 4):
            raise RuntimeError("Ground-load cache lacks its unchanged startup validation audit")
        self._force_cache = cache

    def _step(self, *args, **kwargs):
        self._verify_attachment()
        if self._inside_step:
            raise RuntimeError("Reentrant ground-load step is not supported")
        # Startup initialization can occur after wrapper installation.  Missing
        # or replaced caches still fail BEFORE any physical step is performed.
        self._bind_cache()
        self._force_sum.zero_()
        self._force_count = 0
        self._force_cost_read_in_window = False
        return super()._step(*args, **kwargs)

    def _update(self, *args, **kwargs):
        previous_count = self._count
        result = super()._update(*args, **kwargs)
        # Parent already gates actual native + environment counter increments.
        # Same-counter refreshes during reset/forward MUST NOT query forces.
        if self._inside_step and self._count == previous_count + 1:
            if self._force_count != previous_count:
                raise RuntimeError("Ground-load force samples lost alignment with native substeps")
            self._bind_cache()
            # This frozen function is an instantaneous tensor expression, not
            # RewardManager.compute.  All original cache/sensor/event/finite
            # guards apply at the actual refreshed 5ms state, before averaging.
            sample = reserve.wheel_ground_load_reserve_penalty(self._env, .20)
            if (tuple(sample.shape) != (self._num_envs,) or
                    sample.dtype != self._dtype or sample.device != self._force_sum.device or
                    not bool(self._torch.isfinite(sample).all()) or
                    bool((sample < 0.).any()) or bool((sample > 2.).any())):
                raise RuntimeError("Malformed or nonfinite instantaneous ground-load cost")
            self._force_sum.add_(sample.clone())
            self._force_count += 1
            self._force_samples_total += 1
        return result

    def _require_complete(self):
        super()._require_complete()
        if self._force_count != self._decimation or not self._force_cost_read_in_window:
            raise RuntimeError("Ground-load window requires four samples and exactly one reward read")

    def force_cost(self, scale_fraction):
        """Positive mean(exp(-Fn/(r*mg))) cost; never exp of a mean force."""
        self._verify_attachment()
        scale = reserve._number(scale_fraction, "scale_fraction")
        if scale != .20:
            raise RuntimeError("Ground-load cost scale must remain .20")
        self._bind_cache()
        super()._require_complete()
        if self._force_count != 4 or self._force_cost_read_in_window:
            raise RuntimeError("Ground-load cost requires four samples and exactly one read per window")
        self._force_cost_reads += 1
        self._force_cost_read_in_window = True
        # Division is out-of-place: later zero_() cannot change returned costs.
        return self._force_sum / self._decimation

    def cost(self, std_m_s):
        raise RuntimeError("Inherited wheel-velocity cost is deliberately unused; keep the last-substep vz reward")

    def audit(self):
        lifecycle = super().audit()
        fingerprint = {
            "schema_version": 1, "mode": MODE, "lifecycle_fingerprint_sha256": self._fingerprint,
            "settings": SETTINGS, "sensor_names": list(reserve.SENSOR_NAMES),
            "ground_collider": reserve.GROUND_COLLIDER,
            "reserve_dependency_sha256": RESERVE_DEPENDENCY_SHA256,
            "lifecycle_dependency_sha256": LIFECYCLE_DEPENDENCY_SHA256,
        }
        return {
            "schema_version": 1, "mode": MODE,
            "fingerprint_sha256": hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest(),
            "settings": copy.deepcopy(SETTINGS),
            "formula": "mean_over_4_physics_substeps(sum_wheels(exp(-clamp(ground_normal_force_world_z_N,min=0)/(.20*mg_N))))",
            "lifecycle": lifecycle, "cache_bound": self._force_cache is not None,
            "force_samples_last_control_step": self._force_count,
            "force_samples_total": self._force_samples_total, "force_cost_reads": self._force_cost_reads,
            "force_cost_read_in_last_window": self._force_cost_read_in_window,
            "inherited_velocity_collection_unused": True, "inherited_velocity_cost_reads": self._cost_reads,
            "wheel_vertical_velocity_reward_changed": False,
            "extra_physics_actor_reward_manager_reset_rng_calls": 0,
            "host_synchronization_free": False, "continuous_contact_certified": False,
            "reserve_dependency_sha256": RESERVE_DEPENDENCY_SHA256,
            "lifecycle_dependency_sha256": LIFECYCLE_DEPENDENCY_SHA256,
            "module_sha256": _source_hash(__file__),
        }

    def close(self):
        if self._closed:
            return self.audit()
        super().close()
        for owner, marker in ((self._env, ACCUMULATOR_ATTRIBUTE), (self._scene, _SCENE_MARKER)):
            if getattr(owner, marker, None) is self:
                delattr(owner, marker)
        return self.audit()


def install_ground_load_accumulator(env):
    """Install last, after material/cache audit and any physical recorder."""
    return GroundLoadSubstepAccumulator(env)


def substep_wheel_ground_load_reserve_penalty(env, scale_fraction):
    """Reward entrypoint: one existing reward-manager call, no fallback/install."""
    handler = getattr(env, ACCUMULATOR_ATTRIBUTE, None)
    if type(handler) is not GroundLoadSubstepAccumulator or handler._env is not env:
        raise RuntimeError("All-substep ground-load reward requires its explicit accumulator")
    return handler.force_cost(scale_fraction)
