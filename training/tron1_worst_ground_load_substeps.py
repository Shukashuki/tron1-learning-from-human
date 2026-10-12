"""Prospective worst-substep ground-load cost, not verified policy behavior.

Replace only mean_k(sum_w exp(-Fn_kw/(.20 mg))) with max_k of the
same instantaneous two-wheel sum. This is not a sum of per-wheel maxima.
The frozen lifecycle, cache, four force reads and last LINK-vz term stay held.
An actual runtime fixture is required before training or adoption.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import tron1_ground_load_substeps as mean
from training import tron1_ground_load_reserve as reserve
from training.tron1_substep_rewards import WheelVerticalVelocityAccumulator

MODE = "all_substeps_max_cost"
BASE_SAMPLER_SHA256 = "4df069ea80e37cf7575346201687be934927434db45ea143f5ce797d4979ecd2"
ACCUMULATOR_ATTRIBUTE = mean.ACCUMULATOR_ATTRIBUTE
SETTINGS = dict(mean.SETTINGS)


class WorstGroundLoadSubstepAccumulator(mean.GroundLoadSubstepAccumulator):
    """Same owner and current-window guards; buffer holds a maximum, not a sum."""

    def __init__(self, env):
        if mean._source_hash(mean.__file__) != BASE_SAMPLER_SHA256:
            raise RuntimeError("Worst-load sampler requires its frozen mean-cost baseline")
        super().__init__(env)

    def _update(self, *args, **kwargs):
        previous_count = self._count
        # Exactly the frozen native lifecycle refresh, without the mean
        # collector's force read: one new force read per actual substep.
        result = WheelVerticalVelocityAccumulator._update(self, *args, **kwargs)
        if self._inside_step and self._count == previous_count + 1:
            if self._force_count != previous_count:
                raise RuntimeError("Worst-load samples lost alignment with native substeps")
            self._bind_cache()
            sample = reserve.wheel_ground_load_reserve_penalty(self._env, .20)
            if (tuple(sample.shape) != (self._num_envs,) or
                    sample.dtype != self._dtype or sample.device != self._force_sum.device or
                    not bool(self._torch.isfinite(sample).all()) or
                    bool((sample < 0.).any()) or bool((sample > 2.).any())):
                raise RuntimeError("Malformed or nonfinite instantaneous worst-load cost")
            self._torch.maximum(self._force_sum, sample, out=self._force_sum)
            self._force_count += 1
            self._force_samples_total += 1
        return result

    def force_cost(self, scale_fraction):
        # Parent keeps its complete-window, exact-scale and single-read guards.
        # Discard its mean-shaped return and clone the max buffer directly.
        # Avoid /4 then *4, which could round tiny subnormal float32 values.
        super().force_cost(scale_fraction)
        return self._force_sum.clone()

    def audit(self):
        result = super().audit()
        base_fingerprint = result["fingerprint_sha256"]
        source = mean._source_hash(__file__)
        fingerprint = {"schema_version": 1, "mode": MODE,
            "base_fingerprint_sha256": base_fingerprint,
            "baseline_sampler_sha256": BASE_SAMPLER_SHA256, "module_sha256": source}
        result.update(mode=MODE,
            formula="max_over_4_physics_substeps(sum_wheels(exp(-clamp(ground_normal_force_world_z_N,min=0)/(.20*mg_N))))",
            fingerprint_sha256=hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest(),
            baseline_sampler_sha256=BASE_SAMPLER_SHA256, module_sha256=source,
            quadrature_changed_only=True, full_runtime_fixture_verified=False)
        return result


def install_worst_ground_load_accumulator(env):
    return WorstGroundLoadSubstepAccumulator(env)


def substep_worst_wheel_ground_load_reserve_penalty(env, scale_fraction):
    handler = getattr(env, ACCUMULATOR_ATTRIBUTE, None)
    if type(handler) is not WorstGroundLoadSubstepAccumulator or handler._env is not env:
        raise RuntimeError("Worst-load reward requires its explicit owned accumulator")
    return handler.force_cost(scale_fraction)
