"""Synthetic lifecycle checks only; fake CUDA labels are not actual GPU proof."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import test_ground_load_substeps as fake
from training import tron1_ground_load_reserve as reserve
from training import tron1_worst_ground_load_substeps as worst

fake_runtime = fake.fake_runtime


@pytest.fixture(autouse=True)
def maximum(monkeypatch):
    def apply(left, right, *, out):
        assert left.device == right.device == out.device
        out.value[:] = np.maximum(left.value, right.value)
        return out
    monkeypatch.setattr(fake.Torch, "maximum", staticmethod(apply), raising=False)


class Env(fake.Env):
    def read_force_reward(self):
        self.reward_manager_calls += 1
        self.vz_costs.append(fake.drift.wheel_vertical_velocity_penalty(self, .02))
        if self.force_read_enabled:
            for _ in range(self.force_reads_per_window):
                self.costs.append(worst.substep_worst_wheel_ground_load_reserve_penalty(self, .20))


def install(env):
    fake.initialize(env)
    return worst.install_worst_ground_load_accumulator(env)


def oracle(env):
    mg = getattr(env, reserve.CACHE_ATTRIBUTE).mg_n.value
    samples = np.exp(-np.maximum(env.force_pattern, 0.) / (.20 * mg[None, :, None])).sum(2)
    return samples.max(0), samples.mean(0)


def test_one_unloaded_substep_is_not_diluted_by_other_three():
    env = Env(); handler = install(env)
    assert env.step("original-action") == "original-step-result"
    expected, averaged = oracle(env)
    np.testing.assert_array_equal(env.costs[0].value, expected)
    assert np.all(expected > averaged)
    assert env.force_queries == 4 and env.reward_manager_calls == 1
    assert env.original_steps == 1 and len(env.scene.update_calls) == 4
    assert env.actor_calls == env.rng_calls == env.resets == 0
    assert env.observation_reads == env.command_updates == 1
    assert handler.audit()["mode"] == "all_substeps_max_cost"
    assert handler.audit()["full_runtime_fixture_verified"] is False
    assert handler.audit()["inherited_velocity_cost_reads"] == 0
    assert handler.audit()["force_samples_total"] == 4
    assert handler.audit()["force_cost_reads"] == 1
    handler.close()


def test_maximum_of_two_wheel_sum_not_sum_of_independent_maxima():
    env = Env(); env.force_pattern.fill(100.)
    env.force_pattern[0, :, 0] = 0.; env.force_pattern[1, :, 1] = 0.
    handler = install(env); env.step(None)
    expected, _ = oracle(env)
    np.testing.assert_array_equal(env.costs[0].value, expected)
    assert np.all(expected < 2.)
    handler.close()


def test_same_forces_all_four_steps_keep_original_cost():
    env = Env(); env.force_pattern.fill(50.)
    handler = install(env); env.step(None)
    expected, averaged = oracle(env)
    np.testing.assert_array_equal(env.costs[0].value, expected)
    np.testing.assert_allclose(expected, averaged, rtol=0, atol=0)
    handler.close()


def test_reset_and_next_window_do_not_inherit_old_maximum():
    env = Env(); handler = install(env); env.step(None)
    prior = env.costs[0].value.copy()
    env.reset(); env.force_pattern.fill(1000.); env.step(None)
    np.testing.assert_array_equal(env.costs[-1].value, oracle(env)[0])
    assert np.all(env.costs[-1].value < prior)
    np.testing.assert_array_equal(env.costs[0].value, prior)
    assert env.force_queries == 8
    handler.close()


def test_subnormal_buffer_is_returned_without_divide_multiply_rounding():
    env = Env(); handler = install(env)
    env.before_reward = None
    # Exercise the accessor's complete-window guards after native fake updates.
    # The fake step requires its reward read, so supply it in the callback.
    tiny = np.nextafter(np.float32(0), np.float32(1))
    def consume(env):
        handler._force_sum.value[:] = tiny
        env.costs.append(worst.substep_worst_wheel_ground_load_reserve_penalty(env, .20))
    env.before_reward = consume
    env.step(None)
    np.testing.assert_array_equal(env.costs[0].value, np.full(env.num_envs, tiny, dtype=np.float32))
    handler.close()


@pytest.mark.parametrize("change", ["missing", "duplicate", "incomplete", "duplicate_updates", "wrong_scale", "replacement_cache"])
def test_original_window_guards_remain_enforced(change):
    env = Env(); handler = install(env)
    if change == "missing": env.force_read_enabled = False
    if change == "duplicate": env.force_reads_per_window = 2
    if change == "incomplete": env.executed_substeps = 3
    if change == "duplicate_updates":
        env.duplicate_updates = True
        env.step(None)
        assert env.force_queries == 4
        handler.close(); return
    if change == "wrong_scale":
        env.before_reward = lambda env: worst.substep_worst_wheel_ground_load_reserve_penalty(env, .30)
    if change == "replacement_cache":
        env.step(None)
        setattr(env, reserve.CACHE_ATTRIBUTE, copy.copy(getattr(env, reserve.CACHE_ATTRIBUTE)))
    with pytest.raises(RuntimeError): env.step(None)
    handler.close()
