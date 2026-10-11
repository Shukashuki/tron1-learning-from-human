"""All-substep lifecycle tests without importing Isaac or installed Torch."""
from __future__ import annotations

import ast
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS, MethodType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import tron1_substep_rewards as rewards


class Tensor:
    """Small NumPy-backed stand-in; forbidden host-copy methods fail loudly."""
    def __init__(self, value, *, device="cpu", dtype=None):
        self.value = np.asarray(value, dtype=dtype)
        self.device = device
        self.dtype = self.value.dtype
        self.shape = self.value.shape

    def __getitem__(self, key):
        return Tensor(self.value[key], device=self.device)

    def square(self):
        return Tensor(np.square(self.value), device=self.device)

    def sum(self, dim):
        return Tensor(self.value.sum(axis=dim), device=self.device)

    def add_(self, other):
        assert self.device == other.device
        self.value += other.value
        return self

    def zero_(self):
        self.value.fill(0)
        return self

    def __truediv__(self, scalar):
        return Tensor(self.value / scalar, device=self.device)

    def cpu(self):
        raise AssertionError("No CPU copy is permitted")

    numpy = item = cpu


class Torch:
    @staticmethod
    def zeros(count, *, dtype, device):
        return Tensor(np.zeros(count, dtype=dtype), device=device)

    @staticmethod
    def device(value):
        return value

    @staticmethod
    def is_floating_point(value):
        return np.issubdtype(value.dtype, np.floating)


class Data:
    def __init__(self, env):
        self.env = env
        self.cached = Tensor(np.zeros((env.num_envs, 3, 3), dtype=env.dtype), device=env.device)
        self.reads = 0

    @property
    def body_link_lin_vel_w(self):
        self.reads += 1
        return self.cached

    @property
    def body_lin_vel_w(self):
        raise AssertionError("COM alias must not be read")


class Robot:
    def __init__(self, env):
        self.data = Data(env)

    def find_bodies(self, names, preserve_order):
        assert names == ["wheel_L_Link", "wheel_R_Link"] and preserve_order is True
        return [0, 2], names


class Scene:
    def __init__(self, env):
        self.env = env
        self.robot = Robot(env)
        self.update_calls = []

    def __getitem__(self, name):
        assert name == "robot"
        return self.robot

    def update(self, dt):
        self.update_calls.append(dt)
        # Mimic Isaac lazy cache refresh: values are old until original update.
        self.robot.data.cached = Tensor(self.env.physics_velocity, device=self.env.device)
        return "original-update-result"


class Sim:
    def __init__(self, dt):
        self.dt = dt
        self.current_time_step_index = 1234
        self.current_time = 6.17  # arbitrary warmup offset, unrelated to env count

    def get_physics_dt(self):
        return self.dt


class Env:
    def __init__(self, *, decimation=4, num_envs=2, dtype=np.float32, device="cpu"):
        self.dtype, self.device, self.num_envs = dtype, device, num_envs
        self.cfg = NS(decimation=decimation, sim=NS(dt=.005))
        self.physics_dt, self.step_dt = .005, .005 * decimation
        self.sim = Sim(self.physics_dt)
        self._sim_step_counter = 17
        self.physics_velocity = np.zeros((num_envs, 3, 3), dtype=dtype)
        self.scene = Scene(self)
        self.pattern = np.zeros((decimation, num_envs, 3, 3), dtype=dtype)
        for row in range(decimation):
            self.pattern[row, :, 0, 2] = (.01 + .01 * row) * (-1) ** row
            self.pattern[row, :, 2, 2] = .02 + .01 * row
            self.pattern[row, :, 1, 2] = 100.  # excluded non-wheel body
        if num_envs > 1:
            self.pattern[:, 1] *= 2.
        self.executed_substeps = decimation
        self.counter_increments = [1] * decimation
        self.native_increments = [1] * decimation
        self.time_increments = [self.physics_dt] * decimation
        self.update_dt = self.physics_dt
        self.use_keyword_dt = False
        self.duplicate_updates = False
        self.raise_at = None
        self.before_reward = None
        self.after_reward = None
        self.read_reward = True
        self.std = .02
        self.costs = []
        self.original_steps = 0
        self.arguments = []
        self.resets = self.observation_reads = self.command_updates = 0

    def step(self, action, *, marker=None):
        self.original_steps += 1
        self.arguments.append((action, marker))
        for row in range(self.executed_substeps):
            if self.raise_at == row:
                raise RuntimeError("original-step-failure")
            self._sim_step_counter += self.counter_increments[row]
            self.sim.current_time_step_index += self.native_increments[row]
            self.sim.current_time += self.time_increments[row]
            self.physics_velocity = self.pattern[row].copy()
            result = self.scene.update(dt=self.update_dt) if self.use_keyword_dt else self.scene.update(self.update_dt)
            assert result == "original-update-result"
            if self.duplicate_updates:
                self.scene.update(0.)  # no physics increment, no extra sample
        if self.before_reward:
            self.before_reward(self)
        if self.read_reward:
            self.costs.append(rewards.substep_wheel_vertical_velocity_penalty(self, self.std))
        if self.after_reward:
            self.after_reward(self)
        self.observation_reads += 1
        self.command_updates += 1
        return "original-step-result"


@pytest.fixture(autouse=True)
def fake_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", Torch())


def install(env):
    return rewards.install_wheel_vertical_velocity_accumulator(env)


def expected_cost(env):
    return np.square(env.pattern[:, :, [0, 2], 2]).sum(axis=-1).mean(axis=0) / env.std ** 2


def test_lazy_import_and_no_isaac_dependency(monkeypatch):
    original_import = importlib.import_module
    def reject_torch(name, *args, **kwargs):
        assert name != "torch"
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(importlib, "import_module", reject_torch)
    importlib.reload(rewards)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_mean_of_squares_units_device_and_refresh_order(dtype, device):
    env = Env(dtype=dtype, device=device)
    handler = install(env)
    assert env.step("unchanged-action", marker="unchanged-keyword") == "original-step-result"
    cost = env.costs[-1]
    np.testing.assert_allclose(cost.value, expected_cost(env), rtol=2e-6)
    np.testing.assert_allclose(cost.value * env.step_dt,
                               np.square(env.pattern[:, :, [0, 2], 2]).sum(axis=(0, 2)) * env.physics_dt / env.std ** 2,
                               rtol=2e-6)
    assert cost.device == device and cost.dtype == np.dtype(dtype)
    assert env.arguments == [("unchanged-action", "unchanged-keyword")]
    assert env.original_steps == 1 and len(env.scene.update_calls) == 4
    assert env.observation_reads == env.command_updates == 1
    assert env.scene.robot.data.reads == 5  # install + exactly four fresh samples
    audit = handler.audit()
    assert audit["samples_total"] == audit["samples_last_control_step"] == 4
    assert audit["native_last_step_index"] - audit["native_start_step_index"] == 4
    assert audit["last_start_counter"] == 17 and audit["last_sample_counter"] == 21
    assert audit["steps_completed"] == 1 and audit["last_step_completed"]
    assert not audit["sampling_open"] and not audit["inside_step"]


def test_squared_mean_not_mean_velocity_squared():
    env = Env(num_envs=1)
    env.pattern[:] = 0.
    env.pattern[:, 0, 0, 2] = [.02, -.02, .02, -.02]
    install(env)
    env.step(None)
    assert env.costs[-1].value[0] == pytest.approx(1.)


def test_no_installation_fallback():
    with pytest.raises(RuntimeError, match="without.*installation"):
        rewards.substep_wheel_vertical_velocity_penalty(Env(), .02)


def test_named_link_required_no_com_alias_fallback(monkeypatch):
    env = Env()
    monkeypatch.setattr(env.scene.robot, "find_bodies", lambda *args, **kw: ([0, 2], ["wrong", "names"]))
    with pytest.raises(ValueError, match="named"):
        install(env)


@pytest.mark.parametrize("ids", [[], [0], [0, 0], [0, 3], [True, 2], [-1, 2], [0., 2], None])
def test_bad_explicit_wheel_ids(ids):
    if ids is None:
        ids = 17
    with pytest.raises(ValueError):
        rewards.install_wheel_vertical_velocity_accumulator(Env(), wheel_body_ids=ids)


def test_explicit_wheel_ids_and_permutation():
    env = Env()
    handler = rewards.install_wheel_vertical_velocity_accumulator(env, wheel_body_ids=[2, 0])
    env.step(None)
    np.testing.assert_allclose(env.costs[-1].value, expected_cost(env), rtol=2e-6)
    assert handler.audit()["wheel_body_ids"] == [2, 0]


@pytest.mark.parametrize("std", [0, -.01, True, None, "0.02", float("nan"), float("inf"), .0049, .1001])
def test_invalid_std_disarms_failed_step(std):
    env = Env()
    handler = install(env)
    env.std = std
    with pytest.raises(ValueError):
        env.step(None)
    assert not handler.audit()["inside_step"] and handler.audit()["steps_failed"] == 1
    env.std = .02
    env.step(None)
    assert handler.audit()["steps_completed"] == 1


@pytest.mark.parametrize("std", [.005, .1])
def test_std_bounds(std):
    env = Env()
    env.std = std
    install(env)
    env.step(None)
    np.testing.assert_allclose(env.costs[-1].value, expected_cost(env), rtol=2e-6)


@pytest.mark.parametrize("change", [
    lambda e: setattr(e.cfg, "decimation", 0),
    lambda e: setattr(e.cfg, "decimation", 4.),
    lambda e: setattr(e.cfg, "decimation", True),
    lambda e: setattr(e, "physics_dt", -.005),
    lambda e: setattr(e.cfg.sim, "dt", .006),
    lambda e: setattr(e.sim, "dt", .006),
    lambda e: setattr(e.sim, "dt", float("nan")),
    lambda e: setattr(e, "step_dt", .019),
    lambda e: setattr(e, "num_envs", 0),
    lambda e: setattr(e.sim, "current_time", float("nan")),
    lambda e: setattr(e.sim, "current_time_step_index", 1.),
    lambda e: delattr(e.sim, "current_time_step_index"),
])
def test_invalid_initial_clock_or_size_has_no_wrapper_mutation(change):
    env = Env()
    change(env)
    with pytest.raises(ValueError):
        install(env)
    assert "step" not in vars(env) and "update" not in vars(env.scene)


@pytest.mark.parametrize("change", [
    lambda e: setattr(e.cfg, "decimation", 8),
    lambda e: setattr(e, "step_dt", .04),
    lambda e: setattr(e.sim, "dt", .006),
    lambda e: setattr(e, "num_envs", 3),
])
def test_changed_runtime_clock_or_size_fails_before_original_step(change):
    env = Env()
    handler = install(env)
    change(env)
    with pytest.raises(ValueError):
        env.step(None)
    assert env.original_steps == 0 and not handler.audit()["inside_step"]


@pytest.mark.parametrize("increment", [0, 2, -1])
def test_native_index_zero_double_or_backwards_is_rejected(increment):
    env = Env()
    env.native_increments[1] = increment
    handler = install(env)
    with pytest.raises(RuntimeError, match="Native physics"):
        env.step(None)
    assert handler.audit()["samples_last_control_step"] == 1
    assert not handler.audit()["inside_step"]


@pytest.mark.parametrize("increment", [0., .001, .01, .0050001])
def test_native_time_wrong_delta_rejected_despite_exact_index(increment):
    env = Env()
    env.time_increments[1] = increment
    handler = install(env)
    with pytest.raises(RuntimeError, match="Native physics"):
        env.step(None)
    assert handler.audit()["samples_last_control_step"] == 1


def test_float32_physx_callback_dt_quantization_is_explicitly_supported():
    env = Env()
    env.time_increments = [float(np.float32(env.physics_dt))] * 4
    handler = install(env)
    env.step(None)
    audit = handler.audit()
    assert audit["native_time_float32_dt_rounding_bound_s"] == abs(float(np.float32(.005)) - .005)
    assert audit["native_last_step_index"] == 1238
    assert audit["native_last_time_s"] == pytest.approx(6.17 + 4 * float(np.float32(.005)), abs=1e-14)


@pytest.mark.parametrize("increment", [0, 2, -1])
def test_environment_counter_corruption_rejected(increment):
    env = Env()
    env.counter_increments[1] = increment
    handler = install(env)
    with pytest.raises(RuntimeError):
        env.step(None)
    assert not handler.audit()["inside_step"]


@pytest.mark.parametrize("dt", [0., .004, None, True])
def test_wrong_scene_update_dt_rejected(dt):
    env = Env()
    env.update_dt = dt
    handler = install(env)
    with pytest.raises(ValueError):
        env.step(None)
    assert len(env.scene.update_calls) == 1 and handler.audit()["samples_total"] == 0


def test_keyword_dt_and_same_counter_duplicate_refresh_not_double_counted():
    env = Env()
    env.use_keyword_dt = env.duplicate_updates = True
    handler = install(env)
    env.step(None)
    assert len(env.scene.update_calls) == 8 and handler.audit()["samples_total"] == 4
    np.testing.assert_allclose(env.costs[-1].value, expected_cost(env), rtol=2e-6)


def test_missing_substep_and_early_reward_fail_no_fallback():
    env = Env()
    env.executed_substeps = 3
    handler = install(env)
    with pytest.raises(RuntimeError, match="exactly all"):
        env.step(None)
    assert handler.audit()["samples_last_control_step"] == 3
    with pytest.raises(RuntimeError, match="exactly all"):
        handler.cost(.02)


def test_extra_substep_after_complete_fails():
    env = Env()
    env.executed_substeps = 5
    env.pattern = np.concatenate([env.pattern, env.pattern[-1:]])
    env.counter_increments += [1]
    env.native_increments += [1]
    env.time_increments += [.005]
    handler = install(env)
    with pytest.raises(RuntimeError, match="after.*completed"):
        env.step(None)
    assert handler.audit()["samples_total"] == 4


def test_hidden_native_step_with_same_env_counter_during_update_fails():
    env = Env()
    def hidden_step(e):
        e.sim.current_time_step_index += 1
        e.sim.current_time += e.physics_dt
        e.scene.update(0.)
    env.before_reward = hidden_step
    handler = install(env)
    with pytest.raises(RuntimeError, match="without.*counter"):
        env.step(None)
    assert not handler.audit()["inside_step"]


def test_native_extra_step_without_scene_refresh_fails_at_reward():
    env = Env()
    def hidden_step(e):
        e.sim.current_time_step_index += 1
        e.sim.current_time += e.physics_dt
    env.before_reward = hidden_step
    install(env)
    with pytest.raises(RuntimeError, match="exactly all"):
        env.step(None)


def test_partial_terminal_reset_does_not_enter_cost_or_next_episode():
    env = Env()
    def reset_first(e):
        e.physics_velocity[0] = 1000.
        e.scene.update(0.)
        e.resets += 1
    env.after_reward = reset_first
    handler = install(env)
    env.step(None)
    first_cost = env.costs[-1]
    np.testing.assert_allclose(first_cost.value, expected_cost(env), rtol=2e-6)
    assert env.resets == 1 and handler.audit()["samples_total"] == 4
    env.pattern *= .5
    env.step(None)
    np.testing.assert_allclose(env.costs[-1].value, expected_cost(env), rtol=2e-6)
    np.testing.assert_allclose(first_cost.value, expected_cost(env) * 4., rtol=2e-6)  # no mutable alias
    assert handler.audit()["samples_total"] == 8


def test_external_reset_refresh_is_ignored_and_getter_rejects_outside_step():
    env = Env()
    handler = install(env)
    env.scene.update(0.)
    assert handler.audit()["samples_total"] == 0
    env.step(None)
    env.scene.update(0.)
    assert handler.audit()["samples_total"] == 4
    with pytest.raises(RuntimeError, match="exactly all"):
        handler.cost(.02)


def test_repeat_reward_reads_do_not_recollect_or_consume():
    env = Env()
    values = []
    env.before_reward = lambda e: values.extend([rewards.substep_wheel_vertical_velocity_penalty(e, .02), rewards.substep_wheel_vertical_velocity_penalty(e, .02)])
    handler = install(env)
    env.step(None)
    assert handler.audit()["cost_reads"] == 3 and handler.audit()["samples_total"] == 4
    np.testing.assert_allclose(values[0].value, values[1].value)


def test_disabled_weight_style_no_reward_read_still_validates_exact_count():
    env = Env()
    env.read_reward = False
    handler = install(env)
    env.step(None)
    assert handler.audit()["cost_reads"] == 0 and handler.audit()["steps_completed"] == 1
    env.executed_substeps = 3
    with pytest.raises(RuntimeError, match="exactly all"):
        env.step(None)


def test_original_step_exception_disarms_and_allows_fresh_retry():
    env = Env()
    env.raise_at = 2
    handler = install(env)
    with pytest.raises(RuntimeError, match="original-step-failure"):
        env.step(None)
    assert handler.audit()["samples_total"] == 2 and not handler.audit()["inside_step"]
    env.raise_at = None
    env.step(None)
    np.testing.assert_allclose(env.costs[-1].value, expected_cost(env), rtol=2e-6)
    assert handler.audit()["steps_completed"] == handler.audit()["steps_failed"] == 1


def test_original_scene_update_exception_disarms():
    env = Env()
    env.scene.update = lambda dt: (_ for _ in ()).throw(RuntimeError("original-update-failure"))
    handler = install(env)
    with pytest.raises(RuntimeError, match="original-update-failure"):
        env.step(None)
    assert not handler.audit()["inside_step"] and handler.audit()["samples_total"] == 0
    handler.close()


def test_reentrant_step_fails_and_disarms_outer_call():
    env = Env()
    env.before_reward = lambda e: e.step("recursive")
    handler = install(env)
    with pytest.raises(RuntimeError, match="Reentrant"):
        env.step(None)
    assert env.original_steps == 1 and not handler.audit()["inside_step"]


def test_reentrant_update_fails_without_infinite_recursion():
    env = Env()
    env.scene.update = lambda dt: env.scene.update(dt)
    handler = install(env)
    with pytest.raises(RuntimeError, match="Reentrant scene"):
        env.step(None)
    assert not handler.audit()["inside_step"]


def test_prior_observer_and_bound_subclass_step_are_chained_once_and_restored():
    class SubEnv(Env):
        def step(self, *args, **kw):
            self.subclass_called = True
            return super().step(*args, **kw)
    env = SubEnv()
    prior_update = env.scene.update
    observer_counters = []
    def observer(dt):
        result = prior_update(dt)
        observer_counters.append(env._sim_step_counter)
        return result
    env.scene.update = observer
    handler = install(env)
    env.step(None)
    assert env.subclass_called and observer_counters == [18, 19, 20, 21]
    handler.close()
    assert "step" not in vars(env) and env.scene.update is observer
    assert handler.close()["installed"] is False
    assert not hasattr(env, rewards._ENV_MARKER) and not hasattr(env.scene, rewards._SCENE_MARKER)


def test_double_install_and_shared_scene_owner_are_rejected():
    env = Env()
    handler = install(env)
    with pytest.raises(RuntimeError, match="already installed"):
        install(env)
    second = Env()
    second.scene = env.scene
    with pytest.raises(RuntimeError, match="conflicting"):
        install(second)
    handler.close()
    install(env).close()


@pytest.mark.parametrize("attribute", ["step", "update"])
def test_replaced_wrapper_is_conflict_not_silent_restore(attribute):
    env = Env()
    handler = install(env)
    if attribute == "step":
        env.step = lambda *args, **kw: None
    else:
        env.scene.update = lambda *args, **kw: None
    with pytest.raises(RuntimeError, match="ownership conflict"):
        handler.close()
    with pytest.raises(RuntimeError, match="ownership conflict"):
        handler.cost(.02)
    if attribute == "update":
        with pytest.raises(RuntimeError, match="ownership conflict"):
            env.step(None)


def test_close_inside_step_rejected_but_can_close_after_exception():
    env = Env()
    handler = install(env)
    env.before_reward = lambda e: handler.close()
    with pytest.raises(RuntimeError, match="during.*step"):
        env.step(None)
    handler.close()
    assert "step" not in vars(env) and "update" not in vars(env.scene)


def test_native_clock_rounding_tolerance_does_not_cover_float32_sized_wrong_time():
    env = Env()
    # Several float32 dt ULPs are not the documented nominal-to-f32 quantization.
    env.time_increments[2] += 4 * float(np.spacing(np.float32(.005)))
    install(env)
    with pytest.raises(RuntimeError, match="Native physics"):
        env.step(None)


def test_ast_has_no_host_tensor_copy_or_extra_updates():
    tree = ast.parse((ROOT / "training/tron1_substep_rewards.py").read_text())
    called_attributes = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not called_attributes & {"cpu", "numpy", "item", "reset", "render", "compute", "apply_action", "write_data_to_sim"}
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all("torch" not in ast.unparse(node) and "isaac" not in ast.unparse(node) for node in imports)


@pytest.mark.parametrize("dtype", [np.int32, np.bool_])
def test_nonfloating_initial_velocity_rejected_without_install(dtype):
    env = Env()
    env.scene.robot.data.cached = Tensor(np.zeros((env.num_envs, 3, 3), dtype=dtype))
    with pytest.raises(ValueError, match="floating"):
        install(env)
    assert "step" not in vars(env) and "update" not in vars(env.scene)


def test_initial_velocity_device_mismatch_rejected_without_copy():
    env = Env()
    env.scene.robot.data.cached.device = "cuda:0"
    with pytest.raises(ValueError, match="environment device"):
        install(env)
    assert "step" not in vars(env)


@pytest.mark.parametrize("shape", [(2, 3), (1, 3, 3), (2, 1, 3), (2, 3, 6)])
def test_malformed_initial_link_velocity_shape_rejected(shape):
    env = Env()
    env.scene.robot.data.cached = Tensor(np.zeros(shape, dtype=np.float32))
    with pytest.raises(ValueError):
        install(env)


def test_velocity_dtype_changes_during_step_fail_after_refresh():
    env = Env()
    prior = env.scene.update
    def change_dtype(dt):
        result = prior(dt)
        env.scene.robot.data.cached = Tensor(env.physics_velocity, dtype=np.float64)
        return result
    env.scene.update = change_dtype
    handler = install(env)
    with pytest.raises(ValueError, match="dtype changed"):
        env.step(None)
    assert handler.audit()["samples_total"] == 0 and not handler.audit()["inside_step"]


def test_coherent_but_changed_clock_is_not_silently_adopted():
    env = Env()
    install(env)
    env.physics_dt = env.cfg.sim.dt = env.sim.dt = .006
    env.step_dt = .024
    with pytest.raises(ValueError, match="changed after"):
        env.step(None)
    assert env.original_steps == 0


def test_atomic_install_failure_restores_prior_instance_attributes():
    class RejectScene(Scene):
        def __setattr__(self, name, value):
            if name == "update" and isinstance(value, MethodType):
                raise RuntimeError("read-only-scene-method")
            super().__setattr__(name, value)
    env = Env()
    env.scene = RejectScene(env)
    with pytest.raises(RuntimeError, match="read-only-scene-method"):
        install(env)
    assert "step" not in vars(env)
    assert "update" not in vars(env.scene)
    assert not hasattr(env, rewards._ENV_MARKER)
    assert not hasattr(env.scene, rewards._SCENE_MARKER)


def test_baseexception_also_disarms_sampling():
    env = Env()
    env.before_reward = lambda e: (_ for _ in ()).throw(KeyboardInterrupt())
    handler = install(env)
    with pytest.raises(KeyboardInterrupt):
        env.step(None)
    assert not handler.audit()["inside_step"] and not handler.audit()["sampling_open"]
    handler.close()


@pytest.mark.parametrize("dt", [1e-100, 1e100])
def test_unrepresentable_float32_physics_dt_rejected(dt):
    env = Env()
    env.physics_dt = env.cfg.sim.dt = env.sim.dt = dt
    env.step_dt = dt * 4
    with pytest.raises(ValueError, match="representable"):
        install(env)


def test_ownership_marker_fingerprint_conflict_fails():
    env = Env()
    handler = install(env)
    setattr(env.scene, rewards._SCENE_MARKER, object())
    with pytest.raises(RuntimeError, match="ownership conflict"):
        env.step(None)
    assert handler.audit()["steps_started"] == 0


@pytest.mark.parametrize("decimation", [1, 8, 16])
def test_exact_configured_decimation_not_hardcoded_four(decimation):
    env = Env(decimation=decimation)
    handler = install(env)
    env.step(None)
    np.testing.assert_allclose(env.costs[-1].value, expected_cost(env), rtol=2e-6)
    assert handler.audit()["samples_total"] == decimation
