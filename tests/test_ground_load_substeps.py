"""Synthetic CPU lifecycle/equation checks; no real physics or CUDA allocation."""
from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
# Reuse immutable, simulator-free stand-ins rather than duplicating their clocks.
import test_substep_rewards as lifecycle_fake
import test_ground_load_reserve as reserve_fake
from training import tron1_ground_load_reserve as reserve
from training import tron1_ground_load_substeps as sampling
from training import tron1_drift as drift


class Tensor(reserve_fake.Tensor):
    def zero_(self):
        self.value.fill(0)
        return self

    def add_(self, other):
        assert self.device == other.device
        self.value += other.value
        return self

    def square(self):
        return Tensor(np.square(self.value), device=self.device)


class Torch(reserve_fake.Torch):
    @staticmethod
    def zeros(count, *, device, dtype):
        return Tensor(np.zeros(count, dtype=dtype), device=device)


class VelocityTensor(lifecycle_fake.Tensor):
    def __getitem__(self, key):
        return VelocityTensor(self.value[key], device=self.device)

    def __truediv__(self, scalar):
        return VelocityTensor(self.value / scalar, device=self.device)

    def __pow__(self, exponent):
        return VelocityTensor(self.value ** exponent, device=self.device)


class Env(lifecycle_fake.Env):
    def __init__(self, **kwargs):
        super().__init__(device="cuda:5", **kwargs)
        self.cfg.sim.gravity = (0., 0., -9.81)
        self.scene.env_regex_ns = "/World/envs/env_.*"
        self.scene.robot.body_names = ["wheel_L_Link", "unused_link", "wheel_R_Link"]
        self.native_mass_reads = 0
        self.masses = Tensor(np.tile([10., 0., 12.273], (self.num_envs, 1)), dtype=np.float32)

        def get_masses():
            self.native_mass_reads += 1
            return self.masses

        self.scene.robot.root_physx_view = NS(get_masses=get_masses)
        self.scene.sensors = {}
        for name, body in zip(reserve.SENSOR_NAMES, reserve.WHEEL_BODY_NAMES):
            cfg = reserve_fake.SensorCfg(
                prim_path=self.scene.env_regex_ns + "/Robot/" + body,
                update_period=0., history_length=0, track_air_time=False,
                track_pose=False, track_contact_points=False,
                filter_prim_paths_expr=[reserve.GROUND_COLLIDER], max_contact_data_count_per_prim=32)
            self.scene.sensors[name] = NS(
                cfg=cfg, body_names=[body], num_bodies=1, _sim_physics_dt=.005,
                contact_physx_view=NS(sensor_count=self.num_envs, filter_count=1,
                                      max_contact_data_count=32 * self.num_envs),
                data=NS(force_matrix_w=Tensor(np.zeros((self.num_envs, 1, 1, 3)),
                                             device=self.device, dtype=np.float32)))
        terms = {
            "physics_material": NS(mode="startup", func=reserve_fake.physics_material, params={}),
            "legacy_joint_friction": NS(mode="startup", func=reserve_fake.legacy_joint_friction, params={}),
        }
        self.event_manager = NS(active_terms={"startup": list(terms)}, terms=terms,
                                get_term_cfg=lambda name: terms[name])
        self.force_pattern = np.full((4, self.num_envs, 2), 100., dtype=np.float32)
        self.force_pattern[0, :, 1] = 0.
        self.current_force_row = -1
        self.force_queries = 0
        self.reward_manager_calls = self.actor_calls = self.rng_calls = 0
        self.vz_costs = []
        self.force_read_enabled = True
        self.force_reads_per_window = 1
        self.read_reward = False  # never consume the inherited velocity cost
        self.before_reward = lambda env: env.read_force_reward()
        original_update = self.scene.update

        def update(scene, dt):
            result = original_update(dt)
            scene.robot.data.cached = VelocityTensor(self.physics_velocity, device=self.device)
            # A same-counter reset refresh may clear sensors, as Isaac does.
            for wheel, name in enumerate(reserve.SENSOR_NAMES):
                value = np.zeros((self.num_envs, 1, 1, 3), dtype=np.float32)
                if self.current_force_row >= 0:
                    value[:, 0, 0, 2] = self.force_pattern[self.current_force_row, :, wheel]
                scene.sensors[name].data.force_matrix_w = Tensor(value, device=self.device)
            return result

        self.scene.update = MethodType(update, self.scene)

    def step(self, action, *, marker=None):
        # Match the immutable fake's loop and manager ordering.  No additional
        # physics integration, observation, actor or random-number calls.
        self.original_steps += 1
        self.arguments.append((action, marker))
        for row in range(self.executed_substeps):
            if self.raise_at == row:
                raise RuntimeError("original-step-failure")
            self._sim_step_counter += self.counter_increments[row]
            self.sim.current_time_step_index += self.native_increments[row]
            self.sim.current_time += self.time_increments[row]
            self.physics_velocity = self.pattern[row].copy()
            self.current_force_row = row
            result = self.scene.update(dt=self.update_dt) if self.use_keyword_dt else self.scene.update(self.update_dt)
            assert result == "original-update-result"
            if self.duplicate_updates:
                self.scene.update(0.)
        if self.before_reward:
            self.before_reward(self)
        if self.after_reward:
            self.after_reward(self)
        self.observation_reads += 1
        self.command_updates += 1
        return "original-step-result"

    def read_force_reward(self):
        self.reward_manager_calls += 1
        self.vz_costs.append(drift.wheel_vertical_velocity_penalty(self, .02))
        if self.force_read_enabled:
            for _ in range(self.force_reads_per_window):
                self.costs.append(sampling.substep_wheel_ground_load_reserve_penalty(self, .20))

    def reset(self):
        self.resets += 1
        self.current_force_row = -1
        self.scene.update(0.)


@pytest.fixture(autouse=True)
def fake_runtime(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", Torch)
    monkeypatch.setitem(sys.modules, "training.tron1_tracking", NS(
        mdp=NS(randomize_rigid_body_material=reserve_fake.physics_material),
        clear_legacy_joint_friction=reserve_fake.legacy_joint_friction))
    monkeypatch.setitem(sys.modules, "training.tron1_domain_randomization", NS(
        reset_domain_randomization=reserve_fake.domain_randomization,
        validate_profile=lambda profile: profile))
    monkeypatch.setitem(sys.modules, "isaacsim.core.simulation_manager", NS(
        SimulationManager=NS(get_physics_sim_view=lambda: NS(get_gravity=lambda: (0., 0., -9.81)))))
    monkeypatch.setattr(reserve, "_ground_prim_audit", lambda: {
        "path": reserve.GROUND_COLLIDER, "usd_type": "Mesh", "collision_api_verified": True})
    original = reserve.wheel_ground_load_reserve_penalty

    def measured(env, scale_fraction):
        env.force_queries += 1
        return original(env, scale_fraction)

    monkeypatch.setattr(reserve, "wheel_ground_load_reserve_penalty", measured)


def initialize(env):
    return reserve.initialize_ground_load_reserve(env, copy.deepcopy(sampling.SETTINGS))


def install(env):
    initialize(env)
    return sampling.install_ground_load_accumulator(env)


def oracle(env):
    mg = getattr(env, reserve.CACHE_ATTRIBUTE).mg_n.value
    return np.exp(-np.maximum(env.force_pattern, 0.) / (.20 * mg[None, :, None])).sum(2).mean(0)


def test_lazy_imports_and_frozen_sources():
    tree = ast.parse(Path(sampling.__file__).read_text())
    assert not any(isinstance(node, ast.Import) and any(alias.name in ("torch", "isaaclab")
               for alias in node.names) for node in ast.walk(tree))
    for path, expected in [(reserve.__file__, sampling.RESERVE_DEPENDENCY_SHA256),
                           (ROOT / "training/tron1_substep_rewards.py", sampling.LIFECYCLE_DEPENDENCY_SHA256)]:
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected


def test_mean_of_nonlinear_costs_not_cost_of_mean_force():
    env = Env()
    handler = install(env)
    assert env.step("same-action", marker="same-keyword") == "original-step-result"
    np.testing.assert_array_equal(env.costs[0].value, oracle(env))
    mg = getattr(env, reserve.CACHE_ATTRIBUTE).mg_n.value
    wrong = np.exp(-env.force_pattern.mean(0) / (.20 * mg[:, None])).sum(1)
    assert not np.allclose(env.costs[0].value, wrong)
    assert env.costs[0].device == Torch.device("cuda:5")
    assert env.force_queries == 4 and env.reward_manager_calls == 1
    assert env.original_steps == 1 and len(env.scene.update_calls) == 4
    assert env.arguments == [("same-action", "same-keyword")]
    assert env.actor_calls == env.rng_calls == env.resets == 0
    assert env.observation_reads == env.command_updates == 1
    np.testing.assert_array_equal(env.vz_costs[0].value,
                                 np.square(env.pattern[-1, :, [0, 2], 2] / .02).sum(0))
    assert env.native_mass_reads == 1
    audit = handler.audit()
    assert audit["force_samples_total"] == audit["force_samples_last_control_step"] == 4
    assert audit["force_cost_reads"] == 1 and audit["inherited_velocity_cost_reads"] == 0
    assert audit["lifecycle"]["native_last_step_index"] - audit["lifecycle"]["native_start_step_index"] == 4


def test_deferred_cache_binding_fails_before_physics_then_succeeds():
    env = Env()
    handler = sampling.install_ground_load_accumulator(env)
    with pytest.raises(RuntimeError, match="exact explicitly initialized"):
        env.step(None)
    assert env.original_steps == env.force_queries == 0
    initialize(env)
    env.step(None)
    assert handler.audit()["cache_bound"]


def test_no_fallback_no_inherited_velocity_cost():
    env = Env()
    with pytest.raises(RuntimeError, match="explicit accumulator"):
        sampling.substep_wheel_ground_load_reserve_penalty(env, .20)
    handler = install(env)
    with pytest.raises(RuntimeError, match="deliberately unused"):
        handler.cost(.02)
    with pytest.raises(RuntimeError, match="exactly all"):
        handler.force_cost(.20)


@pytest.mark.parametrize("scale", [True, None, ".2", np.nan, np.inf, .10, .20001])
def test_invalid_reward_scale(scale):
    env = Env()
    install(env)
    env.before_reward = lambda e: sampling.substep_wheel_ground_load_reserve_penalty(e, scale)
    with pytest.raises((RuntimeError, ValueError)):
        env.step(None)


@pytest.mark.parametrize("reads", [0, 2])
def test_exactly_one_cost_read_per_window(reads):
    env = Env()
    handler = install(env)
    env.force_reads_per_window = reads
    with pytest.raises(RuntimeError, match="exactly one"):
        env.step(None)
    assert handler.audit()["lifecycle"]["steps_failed"] == 1
    assert handler.audit()["lifecycle"]["steps_completed"] == 0


def test_same_counter_refresh_and_keyword_arguments_do_not_collect_forces():
    env = Env()
    env.duplicate_updates = env.use_keyword_dt = True
    handler = install(env)
    env.step(None)
    assert env.force_queries == 4 and len(env.scene.update_calls) == 8
    assert handler.audit()["force_samples_total"] == 4


def test_midpoint_reset_and_automatic_done_do_not_leak_windows():
    env = Env()
    handler = install(env)
    env.after_reward = lambda e: e.reset()  # automatic reset after the reward
    env.step(None)
    saved = env.costs[0].value.copy()
    env.reset()  # explicit disarmed reset between windows
    env.force_pattern[:] = 0.
    env.step(None)
    np.testing.assert_array_equal(env.costs[0].value, saved)
    np.testing.assert_array_equal(env.costs[1].value, np.full(env.num_envs, 2., dtype=np.float32))
    assert env.force_queries == 8 and env.resets == 3
    assert handler.audit()["force_samples_total"] == 8


def test_output_and_sensor_buffers_do_not_alias_internal_window():
    env = Env()
    install(env)
    env.step(None)
    got = env.costs[0]
    saved = got.value.copy()
    env.force_pattern[:] = 1000.
    for sensor in env.scene.sensors.values():
        sensor.data.force_matrix_w.value.fill(-123.)
    env.step(None)
    np.testing.assert_array_equal(got.value, saved)
    assert not np.shares_memory(got.value, getattr(env, sampling.ACCUMULATOR_ATTRIBUTE)._force_sum.value)
    assert np.all(env.costs[1].value < got.value)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_force_fails_closed(bad):
    env = Env()
    handler = install(env)
    env.force_pattern[1, 0, 1] = bad
    with pytest.raises(RuntimeError, match="Nonfinite"):
        env.step(None)
    assert handler.audit()["force_samples_total"] == 1
    assert handler.audit()["lifecycle"]["steps_failed"] == 1


def test_negative_force_is_out_of_place_clipped_not_a_fake_contact():
    env = Env()
    install(env)
    env.force_pattern[:] = -1.
    env.step(None)
    np.testing.assert_array_equal(env.costs[0].value, np.full(env.num_envs, 2., dtype=np.float32))
    assert all(np.all(sensor.data.force_matrix_w.value[..., 2] == -1.)
               for sensor in env.scene.sensors.values())


@pytest.mark.parametrize("kind", ["shape", "dtype", "device"])
def test_malformed_refreshed_force_matrices_rejected(kind):
    env = Env()
    initialize(env)
    previous_update = env.scene.update

    def corrupt(*args, **kwargs):
        result = previous_update(*args, **kwargs)
        sensor = env.scene.sensors[reserve.SENSOR_NAMES[0]]
        matrix = sensor.data.force_matrix_w
        if kind == "shape":
            sensor.data.force_matrix_w = Tensor(matrix.value[:, 0], device=env.device)
        elif kind == "dtype":
            sensor.data.force_matrix_w = Tensor(matrix.value, device=env.device, dtype=np.float64)
        else:
            sensor.data.force_matrix_w = Tensor(matrix.value, device="cuda:0")
        return result

    env.scene.update = corrupt
    handler = sampling.install_ground_load_accumulator(env)
    with pytest.raises(RuntimeError, match="shape|float32"):
        env.step(None)
    assert handler.audit()["force_samples_total"] == 0
    assert handler.audit()["lifecycle"]["steps_failed"] == 1


@pytest.mark.parametrize("kind", ["shape", "dtype", "device", "nan", "negative", "too_large"])
def test_malformed_instantaneous_cost_rejected_even_if_dependency_returns_it(monkeypatch, kind):
    env = Env()
    handler = install(env)
    bad = Tensor(np.ones(env.num_envs, dtype=np.float32), device=env.device)
    if kind == "shape":
        bad = Tensor(np.ones((env.num_envs, 1), dtype=np.float32), device=env.device)
    elif kind == "dtype":
        bad = Tensor(np.ones(env.num_envs, dtype=np.float64), device=env.device)
    elif kind == "device":
        bad = Tensor(np.ones(env.num_envs, dtype=np.float32), device="cuda:0")
    else:
        bad.value[0] = {"nan": np.nan, "negative": -1., "too_large": 2.1}[kind]
    monkeypatch.setattr(reserve, "wheel_ground_load_reserve_penalty", lambda *args: bad)
    with pytest.raises(RuntimeError, match="Malformed"):
        env.step(None)
    assert handler.audit()["force_samples_total"] == 0


@pytest.mark.parametrize("change", [
    lambda e: setattr(e.cfg, "decimation", 8),
    lambda e: setattr(e, "step_dt", .04),
])
def test_unsupported_initial_clock_rolls_back_installed_methods(change):
    env = Env()
    env.cfg.decimation = 8
    env.step_dt = .04
    change(env)
    step, update = env.step, env.scene.update
    with pytest.raises(ValueError, match="four 5ms"):
        sampling.install_ground_load_accumulator(env)
    assert env.step == step and env.scene.update == update
    assert not hasattr(env, sampling.ACCUMULATOR_ATTRIBUTE)
    assert not hasattr(env, "_tron1_substep_wheel_velocity_accumulator")


def test_unsupported_float64_initial_velocity_rolls_back():
    env = Env(dtype=np.float64)
    step, update = env.step, env.scene.update
    with pytest.raises(ValueError, match="float32"):
        sampling.install_ground_load_accumulator(env)
    assert env.step == step and env.scene.update == update
    assert not hasattr(env.scene, "_tron1_ground_load_substep_owner")


def test_changed_dependency_source_rejected_before_any_wrapper(monkeypatch):
    env = Env()
    step, update = env.step, env.scene.update
    monkeypatch.setattr(sampling, "_source_hash", lambda path: "0" * 64)
    with pytest.raises(RuntimeError, match="frozen dependencies"):
        sampling.install_ground_load_accumulator(env)
    assert env.step == step and env.scene.update == update


@pytest.mark.parametrize("change", [
    lambda e: setattr(e.scene.sensors[reserve.SENSOR_NAMES[0]], "_sim_physics_dt", .02),
    lambda e: setattr(e.scene.sensors[reserve.SENSOR_NAMES[0]].cfg, "filter_prim_paths_expr", ["/wrong"]),
    lambda e: setattr(e.scene.sensors[reserve.SENSOR_NAMES[0]].cfg, "max_contact_data_count_per_prim", 1),
    lambda e: setattr(e.scene.sensors[reserve.SENSOR_NAMES[0]].contact_physx_view, "max_contact_data_count", 1),
    lambda e: setattr(e.scene.sensors[reserve.SENSOR_NAMES[0]].contact_physx_view, "filter_count", 2),
    lambda e: setattr(e.scene.sensors[reserve.SENSOR_NAMES[0]], "body_names", ["wrong"]),
])
def test_sensor_dt_filter_capacity_and_body_guards(change):
    env = Env()
    handler = install(env)
    change(env)
    with pytest.raises(RuntimeError):
        env.step(None)
    assert handler.audit()["force_cost_reads"] == 0


@pytest.mark.parametrize("change", [
    lambda e: setattr(e.cfg, "decimation", 8),
    lambda e: setattr(e, "step_dt", .04),
    lambda e: setattr(e.sim, "dt", .006),
    lambda e: setattr(e.sim, "current_time", np.nan),
    lambda e: setattr(e, "num_envs", 3),
])
def test_runtime_clock_changes_fail_before_original_step(change):
    env = Env()
    install(env)
    change(env)
    with pytest.raises((RuntimeError, ValueError)):
        env.step(None)
    assert env.original_steps == 0


@pytest.mark.parametrize("field,value", [("native_increments", 0), ("native_increments", 2),
    ("time_increments", .01), ("counter_increments", 0), ("counter_increments", 2)])
def test_native_and_environment_clock_corruption(field, value):
    env = Env()
    handler = install(env)
    getattr(env, field)[1] = value
    with pytest.raises(RuntimeError):
        env.step(None)
    assert handler.audit()["force_samples_total"] == 1


def test_float32_native_callback_dt_bound_is_inherited():
    env = Env()
    env.time_increments = [float(np.float32(.005))] * 4
    handler = install(env)
    env.step(None)
    assert handler.audit()["lifecycle"]["native_time_float32_dt_rounding_bound_s"] == abs(float(np.float32(.005)) - .005)


@pytest.mark.parametrize("steps", [3, 5])
def test_missing_and_extra_physics_samples_rejected(steps):
    env = Env()
    install(env)
    env.executed_substeps = steps
    if steps == 5:
        env.pattern = np.concatenate([env.pattern, env.pattern[-1:]])
        env.force_pattern = np.concatenate([env.force_pattern, env.force_pattern[-1:]])
        env.counter_increments += [1]
        env.native_increments += [1]
        env.time_increments += [.005]
    with pytest.raises(RuntimeError):
        env.step(None)


@pytest.mark.parametrize("change", [
    lambda c: c.settings.update(weight=-.5),
    lambda c: c.settings.update(scale_fraction=.1),
    lambda c: c.audit.update(module_sha256="0" * 64),
    lambda c: c.audit.update(sampling="all_substeps_mean_cost"),
    lambda c: c.audit.update(raw_normal_forces_thresholded=True),
    lambda c: c.audit.update(schema_version=True),
])
def test_exact_original_cache_settings_and_validation_required(change):
    env = Env()
    install(env)
    change(getattr(env, reserve.CACHE_ATTRIBUTE))
    with pytest.raises(RuntimeError):
        env.step(None)
    assert env.original_steps == 0


def test_cache_replacement_and_wrong_environment_identity_rejected():
    env = Env()
    handler = install(env)
    env.step(None)
    original = getattr(env, reserve.CACHE_ATTRIBUTE)
    setattr(env, reserve.CACHE_ATTRIBUTE, copy.copy(original))
    with pytest.raises(RuntimeError, match="changed after binding"):
        env.step(None)
    setattr(env, reserve.CACHE_ATTRIBUTE, original)
    env.scene.robot = object()
    with pytest.raises(RuntimeError, match="live environment"):
        env.step(None)
    assert handler.audit()["force_cost_reads"] == 1


def test_cache_lookalike_rejected_before_physics():
    env = Env()
    install(env)
    real = getattr(env, reserve.CACHE_ATTRIBUTE)
    setattr(env, reserve.CACHE_ATTRIBUTE, NS(**vars(real)))
    with pytest.raises(RuntimeError, match="exact explicitly"):
        env.step(None)
    assert env.original_steps == 0


def test_observer_arguments_single_call_and_lifo_cleanup():
    env = Env()
    initial_step = env.step
    initial_update = env.scene.update
    observed = []

    def observer(*args, **kwargs):
        observed.append((args, kwargs))
        return initial_update(*args, **kwargs)

    env.scene.update = observer
    handler = install(env)
    with pytest.raises(RuntimeError, match="already"):
        sampling.install_ground_load_accumulator(env)
    env.step(None)
    assert observed == [((.005,), {})] * 4
    handler.close()
    assert env.step == initial_step and env.scene.update is observer
    assert not hasattr(env, sampling.ACCUMULATOR_ATTRIBUTE)
    assert not hasattr(env.scene, "_tron1_ground_load_substep_owner")
    assert handler.close()["lifecycle"]["installed"] is False
    with pytest.raises(RuntimeError, match="closed"):
        handler.force_cost(.20)


@pytest.mark.parametrize("kind", ["step", "scene", "force_marker"])
def test_conflicting_owner_is_never_overwritten_or_removed(kind):
    env = Env()
    handler = install(env)
    sentinel = object()
    if kind == "step":
        env.step = sentinel
    elif kind == "scene":
        env.scene.update = sentinel
    else:
        setattr(env, sampling.ACCUMULATOR_ATTRIBUTE, sentinel)
    with pytest.raises(RuntimeError, match="ownership"):
        handler.close()
    assert (env.step if kind == "step" else env.scene.update if kind == "scene"
            else getattr(env, sampling.ACCUMULATOR_ATTRIBUTE)) is sentinel


def test_failed_original_step_disarms_and_close_restores_without_reset():
    env = Env()
    step, update = env.step, env.scene.update
    handler = install(env)
    env.raise_at = 2
    with pytest.raises(RuntimeError, match="original-step-failure"):
        env.step(None)
    assert not handler.audit()["lifecycle"]["inside_step"]
    assert handler.audit()["force_samples_total"] == 2
    handler.close()
    assert env.step == step and env.scene.update == update
    assert env.resets == 0
