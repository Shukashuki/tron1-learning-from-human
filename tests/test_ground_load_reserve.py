"""Ground-only reserve reward tests without installed Torch or simulator imports."""
from __future__ import annotations

import ast
import copy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import tron1_ground_load_reserve as reserve


class Device:
    def __init__(self, value):
        self.value = str(value)
        split = self.value.split(":")
        self.type, self.index = split[0], int(split[1]) if len(split) == 2 else None

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return self.value == str(other)


class Tensor:
    copies = []

    def __init__(self, value, *, device="cpu", dtype=None):
        self.value = np.asarray(value, dtype=dtype)
        self.device = Device(device)
        self.dtype, self.shape = self.value.dtype, self.value.shape

    def __getitem__(self, index):
        return Tensor(self.value[index], device=self.device)

    def __bool__(self):
        return bool(self.value)

    def __float__(self):
        Tensor.copies.append("float")
        return float(self.value)

    def __lt__(self, other):
        return Tensor(self.value < other, device=self.device)

    def __gt__(self, other):
        return Tensor(self.value > other, device=self.device)

    def __neg__(self):
        return Tensor(-self.value, device=self.device)

    def __mul__(self, other):
        return Tensor(self.value * other, device=self.device)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            assert self.device == other.device
            other = other.value
        return Tensor(self.value / other, device=self.device)

    def all(self):
        return Tensor(self.value.all(), device=self.device)

    def any(self):
        return Tensor(self.value.any(), device=self.device)

    def min(self):
        return Tensor(self.value.min(), device=self.device)

    def max(self):
        return Tensor(self.value.max(), device=self.device)

    def sum(self, dim):
        return Tensor(self.value.sum(axis=dim), device=self.device)

    def clone(self):
        return Tensor(self.value.copy(), device=self.device)

    def detach(self):
        return self

    def to(self, *, device=None, dtype=None):
        Tensor.copies.append((str(self.device), str(device or self.device), str(dtype)))
        return Tensor(self.value, device=device or self.device, dtype=dtype)

    def tolist(self):
        Tensor.copies.append("tolist")
        return self.value.tolist()

    def cpu(self):
        raise AssertionError("No hot-path CPU copy allowed")

    numpy = item = cpu


class Torch:
    float32, float64 = np.dtype("float32"), np.dtype("float64")
    device = Device

    @staticmethod
    def is_floating_point(tensor):
        return np.issubdtype(tensor.dtype, np.floating)

    @staticmethod
    def isfinite(tensor):
        return Tensor(np.isfinite(tensor.value), device=tensor.device)

    @staticmethod
    def stack(tensors, dim):
        assert all(tensor.device == tensors[0].device for tensor in tensors)
        return Tensor(np.stack([tensor.value for tensor in tensors], axis=dim), device=tensors[0].device)

    @staticmethod
    def exp(tensor):
        return Tensor(np.exp(tensor.value), device=tensor.device)

    @staticmethod
    def clamp(tensor, min):
        return Tensor(np.maximum(tensor.value, min), device=tensor.device)


class SensorCfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Scene:
    env_regex_ns = "/World/envs/env_.*"

    def __getitem__(self, key):
        assert key == "robot"
        return self.robot


def physics_material(*args):
    pass


def legacy_joint_friction(*args):
    pass


def domain_randomization(*args):
    pass


def make_env():
    num_envs, device = 2, "cuda:5"
    scene = Scene()
    scene.sensors = {}
    for name, body in zip(reserve.SENSOR_NAMES, reserve.WHEEL_BODY_NAMES):
        cfg = SensorCfg(prim_path=scene.env_regex_ns + "/Robot/" + body,
                        update_period=0., history_length=0,
                        filter_prim_paths_expr=[reserve.GROUND_COLLIDER],
                        track_air_time=False, track_pose=False, track_contact_points=False,
                        max_contact_data_count_per_prim=32)
        scene.sensors[name] = NS(cfg=cfg, body_names=[body], num_bodies=1,
                                _sim_physics_dt=.005,
                                contact_physx_view=NS(sensor_count=num_envs, filter_count=1,
                                                      max_contact_data_count=64),
                                data=NS(force_matrix_w=Tensor(np.zeros((num_envs, 1, 1, 3)),
                                                              device=device, dtype=np.float32)))
    masses = Tensor([[0., 10., 12.273], [0., 8., 15.]], dtype=np.float32)
    native = NS(masses=masses, mass_reads=0)

    def get_masses():
        native.mass_reads += 1
        return native.masses

    native.get_masses = get_masses
    scene.robot = NS(body_names=["world", *reserve.WHEEL_BODY_NAMES], root_physx_view=native)
    terms = {
        "physics_material": NS(mode="startup", func=physics_material, params={}),
        "legacy_joint_friction": NS(mode="startup", func=legacy_joint_friction, params={}),
    }
    events = NS(active_terms={"startup": list(terms)}, terms=terms)
    events.get_term_cfg = lambda name: events.terms[name]
    return NS(num_envs=num_envs, device=device, physics_dt=.005, step_dt=.02,
              cfg=NS(decimation=4, sim=NS(dt=.005, gravity=(0., 0., -9.81))),
              sim=NS(get_physics_dt=lambda: .005), scene=scene, event_manager=events)


@pytest.fixture
def runtime(monkeypatch):
    Tensor.copies = []
    monkeypatch.setitem(sys.modules, "torch", Torch)
    monkeypatch.setitem(sys.modules, "isaaclab.sensors", NS(ContactSensorCfg=SensorCfg))
    monkeypatch.setitem(sys.modules, "training.tron1_tracking", NS(
        mdp=NS(randomize_rigid_body_material=physics_material),
        clear_legacy_joint_friction=legacy_joint_friction))
    monkeypatch.setitem(sys.modules, "training.tron1_domain_randomization", NS(
        reset_domain_randomization=domain_randomization,
        validate_profile=lambda profile: {} if profile == {"safe": True} else (_ for _ in ()).throw(ValueError("invalid motor profile"))))
    native = NS(gravity=(0., 0., float(np.float32(-9.81))), reads=0)

    def get_gravity():
        native.reads += 1
        return native.gravity

    view = NS(get_gravity=get_gravity)
    monkeypatch.setitem(sys.modules, "isaacsim.core.simulation_manager", NS(
        SimulationManager=NS(get_physics_sim_view=lambda: view)))
    monkeypatch.setattr(reserve, "_ground_prim_audit", lambda: {
        "path": reserve.GROUND_COLLIDER, "usd_type": "Mesh", "collision_api_verified": True})
    return native


@pytest.mark.parametrize("weight,scale", [(-2., .05), (-1., .20), (0., .30)])
def test_settings_copy_and_bounds(weight, scale):
    incoming = {"weight": weight, "scale_fraction": scale}
    got = reserve.validate_load_reserve_settings(incoming)
    assert got == incoming and got is not incoming


@pytest.mark.parametrize("settings", [None, [], {}, {"weight": -1},
    {"weight": -1, "scale_fraction": .2, "extra": 0},
    {"weight": True, "scale_fraction": .2}, {"weight": -.5, "scale_fraction": False},
    {"weight": "-1", "scale_fraction": .2}, {"weight": -1, "scale_fraction": ".2"},
    {"weight": np.nan, "scale_fraction": .2}, {"weight": -1, "scale_fraction": np.inf},
    {"weight": 10**1000, "scale_fraction": .2},
    {"weight": -2.01, "scale_fraction": .2}, {"weight": .01, "scale_fraction": .2},
    {"weight": -1, "scale_fraction": .049}, {"weight": -1, "scale_fraction": .301}])
def test_settings_fail_closed(settings):
    with pytest.raises(ValueError):
        reserve.validate_load_reserve_settings(settings)


def test_configuration_adds_exactly_two_no_points_or_history(runtime):
    sentinel = object()
    cfg = NS(contact_forces=sentinel)
    reserve.configure_training_ground_force_sensors(cfg)
    assert set(vars(cfg)) == {"contact_forces", *reserve.SENSOR_NAMES}
    assert cfg.contact_forces is sentinel
    for name, body in zip(reserve.SENSOR_NAMES, reserve.WHEEL_BODY_NAMES):
        sensor = getattr(cfg, name)
        assert sensor.prim_path == "{ENV_REGEX_NS}/Robot/" + body
        assert sensor.filter_prim_paths_expr == [reserve.GROUND_COLLIDER]
        assert sensor.update_period == 0 and sensor.history_length == 0
        assert sensor.track_contact_points is sensor.track_air_time is sensor.track_pose is False
        assert sensor.max_contact_data_count_per_prim == 32


@pytest.mark.parametrize("name", reserve.SENSOR_NAMES)
def test_configuration_no_partial_overwrite(name, runtime):
    existing = object()
    cfg = NS(**{name: existing})
    with pytest.raises(ValueError, match="overwrite"):
        reserve.configure_training_ground_force_sensors(cfg)
    assert vars(cfg) == {name: existing}


def initialize(env):
    return reserve.initialize_ground_load_reserve(env, {"weight": -1., "scale_fraction": .2})


def test_startup_live_cpu_mass_double_reduce_gpu_cache_json_audit(runtime):
    env = make_env()
    audit = initialize(env)
    cache = getattr(env, reserve.CACHE_ATTRIBUTE)
    assert str(cache.mg_n.device) == env.device and cache.mg_n.dtype == np.dtype("float32")
    expected = env.scene.robot.root_physx_view.masses.value.astype(np.float64).sum(1) * abs(runtime.gravity[2])
    np.testing.assert_array_equal(cache.mg_n.value, expected.astype(np.float32))
    assert audit["mass_kg_range"] == [float(expected[0] / abs(runtime.gravity[2])), 23.]
    assert audit["first_environment_link_mass_kg"][0] == 0.
    assert audit["sensor_matrix_shape"] == [2, 1, 1, 3]
    assert audit["sampling"] == "last_physics_substep"
    assert audit["scale_is_minimum_force"] is False
    assert audit["startup_loaded_contact_required"] is False
    assert audit["host_synchronization_free"] is False
    assert audit["loaded_normal_vector_and_raw_aggregate_runtime_smoke_required"] is True
    json.dumps(audit, allow_nan=False)
    audit["settings"]["scale_fraction"] = .3
    assert cache.settings["scale_fraction"] == .2 and cache.audit["settings"]["scale_fraction"] == .2
    assert runtime.reads == env.scene.robot.root_physx_view.mass_reads == 1
    with pytest.raises(RuntimeError, match="already"):
        initialize(env)


def test_compatible_evaluation_point_sensors_do_not_change(runtime):
    env = make_env()
    for sensor in env.scene.sensors.values():
        sensor.cfg.track_contact_points = True
    audit = initialize(env)
    assert audit["sensor_track_contact_points"] == [True, True]


def set_loads(env, left, right):
    for name, values in zip(reserve.SENSOR_NAMES, (left, right)):
        env.scene.sensors[name].data.force_matrix_w.value[:, 0, 0, 2] = values


def test_formula_current_matrix_no_threshold_no_raw_mutation_no_hot_copy(runtime):
    env = make_env()
    initialize(env)
    before_copies = list(Tensor.copies)
    cache = getattr(env, reserve.CACHE_ATTRIBUTE)
    set_loads(env, [0., 2.], [0., 4.])  # sub-5N loads must still influence cost
    matrices = [sensor.data.force_matrix_w.value.copy() for sensor in env.scene.sensors.values()]
    first = reserve.wheel_ground_load_reserve_penalty(env, .2)
    expected = np.exp(-np.array([[0., 0.], [2., 4.]]) / (.2 * cache.mg_n.value[:, None])).sum(1)
    np.testing.assert_allclose(first.value, expected, rtol=1.e-7)
    assert first.value[0] == 2. and first.value[1] < 2.
    for original, sensor in zip(matrices, env.scene.sensors.values()):
        np.testing.assert_array_equal(original, sensor.data.force_matrix_w.value)
    set_loads(env, [120., -2.], [110., -4.])
    second = reserve.wheel_ground_load_reserve_penalty(env, .2)
    assert second.value[0] < .2 and second.value[1] == 2.
    assert Tensor.copies == before_copies
    assert runtime.reads == env.scene.robot.root_physx_view.mass_reads == 1


def test_balanced_load_soft_preference_not_hard_equal_split(runtime):
    env = make_env()
    initialize(env)
    set_loads(env, [100., 200.], [100., 0.])
    costs = reserve.wheel_ground_load_reserve_penalty(env, .2).value
    assert 0. < costs[0] < costs[1] < 2.


def test_reset_does_not_add_history_or_recache(runtime):
    env = make_env()
    initialize(env)
    cache = getattr(env, reserve.CACHE_ATTRIBUTE)
    set_loads(env, [100., 100.], [100., 100.])
    reserve.wheel_ground_load_reserve_penalty(env, .2)
    set_loads(env, [0., 0.], [0., 0.])  # emulate fresh postreset sensor buffers
    assert np.all(reserve.wheel_ground_load_reserve_penalty(env, .2).value == 2.)
    assert getattr(env, reserve.CACHE_ATTRIBUTE) is cache
    assert env.scene.robot.root_physx_view.mass_reads == 1


@pytest.mark.parametrize("change", [
    lambda e: setattr(e, "device", "cpu"),
    lambda e: setattr(e, "device", "cuda"),
    lambda e: setattr(e, "num_envs", 0),
    lambda e: setattr(e.cfg.sim, "dt", .0025),
    lambda e: setattr(e, "physics_dt", .0025),
    lambda e: setattr(e.sim, "get_physics_dt", lambda: .02),
    lambda e: setattr(e.cfg, "decimation", 8),
    lambda e: setattr(e.cfg, "decimation", True),
    lambda e: setattr(e, "step_dt", .025),
    lambda e: setattr(e.scene.robot, "body_names", ["world", "wheel_L_Link", "wrong"]),
    lambda e: e.scene.sensors.pop("wheel_R_ground"),
])
def test_startup_environment_guards(change, runtime):
    env = make_env()
    change(env)
    with pytest.raises((RuntimeError, ValueError)):
        initialize(env)
    assert not hasattr(env, reserve.CACHE_ATTRIBUTE)


@pytest.mark.parametrize("change", [
    lambda s: setattr(s.cfg, "update_period", .02),
    lambda s: setattr(s.cfg, "history_length", 1),
    lambda s: setattr(s.cfg, "track_air_time", True),
    lambda s: setattr(s.cfg, "filter_prim_paths_expr", ["/World/Robot"]),
    lambda s: setattr(s.cfg, "filter_prim_paths_expr", [reserve.GROUND_COLLIDER] * 2),
    lambda s: setattr(s.cfg, "prim_path", "/World/envs/env_.*/Robot/.*"),
    lambda s: setattr(s, "body_names", ["wheel_R_Link"]),
    lambda s: setattr(s, "num_bodies", 2),
    lambda s: setattr(s.contact_physx_view, "filter_count", 2),
    lambda s: setattr(s.contact_physx_view, "sensor_count", 1),
    lambda s: setattr(s.contact_physx_view, "max_contact_data_count", 16),
    lambda s: setattr(s.cfg, "max_contact_data_count_per_prim", 4),
    lambda s: setattr(s, "_sim_physics_dt", .02),
    lambda s: setattr(s.data, "force_matrix_w", None),
    lambda s: setattr(s.data, "force_matrix_w", Tensor(np.zeros((2, 2, 1, 3)), device="cuda:5", dtype=np.float32)),
    lambda s: setattr(s.data, "force_matrix_w", Tensor(np.zeros((2, 1, 1, 3)), device="cpu", dtype=np.float32)),
    lambda s: setattr(s.data, "force_matrix_w", Tensor(np.zeros((2, 1, 1, 3)), device="cuda:5", dtype=np.float64)),
    lambda s: s.data.force_matrix_w.value.fill(np.nan),
    lambda s: s.data.force_matrix_w.value.__setitem__((0, 0, 0, 2), -1.),
])
def test_startup_sensor_guards(change, runtime):
    env = make_env()
    change(env.scene.sensors["wheel_L_ground"])
    with pytest.raises((RuntimeError, ValueError)):
        initialize(env)


@pytest.mark.parametrize("mass", [
    Tensor([[1., 2.]], dtype=np.float32),
    Tensor([[0., 1., 2.], [0., 1., 2.]], device="cuda:5", dtype=np.float32),
    Tensor([[0, 1, 2], [0, 1, 2]], dtype=np.int64),
    Tensor([[0., -1., 2.], [0., 1., 2.]], dtype=np.float32),
    Tensor([[0., np.inf, 2.], [0., 1., 2.]], dtype=np.float32),
    Tensor([[0., np.nan, 2.], [0., 1., 2.]], dtype=np.float32),
    Tensor([[0., 0., 0.], [0., 1., 2.]], dtype=np.float32),
])
def test_live_mass_guards(mass, runtime):
    env = make_env()
    env.scene.robot.root_physx_view.masses = mass
    with pytest.raises(RuntimeError):
        initialize(env)


@pytest.mark.parametrize("gravity", [None, (0., 0.), (1., 0., -9.81),
    (0., 1., -9.81), (0., 0., 0.), (0., 0., 9.81), (0., 0., np.nan), (0., 0., -1.)])
def test_native_gravity_guards(gravity, runtime):
    runtime.gravity = gravity
    with pytest.raises(RuntimeError):
        initialize(make_env())


@pytest.mark.parametrize("valid,collision,prim_type,accepted", [
    (True, True, "Mesh", True), (True, True, "Plane", True),
    (False, True, "Mesh", False), (True, False, "Mesh", False),
    (True, True, "Cube", False),
])
def test_ground_path_requires_actual_collision_prim(valid, collision, prim_type, accepted, runtime, monkeypatch):
    class Prim:
        def IsValid(self): return valid
        def HasAPI(self, api): return collision
        def GetTypeName(self): return prim_type
    queried_paths = []
    def query(path):
        queried_paths.append(path)
        return Prim()
    stage = NS(GetPrimAtPath=query)
    monkeypatch.setitem(sys.modules, "omni", NS(usd=NS(get_context=lambda: NS(get_stage=lambda: stage))))
    monkeypatch.setitem(sys.modules, "omni.usd", sys.modules["omni"].usd)
    monkeypatch.setitem(sys.modules, "pxr", NS(UsdPhysics=NS(CollisionAPI=object())))
    # Use the actual function despite the fixture's simulator-free substitute.
    function_ast = ast.parse(Path(reserve.__file__).read_text())
    function = next(n for n in function_ast.body if isinstance(n, ast.FunctionDef) and n.name == "_ground_prim_audit")
    namespace = {"GROUND_COLLIDER": reserve.GROUND_COLLIDER}
    exec(compile(ast.Module(body=[function], type_ignores=[]), reserve.__file__, "exec"), namespace)
    if accepted:
        audit = namespace["_ground_prim_audit"]()
        assert audit["usd_type"] == prim_type and audit["native_cooked_geometry_readback_performed"] is False
    else:
        with pytest.raises(RuntimeError, match="collider|collision prim"):
            namespace["_ground_prim_audit"]()
    assert queried_paths == [reserve.GROUND_COLLIDER]


def test_motor_only_reset_event_allowed(runtime):
    env = make_env()
    env.event_manager.active_terms["reset"] = ["domain_randomization"]
    env.event_manager.terms["domain_randomization"] = NS(mode="reset", func=domain_randomization,
                                                         params={"profile": {"safe": True}})
    audit = initialize(env)
    assert len(audit["active_event_identity_audit"]) == 3


def test_installed_manager_initializes_exact_material_class(runtime, monkeypatch):
    class Material:
        pass
    tracking = importlib.import_module("training.tron1_tracking")
    monkeypatch.setattr(tracking.mdp, "randomize_rigid_body_material", Material)
    env = make_env()
    env.event_manager.terms["physics_material"].func = Material()
    audit = initialize(env)
    assert audit["active_event_identity_audit"][0]["initialized_exact_class_instance"] is True


@pytest.mark.parametrize("wrong", ["subclass", "lookalike"])
def test_material_class_subclass_or_name_lookalike_rejected(wrong, runtime, monkeypatch):
    class Material:
        pass
    class Subclass(Material):
        pass
    Lookalike = type("Material", (), {})
    monkeypatch.setattr(importlib.import_module("training.tron1_tracking").mdp,
                        "randomize_rigid_body_material", Material)
    env = make_env()
    env.event_manager.terms["physics_material"].func = Subclass() if wrong == "subclass" else Lookalike()
    with pytest.raises(RuntimeError, match="identity"):
        initialize(env)


@pytest.mark.parametrize("name", ["mass", "gravity", "push", "center_of_mass", "future_unknown"])
def test_mass_gravity_or_future_event_rejected(name, runtime):
    env = make_env()
    env.event_manager.active_terms["reset"] = [name]
    env.event_manager.terms[name] = NS(mode="reset", func=domain_randomization, params={})
    with pytest.raises(RuntimeError, match="Unsafe"):
        initialize(env)


@pytest.mark.parametrize("change", [
    lambda e: setattr(e.event_manager.terms["physics_material"], "func", lambda *a: None),
    lambda e: setattr(e.event_manager.terms["legacy_joint_friction"], "func", lambda *a: None),
    lambda e: setattr(e.event_manager.terms["legacy_joint_friction"], "mode", "reset"),
    lambda e: e.event_manager.active_terms["startup"].remove("legacy_joint_friction"),
    lambda e: e.event_manager.active_terms["startup"].append("physics_material"),
])
def test_same_label_wrong_identity_or_event_mode_rejected(change, runtime):
    env = make_env()
    change(env)
    with pytest.raises(RuntimeError):
        initialize(env)


@pytest.mark.parametrize("params", [{}, {"profile": {}}, {"profile": {"safe": True}, "mass": 1.}])
def test_unsafe_motor_event_configuration_rejected(params, runtime):
    env = make_env()
    env.event_manager.active_terms["reset"] = ["domain_randomization"]
    env.event_manager.terms["domain_randomization"] = NS(mode="reset", func=domain_randomization, params=params)
    with pytest.raises((RuntimeError, ValueError)):
        initialize(env)


def test_reward_cannot_implicitly_initialize(runtime):
    env = make_env()
    with pytest.raises(RuntimeError, match="explicit"):
        reserve.wheel_ground_load_reserve_penalty(env, .2)
    assert env.scene.robot.root_physx_view.mass_reads == runtime.reads == 0


@pytest.mark.parametrize("change", [
    lambda e: setattr(e, "num_envs", 1),
    lambda e: setattr(e, "device", "cuda:4"),
    lambda e: setattr(e.cfg.sim, "gravity", (0., 0., -10.)),
    lambda e: setattr(e.scene, "robot", copy.copy(e.scene.robot)),
    lambda e: e.scene.sensors.__setitem__("wheel_L_ground", copy.copy(e.scene.sensors["wheel_L_ground"])),
    lambda e: setattr(e.scene.sensors["wheel_L_ground"].cfg, "history_length", 1),
    lambda e: e.scene.sensors["wheel_L_ground"].data.force_matrix_w.value.fill(np.inf),
    lambda e: e.event_manager.active_terms.__setitem__("interval", ["gravity"]),
    lambda e: setattr(e.cfg, "decimation", 8),
])
def test_runtime_stale_environment_rejected(change, runtime):
    env = make_env()
    initialize(env)
    change(env)
    with pytest.raises((RuntimeError, ValueError)):
        reserve.wheel_ground_load_reserve_penalty(env, .2)


@pytest.mark.parametrize("scale", [.1, True, np.nan, np.inf, "0.2"])
def test_runtime_stale_or_malformed_scale_rejected(scale, runtime):
    env = make_env()
    initialize(env)
    with pytest.raises((RuntimeError, ValueError)):
        reserve.wheel_ground_load_reserve_penalty(env, scale)


@pytest.mark.parametrize("change", [
    lambda c: setattr(c.mg_n, "device", Device("cpu")),
    lambda c: setattr(c.mg_n, "shape", (1,)),
    lambda c: setattr(c.mg_n, "dtype", np.dtype("float64")),
    lambda c: c.mg_n.value.fill(np.nan),
    lambda c: c.mg_n.value.fill(0.),
])
def test_runtime_invalid_cached_mg_rejected(change, runtime):
    env = make_env()
    initialize(env)
    change(getattr(env, reserve.CACHE_ATTRIBUTE))
    with pytest.raises(RuntimeError):
        reserve.wheel_ground_load_reserve_penalty(env, .2)


def test_no_eager_torch_isaac_or_hot_host_array_conversion():
    tree = ast.parse(Path(reserve.__file__).read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert not any(n.name.startswith(("torch", "isaac", "omni", "pxr")) for n in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not node.module.startswith(("torch", "isaac", "omni", "pxr"))
    reward = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "wheel_ground_load_reserve_penalty")
    assert not {"cpu", "numpy", "item", "tolist", "get_masses", "get_gravity"} & {
        n.attr for n in ast.walk(reward) if isinstance(n, ast.Attribute)}
    assert not {"step", "reset", "update"} & {n.attr for n in ast.walk(reward) if isinstance(n, ast.Attribute)}
