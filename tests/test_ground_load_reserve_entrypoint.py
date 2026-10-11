"""Opt-in load-reserve integration; real AST/configuration, no Isaac or physics."""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import sys
import types
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from training import tron1_drift as drift
from eval_tracking_mujoco import read_contract
from test_car_drift_task import (
    configuration_inputs, extract_configuration_factory, material_audit_environment,
    task_profile,
)

KEY = "wheel_ground_load_reserve"
MODULE_PATH = "training/tron1_ground_load_reserve.py"
SETTINGS = {"weight": -.4, "scale_fraction": .05}


def protected_nodes(filename):
    tree = ast.parse((ROOT / "scripts" / filename).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    return next(n for n in main.body if isinstance(n, ast.Try) and n.finalbody).body


def execute(nodes, namespace):
    tree = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(tree, "<actual-load-reserve-entrypoint>", "exec"), namespace)


def pin_node(filename):
    matches = [n for n in protected_nodes(filename) if isinstance(n, ast.If)
               and ast.unparse(n.test) == "task_profile is not None and 'wheel_ground_load_reserve' in task_profile"]
    assert len(matches) == 1
    return matches[0]


def audit_node(filename):
    return next(n for n in protected_nodes(filename) if isinstance(n, ast.If)
                and any(isinstance(c, ast.ImportFrom) and c.module == "training.tron1_drift"
                        and any(a.name == "audit_isaac_materials" for a in c.names)
                        for c in ast.walk(n)))


@pytest.mark.parametrize("filename", ["train_tracking.py", "eval_tracking.py"])
@pytest.mark.parametrize("profile", [None, {}, {"task": "car_drift"}])
def test_historical_entrypoints_add_no_source_fields_or_new_imports(filename, profile, monkeypatch):
    monkeypatch.setitem(sys.modules, "training.tron1_ground_load_reserve", None)
    manifest = {"source_sha256": {"historical.py": "historical-hash"}}
    report = {"historical": True}
    calls = []
    before = copy.deepcopy((manifest, report, profile))
    execute([pin_node(filename)], {"task_profile": profile, "manifest": manifest,
            "report": report, "ROOT": ROOT, "sha256": lambda p: calls.append(p)})
    assert (manifest, report, profile) == before
    assert calls == []


@pytest.mark.parametrize("filename", ["train_tracking.py", "eval_tracking.py"])
def test_explicit_entrypoint_pins_only_the_new_module(filename):
    profile = {KEY: copy.deepcopy(SETTINGS)}
    manifest = {"source_sha256": {"historical.py": "historical-hash"}}
    report = {"contact_material_audit": {"ground_load_reserve": {"startup": "checked"}}}
    before_profile = copy.deepcopy(profile)
    calls = []

    def digest(path):
        calls.append(path)
        return "exact-module-source-hash"

    execute([pin_node(filename)], {"task_profile": profile, "manifest": manifest,
            "report": report, "ROOT": ROOT, "sha256": digest})
    assert calls == [ROOT / MODULE_PATH]
    assert profile == before_profile
    assert report["contact_material_audit"]["ground_load_reserve"] == {"startup": "checked"}
    if filename == "train_tracking.py":
        assert manifest["source_sha256"] == {"historical.py": "historical-hash",
                                              MODULE_PATH: "exact-module-source-hash"}
        assert "ground_load_reserve_module_sha256" not in report
    else:
        assert report["ground_load_reserve_module_sha256"] == "exact-module-source-hash"
        assert manifest["source_sha256"] == {"historical.py": "historical-hash"}
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(pin_node(filename)))


@pytest.mark.parametrize("filename,policy_call", [("train_tracking.py", "learn"),
                                                  ("eval_tracking.py", "export_policy_as_jit")])
def test_startup_cache_audit_precedes_wrappers_and_policy_use(filename, policy_call):
    nodes = protected_nodes(filename)
    audit = audit_node(filename)
    calls = [n for n in ast.walk(ast.Module(body=nodes, type_ignores=[])) if isinstance(n, ast.Call)]
    policy = next(n for n in calls if (isinstance(n.func, ast.Name) and n.func.id == policy_call)
                  or (isinstance(n.func, ast.Attribute) and n.func.attr == policy_call))
    wrapped = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "RslRlVecEnvWrapper")
    assert audit.end_lineno < wrapped.lineno < policy.lineno
    assert "contact_material_audit = audit_isaac_materials(env, task_profile)" in ast.unparse(audit)
    # Initialization belongs to the material audit, not an extra step observer.
    source = ast.parse((ROOT / "training/tron1_drift.py").read_text())
    factory = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "audit_isaac_materials")
    reserve = next(n for n in factory.body if isinstance(n, ast.If)
                   and ast.unparse(n.test) == "'wheel_ground_load_reserve' in profile")
    assert "audit['ground_load_reserve'] = initialize_ground_load_reserve(env, profile['wheel_ground_load_reserve'])" in ast.unparse(reserve)


@pytest.mark.parametrize("filename,target", [("train_tracking.py", "manifest"),
                                             ("eval_tracking.py", "report")])
def test_existing_metadata_dictionary_includes_the_startup_audit(filename, target):
    assignment = next(n for n in protected_nodes(filename) if isinstance(n, ast.Assign)
                      and isinstance(n.value, ast.Dict)
                      and any(isinstance(t, ast.Name) and t.id == target for t in n.targets))
    value = next(v for k, v in zip(assignment.value.keys, assignment.value.values)
                 if isinstance(k, ast.Constant) and k.value == "contact_material_audit")
    assert isinstance(value, ast.Name) and value.id == "contact_material_audit"


@pytest.mark.parametrize("filename", ["train_tracking.py", "eval_tracking.py"])
@pytest.mark.parametrize("fail", [False, True])
def test_real_startup_audit_records_cache_or_aborts_before_policy_use(
        filename, fail, material_audit_environment, monkeypatch):
    from training import tron1_ground_load_reserve as reserve
    seen = []
    cache_audit = {"startup_mass_cache": "validated", "sampling": "last_substep"}

    def initialize(env, settings):
        seen.append(("initialize", copy.deepcopy(settings)))
        if fail:
            raise RuntimeError("unsafe startup mass cache")
        return copy.deepcopy(cache_audit)

    monkeypatch.setattr(reserve, "initialize_ground_load_reserve", initialize)
    profile = {**task_profile(), KEY: copy.deepcopy(SETTINGS)}
    namespace = {"env": material_audit_environment(500.), "task_profile": profile,
                 "contact_material_audit": None, "policy_use": lambda: seen.append("policy-use")}
    marker = ast.Expr(value=ast.Call(func=ast.Name(id="policy_use", ctx=ast.Load()), args=[], keywords=[]))
    if fail:
        with pytest.raises(RuntimeError, match="unsafe startup mass cache"):
            execute([audit_node(filename), marker], namespace)
        assert "policy-use" not in seen
        assert namespace["contact_material_audit"] is None
    else:
        execute([audit_node(filename), marker], namespace)
        assert namespace["contact_material_audit"]["ground_load_reserve"] == cache_audit
        assert seen == [("initialize", SETTINGS), "policy-use"]
    assert profile[KEY] == SETTINGS


def test_old_core_audit_and_contract_do_not_import_or_activate_reserve(material_audit_environment, monkeypatch):
    monkeypatch.setitem(sys.modules, "training.tron1_ground_load_reserve", None)
    contract = read_contract()
    before = copy.deepcopy(contract)
    profile = task_profile()
    combined = drift.apply_task_profile_to_contract(contract, profile)
    assert {k: v for k, v in combined.items() if k != "task_profile"} == before
    assert contract == before
    assert (contract["actor_dim"], contract["critic_dim"], contract["action_dim"]) == (51, 120, 8)
    assert contract["physics_dt"] == .005 and contract["decimation"] == 4
    assert "ground_load_reserve" not in drift.audit_isaac_materials(material_audit_environment(500.), profile)
    assert drift.install_drift_reward_sampling(object(), profile) is None


@pytest.fixture
def sensor_cfg(monkeypatch):
    sensors = types.ModuleType("isaaclab.sensors")

    def constructor(**kwargs):
        return NS(**{"track_contact_points": False, **kwargs})

    sensors.ContactSensorCfg = constructor
    monkeypatch.setitem(sys.modules, "isaaclab.sensors", sensors)
    return constructor


@pytest.mark.parametrize("eval_mode", [False, True])
def test_explicit_core_reward_and_train_eval_sensor_split(configuration_inputs, sensor_cfg, eval_mode, monkeypatch):
    from training import tron1_ground_load_reserve as reserve
    if eval_mode:
        tracking = types.ModuleType("training.tron1_tracking")
        tracking.NON_WHEEL_BODY_NAMES = read_contract()["orientation_body_names"]
        monkeypatch.setitem(sys.modules, "training.tron1_tracking", tracking)
    profile = {**task_profile(), KEY: copy.deepcopy(SETTINGS)}
    cfg = extract_configuration_factory()(*configuration_inputs, num_envs=4,
                                           task_profile=profile, eval_mode=eval_mode)
    reward = cfg.rewards.drift_wheel_ground_load_reserve
    assert reward.func is reserve.wheel_ground_load_reserve_penalty
    assert reward.weight == SETTINGS["weight"]
    assert reward.params == {"scale_fraction": SETTINGS["scale_fraction"]}
    wheels = [getattr(cfg.scene, name) for name in ("wheel_L_ground", "wheel_R_ground")]
    for sensor in wheels:
        assert sensor.filter_prim_paths_expr == ["/World/ground/terrain/GroundPlane/CollisionPlane"]
        assert sensor.update_period == 0. and sensor.history_length == 0
        assert sensor.track_contact_points is eval_mode
    nonwheel = [name for name in vars(cfg.scene) if name.startswith("nonwheel_ground_")]
    assert len(nonwheel) == (7 if eval_mode else 0)
    assert cfg.scene.contact_forces is not None
    assert cfg.sim.dt == .005 and cfg.decimation == 4
    assert cfg.scene.robot.actuators["legs"].stiffness == 500.
    assert cfg.scene.robot.actuators["legs"].damping == 10.
    assert drift.install_drift_reward_sampling(object(), profile) is None


def test_old_training_configuration_has_no_reserve_reward_or_sensors(configuration_inputs, monkeypatch):
    monkeypatch.setitem(sys.modules, "training.tron1_ground_load_reserve", None)
    cfg = extract_configuration_factory()(*configuration_inputs, num_envs=4, task_profile=task_profile())
    assert not hasattr(cfg.rewards, "drift_wheel_ground_load_reserve")
    assert not any(name in vars(cfg.scene) for name in ("wheel_L_ground", "wheel_R_ground"))
    assert not any(name.startswith("nonwheel_ground_") for name in vars(cfg.scene))
    assert cfg.scene.contact_forces is not None
