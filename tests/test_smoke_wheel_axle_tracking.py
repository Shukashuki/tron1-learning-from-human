"""Synthetic CPU fixture/source tests, never an actual GPU or physics PASS."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import types
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_wheel_axle_tracking.py"
spec = importlib.util.spec_from_file_location("axle_smoke_under_test", SCRIPT)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)
from training import tron1_wheel_axle_tracking as axle


def quaternion_x(angles):
    value = np.zeros(np.shape(angles) + (4,), dtype=np.float64)
    value[..., 0] = np.cos(np.asarray(angles) / 2)
    value[..., 1] = np.sin(np.asarray(angles) / 2)
    return value


def evidence(count=8):
    angles = np.broadcast_to(np.array([.2, .4]), (16, count, 2)).copy()
    actual = quaternion_x(angles).astype(np.float32)
    reference = np.zeros_like(actual); reference[..., 0] = 1
    expected = smoke.numpy_axle_reward(reference, actual)
    return {"axle_reference_quaternion_wxyz": reference,
            "axle_actual_quaternion_wxyz": actual,
            "axle_weighted_reward_density": expected.astype(np.float32),
            "axle_integrated_reward_contribution": (expected * .02).astype(np.float32)}


def old_reward(environment, **params):
    raise AssertionError("Metadata tests must not compute any reward")


def rewards():
    return NS(motion_body_ori=NS(func=old_reward, weight=1., params={"command_name": "motion", "std": .4}),
              drift_wheel_ground_load_reserve=NS(func=old_reward, weight=-1., params={"scale_fraction": .2}),
              zero=NS(func=old_reward, weight=0., params={}), inactive=None)


class Manager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.active_terms = [name for name, term in vars(cfg).items()
                             if term is not None and term.weight != 0]
        self._step_reward = np.zeros((8, len(self.active_terms)))
    def get_term_cfg(self, name):
        return getattr(self.cfg, name)


def configured():
    baseline = rewards(); current = copy.deepcopy(baseline)
    current.drift_wheel_axle_orientation = NS(func=axle.wheel_axle_orientation_tracking,
                                             weight=1., params={"command_name": "motion", "std": .4})
    return baseline, current


def main_tree():
    return next(node for node in ast.parse(SCRIPT.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "main")


def execute_definition(node, namespace):
    module = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]))
    exec(compile(module, "<actual-fixture-source>", "exec"), namespace)


def test_import_is_inert_no_torch_isaac_mujoco_or_files(tmp_path):
    code = ("import importlib.util,sys; "
            f"s=importlib.util.spec_from_file_location('inert',{str(SCRIPT)!r}); "
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m); "
            "assert not any(k.startswith(('torch','isaac','mujoco','omni')) for k in sys.modules)")
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())


def test_original620_source_is_unchanged():
    assert hashlib.sha256((ROOT / "scripts/smoke_dense_ground_load.py").read_bytes()).hexdigest() == (
        "620c8f2906114cf4cf7aa375413d43034b416acab091ca144a792801436c3d08")


@pytest.mark.parametrize("count", [8, 2048])
def test_independent_saved_LAST_replay_and_dt_integration(count):
    arrays = evidence(count)
    result = smoke.validate_saved_axle_evidence(arrays, count)
    assert result["status"] == "recorded_LAST_quaternion_arithmetic_verified"
    assert result["absolute_tolerance"] == 1e-6
    assert result["cpu_bit_exact_claimed"] is False
    assert result["physics_or_actor_executed"] is False
    known = math.exp(-(.2**2 + .4**2) / 2 / .4**2)
    assert np.max(np.abs(arrays["axle_weighted_reward_density"] - known)) < 1e-7
    assert np.max(np.abs(arrays["axle_integrated_reward_contribution"] - .02 * known)) < 1e-8


@pytest.mark.parametrize("key", ["axle_weighted_reward_density", "axle_integrated_reward_contribution"])
def test_density_and_integrated_units_not_interchangeable(key):
    arrays = evidence()
    arrays[key] *= .02 if key.endswith("density") else 50.
    with pytest.raises(ValueError, match="exceeds"):
        smoke.validate_saved_axle_evidence(arrays, 8)


@pytest.mark.parametrize("bad", [None, np.zeros((8, 2, 4)), np.ones((8, 3, 4)),
                                 np.ones((8, 2, 4), dtype=np.int64), np.full((8, 2, 4), np.nan)])
def test_invalid_quaternion_arrays_fail_closed(bad):
    with pytest.raises(ValueError):
        smoke.numpy_axle_reward(bad, bad)


def test_directed_reversal_is_not_unoriented_match():
    reference = np.zeros((1, 2, 4)); reference[..., 0] = 1
    actual = quaternion_x(np.full((1, 2), math.pi))
    assert smoke.numpy_axle_reward(reference, actual)[0] < 1e-20
    assert np.array_equal(smoke.numpy_axle_reward(reference, -3 * reference), np.ones(1))


@pytest.mark.parametrize("count,device", [(8, "cuda:0"), (2048, "cuda:7")])
def test_valid_explicit_cli_not_hardcoded_gpu(count, device):
    smoke.validate_cli(NS(num_envs=count, device=device, protocol_sha256="a" * 64))


@pytest.mark.parametrize("count,device,sha", [(True, "cuda:5", "a"*64), (16, "cuda:5", "a"*64),
    (8, "cuda", "a"*64), (8, "cpu", "a"*64), (8, "cuda:05", "a"*64),
    (8, "cuda:8", "a"*64), (8, "cuda:-1", "a"*64), (8, "cuda:2", "A"*64),
    (8, "cuda:2", "not-a-sha")])
def test_bad_cli_values_rejected(count, device, sha):
    with pytest.raises(ValueError):
        smoke.validate_cli(NS(num_envs=count, device=device, protocol_sha256=sha))


def test_all_original_live_terms_and_exact_unwrapped_axle_identity():
    baseline, current = configured()
    manager = Manager(current)
    ledger = smoke.validate_live_ledger(manager, baseline, axle.wheel_axle_orientation_tracking)
    assert len(ledger) == 3
    assert current.drift_wheel_axle_orientation.func is axle.wheel_axle_orientation_tracking
    def capture(*args):
        raise AssertionError("Metadata only")
    current.drift_wheel_ground_load_reserve.func = capture
    smoke.assert_armed_ledger(manager, ledger, capture, axle.wheel_axle_orientation_tracking)
    # A deliberately forged metadata name is not actual function identity.
    def forged(*args):
        raise AssertionError("Must be rejected, never called")
    forged.__module__ = axle.wheel_axle_orientation_tracking.__module__
    forged.__qualname__ = axle.wheel_axle_orientation_tracking.__qualname__
    current.drift_wheel_axle_orientation.func = forged
    with pytest.raises(ValueError):
        smoke.assert_armed_ledger(manager, ledger, capture, axle.wheel_axle_orientation_tracking)


@pytest.mark.parametrize("mutation", ["old_weight", "old_weight_type", "old_param", "new_weight", "new_std", "new_function", "second_term"])
def test_live_ledger_rejects_any_other_reward_change(mutation):
    baseline, current = configured()
    if mutation == "old_weight": current.motion_body_ori.weight = 2.
    elif mutation == "old_weight_type": current.motion_body_ori.weight = 1
    elif mutation == "old_param": current.motion_body_ori.params["std"] = .3
    elif mutation == "new_weight": current.drift_wheel_axle_orientation.weight = 2.
    elif mutation == "new_std": current.drift_wheel_axle_orientation.params["std"] = .3
    elif mutation == "new_function": current.drift_wheel_axle_orientation.func = old_reward
    else: current.second = NS(func=old_reward, weight=1., params={})
    with pytest.raises(ValueError):
        smoke.validate_live_ledger(Manager(current), baseline, axle.wheel_axle_orientation_tracking)


def test_metadata_only_resolve_private_baseline_selection_not_live_config():
    class SceneEntityCfg:
        def __init__(self): self.body_names = ["wheel_L_Link"]; self.body_ids = slice(None)
        def resolve(self, scene):
            assert scene is marker
            self.body_ids = [8]
    SceneEntityCfg.__module__ = "isaaclab.managers.scene_entity_cfg"
    marker = object()
    live = rewards(); live.motion_body_ori.params["asset_cfg"] = SceneEntityCfg()
    baseline = copy.deepcopy(live)
    before = smoke.configuration_ledger(live)
    smoke.resolve_baseline_parameters(baseline, marker)
    assert smoke.configuration_ledger(live) == before
    assert baseline.motion_body_ori.params["asset_cfg"].body_ids == [8]
    assert isinstance(live.motion_body_ori.params["asset_cfg"].body_ids, slice)


def test_stored_reward_is_read_only_weighted_density_no_reward_execution():
    baseline, current = configured(); manager = Manager(current)
    manager._step_reward[:] = np.arange(24).reshape(8, 3)
    before = manager._step_reward.copy()
    value = smoke.stored_density(manager, axle.TERM_NAME, 8).copy()
    assert np.array_equal(value, before[:, 2]) and np.array_equal(manager._step_reward, before)
    with pytest.raises(ValueError): smoke.stored_density(manager, axle.TERM_NAME, 2048)


class Array(np.ndarray):
    def __new__(cls, value): return np.asarray(value).view(cls)
    def clone(self): return Array(self.copy())
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return np.asarray(self)


def test_actual_observer_AST_saves_LAST_before_target_advance_and_reset_without_reward_calls():
    # Execute the actual nested observer, with two synthetic environments and
    # the2048 observer mode (therefore no raw point API). This is not simulation.
    observe = next(n for n in ast.walk(main_tree()) if isinstance(n, ast.FunctionDef) and n.name == "observe")
    q = Array(np.zeros((2, 9, 4))); q[..., 0] = 1
    command = NS(body_quat_relative_w=q.clone(), robot_body_quat_w=q.clone())
    state = {"active": True, "samples": [], "last_counter": 0, "last_native": (100, 2.), "last_quats": None}
    env = NS(_sim_step_counter=0, sim=NS(index=100, time=2.))
    calls = []
    def update(*args, **kwargs): calls.append("original_scene_update"); return "original-result"
    force = Array(np.zeros((2, 1, 1, 3)))
    namespace = {"original_update": update, "current": state, "env": env,
        "_native_clock": lambda sim: (sim.index, sim.time), "_float32": lambda v: float(np.float32(v)),
        "math": math, "args": NS(num_envs=2048), "torch": NS(stack=lambda values, dim: Array(np.stack(values, axis=dim))),
        "cache": NS(sensors=[NS(data=NS(force_matrix_w=force)) for _ in range(2)]),
        "command": command, "wheel_indices": [7, 8], "raw_vectors": [], "counters": [], "native_times": []}
    execute_definition(observe, namespace)
    for count in range(1, 5):
        env._sim_step_counter = count; env.sim.index += 1; env.sim.time += float(np.float32(.005))
        command.robot_body_quat_w[:, 7:9] = quaternion_x(np.full((2, 2), count * .05))
        assert namespace["observe"](None, .005) == "original-result"
        assert (state["last_quats"] is None) is (count < 4)
    saved_reference, saved_actual = state["last_quats"]
    command.body_quat_relative_w[:] = 99  # later command.compute
    command.robot_body_quat_w[:] = 77   # later auto-reset
    assert np.all(saved_reference[..., 0] == 1) and np.all(saved_reference[..., 1:] == 0)
    assert np.allclose(saved_actual, quaternion_x(np.full((2, 2), .2)))
    assert calls == ["original_scene_update"] * 4 and len(state["samples"]) == 4


def test_actual_source_preserves_force_probe_and_no_axle_reward_wrapper_extra_compute_or_RNG():
    node = main_tree(); all_calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
    sets = [n for n in all_calls if isinstance(n.func, ast.Attribute) and n.func.attr == "set_term_cfg"]
    assert len(sets) == 1 and ast.literal_eval(sets[0].args[0]) == "drift_wheel_ground_load_reserve"
    attrs = [n.func.attr for n in all_calls if isinstance(n.func, ast.Attribute)]
    assert not set(attrs) & {"compute", "learn", "mj_step", "load_state_dict", "rand", "randn"}
    assert attrs.count("step") == 1 and attrs.count("reset") == 2
    installed = next(n for n in node.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                     and isinstance(n.value.func, ast.Name) and n.value.func.id == "validate_installed_sources")
    import_app = next(n for n in node.body if isinstance(n, ast.ImportFrom) and n.module == "isaaclab.app")
    assert installed.lineno < import_app.lineno
    # Historical direct force-return alias probe stays intact.
    source = SCRIPT.read_text()
    for snippet in ("costs.append(value)", "return value", "torch.equal(expected, current[\"cost\"])",
                    "zip(costs[:-1], windows)", "if args.num_envs == 8:", "wrapped.step(torch.zeros(args.num_envs, 8"):
        assert snippet in source


def test_actual_science_snapshots_existing_resolved_baseline_without_extra_state_calls():
    tree = main_tree()
    pair = next((key, value) for node in ast.walk(tree) if isinstance(node, ast.Dict)
                for key, value in zip(node.keys, node.values)
                if isinstance(key, ast.Constant) and key.value == "baseline_dense_reward_ledger_resolved")
    value = pair[1]
    assert isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
    assert value.func.id == "configuration_ledger" and len(value.args) == 1 and not value.keywords
    assert ast.unparse(value.args[0]) == "baseline_cfg.rewards"
    resolves = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "resolve_baseline_parameters"]
    assert len(resolves) == 1 and resolves[0].lineno < value.lineno
    baseline = rewards(); before = copy.deepcopy(smoke.configuration_ledger(baseline))
    expression = ast.fix_missing_locations(ast.Expression(body=copy.deepcopy(value)))
    result = eval(compile(expression, "<actual-science-baseline-snapshot>", "eval"),
                  {"configuration_ledger": smoke.configuration_ledger, "baseline_cfg": NS(rewards=baseline)})
    assert result == before and smoke.configuration_ledger(baseline) == before


def test_expected_installed_source_bytes_checked_without_importing_sdk(monkeypatch, tmp_path):
    sdk = tmp_path / "isaaclab"; sdk.mkdir(); (sdk / "__init__.py").touch()
    pins = {}
    for relative in smoke.INSTALLED_SOURCE_SHA256:
        path = sdk / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("synthetic-source")
        pins[relative] = smoke.sha(path)
    monkeypatch.setattr(smoke.importlib.util, "find_spec", lambda name: NS(origin=str(sdk / "__init__.py")))
    monkeypatch.setattr(smoke, "INSTALLED_SOURCE_SHA256", pins)
    assert smoke.validate_installed_sources() == pins
    (sdk / "managers/reward_manager.py").write_text("different")
    with pytest.raises(RuntimeError): smoke.validate_installed_sources()


@pytest.mark.parametrize("mode", ["num_envs", "device", "mask", "protocol_sha", "existing_output"])
def test_rejected_actual_main_CLI_never_launches_app_or_overwrites(monkeypatch, tmp_path, mode):
    app_module = types.ModuleType("isaaclab.app")
    class AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--device", default="cuda:5")
            parser.add_argument("--headless", action="store_true")
        def __init__(self, *args, **kwargs):
            raise AssertionError("Rejected CLI must not launch an AppLauncher")
    app_module.AppLauncher = AppLauncher
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)
    monkeypatch.setattr(smoke, "validate_installed_sources", lambda: smoke.INSTALLED_SOURCE_SHA256.copy())
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    motion, asset = tmp_path / "motion.npz", tmp_path / "asset.usd"
    motion.write_bytes(b"synthetic-no-physics"); asset.write_bytes(b"synthetic-no-physics")
    protocol = tmp_path / "prospective.json"
    protocol.write_text(json.dumps({"trial_id": "trial017",
        "task_profile_sha256": smoke.sha(ROOT / "config/car_drift_wheel_axle_task.json"),
        "unchanged": {"motion_sha256": smoke.sha(motion), "robot_asset_sha256": smoke.sha(asset)},
        "expected_historical_source_sha256": {}, "expected_new_source_sha256": {},
        "installed_isaaclab_manager_sources": {"source_sha256": smoke.INSTALLED_SOURCE_SHA256}}))
    output = tmp_path / "output"
    args = ["smoke", "--motion-file", str(motion), "--asset-path", str(asset),
            "--task-profile", str(ROOT / "config/car_drift_wheel_axle_task.json"),
            "--protocol", str(protocol), "--protocol-sha256", smoke.sha(protocol),
            "--output-dir", str(output), "--num-envs", "8", "--device", "cuda:5"]
    if mode == "num_envs": args[args.index("--num-envs") + 1] = "16"
    elif mode == "device": args[args.index("--device") + 1] = "cpu"
    elif mode == "mask": monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    elif mode == "protocol_sha": args[args.index("--protocol-sha256") + 1] = "0" * 64
    else:
        output.mkdir(); (output / "user-owned").write_text("preserve")
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises((ValueError, SystemExit)):
        smoke.main()
    if mode == "existing_output":
        assert (output / "user-owned").read_text() == "preserve"
        assert [p.name for p in output.iterdir()] == ["user-owned"]
    else:
        assert not output.exists()


def test_GPU_invariance_routine_arithmetic_with_fake_tensor_CPU_only():
    fake_torch = NS(tensor=lambda value, device: Array(value))
    fake_axle = NS(wheel_axle_orientation_reward=lambda q, p, std: Array(smoke.numpy_axle_reward(q, p)))
    arrays, audit = smoke.deterministic_gpu_invariance(fake_torch, np, fake_axle, "synthetic-not-GPU")
    assert arrays["invariance_reward"].shape == (4, 2)
    assert audit["new_environment_reward_manager_physics_reset_observation_rng_calls"] == 0
    assert audit["independent_local_Y_spin"] and audit["common_world_yaw"]
    # The returned producer label is a proposed schema, not runtime evidence.


@pytest.mark.parametrize("conflict,fail_env,fail_handler", [
    (False, False, False), (True, False, False), (True, True, False),
    (False, True, False), (False, False, True)])
def test_actual_finally_always_closes_env_and_app_preserves_ownership_failure(conflict, fail_env, fail_handler):
    block = next(n for n in main_tree().body if isinstance(n, ast.Try) and n.finalbody)
    events = []
    observer, original, unknown = object(), object(), object()
    scene = NS(update=unknown if conflict else observer)
    def close_env():
        events.append("environment")
        if fail_env: raise OSError("environment failure")
    def close_handler():
        events.append("sampler")
        if fail_handler: raise OSError("sampler failure")
        return {"closed": True}
    namespace = {"handler": NS(close=close_handler), "env": NS(scene=scene, close=close_env),
        "observer_wrapper": observer, "previous_update_attribute": original, "_MISSING": smoke._MISSING,
        "app": NS(close=lambda **kwargs: events.append("app")), "output": Path("not-written"),
        "new_json": lambda *args: events.append("durable_teardown"),
        "close_environment_preserving_pending": smoke.close_environment_preserving_pending}
    node = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(block.finalbody), type_ignores=[]))
    if conflict or fail_env or fail_handler:
        expected = RuntimeError if conflict else OSError
        with pytest.raises(expected) as error:
            exec(compile(node, "<actual-fixture-finally>", "exec"), namespace)
        if conflict:
            assert "unknown scene observer" in str(error.value)
            assert scene.update is unknown  # never overwrite an unknown wrapper
    else:
        exec(compile(node, "<actual-fixture-finally>", "exec"), namespace)
        assert scene.update is original
    assert events[:2] == ["sampler", "environment"] and events[-1] == "app"


def test_real_Torch_CPU_math_via_installed_interpreter_no_CUDA():
    code = f"""
import importlib.util,sys,numpy as np,torch
sys.path.insert(0,{str(ROOT)!r})
spec=importlib.util.spec_from_file_location('smoke',{str(SCRIPT)!r})
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from training import tron1_wheel_axle_tracking as axle
assert not torch.cuda.is_initialized()
arrays,audit=m.deterministic_gpu_invariance(torch,np,axle,'cpu')
assert np.allclose(arrays['invariance_reward'],m.numpy_axle_reward(arrays['invariance_reference_quaternion_wxyz'],arrays['invariance_actual_quaternion_wxyz']),rtol=0,atol=1e-6)
assert not torch.cuda.is_initialized()
"""
    python = Path("/home/roy/miniconda3/envs/retarget_sim/bin/python")
    assert python.is_file(), "Required actual Torch CPU arithmetic interpreter missing"
    result = subprocess.run([str(python), "-B", "-c", code], cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
