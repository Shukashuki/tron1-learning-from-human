"""Pure/native-readback stubs only: no app, real CUDA, or simulator launch."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.tron1_runtime_device import validate_device_audit, validate_simulation_device


def readbacks(index=3):
    device = f"cuda:{index}"
    return {"requested_device": device, "environment_device": device,
            "native_physics_backend_device_ordinal": index,
            "physics_sim_view_frontend_device": device, "torch_current_device": index,
            "robot_native_transform_tensor_device": device, "physics_cudaDevice_setting": index,
            "physics_sceneMultiGPUMode_setting": 0, "cuda_visible_devices": None}


@pytest.mark.parametrize("index", [0, 2, 3, 7])
def test_exact_unmasked_native_device_audit_is_serializable_and_read_only(index):
    values = readbacks(index)
    before = copy.deepcopy(values)
    audit = validate_device_audit(values)
    assert values == before
    assert audit["status"] == "verified" and audit["device_index"] == index
    assert audit["index_scope"] == "unmasked_cuda_ordinal"
    assert audit["native_and_tensor_device_consistency_verified"] is True
    assert audit["physical_mapping_verified"] is False
    assert audit["exclusivity_verified"] is False
    assert json.loads(json.dumps(audit, allow_nan=False)) == audit


@pytest.mark.parametrize("name", ["requested_device", "environment_device",
                                  "physics_sim_view_frontend_device", "robot_native_transform_tensor_device"])
@pytest.mark.parametrize("value", [None, True, False, "cuda", "cpu", "cuda:0", "cuda:-1", "cuda:03", "cuda:3 ", 3])
def test_missing_ambiguous_or_mismatched_devices_fail(name, value):
    with pytest.raises(ValueError):
        validate_device_audit({**readbacks(), name: value})


@pytest.mark.parametrize("name", ["native_physics_backend_device_ordinal", "torch_current_device",
                                  "physics_cudaDevice_setting"])
@pytest.mark.parametrize("value", [None, True, False, -1, 0, 2, 3., "3"])
def test_noninteger_unknown_or_wrong_native_ordinals_fail(name, value):
    with pytest.raises(ValueError):
        validate_device_audit({**readbacks(), name: value})


@pytest.mark.parametrize("value", [None, True, False, 1, -1, 0., "0"])
def test_multi_gpu_must_be_explicit_integer_zero(value):
    with pytest.raises(ValueError):
        validate_device_audit({**readbacks(), "physics_sceneMultiGPUMode_setting": value})


@pytest.mark.parametrize("value", ["", "3", "0,1,2,3", "GPU-example", "-1", True, []])
def test_any_cuda_visibility_mask_rejected_when_unmasked_required(value):
    with pytest.raises(ValueError):
        validate_device_audit({**readbacks(), "cuda_visible_devices": value})


def test_explicit_mask_opt_out_does_not_claim_physical_mapping_or_exclusivity():
    audit = validate_device_audit({**readbacks(), "cuda_visible_devices": "3"}, required_unmasked=False)
    assert audit["index_scope"] == "cuda_visible_logical_gpu"
    assert audit["physical_mapping_verified"] is False and audit["exclusivity_verified"] is False


@pytest.mark.parametrize("value", [None, [], True, "device"])
def test_nonobject_readback_inputs_fail(value):
    with pytest.raises(ValueError):
        validate_device_audit(value)


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_unmasked_option_cannot_be_coerced(value):
    with pytest.raises(ValueError):
        validate_device_audit(readbacks(), required_unmasked=value)


@pytest.mark.parametrize("name", list(readbacks()))
def test_every_required_readback_is_required(name):
    values = readbacks()
    del values[name]
    with pytest.raises(ValueError):
        validate_device_audit(values)


def test_imports_are_lazy_and_no_state_writes_or_device_selection():
    tree = ast.parse((ROOT / "training/tron1_runtime_device.py").read_text())
    top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = {alias.name for node in top_imports for alias in node.names}
    assert not names & {"torch", "carb", "isaaclab", "isaacsim"}
    called = {node.func.attr for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not called & {"set_device", "cpu", "numpy", "item", "step", "reset", "update",
                         "write_root_pose_to_sim", "write_root_velocity_to_sim", "apply_action"}


def runtime_env(monkeypatch, *, mask=None, native_index=3):
    calls = []
    monkeypatch.setitem(sys.modules, "torch", NS(cuda=NS(current_device=lambda: 3)))
    settings = {"/physics/cudaDevice": 3, "/physics/sceneMultiGPUMode": 0}
    monkeypatch.setitem(sys.modules, "carb", NS(settings=NS(get_settings=lambda: NS(get=settings.get))))
    if mask is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    def transforms():
        calls.append("get_root_transforms")
        return NS(device="cuda:3")
    env = NS(device="cuda:3", sim=NS(physics_sim_view=NS(device_ordinal=native_index, device="cuda:3")),
             scene={"robot": NS(root_physx_view=NS(get_root_transforms=transforms))})
    return env, calls


def test_runtime_reads_native_and_tensor_device_once_without_writing_environment(monkeypatch):
    env, calls = runtime_env(monkeypatch)
    native_before = dict(vars(env.sim.physics_sim_view))
    audit = validate_simulation_device(env, requested_device="cuda:3")
    assert not hasattr(env.scene["robot"].root_physx_view, "get_transforms")
    assert calls == ["get_root_transforms"]
    assert vars(env.sim.physics_sim_view) == native_before
    assert audit["requested_device"] == audit["robot_native_transform_tensor_device"] == "cuda:3"


def test_runtime_default_request_comes_from_environment_device(monkeypatch):
    env, _ = runtime_env(monkeypatch)
    assert validate_simulation_device(env)["requested_device"] == env.device


@pytest.mark.parametrize("change", ["requested", "native", "mask"])
def test_runtime_native_requested_or_visibility_mismatch_fails(monkeypatch, change):
    env, _ = runtime_env(monkeypatch, mask="3" if change == "mask" else None,
                         native_index=2 if change == "native" else 3)
    with pytest.raises(ValueError):
        validate_simulation_device(env, requested_device="cuda:2" if change == "requested" else "cuda:3")
