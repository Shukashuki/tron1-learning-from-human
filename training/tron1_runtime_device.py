"""Opt-in, read-only native PhysX/CUDA device consistency audit.

Importing this module does not import Torch, Carb, or Isaac. This checks device
readbacks, not GPU exclusivity: a process context on a GPU is not proof that
physics executes there. No simulator state, actions, or observations are written.
"""
from __future__ import annotations

from numbers import Integral
import os
import re


def _cuda_device(value, label):
    if not isinstance(value, str) or re.fullmatch(r"cuda:(0|[1-9][0-9]*)", value) is None:
        raise ValueError(f"{label} must be an explicit cuda:N device")
    return int(value.partition(":")[2])


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{label} must be a nonnegative, non-boolean integer")
    return int(value)


def validate_device_audit(values, *, required_unmasked=True):
    """Validate pure readbacks and return JSON-serializable audit metadata.

    This checks native/tensor CUDA ordinal consistency only. Even unmasked CUDA
    ordinals do not prove an nvidia-smi index or pinned physical GPU UUID mapping.
    Unknown settings (including None for multi-GPU mode) fail closed.
    """
    if type(required_unmasked) is not bool:
        raise ValueError("required_unmasked must be a boolean")
    if not isinstance(values, dict):
        raise ValueError("Device audit requires a readback object")
    requested = values.get("requested_device")
    index = _cuda_device(requested, "requested_device")
    devices = ("environment_device", "physics_sim_view_frontend_device",
               "robot_native_transform_tensor_device")
    out = {"schema_version": 1, "status": "verified", "requested_device": requested}
    for name in devices:
        if _cuda_device(values.get(name), name) != index:
            raise ValueError(f"{name} differs from requested_device")
        out[name] = values[name]
    ordinals = ("native_physics_backend_device_ordinal", "torch_current_device",
                "physics_cudaDevice_setting")
    for name in ordinals:
        ordinal = _integer(values.get(name), name)
        if ordinal != index:
            raise ValueError(f"{name} differs from requested_device")
        out[name] = ordinal
    mode = _integer(values.get("physics_sceneMultiGPUMode_setting"),
                    "physics_sceneMultiGPUMode_setting")
    if mode != 0:
        raise ValueError("Native scene multi-GPU mode must be explicitly zero")
    out["physics_sceneMultiGPUMode_setting"] = mode
    if "cuda_visible_devices" not in values:
        raise ValueError("CUDA_VISIBLE_DEVICES readback is required, even when unset")
    visible = values["cuda_visible_devices"]
    if visible is not None and not isinstance(visible, str):
        raise ValueError("CUDA_VISIBLE_DEVICES readback must be a string or None")
    if required_unmasked and visible is not None:
        raise ValueError("Unmasked device audit requires CUDA_VISIBLE_DEVICES to be unset")
    return {**out, "cuda_visible_devices": visible, "required_unmasked": required_unmasked,
            "device_index": index,
            "index_scope": "unmasked_cuda_ordinal" if visible is None else "cuda_visible_logical_gpu",
            "native_and_tensor_device_consistency_verified": True,
            "physical_mapping_verified": False, "exclusivity_verified": False}


def validate_simulation_device(env, *, required_unmasked=True, requested_device=None):
    """Read native/tensor/settings device fields without stepping or changing them.

    Pass the launch CLI device as requested_device to check it independently of
    env.device. Without it, env.device is the declared request. CUDA-only: there
    is no implicit CPU fallback or current-device selection.
    """
    import torch
    import carb

    environment_device = str(env.device)
    view = env.sim.physics_sim_view
    settings = carb.settings.get_settings()
    # TRON1 is an ArticulationView, not the RigidBodyView used by isolated
    # wheel benchmarks. Its native transform readback has a different API.
    transforms = env.scene["robot"].root_physx_view.get_root_transforms()
    return validate_device_audit({
        "requested_device": environment_device if requested_device is None else requested_device,
        "environment_device": environment_device,
        "native_physics_backend_device_ordinal": view.device_ordinal,
        "physics_sim_view_frontend_device": str(view.device),
        "torch_current_device": torch.cuda.current_device(),
        "robot_native_transform_tensor_device": str(transforms.device),
        "physics_cudaDevice_setting": settings.get("/physics/cudaDevice"),
        "physics_sceneMultiGPUMode_setting": settings.get("/physics/sceneMultiGPUMode"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }, required_unmasked=required_unmasked)
