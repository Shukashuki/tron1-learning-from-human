"""Prospective trial019 tight-axle fixture; no actor or PPO.

The passive observer clones filtered forces at each actual native substep.
It replays LAST yaw-reanchored axle rewards from saved quaternions and independently
replays held force costs. For eight environments only, it compares
the filtered matrix to raw ground-normal point-force sums. Extra reads may
synchronize the host; this fixture is not a training performance benchmark.
"""
from __future__ import annotations

import argparse
import copy
import inspect
import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
if hashlib.sha256((ROOT / "scripts/smoke_worst_ground_load_tracking.py").read_bytes()).hexdigest() != (
        "8fa4ba0f3746bc7b4d3354eeaf2485a17866923edc44e883314decdc30665d3f"):
    raise RuntimeError("Frozen baseline018 fixture source mismatch")
_MISSING = object()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


INSTALLED_SOURCE_SHA256 = {
    "managers/reward_manager.py": "5d41d6a376cfb456849d861c3232694ea5662cc38970e2e2c7e5152f156d88d7",
    "managers/scene_entity_cfg.py": "1f187fb964c5ecb77865d2d24f2ff61468e667c4d4e170b2020145888a0eb689",
    "envs/manager_based_rl_env.py": "8ec2759541c8320ed725411d49564f5617b3ab6b9d340c860bededee13d557d4",
}


def validate_installed_sources():
    # find_spec of the top-level package locates sources without executing Kit.
    spec = importlib.util.find_spec("isaaclab")
    if spec is None or spec.origin is None:
        raise RuntimeError("The predeclared actual IsaacLab source installation is required")
    source_root = Path(spec.origin).resolve().parent
    actual = {name: sha(source_root / name) for name in INSTALLED_SOURCE_SHA256}
    if actual != INSTALLED_SOURCE_SHA256:
        raise RuntimeError("Actual installed RewardManager/scene-selection/lifecycle sources changed")
    return actual


def validate_cli(args):
    if (type(args.num_envs) is not int or args.num_envs not in (8, 2048)
            or type(args.device) is not str or not args.device.startswith("cuda:")
            or not args.device[5:].isdigit() or not 0 <= int(args.device[5:]) <= 7
            or args.device != "cuda:" + str(int(args.device[5:]))):
        raise ValueError("Fixture requires explicit unmasked cuda:0..7 and8 or2048 environments")
    value = args.protocol_sha256
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Exact lower-case SHA256 pin required")


def numpy_axle_reward(reference, actual):
    """Independent float64 rotation of local+Y; never imports Torch/Isaac."""
    import numpy as np
    reference, actual = np.asarray(reference), np.asarray(actual)
    if (reference.shape != actual.shape or reference.ndim < 3 or reference.shape[-2:] != (2, 4)
            or any(size == 0 for size in reference.shape)
            or not np.issubdtype(reference.dtype, np.floating) or not np.issubdtype(actual.dtype, np.floating)
            or not np.isfinite(reference).all() or not np.isfinite(actual).all()):
        raise ValueError("Finite matching wxyz quaternion arrays [...,2,4] required")
    axes = []
    for value in (reference, actual):
        q = value.astype(np.float64)
        lengths = np.linalg.norm(q, axis=-1, keepdims=True)
        if np.any(lengths <= 0) or not np.isfinite(lengths).all():
            raise ValueError("Nonzero finite quaternion norms required")
        q = q / lengths
        vector = np.broadcast_to(np.array([0., 1., 0.]), q.shape[:-1] + (3,))
        axis = vector + 2 * np.cross(q[..., 1:], np.cross(q[..., 1:], vector) + q[..., :1] * vector)
        axes.append(axis / np.linalg.norm(axis, axis=-1, keepdims=True))
    sine = np.linalg.norm(np.cross(*axes), axis=-1)
    cosine = np.clip((axes[0] * axes[1]).sum(axis=-1), -1., 1.)
    angle = np.arctan2(sine, cosine)
    return np.exp(-np.mean(angle * angle, axis=-1) / .2**2)


def validate_saved_axle_evidence(arrays, num_envs):
    import numpy as np
    if type(num_envs) is not int or num_envs not in (8, 2048):
        raise ValueError("Exact8/2048 fixture environment count required")
    q, p = [np.asarray(arrays[k]) for k in (
        "axle_reference_quaternion_wxyz", "axle_actual_quaternion_wxyz")]
    if q.shape != (16, num_envs, 2, 4) or p.shape != q.shape:
        raise ValueError("Exactly16 LAST quaternion windows required")
    expected = numpy_axle_reward(q, p)
    errors = {}
    for key, scale in (("axle_weighted_reward_density", 1.),
                       ("axle_integrated_reward_contribution", .02)):
        value = np.asarray(arrays[key])
        if value.shape != (16, num_envs) or not np.isfinite(value).all():
            raise ValueError("Finite original RewardManager stored rows required")
        error = float(np.max(np.abs(value.astype(np.float64) - expected * scale)))
        if error > 1e-6:
            raise ValueError("Independent axle replay exceeds1e-6")
        errors[key + "_maximum_absolute_error"] = error
    return {"status": "recorded_LAST_quaternion_arithmetic_verified", **errors,
            "absolute_tolerance": 1e-6, "cpu_bit_exact_claimed": False,
            "physics_or_actor_executed": False}


def _parameter_snapshot(value):
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Finite reward metadata required")
        return value
    if type(value) is list:
        return [_parameter_snapshot(x) for x in value]
    if type(value) is tuple:
        return {"tuple": [_parameter_snapshot(x) for x in value]}
    if type(value) is slice:
        return {"slice": [_parameter_snapshot(x) for x in (value.start, value.stop, value.step)]}
    if type(value) is dict and all(type(k) is str for k in value):
        return {k: _parameter_snapshot(v) for k, v in value.items()}
    cls = type(value)
    if cls.__name__ == "SceneEntityCfg" and cls.__module__ in (
            "isaaclab.managers", "isaaclab.managers.scene_entity_cfg"):
        if any(k.startswith("_") for k in vars(value)):
            raise ValueError("Only public stored scene-selection fields supported")
        return {"SceneEntityCfg": cls.__module__ + "." + cls.__qualname__,
                "fields": _parameter_snapshot(vars(value))}
    raise ValueError("Unsupported reward parameter type")


def _term_metadata(name, term):
    if (not callable(term.func) or isinstance(term.weight, bool)
            or type(term.weight) not in (float, int) or not math.isfinite(term.weight)):
        raise ValueError("Stable reward function and finite numeric weight required")
    return {"name": name, "function": term.func.__module__ + "." + term.func.__qualname__,
            "weight": term.weight, "params": _parameter_snapshot(term.params)}


def configuration_ledger(rewards):
    return [_term_metadata(name, value) for name, value in vars(rewards).items()
            if value is not None and hasattr(value, "func") and value.weight != 0]


def _same_metadata(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(_same_metadata(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(_same_metadata(a, b) for a, b in zip(left, right))
    return left == right


def assert_tight_only(baseline, current):
    by_name = {v["name"]: v for v in current}
    baseline_names = {v["name"] for v in baseline}
    if (len(by_name) != len(current) or len(baseline_names) != len(baseline)
            or set(by_name) != baseline_names
            or not {"drift_wheel_axle_orientation", "drift_wheel_ground_load_reserve"} <= baseline_names):
        raise ValueError("Exactly the same reward terms required")
    for previous in baseline:
        expected = copy.deepcopy(previous)
        if previous["name"] == "drift_wheel_axle_orientation":
            if (previous["function"] != "training.tron1_wheel_axle_tracking.wheel_axle_orientation_tracking"
                    or not _same_metadata(previous["params"], {"command_name": "motion", "std": .4})
                    or not _same_metadata(previous["weight"], 1.0)):
                raise ValueError("Held018 axle baseline identity/scale/weight required")
            expected["function"] = "training.tron1_tight_axle_tracking.wheel_axle_orientation_tracking"
            expected["params"]["std"] = .2
        if not _same_metadata(by_name[previous["name"]], expected):
            raise ValueError("Only axle callable and angular std .4 to .2 may change")


def resolve_baseline_parameters(rewards, scene):
    # Installed manager resolves only top-level SceneEntityCfg parameters.
    # Resolve private copied configurations, never live state or rewards.
    for term in vars(rewards).values():
        if term is None or not hasattr(term, "params"):
            continue
        for value in term.params.values():
            if type(value).__name__ == "SceneEntityCfg" and type(value).__module__ in (
                    "isaaclab.managers", "isaaclab.managers.scene_entity_cfg"):
                value.resolve(scene)


def validate_live_ledger(manager, baseline_rewards, axle_function):
    names = list(manager.active_terms)
    live = [_term_metadata(name, manager.get_term_cfg(name)) for name in names]
    assert_tight_only(configuration_ledger(baseline_rewards), live)
    for name in names:
        expected = axle_function if name == "drift_wheel_axle_orientation" else getattr(baseline_rewards, name).func
        if manager.get_term_cfg(name).func is not expected:
            raise ValueError("Actual RewardManager function identity changed")
    return live


def assert_armed_ledger(manager, canonical, force_capture, axle_function):
    # Check actual records, do not rewrite their identities to masquerade as
    # original terms. Exactly the known620 force probe is explicitly permitted.
    names = list(manager.active_terms)
    if names != [value["name"] for value in canonical]:
        raise ValueError("Actual active reward order changed")
    for expected in canonical:
        term = manager.get_term_cfg(expected["name"])
        actual = _term_metadata(expected["name"], term)
        if expected["name"] == "drift_wheel_ground_load_reserve":
            if (term.func is not force_capture or not _same_metadata(actual["weight"], expected["weight"])
                    or not _same_metadata(actual["params"], expected["params"])):
                raise ValueError("Unknown force telemetry wrapper/weight/parameter change")
        elif not _same_metadata(actual, expected) or (expected["name"] == "drift_wheel_axle_orientation" and term.func is not axle_function):
            raise ValueError("A live reward changed while the fixture was armed")


def stored_density(manager, name, num_envs):
    names = list(manager.active_terms)
    if len(set(names)) != len(names) or name not in names or tuple(manager._step_reward.shape) != (num_envs, len(names)):
        raise ValueError("Actual installed weighted per-second reward ledger layout mismatch")
    return manager._step_reward[:, names.index(name)]


def close_environment_preserving_pending(env):
    """Always try env.close; do not replace an existing ownership failure."""
    pending = sys.exc_info()[1]
    try:
        env.close()
    except BaseException as error:
        if pending is None:
            raise
        if hasattr(pending, "add_note"):
            pending.add_note("Environment close also failed: " + type(error).__name__ + ": " + str(error))


def deterministic_gpu_invariance(torch, np, axle, device):
    # Explicit constants, no random tensors or environment/reward-manager call.
    q = np.array([[[1., 0., 0., 0.], [1., 0., 0., 0.]],
                  [[.98, .1, .05, -.1], [.96, -.2, .1, .05]]], dtype=np.float32)
    p = np.array([[[.98, .2, 0., 0.], [.98, -.2, 0., 0.]],
                  [[.97, -.1, .15, .1], [.98, .05, -.1, -.15]]], dtype=np.float32)
    def mul(a, b):
        aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
        return np.concatenate((aw*bw-(av*bv).sum(-1, keepdims=True),
                               aw*bv+bw*av+np.cross(av, bv)), axis=-1).astype(np.float32)
    def rotation(axis, angles):
        value = np.zeros((2, 2, 4), dtype=np.float32)
        value[..., 0] = np.cos(angles/2)
        value[..., axis + 1] = np.sin(angles/2)
        return value
    spin = np.array([[.7, -1.2], [2.3, -3.1]], dtype=np.float32)
    yaw = rotation(2, np.array([[.4, .4], [-.9, -.9]], dtype=np.float32))
    cases = [(q, p), (mul(q, rotation(1, spin)), mul(p, rotation(1, -spin))),
             (-2*q, 3*p), (mul(yaw, q), mul(yaw, p))]
    rows = []
    for reference, actual in cases:
        value = axle.wheel_axle_orientation_reward(torch.tensor(reference, device=device),
                                                   torch.tensor(actual, device=device), .2)
        rows.append(value.detach().cpu().numpy().copy())
    values = np.asarray(rows)
    references, actuals = np.asarray([c[0] for c in cases]), np.asarray([c[1] for c in cases])
    if (not np.allclose(values, numpy_axle_reward(references, actuals), rtol=0., atol=1e-6)
            or not np.allclose(values, values[:1], rtol=0., atol=1e-6)):
        raise RuntimeError("Deterministic GPU quaternion/axis invariance failed")
    return {"invariance_reference_quaternion_wxyz": references,
            "invariance_actual_quaternion_wxyz": actuals, "invariance_reward": values}, {
        "status": "constant_GPU_tensor_invariance_verified", "absolute_tolerance": 1e-6,
        "independent_local_Y_spin": True, "quaternion_sign_and_scale": True, "common_world_yaw": True,
        "new_environment_reward_manager_physics_reset_observation_rng_calls": 0,
        "host_synchronization_performed": True}


def validate_saved_worst_cost_evidence(arrays, num_envs):
    """Independent float64 max-of-sums replay; not a new simulator call."""
    import numpy as np
    forces = np.asarray(arrays["force_matrix_world_n"])
    mg = np.asarray(arrays["mg_n"])
    if (forces.shape != (64, num_envs, 2, 3) or mg.shape != (num_envs,)
            or not np.isfinite(forces).all() or not np.isfinite(mg).all() or np.any(mg <= 0)):
        raise ValueError("Exactly64 refreshed force samples and positive per-env mg required")
    samples = np.exp(-np.maximum(forces[:, :, :, 2].astype(np.float64), 0.) /
                     (.20 * mg.astype(np.float64)[None, :, None])).sum(-1)
    expected = samples.reshape(16, 4, num_envs).max(1)
    errors = {}
    for key, scale in (("cost", 1.), ("ground_load_weighted_reward_density", -1.),
                       ("ground_load_integrated_reward_contribution", -.02)):
        value = np.asarray(arrays[key])
        if value.shape != (16, num_envs) or not np.isfinite(value).all():
            raise ValueError("Finite16 actual cost/RewardManager rows required")
        error = float(np.max(np.abs(value.astype(np.float64) - expected * scale)))
        if error > 1e-6:
            raise ValueError("Independent worst-load replay exceeds1e-6")
        errors[key + "_maximum_absolute_error"] = error
    return {"status": "recorded_max_of_two_wheel_sums_arithmetic_verified", **errors,
        "absolute_tolerance": 1e-6, "CPU_bit_exact_claimed": False,
        "new_physics_or_actor_executed": False}


def main():
    installed_sources = validate_installed_sources()
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--asset-path", type=Path, required=True)
    parser.add_argument("--task-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, choices=(8, 2048), default=8)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from training.tron1_tight_axle_trial import load_tight_axle_profile
    profile = load_tight_axle_profile(args.task_profile)
    validate_cli(args)
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        parser.error("Native physical CUDA ordinals require no mask")
    if sha(args.protocol) != args.protocol_sha256:
        parser.error("Prospective protocol SHA mismatch")
    protocol = json.loads(args.protocol.read_text())
    if protocol["installed_isaaclab_manager_sources"]["source_sha256"] != INSTALLED_SOURCE_SHA256:
        parser.error("Prospective installed manager source pins mismatch")
    if (protocol["trial_id"] != "trial019" or sha(args.task_profile) != protocol["task_profile_sha256"]
            or sha(args.motion_file) != protocol["unchanged"]["motion_sha256"]
            or sha(args.asset_path) != protocol["unchanged"]["robot_asset_sha256"]):
        parser.error("Fixture source/profile/reference identity mismatch")
    source_pins = {**protocol["expected_historical_source_sha256"], **protocol["expected_new_source_sha256"]}
    if not all(sha(ROOT / name) == value for name, value in source_pins.items()):
        parser.error("Current-source protocol pins mismatch")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("Fixture output must be new; never overwrite or restart")
    output.mkdir(parents=True, exist_ok=False)
    new_json(output / "launch_manifest.json", {
        "schema_version": 1, "trial_id": "trial019", "protocol_sha256": sha(args.protocol),
        "profile_sha256": sha(args.task_profile), "source_sha256": source_pins,
        "script_sha256": sha(__file__), "num_envs": args.num_envs, "device": args.device,
        "policy_steps": 16, "physical_samples": 64, "no_actor_or_PPO": True,
        "point_force_cross_check": args.num_envs == 8,
        "installed_isaaclab_source_sha256": installed_sources,
        "runtime_verified_by_this_metadata": False,
    })
    validate_installed_sources()  # Recheck after output and immediately before AppLauncher.
    app = env = handler = None
    observer_wrapper = original_update = None
    previous_update_attribute = _MISSING
    started = time.monotonic()
    try:
        app = AppLauncher(args).app
        import numpy as np
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from training.tron1_tracking import make_env_cfg
        from training.tron1_domain_randomization import default_profile
        from training.tron1_runtime_device import validate_simulation_device
        from training.tron1_drift import audit_isaac_materials
        from training.tron1_worst_ground_load_trial import configure_worst_ground_load_trial
        from training.tron1_tight_axle_trial import configure_tight_axle_trial, baseline_worst_profile
        from training.tron1_wheel_axle_trial import baseline_dense_profile
        from training import tron1_tight_axle_tracking as axle
        from training.tron1_worst_ground_load_substeps import (
            install_worst_ground_load_accumulator, substep_worst_wheel_ground_load_reserve_penalty,
        )
        from training import tron1_ground_load_reserve as reserve
        from training.tron1_substep_rewards import _native_clock, _float32
        from training.tron1_contact_slip import sample_raw_isaac_ground_contacts

        cfg = make_env_cfg(args.motion_file, args.asset_path, args.num_envs, args.device,
                           dr_profile=default_profile(42), task_profile=baseline_dense_profile())
        baseline_cfg = copy.deepcopy(cfg)
        configure_worst_ground_load_trial(baseline_cfg, baseline_worst_profile())
        baseline_ledger = configuration_ledger(baseline_cfg.rewards)
        configure_tight_axle_trial(cfg, profile)
        assert_tight_only(baseline_ledger, configuration_ledger(cfg.rewards))
        cfg.seed = 42
        env = ManagerBasedRLEnv(cfg=cfg)
        native_audit = validate_simulation_device(env, requested_device=args.device)
        manager_source = inspect.getsourcefile(type(env.reward_manager))
        if manager_source is None or sha(manager_source) != INSTALLED_SOURCE_SHA256["managers/reward_manager.py"]:
            raise RuntimeError("Actual installed RewardManager source pin mismatch")
        metadata_clock = (env._sim_step_counter, _native_clock(env.sim))
        resolve_baseline_parameters(baseline_cfg.rewards, env.scene)
        if metadata_clock != (env._sim_step_counter, _native_clock(env.sim)):
            raise RuntimeError("Copied baseline metadata resolution advanced physics")
        live_ledger = validate_live_ledger(env.reward_manager, baseline_cfg.rewards, axle.wheel_axle_orientation_tracking)
        invariance_arrays, invariance_audit = deterministic_gpu_invariance(torch, np, axle, args.device)
        command = env.command_manager.get_term("motion")
        body_names = list(command.cfg.body_names)
        if len(body_names) != 9 or set(body_names) != set(axle.BODY_NAMES):
            raise RuntimeError("Expected the named nine-body tracking subset")
        wheel_indices = [body_names.index(name) for name in axle.WHEEL_BODY_NAMES]
        material_audit = audit_isaac_materials(env, profile)
        cache = getattr(env, reserve.CACHE_ATTRIBUTE)
        mg = cache.mg_n.clone()
        windows, costs, raw_vectors, point_vectors, counters, native_times = [], [], [], [], [], []
        step_wall_times = []
        target_quats, actual_quats, axle_densities, axle_contributions = [], [], [], []
        retained_axle, window_native_baselines = [], []
        load_densities, load_contributions = [], []
        raw_contact_buffers = {name: [] for name in ("normal_force", "points", "normals", "separation", "counts", "starts")}
        current = {"active": False, "samples": [], "cost": None, "last_counter": None, "last_quats": None}
        original_update = env.scene.update
        previous_update_attribute = vars(env.scene).get("update", _MISSING)

        def observe(scene, *positional, **keyword):
            result = original_update(*positional, **keyword)
            if not current["active"]:
                return result
            counter = env._sim_step_counter
            if counter == current["last_counter"]:
                if _native_clock(env.sim) != current["last_native"]:
                    raise RuntimeError("Native physics advanced on a same-counter refresh")
                return result
            index, clock = _native_clock(env.sim)
            previous_index, previous_clock = current["last_native"]
            dt = positional[0] if positional else keyword.get("dt")
            bound = abs(_float32(.005) - .005) + 8 * math.ulp(max(abs(clock), abs(previous_clock), .005))
            if (counter != current["last_counter"] + 1 or index != previous_index + 1
                    or abs(clock - previous_clock - .005) > bound or dt != .005
                    or len(current["samples"]) >= 4):
                raise RuntimeError("Independent observer requires four consecutive native5ms samples")
            force = torch.stack([sensor.data.force_matrix_w[:, 0, 0].clone()
                                 for sensor in cache.sensors], dim=1)
            current["samples"].append(force)
            if len(current["samples"]) == 4:
                current["last_quats"] = (command.body_quat_relative_w[:, wheel_indices].clone(),
                                         command.robot_body_quat_w[:, wheel_indices].clone())
            if args.num_envs == 8:
                buffers_by_wheel = [tuple(value.detach().cpu().numpy().copy() for value in
                                          sensor.contact_physx_view.get_contact_data(dt=.005))
                                    for sensor in cache.sensors]
                for wheel_buffers in buffers_by_wheel:
                    raw_fn, raw_points, raw_normals, separation, counts, starts = wheel_buffers
                    raw_fn, raw_normals = raw_fn.reshape(-1), raw_normals.reshape(-1, 3)
                    for count, start in zip(counts.reshape(-1), starts.reshape(-1)):
                        count, start = int(count), int(start)
                        if count < 0 or (count and (start < 0 or start + count > len(raw_fn))):
                            raise RuntimeError("Raw native contact range exceeds the allocated buffer")
                        if count:
                            loaded = raw_fn[start:start + count] > 0.
                            normals = raw_normals[start:start + count][loaded]
                            if (not np.allclose(np.linalg.norm(normals, axis=1), 1., rtol=0., atol=2e-5)
                                    or np.any(normals[:, 2] < .9999)):
                                raise RuntimeError("Actual flat-ground contact normals are not unit/upward")
                for buffer_index, name in enumerate(raw_contact_buffers):
                    raw_contact_buffers[name].append(np.stack([values[buffer_index] for values in buffers_by_wheel]))
                points = sample_raw_isaac_ground_contacts(env)["wheel_ground_contact_force_w_n"]
                actual = force.detach().cpu().numpy()
                if not np.allclose(points, actual, rtol=2e-5, atol=2e-3):
                    raise RuntimeError("Filtered force differs from raw ground-normal point-force sum")
                point_vectors.append(points.copy())
            raw_vectors.append(force)
            counters.append((counter, index))
            native_times.append(clock)
            current["last_counter"], current["last_native"] = counter, (index, clock)
            return result

        observer_wrapper = MethodType(observe, env.scene)
        env.scene.update = observer_wrapper
        handler = install_worst_ground_load_accumulator(env)
        term = env.reward_manager.get_term_cfg("drift_wheel_ground_load_reserve")
        original_reward = term.func
        weight, parameters = term.weight, dict(term.params)
        if (original_reward is not substep_worst_wheel_ground_load_reserve_penalty
                or weight != -1. or parameters != {"scale_fraction": .20}):
            raise RuntimeError("Actual dense reserve weight/parameters mismatch")

        def capture(environment, scale_fraction):
            if environment is not env or not current["active"] or len(current["samples"]) != 4 or current["cost"] is not None:
                raise RuntimeError("Existing reward must read once after four independent samples")
            value = original_reward(environment, scale_fraction)
            current["cost"] = value.clone()
            costs.append(value)  # retain to detect later in-place aliasing
            return value

        term.func = capture
        env.reward_manager.set_term_cfg("drift_wheel_ground_load_reserve", term)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.)
        wrapped.reset()
        automatic_dones = 0
        reset_audits = []
        for window_index in range(16):
            if window_index == 8:
                before = handler.audit()
                native_before = _native_clock(env.sim)
                wrapped.reset()
                after = handler.audit()
                if before != after or native_before != _native_clock(env.sim):
                    raise RuntimeError("Explicit reset changed sample counters or stepped physics")
                reset_audits.append({"before": before, "after": after,
                                     "native_before": list(native_before), "native_after": list(_native_clock(env.sim))})
            current.update(active=True, samples=[], cost=None, last_quats=None, last_counter=env._sim_step_counter,
                           last_native=_native_clock(env.sim))
            window_native_baselines.append((env._sim_step_counter, *current["last_native"]))
            torch.cuda.synchronize(args.device)
            step_started = time.monotonic()
            obs, rewards, dones, extras = wrapped.step(torch.zeros(args.num_envs, 8, device=args.device))
            torch.cuda.synchronize(args.device)
            step_wall_times.append(time.monotonic() - step_started)
            current["active"] = False
            if len(current["samples"]) != 4 or current["cost"] is None or current["last_quats"] is None:
                raise RuntimeError("Incomplete actual force window")
            axle_density = stored_density(env.reward_manager, axle.TERM_NAME, args.num_envs).clone()
            axle_contribution = axle_density * env.step_dt
            target, actual = current["last_quats"]
            replay = numpy_axle_reward(target.detach().cpu().numpy(), actual.detach().cpu().numpy())
            if not np.allclose(axle_density.detach().cpu().numpy(), replay, rtol=0., atol=1e-6):
                raise RuntimeError("Actual LAST axle RewardManager density does not replay within1e-6")
            if not np.allclose(axle_contribution.detach().cpu().numpy(), replay * .02, rtol=0., atol=1e-6):
                raise RuntimeError("Actual integrated axle term does not replay within1e-6")
            target_quats.append(target); actual_quats.append(actual)
            axle_densities.append(axle_density); axle_contributions.append(axle_contribution)
            if any(not torch.equal(old, saved) for old, saved in zip(axle_densities[:-1], retained_axle)):
                raise RuntimeError("Saved axle reward-density clones aliased subsequent windows")
            retained_axle.append(axle_density.clone())
            assert_armed_ledger(env.reward_manager, live_ledger, capture, axle.wheel_axle_orientation_tracking)
            expected = torch.stack([torch.exp(-torch.clamp(force[:, :, 2], min=0.) / (.20 * mg[:, None])).sum(dim=1)
                                    for force in current["samples"]]).amax(dim=0)
            if not torch.equal(expected, current["cost"]):
                raise RuntimeError("Independent GPU max-of-sums cost is not bit-exact")
            load_density = stored_density(env.reward_manager, "drift_wheel_ground_load_reserve", args.num_envs).clone()
            load_contribution = load_density * env.step_dt
            if not torch.allclose(load_density, -current["cost"], rtol=0., atol=1e-6):
                raise RuntimeError("Actual RewardManager worst-load density does not replay")
            if not torch.allclose(load_contribution, -current["cost"] * .02, rtol=0., atol=1e-6):
                raise RuntimeError("Actual worst-load integration must apply policy_dt once")
            load_densities.append(load_density); load_contributions.append(load_contribution)
            if (tuple(obs["policy"].shape) != (args.num_envs, 51)
                    or tuple(obs["critic"].shape) != (args.num_envs, 120)
                    or not all(bool(torch.isfinite(value).all()) for value in obs.values())
                    or not bool(torch.isfinite(rewards).all())):
                raise RuntimeError("Original observation dimensions or finite state changed")
            if any(not torch.equal(old, saved) for old, saved in zip(costs[:-1], windows)):
                raise RuntimeError("Returned force costs aliased subsequent windows")
            windows.append(current["cost"])
            automatic_dones += int(dones.sum().item())
        final_audit = handler.audit()
        if final_audit["force_samples_total"] != 64 or final_audit["force_cost_reads"] != 16:
            raise RuntimeError("Dense force sampling count mismatch")
        arrays = {"force_matrix_world_n": torch.stack(raw_vectors).cpu().numpy(),
                  "mg_n": mg.cpu().numpy(), "cost": torch.stack(windows).cpu().numpy(),
                  "ground_load_weighted_reward_density": torch.stack(load_densities).cpu().numpy(),
                  "ground_load_integrated_reward_contribution": torch.stack(load_contributions).cpu().numpy(),
                  "counter_env_native": np.asarray(counters), "native_time_s": np.asarray(native_times),
                  "policy_step_wall_time_s": np.asarray(step_wall_times),
                  "window_start_env_native_time": np.asarray(window_native_baselines),
                  "axle_reference_quaternion_wxyz": torch.stack(target_quats).cpu().numpy(),
                  "axle_actual_quaternion_wxyz": torch.stack(actual_quats).cpu().numpy(),
                  "axle_weighted_reward_density": torch.stack(axle_densities).cpu().numpy(),
                  "axle_integrated_reward_contribution": torch.stack(axle_contributions).cpu().numpy(),
                  **invariance_arrays}
        offline_axle_audit = validate_saved_axle_evidence(arrays, args.num_envs)
        offline_worst_audit = validate_saved_worst_cost_evidence(arrays, args.num_envs)
        validate_installed_sources()
        if not all(sha(ROOT / name) == value for name, value in source_pins.items()):
            raise RuntimeError("Prospective source bytes changed during the fixture")
        if point_vectors:
            arrays["raw_point_normal_force_world_n"] = np.asarray(point_vectors)
            arrays.update({"raw_contact_" + name: np.asarray(values) for name, values in raw_contact_buffers.items()})
        with (output / "window_evidence.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush(); os.fsync(stream.fileno())
        new_json(output / "science_report.json", {
            "schema_version": 1, "status": "tight_axle_fixture_pass", "num_envs": args.num_envs,
            "native_device_audit": native_audit, "contact_material_audit": material_audit,
            "sampler_final_audit": final_audit, "independent_gpu_cost_bit_exact": True,
            "raw_point_force_comparison_performed": args.num_envs == 8,
            "raw_point_buffers_retained": args.num_envs == 8,
            "raw_loaded_normals_unit_upward_verified": args.num_envs == 8,
            "explicit_reset_audits": reset_audits, "automatic_reset_dones": automatic_dones,
            "no_new_actor_or_PPO": True, "unchanged_51_120_8_dimensions": True,
            "inherited_vz_cost_unused": True, "reward_capture_exception": "original620 fixture-only dense-force capture; new axle func never wrapped",
            "window_evidence_sha256": sha(output / "window_evidence.npz"),
            "wall_time_s": time.monotonic() - started, "formal_training_or_behavior_success": False,
            "fixture_policy_step_wall_time_s": step_wall_times,
            "step_timing_scope": "synchronized fixture wrapped.step; observer/reward capture overhead included, not a production benchmark",
            "performance_benchmark": False, "physical_GPU_isolation_claimed": False,
            "live_reward_ledger_before_force_capture": live_ledger, "all_other_original_reward_terms_unchanged": True,
            "only_axle_angular_scale_changed": True,
            "worst_load_RewardManager_density_and_integration_verified": True,
            "baseline_worst_load_reward_ledger_resolved": configuration_ledger(baseline_cfg.rewards),
            "trial019_axle_reward_never_wrapped_or_replaced_after_manager_creation": True,
            "installed_isaaclab_source_sha256": validate_installed_sources(),
            "reward_manager_step_store_units": "weighted_per_second_density; integrated=value*.02",
            "quaternion_capture_scope": "named two-wheel LAST scene.update before rewards/reset/command.compute",
            "axle_offline_replay": offline_axle_audit, "deterministic_gpu_invariance": invariance_audit,
            "worst_load_offline_replay": offline_worst_audit,
            "saved_axle_reward_clone_nonalias_verified": True,
            "original_dense_force_return_nonalias_verified": True,
            "production_2048_training_force_or_behavior_verified": False,
            "copied_baseline_metadata_resolution_physics_steps": 0,
            "source_bytes_unchanged_at_evidence_write": True,
        })
    except BaseException as error:
        new_json(output / "failure_report.json", {
            "schema_version": 1, "status": "tight_axle_fixture_failed",
            "error": type(error).__name__ + ": " + str(error),
            "wall_time_s": time.monotonic() - started,
            "native_app_close_return_claimed": False, "formal_training_or_behavior_success": False,
        })
        raise
    finally:
        closed_audit = None
        try:
            closed_audit = handler.close() if handler is not None else None
        finally:
            try:
                try:
                    if env is not None and observer_wrapper is not None:
                        if env.scene.update is not observer_wrapper:
                            raise RuntimeError("Refusing to replace an unknown scene observer at teardown")
                        if previous_update_attribute is _MISSING:
                            delattr(env.scene, "update")
                        else:
                            env.scene.update = previous_update_attribute
                finally:
                    if env is not None:
                        close_environment_preserving_pending(env)
                if env is not None:
                    new_json(output / "teardown_report.json", {
                        "schema_version": 1, "status": "sampler_observer_environment_closed",
                        "sampler_closed_audit": closed_audit, "native_app_close_return_claimed": False,
                    })
            finally:
                if app is not None:
                    app.close(skip_cleanup=True)


if __name__ == "__main__":
    main()
