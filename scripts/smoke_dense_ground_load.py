"""Bounded actual-Isaac trial016 force-sampling fixture; no actor or PPO.

The passive observer clones filtered forces at each actual native substep.
It independently replays the costs and (for eight environments only) compares
the filtered matrix to raw ground-normal point-force sums. Extra reads may
synchronize the host; this fixture is not a training performance benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
_MISSING = object()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
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
    from training.tron1_ground_load_substeps_trial import load_dense_load_profile
    profile = load_dense_load_profile(args.task_profile)
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        parser.error("Native physical CUDA ordinals require no mask")
    if sha(args.protocol) != args.protocol_sha256:
        parser.error("Prospective protocol SHA mismatch")
    protocol = json.loads(args.protocol.read_text())
    if (protocol["trial_id"] != "trial016" or sha(args.task_profile) != protocol["task_profile_sha256"]
            or sha(args.motion_file) != protocol["unchanged"]["motion_sha256"]
            or sha(args.asset_path) != protocol["unchanged"]["robot_asset_sha256"]):
        parser.error("Fixture source/profile/reference identity mismatch")
    source_pins = protocol["expected_source_sha256"]
    if not all(sha(ROOT / name) == value for name, value in source_pins.items()):
        parser.error("Current-source protocol pins mismatch")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("Fixture output must be new; never overwrite or restart")
    output.mkdir(parents=True, exist_ok=False)
    new_json(output / "launch_manifest.json", {
        "schema_version": 1, "trial_id": "trial016", "protocol_sha256": sha(args.protocol),
        "profile_sha256": sha(args.task_profile), "source_sha256": source_pins,
        "script_sha256": sha(__file__), "num_envs": args.num_envs, "device": args.device,
        "policy_steps": 16, "physical_samples": 64, "no_actor_or_PPO": True,
        "point_force_cross_check": args.num_envs == 8,
    })
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
        from training.tron1_ground_load_substeps_trial import configure_dense_ground_load
        from training.tron1_ground_load_substeps import (
            install_ground_load_accumulator, substep_wheel_ground_load_reserve_penalty,
        )
        from training import tron1_ground_load_reserve as reserve
        from training.tron1_substep_rewards import _native_clock, _float32
        from training.tron1_contact_slip import sample_raw_isaac_ground_contacts

        cfg = make_env_cfg(args.motion_file, args.asset_path, args.num_envs, args.device,
                           dr_profile=default_profile(42), task_profile=profile)
        configure_dense_ground_load(cfg, profile)
        cfg.seed = 42
        env = ManagerBasedRLEnv(cfg=cfg)
        native_audit = validate_simulation_device(env, requested_device=args.device)
        material_audit = audit_isaac_materials(env, profile)
        cache = getattr(env, reserve.CACHE_ATTRIBUTE)
        mg = cache.mg_n.clone()
        windows, costs, raw_vectors, point_vectors, counters, native_times = [], [], [], [], [], []
        step_wall_times = []
        raw_contact_buffers = {name: [] for name in ("normal_force", "points", "normals", "separation", "counts", "starts")}
        current = {"active": False, "samples": [], "cost": None, "last_counter": None}
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
        handler = install_ground_load_accumulator(env)
        term = env.reward_manager.get_term_cfg("drift_wheel_ground_load_reserve")
        original_reward = term.func
        weight, parameters = term.weight, dict(term.params)
        if (original_reward is not substep_wheel_ground_load_reserve_penalty
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
                reset_audits.append(before)
            current.update(active=True, samples=[], cost=None, last_counter=env._sim_step_counter,
                           last_native=_native_clock(env.sim))
            torch.cuda.synchronize(args.device)
            step_started = time.monotonic()
            obs, rewards, dones, extras = wrapped.step(torch.zeros(args.num_envs, 8, device=args.device))
            torch.cuda.synchronize(args.device)
            step_wall_times.append(time.monotonic() - step_started)
            current["active"] = False
            if len(current["samples"]) != 4 or current["cost"] is None:
                raise RuntimeError("Incomplete actual force window")
            total = torch.zeros_like(current["cost"])
            for force in current["samples"]:
                total.add_(torch.exp(-torch.clamp(force[:, :, 2], min=0.) / (.20 * mg[:, None])).sum(dim=1))
            expected = total / 4
            if not torch.equal(expected, current["cost"]):
                raise RuntimeError("Independent GPU cost is not bit-exact")
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
                  "counter_env_native": np.asarray(counters), "native_time_s": np.asarray(native_times),
                  "policy_step_wall_time_s": np.asarray(step_wall_times)}
        if point_vectors:
            arrays["raw_point_normal_force_world_n"] = np.asarray(point_vectors)
            arrays.update({"raw_contact_" + name: np.asarray(values) for name, values in raw_contact_buffers.items()})
        with (output / "window_evidence.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        new_json(output / "science_report.json", {
            "schema_version": 1, "status": "dense_ground_load_fixture_pass", "num_envs": args.num_envs,
            "native_device_audit": native_audit, "contact_material_audit": material_audit,
            "sampler_final_audit": final_audit, "independent_gpu_cost_bit_exact": True,
            "raw_point_force_comparison_performed": args.num_envs == 8,
            "raw_point_buffers_retained": args.num_envs == 8,
            "raw_loaded_normals_unit_upward_verified": args.num_envs == 8,
            "explicit_reset_audits": reset_audits, "automatic_reset_dones": automatic_dones,
            "no_new_actor_or_PPO": True, "unchanged_51_120_8_dimensions": True,
            "inherited_vz_cost_unused": True, "reward_capture_exception": "one fixture-only function wrapper; weight/params held",
            "window_evidence_sha256": sha(output / "window_evidence.npz"),
            "wall_time_s": time.monotonic() - started, "formal_training_or_behavior_success": False,
            "fixture_policy_step_wall_time_s": step_wall_times,
            "step_timing_scope": "synchronized fixture wrapped.step; observer/reward capture overhead included, not a production benchmark",
            "performance_benchmark": False, "physical_GPU_isolation_claimed": False,
        })
    except BaseException as error:
        new_json(output / "failure_report.json", {
            "schema_version": 1, "status": "dense_ground_load_fixture_failed",
            "error": type(error).__name__ + ": " + str(error),
            "wall_time_s": time.monotonic() - started,
            "native_app_close_return_claimed": False, "formal_training_or_behavior_success": False,
        })
        raise
    finally:
        try:
            closed_audit = handler.close() if handler is not None else None
        finally:
            try:
                if env is not None:
                    if observer_wrapper is not None:
                        if env.scene.update is not observer_wrapper:
                            raise RuntimeError("Refusing to replace an unknown scene observer at teardown")
                        if previous_update_attribute is _MISSING:
                            delattr(env.scene, "update")
                        else:
                            env.scene.update = previous_update_attribute
                    env.close()
                    new_json(output / "teardown_report.json", {
                        "schema_version": 1, "status": "sampler_observer_environment_closed",
                        "sampler_closed_audit": closed_audit, "native_app_close_return_claimed": False,
                    })
            finally:
                if app is not None:
                    app.close(skip_cleanup=True)


if __name__ == "__main__":
    main()
