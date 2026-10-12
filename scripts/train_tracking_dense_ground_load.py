"""Trial016: dense ground-load costs with held initial PPO scalar-LR sync.

Targets the inspected Isaac Lab 2.3 / RSL-RL 3 environment. No registry upload,
package installation, or privileged root trajectory playback is performed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE_TRAINING_SOURCE_SHA256 = "ea9e4cd14f25ff3ba5202b9ba43ac0f08320418e2311cbae205c8449f3a1da9c"
BASE_LR_TRAINING_SOURCE_SHA256 = "9bd544b82e211633bb37c3fd8a658448d6d0e9b90f380f01a78dd9e8de02b8cb"
RSL_RESUME_SOURCE_SHA256 = {
    "runners/on_policy_runner.py": "e51e4dd49664ccc8743a16b71d2971d392f40a6988136f10fe8c5f03ab6a83b5",
    "algorithms/ppo.py": "deafc8c947eba4df3e91b393869426cdab8d7b71e05974c3734125d2331d7d1c",
    "networks/normalization.py": "7db82f8021d62a660b4229782b52ac47e32d781206dba709d6bc436bf13f6672",
}


def ppo_config(seed=42, save_interval=100):
    return {
        "seed": seed, "num_steps_per_env": 24, "save_interval": save_interval,
        "logger": "tensorboard", "experiment_name": "tron1_wf_single_jump",
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
        "policy": {
            "class_name": "ActorCritic", "init_noise_std": 0.5,
            "noise_std_type": "log", "actor_obs_normalization": True,
            "critic_obs_normalization": True, "actor_hidden_dims": [256, 128, 128],
            "critic_hidden_dims": [256, 128, 128], "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO", "value_loss_coef": 1.0,
            "use_clipped_value_loss": True, "clip_param": 0.2,
            "entropy_coef": 0.005, "num_learning_epochs": 5,
            "num_mini_batches": 4, "learning_rate": 0.001,
            "schedule": "adaptive", "gamma": 0.99, "lam": 0.95,
            "desired_kl": 0.01, "max_grad_norm": 1.0,
        },
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_installed_resume_sources():
    """Read-only exact RSL3.0.1 source guard, before load/scalar mutation/learn."""
    import rsl_rl
    version = importlib.metadata.version("rsl-rl-lib")
    if version != "3.0.1":
        raise RuntimeError(f"Resume LR sync requires inspected RSL3.0.1, got {version}")
    module_file = getattr(rsl_rl, "__file__", None)
    if module_file is None:
        raise RuntimeError("RSL package source location is unavailable")
    source_root = Path(module_file).resolve().parent
    actual = {name: sha256(source_root / name) for name in RSL_RESUME_SOURCE_SHA256}
    if actual != RSL_RESUME_SOURCE_SHA256:
        raise RuntimeError("Installed RSL runner/PPO/normalization source hash mismatch")
    return {
        "schema_version": 1, "rsl_rl_version": version, "source_root": str(source_root),
        "expected_source_sha256": dict(RSL_RESUME_SOURCE_SHA256),
        "actual_source_sha256": actual, "verified_before_runner_load": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--asset-path", type=Path, default=ROOT / "assets/robots/WF_TRON1A/WF_TRON1A.usd")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-steps", type=int, default=8)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--terrain-file", type=Path, help="Explicit matched static terrain JSON; default is the existing flat plane")
    parser.add_argument("--task-profile", type=Path, help="Optional explicit drift reference/reward/material contract")
    parser.add_argument("--verify-unmasked-device", action="store_true",
                        help="Opt-in strict native PhysX/Torch/tensor CUDA-ordinal readback audit; requires no CUDA_VISIBLE_DEVICES mask")
    domains = parser.add_mutually_exclusive_group()
    domains.add_argument("--domain-randomization", choices=("none", "motor-v1"), default="none")
    domains.add_argument("--domain-profile", type=Path, help="Explicit bounded domain-randomization JSON profile")
    resume_parser = argparse.ArgumentParser(add_help=False)
    resume_parser.add_argument("--resume", type=Path)
    resume_preflight, _ = resume_parser.parse_known_args()
    if resume_preflight.resume is None:
        parser.error("This explicit LR-sync entrypoint requires --resume before AppLauncher")
    if sha256(ROOT / "scripts/train_tracking.py") != BASE_TRAINING_SOURCE_SHA256:
        parser.error("Frozen base training entrypoint SHA mismatch")
    if sha256(ROOT / "scripts/train_tracking_lr_sync.py") != BASE_LR_TRAINING_SOURCE_SHA256:
        parser.error("Frozen LR-sync baseline SHA mismatch")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.num_envs, args.iterations, args.save_interval) < 1 or args.smoke_steps < 0:
        parser.error("num-envs/iterations/save-interval must be positive and smoke-steps nonnegative")
    if (args.num_envs, args.iterations, args.save_interval, args.seed, args.smoke_steps) != (2048, 600, 600, 42, 8):
        parser.error("Trial016 requires the predeclared 2048/600/600/42/8 budget")
    if (args.task_profile is None or args.terrain_file is not None or args.domain_profile is not None
            or args.domain_randomization != "motor-v1" or not args.verify_unmasked_device):
        parser.error("Trial016 requires explicit profile, motor-v1, native-device audit and no terrain/domain overrides")
    sys.path.insert(0, str(ROOT))
    from training.tron1_ground_load_substeps_trial import load_dense_load_profile
    load_dense_load_profile(args.task_profile)  # pure source/profile preflight before application/output
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new output directory; prior experiments are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"status": "initializing", "jump_success_claimed": False}
    app = env = None
    reward_accumulator = None
    try:
        launcher = AppLauncher(args)
        app = launcher.app
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        sys.path.insert(0, str(ROOT))
        from training.tron1_resume_lr import restore_resume_learning_rate
        resume_source_audit = audit_installed_resume_sources()
        from training.tron1_tracking import make_env_cfg, observation_action_contract
        from training.tron1_domain_randomization import default_profile, validate_profile, domain_randomization_audit
        from training.tron1_ground_load_substeps_trial import load_dense_load_profile, configure_dense_ground_load
        if args.task_profile is None:
            raise ValueError("Dense-load entry requires an explicit trial016 task profile")
        task_profile = load_dense_load_profile(args.task_profile)

        profile = (validate_profile(json.loads(args.domain_profile.read_text())) if args.domain_profile else
                   default_profile(args.seed) if args.domain_randomization == "motor-v1" else None)
        terrain_spec = None
        if args.terrain_file:
            from training.tron1_terrain import load_terrain
            terrain_spec = load_terrain(args.terrain_file)
        cfg = make_env_cfg(args.motion_file, args.asset_path, args.num_envs, args.device,
                           dr_profile=profile, terrain_spec=terrain_spec, task_profile=task_profile)
        configure_dense_ground_load(cfg, task_profile)
        cfg.seed = args.seed
        env = ManagerBasedRLEnv(cfg=cfg)
        runtime_device_audit = None
        if getattr(args, "verify_unmasked_device", False):
            from training.tron1_runtime_device import validate_simulation_device
            runtime_device_audit = validate_simulation_device(env, requested_device=args.device)
        contact_material_audit = None
        if task_profile is not None:
            from training.tron1_drift import audit_isaac_materials
            contact_material_audit = audit_isaac_materials(env, task_profile)
        # Bind only after the original material audit initializes the live mg cache.
        from training.tron1_ground_load_substeps import install_ground_load_accumulator
        reward_accumulator = install_ground_load_accumulator(env)
        terrain_audit = None
        if terrain_spec is not None:
            from training.tron1_terrain import terrain_runtime_audit
            terrain_audit = terrain_runtime_audit(env, terrain_spec)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        contract = observation_action_contract()
        if task_profile is not None:
            from training.tron1_drift import apply_task_profile_to_contract
            contract = apply_task_profile_to_contract(contract, task_profile)
        contract["domain_randomization"] = profile if profile else "disabled; nominal actuator parameters"
        if terrain_spec is not None:
            contract["terrain"] = terrain_spec
        obs, _ = wrapped.reset()
        for group, size in (("policy", contract["actor_dim"]), ("critic", contract["critic_dim"])):
            if tuple(obs[group].shape) != (args.num_envs, size):
                raise ValueError(f"Unexpected {group} shape: {obs[group].shape}")
        if env.action_manager.total_action_dim != contract["action_dim"]:
            raise ValueError("Action dimension differs from deployment contract")
        for _ in range(args.smoke_steps):
            obs, rewards, dones, extras = wrapped.step(torch.zeros(args.num_envs, 8, device=args.device))
            if not all(torch.isfinite(obs[key]).all().item() for key in obs.keys()):
                raise FloatingPointError("Non-finite observations in smoke test")
            if not torch.isfinite(rewards).all().item():
                raise FloatingPointError("Non-finite rewards in smoke test")
        wrapped.reset()

        config = ppo_config(args.seed, args.save_interval)
        write_json(output / "runner_config.json", config)
        manifest = {
            "method": "official BeyondMimic tracking MDP + TRON1 wheel-aware adaptation + RSL-RL PPO",
            "upstream_revision": "cd65172032893724b445448818c34165846d847d",
            "host": platform.node(), "python": sys.version,
            "gpu": torch.cuda.get_device_name(args.device) if str(args.device).startswith("cuda") else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "versions": {name: importlib.metadata.version(name) for name in
                         ("torch", "isaacsim", "isaaclab", "rsl-rl-lib")},
            "motion_file": str(args.motion_file.resolve()),
            "motion_sha256": sha256(args.motion_file),
            "asset_file": str(args.asset_path.resolve()), "asset_sha256": sha256(args.asset_path),
            "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                              (ROOT / "training/tron1_tracking.py", ROOT / "training/tron1_domain_randomization.py",
                               ROOT / "training/tron1_terrain.py",
                               ROOT / "training/tron1_drift.py", ROOT / "training/tron1_contact_slip.py",
                               Path(__file__).resolve())},
            "num_envs": args.num_envs, "iterations_requested": args.iterations,
            "seed": args.seed, "smoke_steps_passed": args.smoke_steps,
            "joint_names": env.scene["robot"].joint_names,
            "body_names": env.scene["robot"].body_names,
            "reference_name_mapping": env.command_manager.get_term("motion").reference_name_mapping,
            "contract": contract, "resume": str(args.resume) if args.resume else None,
            "resume_sha256": sha256(args.resume) if args.resume else None,
            "terrain": terrain_spec,
            "task_profile": task_profile,
            "task_profile_sha256": sha256(args.task_profile) if args.task_profile else None,
            "contact_material_audit": contact_material_audit,
            "terrain_runtime_audit": terrain_audit,
            "terrain_file_sha256": sha256(args.terrain_file) if args.terrain_file else None,
            "legacy_joint_friction_audit": env.legacy_joint_friction_audit,
            "domain_randomization_profile": profile,
            "domain_profile_file_sha256": sha256(args.domain_profile) if args.domain_profile else None,
            "domain_randomization_initial_audit": domain_randomization_audit(env),
            "evaluation_required": "Separate deterministic rollout from frame 0, without airborne resets",
        }
        if reward_accumulator is not None:
            manifest["source_sha256"]["training/tron1_substep_rewards.py"] = sha256(ROOT / "training/tron1_substep_rewards.py")
            manifest["reward_sampling_audit"] = reward_accumulator.audit()
            manifest["dense_ground_load_sampling"] = "all_substeps_mean_cost"
            manifest["baseline_entry_source_sha256"] = BASE_LR_TRAINING_SOURCE_SHA256
            for name in ("training/tron1_ground_load_substeps.py",
                         "training/tron1_ground_load_substeps_trial.py",
                         "scripts/train_tracking_lr_sync.py"):
                manifest["source_sha256"][name] = sha256(ROOT / name)
        if task_profile is not None and "wheel_ground_load_reserve" in task_profile:
            manifest["source_sha256"]["training/tron1_ground_load_reserve.py"] = sha256(
                ROOT / "training/tron1_ground_load_reserve.py")
        if runtime_device_audit is not None:
            manifest["runtime_device_audit"] = runtime_device_audit
            manifest["source_sha256"]["training/tron1_runtime_device.py"] = sha256(ROOT / "training/tron1_runtime_device.py")
        write_json(output / "manifest.json", manifest)
        # RSL-RL consumes (pops) class_name fields, so pass a private copy.
        runner = OnPolicyRunner(wrapped, copy.deepcopy(config), log_dir=str(output), device=args.device)
        if args.resume:
            runner.load(str(args.resume.resolve()), map_location=args.device)
        resume_learning_rate_audit = restore_resume_learning_rate(
            runner.alg, expected_scalar_before=0.001, expected_loaded_lr=1e-5,
        )
        manifest["source_sha256"]["scripts/train_tracking.py"] = sha256(ROOT / "scripts/train_tracking.py")
        manifest["source_sha256"]["training/tron1_resume_lr.py"] = sha256(ROOT / "training/tron1_resume_lr.py")
        manifest["resume_source_audit"] = resume_source_audit
        manifest["resume_learning_rate_audit"] = resume_learning_rate_audit
        write_json(output / "manifest.json", manifest)
        write_json(output / "resume_learning_rate_audit.json", resume_learning_rate_audit)
        print("TRON1_RESUME_LEARNING_RATE_AUDIT", json.dumps(resume_learning_rate_audit), flush=True)
        report.update(status="training", num_envs=args.num_envs, iterations_requested=args.iterations)
        write_json(output / "run_status.json", report)
        print("TRON1_TRAINING_STARTED", json.dumps(report), flush=True)
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        runner.save(str(output / "model_final.pt"))
        if runner.writer is not None:
            runner.writer.flush()
            runner.writer.close()
        report.update(status="training_completed", elapsed_seconds=time.monotonic() - started,
                      last_iteration=runner.current_learning_iteration,
                      training_environment_steps=runner.tot_timesteps,
                      checkpoint=str(output / "model_final.pt"))
        write_json(output / "domain_randomization_final_audit.json", domain_randomization_audit(env))
        if reward_accumulator is not None:
            write_json(output / "reward_sampling_final_audit.json", reward_accumulator.audit())
        print("TRON1_TRAINING_COMPLETED", json.dumps(report), flush=True)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      elapsed_seconds=time.monotonic() - started)
        raise
    finally:
        try:
            write_json(output / "run_status.json", report)
        finally:
            try:
                if reward_accumulator is not None:
                    reward_accumulator.close()
            finally:
                try:
                    if env is not None:
                        env.close()
                finally:
                    if app is not None:
                        # All checkpoints/logs are flushed above. No Replicator capture is
                        # used; its default shutdown path hangs on this headless host.
                        app.close(skip_cleanup=True)


if __name__ == "__main__":
    main()


