"""Train one TRON1 WF motion with BeyondMimic's MDP and local RSL-RL PPO.

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
    parser.add_argument("--resume", type=Path)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.num_envs, args.iterations, args.save_interval) < 1 or args.smoke_steps < 0:
        parser.error("num-envs/iterations/save-interval must be positive and smoke-steps nonnegative")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new output directory; prior experiments are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"status": "initializing", "jump_success_claimed": False}
    app = env = None
    try:
        launcher = AppLauncher(args)
        app = launcher.app
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        sys.path.insert(0, str(ROOT / "training"))
        from tron1_tracking import make_env_cfg, observation_action_contract

        cfg = make_env_cfg(args.motion_file, args.asset_path, args.num_envs, args.device)
        cfg.seed = args.seed
        env = ManagerBasedRLEnv(cfg=cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        contract = observation_action_contract()
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
                              (ROOT / "training/tron1_tracking.py", Path(__file__).resolve())},
            "num_envs": args.num_envs, "iterations_requested": args.iterations,
            "seed": args.seed, "smoke_steps_passed": args.smoke_steps,
            "joint_names": env.scene["robot"].joint_names,
            "body_names": env.scene["robot"].body_names,
            "reference_name_mapping": env.command_manager.get_term("motion").reference_name_mapping,
            "contract": contract, "resume": str(args.resume) if args.resume else None,
            "legacy_joint_friction_audit": env.legacy_joint_friction_audit,
            "evaluation_required": "Separate deterministic rollout from frame 0, without airborne resets",
        }
        write_json(output / "manifest.json", manifest)
        # RSL-RL consumes (pops) class_name fields, so pass a private copy.
        runner = OnPolicyRunner(wrapped, copy.deepcopy(config), log_dir=str(output), device=args.device)
        if args.resume:
            runner.load(str(args.resume.resolve()), map_location=args.device)
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
        print("TRON1_TRAINING_COMPLETED", json.dumps(report), flush=True)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      elapsed_seconds=time.monotonic() - started)
        raise
    finally:
        write_json(output / "run_status.json", report)
        if env is not None:
            env.close()
        if app is not None:
            # All checkpoints/logs are flushed above. No Replicator capture is
            # used; its default shutdown path hangs on this headless host.
            app.close(skip_cleanup=True)


if __name__ == "__main__":
    main()
