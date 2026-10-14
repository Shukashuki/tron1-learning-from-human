"""Exploratory PPO residual fine-tuning on evaluator-identical MuJoCo physics.

The normalized Isaac-trained actor is frozen. A bounded 51D->8D residual and
critic are trained on full physical episodes. MuJoCo assessments are exploratory;
the resulting actor MUST be exported and evaluated independently in Isaac for
any sim2sim claim. This script never changes the frozen acceptance gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from assess_drift_turn import assess, completed_from_report, physical_episode
from drift_mujoco_training_env import MujocoDriftEnv
from eval_tracking_mujoco import run_evaluation, sha256
from training.drift_mujoco_residual_reward import reward_components
from training.drift_mujoco_streak import DriftStreakTracker


class ResidualActorCritic(nn.Module):
    def __init__(self, base_actor):
        super().__init__()
        self.base_actor = base_actor.eval()
        for parameter in self.base_actor.parameters():
            parameter.requires_grad_(False)
        self.residual = nn.Sequential(nn.Linear(51, 64), nn.Tanh(),
                                      nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 8))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.critic = nn.Sequential(nn.Linear(51, 64), nn.Tanh(),
                                    nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        # The frozen actor completed 8/8 noisy full clips at this initial
        # standard deviation, versus 6/8 at log_std=-3.9 (same seed/model).
        self.log_std = nn.Parameter(torch.full((8,), -5.3))

    def forward(self, observation):
        with torch.no_grad():
            base = self.base_actor(observation)
        mean = .12 * torch.tanh(self.residual(observation))
        value = self.critic(observation).squeeze(-1)
        return base, mean, value

    def distribution(self, mean):
        return torch.distributions.Normal(mean, torch.exp(self.log_std).expand_as(mean))


def gae(rewards, values, dones, gamma=.995, lam=.95):
    if not (len(rewards) == len(values) == len(dones)):
        raise ValueError("Aligned reward, critic and termination arrays required")
    advantages = np.zeros(len(rewards), dtype=np.float32)
    carry = 0.
    for index in range(len(rewards) - 1, -1, -1):
        next_value = 0. if index == len(rewards) - 1 or dones[index] else values[index + 1]
        continuation = 0. if dones[index] else 1.
        delta = rewards[index] + gamma * next_value * continuation - values[index]
        carry = delta + gamma * lam * continuation * carry
        advantages[index] = carry
    return advantages, advantages + np.asarray(values, dtype=np.float32)


def collect(policy, env, episodes, device, rng):
    memory = {name: [] for name in ("observation", "residual", "logp", "value", "reward", "done")}
    completed, episode_rewards, best_turns = 0, [], []
    for _ in range(episodes):
        observation = env.reset()
        streak = DriftStreakTracker()
        cumulative = 0.
        while True:
            state = torch.as_tensor(observation[None], dtype=torch.float32, device=device)
            with torch.no_grad():
                base, mean, value = policy(state)
                distribution = policy.distribution(mean)
                residual = distribution.sample()
                logp = distribution.log_prob(residual).sum(-1)
                action = (base + residual).cpu().numpy()[0]
            target = env.reference.sample(env.frame)
            next_observation, done, info = env.step(action)
            parts = reward_components(target, info["state"], info["substeps"],
                                      residual.cpu().numpy()[0])
            record_gain = sum(streak.observe(sample) for sample in info["substeps"])
            if done:
                record_gain += streak.finish()
            reward = (parts["reward"] + 10.*record_gain
                      - (2. if done and info["termination_terms"] else 0.))
            for name, item in (("observation", observation), ("residual", residual.cpu().numpy()[0]),
                               ("logp", float(logp.item())), ("value", float(value.item())),
                               ("reward", reward), ("done", done)):
                memory[name].append(item)
            cumulative += reward
            if done:
                completed += int(info["completed_full_reference"])
                episode_rewards.append(cumulative)
                best_turns.append(streak.largest_qualifying_turn_deg)
                break
            observation = next_observation
    # The generator is seeded independently from torch for deterministic batch
    # permutation, while actor exploration is seeded with torch.manual_seed.
    memory["permutation_seed"] = int(rng.integers(0, 2**31))
    return memory, {"episodes": episodes, "full_clip_episodes": completed,
                    "mean_episode_reward": float(np.mean(episode_rewards)),
                    "mean_largest_qualifying_turn_deg": float(np.mean(best_turns)),
                    "mean_policy_steps": len(memory["reward"]) / episodes}


def update(policy, optimizer, memory, device, *, epochs=4, minibatch=256):
    observations = torch.as_tensor(np.asarray(memory["observation"]), dtype=torch.float32, device=device)
    sampled = torch.as_tensor(np.asarray(memory["residual"]), dtype=torch.float32, device=device)
    previous_logp = torch.as_tensor(memory["logp"], dtype=torch.float32, device=device)
    advantages, returns = gae(memory["reward"], memory["value"], memory["done"])
    advantages = torch.as_tensor(advantages, device=device)
    returns = torch.as_tensor(returns, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)
    rng = np.random.default_rng(memory["permutation_seed"])
    losses = []
    for _ in range(epochs):
        for indices in np.array_split(rng.permutation(len(observations)),
                                      max(1, int(np.ceil(len(observations) / minibatch)))):
            indices = torch.as_tensor(indices, dtype=torch.long, device=device)
            _, mean, value = policy(observations[indices])
            distribution = policy.distribution(mean)
            logp = distribution.log_prob(sampled[indices]).sum(-1)
            ratio = torch.exp(logp - previous_logp[indices])
            surrogate = torch.minimum(ratio * advantages[indices],
                                      torch.clamp(ratio, .8, 1.2) * advantages[indices])
            actor_loss = -surrogate.mean()
            critic_loss = torch.mean((value - returns[indices])**2)
            entropy = distribution.entropy().sum(-1).mean()
            loss = actor_loss + .25*critic_loss - .001*entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], .5)
            optimizer.step()
            with torch.no_grad():
                policy.log_std.clamp_(-6., -3.)
            losses.append((float(actor_loss.detach()), float(critic_loss.detach()), float(entropy.detach())))
    return {"actor_loss": float(np.mean([row[0] for row in losses])),
            "critic_loss": float(np.mean([row[1] for row in losses])),
            "entropy": float(np.mean([row[2] for row in losses])),
            "residual_std": torch.exp(policy.log_std).detach().cpu().numpy().tolist()}


def evaluate(policy, args, output, device, reference):
    def actor(observation):
        with torch.no_grad():
            state = torch.as_tensor(observation[None], dtype=torch.float32, device=device)
            base, mean, _ = policy(state)
            return (base + mean).cpu().numpy()[0]

    report = run_evaluation(actor, args.motion_file, args.model, output,
                            contract_path=args.contract, no_visual_mesh=True)
    with np.load(output / "rollout.npz", allow_pickle=False) as archive:
        result = assess(physical_episode(archive, "mujoco"), reference)
    complete = completed_from_report(report, "mujoco")
    result["full_reference_completed"] = complete
    result["status"] = "PASS" if complete and result["status"] == "PASS" else "FAIL"
    result["rollout_sha256"] = sha256(output / "rollout.npz")
    (output / "turn_assessment.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("policy", "motion-file", "robot-reference", "model", "contract"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=40)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Preserving existing residual training run")
    if min(args.updates, args.episodes_per_update, args.eval_every, args.torch_threads) < 1:
        parser.error("Updates, episodes, evaluation cadence and torch threads must be positive")
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    base_actor = torch.jit.load(str(args.policy), map_location=device).eval()
    policy = ResidualActorCritic(base_actor).to(device)
    optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=3e-4)
    env = MujocoDriftEnv(args.motion_file, args.model, args.contract)
    with np.load(args.robot_reference, allow_pickle=False) as archive:
        reference = {key: archive[key].copy() for key in archive.files}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"kind": "exploratory_mujoco_ppo_residual_not_sim2sim",
                "seed": args.seed, "device": str(device), "torch_threads": args.torch_threads,
                "updates": args.updates,
                "initial_residual_gaussian_log_std": -5.3,
                "episodes_per_update": args.episodes_per_update, "eval_every": args.eval_every,
                "policy_sha256": sha256(args.policy), "motion_sha256": sha256(args.motion_file),
                "robot_reference_sha256": sha256(args.robot_reference),
                "model_sha256": sha256(args.model), "contract_sha256": sha256(args.contract),
                "trainer_sha256": sha256(__file__),
                "environment_sha256": sha256(Path(__file__).with_name("drift_mujoco_training_env.py")),
                "reward_sha256": sha256(Path(__file__).resolve().parents[1] / "training/drift_mujoco_residual_reward.py"),
                "streak_sha256": sha256(Path(__file__).resolve().parents[1] / "training/drift_mujoco_streak.py"),
                "streak_record_improvement_reward_per_degree": 10.0,
                "policy_observation_dim": 51, "policy_action_dim": 8,
                "no_isaac_training_or_validation": True,
                "limitations": ["Any MuJoCo PASS remains provisional until the same exported actor passes paired Isaac/MuJoCo assessment.",
                                "This fixed-reference residual PPO run is exploratory, not a generalization or hardware-safety result."]}
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for number in range(1, args.updates + 1):
        memory, collected = collect(policy, env, args.episodes_per_update, device, rng)
        learned = update(policy, optimizer, memory, device)
        row = {"update": number, **collected, **learned}
        if number == 1 or number % args.eval_every == 0 or number == args.updates:
            outcome = evaluate(policy, args, args.output_dir / f"eval_{number:04d}", device, reference)
            row["evaluation"] = {key: outcome[key] for key in (
                "status", "full_reference_completed", "whole_core_min_speed_m_s",
                "core_path_rmse_m", "core_fraction_two_wheel_support_ok",
                "longest_qualifying_duration_s", "largest_qualifying_course_turn_deg", "rollout_sha256")}
            torch.save({"update": number, "state_dict": policy.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "manifest": metadata},
                       args.output_dir / f"checkpoint_{number:04d}.pt")
        with (args.output_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
