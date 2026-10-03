"""One bounded motion trial: PPO then deterministic Isaac evaluation.

Each stage uses an explicit timeout and its own log. A completed process is
not a successful skill; independent task/sim2sim assessment is still required.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def commands(args):
    base = [sys.executable]
    train = base + [str(ROOT / "scripts/train_tracking.py"),
        "--motion-file", str(args.motion_file), "--asset-path", str(args.asset_path),
        "--output-dir", str(args.output_dir / "training"), "--num-envs", str(args.num_envs),
        "--iterations", str(args.iterations), "--save-interval", str(args.iterations),
        "--seed", str(args.seed), "--resume", str(args.resume),
        "--domain-randomization", "motor-v1", "--device", "cuda:0", "--headless"]
    evaluate = base + [str(ROOT / "scripts/eval_tracking.py"),
        "--motion-file", str(args.motion_file), "--asset-path", str(args.asset_path),
        "--checkpoint", str(args.output_dir / "training/model_final.pt"),
        "--runner-config", str(args.output_dir / "training/runner_config.json"),
        "--output-dir", str(args.output_dir / "isaac"), "--num-envs", "1",
        "--seed", str(args.seed), "--device", "cuda:0", "--headless"]
    if args.terrain_file:
        for command in (train, evaluate):
            command += ["--terrain-file", str(args.terrain_file)]
    return [("training", train), ("isaac_evaluation", evaluate)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("motion-file", "asset-path", "resume", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--terrain-file", type=Path)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage-timeout-s", type=float, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.num_envs, args.iterations, args.stage_timeout_s) <= 0:
        parser.error("Environment count, updates and timeout must be positive")
    for name in ("motion_file", "asset_path", "resume", "output_dir", "terrain_file"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
            if name != "output_dir" and not value.is_file():
                parser.error(f"Missing {name}: {value}")
    plan = commands(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output must be new or empty; prior trials are preserved")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "stages": [], "skill_success_claimed": False,
              "sim2sim_evaluated": False, "iterations_per_motion": args.iterations,
              "num_envs": args.num_envs, "seed": args.seed,
              "selection_rule": "Predesignated final checkpoint, no best-of-evaluation selection.",
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    path = args.output_dir / "trial_status.json"
    def save():
        path.write_text(json.dumps(report, indent=2) + "\n")
    save()
    try:
        for stage, command in plan:
            entry = {"stage": stage, "command": command, "status": "running"}
            report["stages"].append(entry)
            save()
            started = time.monotonic()
            with (args.output_dir / (stage + ".log")).open("x") as log:
                try:
                    result = subprocess.run(command, cwd=ROOT, stdout=log,
                                            stderr=subprocess.STDOUT,
                                            timeout=args.stage_timeout_s, check=False)
                except subprocess.TimeoutExpired:
                    entry.update(status="timeout", elapsed_seconds=time.monotonic() - started)
                    raise RuntimeError(f"{stage} exceeded its bounded timeout")
            entry.update(returncode=result.returncode, elapsed_seconds=time.monotonic() - started,
                         status="completed" if result.returncode == 0 else "failed")
            save()
            if result.returncode:
                raise RuntimeError(f"{stage} exited {result.returncode}; see its log")
        report["status"] = "training_and_isaac_evaluation_completed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
