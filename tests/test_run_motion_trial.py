"""The bounded suite uses final checkpoints and a shared training/eval scene."""
from argparse import Namespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_motion_trial import commands


def arguments(tmp_path, terrain=None):
    return Namespace(motion_file=tmp_path / "motion.npz", asset_path=tmp_path / "robot.usda",
                     output_dir=tmp_path / "trial", resume=tmp_path / "resume.pt",
                     num_envs=2048, iterations=600, seed=42, terrain_file=terrain)


def value(command, option):
    return command[command.index(option) + 1]


def test_same_reference_scene_and_predesignated_final_checkpoint(tmp_path):
    args = arguments(tmp_path, tmp_path / "terrain.json")
    (train_stage, train), (eval_stage, evaluate) = commands(args)
    assert (train_stage, eval_stage) == ("training", "isaac_evaluation")
    for command in (train, evaluate):
        assert value(command, "--motion-file") == str(args.motion_file)
        assert value(command, "--terrain-file") == str(args.terrain_file)
        assert value(command, "--seed") == "42"
        assert "--headless" in command
    assert value(train, "--iterations") == "600"
    assert value(train, "--num-envs") == "2048"
    assert value(train, "--domain-randomization") == "motor-v1"
    assert value(train, "--resume") == str(args.resume)
    assert value(evaluate, "--num-envs") == "1"
    assert value(evaluate, "--checkpoint") == str(args.output_dir / "training/model_final.pt")
    assert "--resume" not in evaluate


def test_flat_floor_omits_terrain_and_does_not_create_output(tmp_path):
    args = arguments(tmp_path)
    for _, command in commands(args):
        assert "--terrain-file" not in command
    assert not args.output_dir.exists()
