"""Renderer comparison interface and terminal bounds; no GL context or physics."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

for dependency in ("mujoco", "matplotlib", "scipy", "imageio_ffmpeg"):
    pytest.importorskip(dependency)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_tracking as render


NAMES = [f"{joint}_{side}_Joint" for joint in ("abad", "hip", "knee", "wheel") for side in ("L", "R")]


def clip(end=2., offset=0.):
    times = np.linspace(0., end, 11)
    root = np.column_stack((times + offset, np.zeros(len(times)), 1. + times * .1))
    quat = np.tile([1., 0., 0., 0.], (len(times), 1))
    joints = times[:, None] * np.arange(1, 9)[None, :] / 10.
    return render.PoseClip(times, root, quat, joints, NAMES)


def required_args():
    return ["--trajectory", "isaac/trajectory.npz", "--motion-file", "motion.npz", "--output-dir", "preview"]


def test_cli_defaults_preserve_reference_mode_and_old_chart_option():
    args = render.make_parser().parse_args(required_args())
    assert args.comparison_trajectory is None
    assert args.comparison_label == "MuJoCo PPO"
    assert args.primary_label == "Isaac PhysX"
    assert args.playback_speed == .5
    legacy = render.make_parser().parse_args(required_args() + ["--mujoco-trajectory", "mujoco/rollout.npz"])
    assert legacy.mujoco_trajectory == Path("mujoco/rollout.npz")
    assert legacy.comparison_trajectory is None


def test_cli_accepts_two_recorded_rollouts_and_custom_labels():
    args = render.make_parser().parse_args(required_args() + [
        "--comparison-trajectory", "mujoco/rollout.npz", "--comparison-label", "MuJoCo 3.8.1",
        "--primary-label", "PhysX PPO",
    ])
    assert args.comparison_trajectory == Path("mujoco/rollout.npz")
    assert args.comparison_label == "MuJoCo 3.8.1"
    assert args.primary_label == "PhysX PPO"


def test_default_right_panel_is_reference():
    actual, reference = clip(1.), clip(2., offset=5.)
    left, right, end = render.panel_clips(actual, reference)
    assert left is actual and right is reference
    assert end == 2.


def test_comparison_right_panel_is_measured_rollout_not_reference():
    actual, reference, measured = clip(2.), clip(2.2, offset=5.), clip(1.5, offset=10.)
    left, right, end = render.panel_clips(actual, reference, measured)
    assert left is actual and right is measured
    assert end == 2.2  # The reference height curve remains on the original clock.
    np.testing.assert_allclose(right.sample(.5)[0], [10.5, 0., 1.05])


def test_longer_comparison_extends_video_clock_not_other_robot_motion():
    actual, reference, measured = clip(1.), clip(1.5), clip(2.5)
    left, right, end = render.panel_clips(actual, reference, measured)
    assert end == 2.5
    np.testing.assert_allclose(left.sample(end)[0], left.root[-1])
    assert render.terminal_annotation(left, end, "early_termination") == "RECORDING ENDED\nearly_termination"
    assert render.terminal_annotation(right, end, "motion_end") == ""


def test_early_terminal_freezes_full_pose_and_is_explicitly_labeled():
    measured = clip(1.52)
    at_terminal = measured.sample(1.52)
    later = measured.sample(4.42)
    for original, held in zip(at_terminal, later):
        np.testing.assert_allclose(original, held)
    assert render.terminal_annotation(measured, 1.52, "ee_body_pos") == ""
    assert "RECORDING ENDED" in render.terminal_annotation(measured, 1.54, "ee_body_pos")
    assert "ee_body_pos" in render.terminal_annotation(measured, 1.54, "ee_body_pos")
    assert measured.times[-1] == 1.52  # The source timeline itself never grows.


def test_each_panel_has_its_own_terminal_boundary():
    left, right = clip(1.8), clip(1.2)
    assert render.terminal_annotation(left, 1.5, "left reason") == ""
    assert render.terminal_annotation(right, 1.5, "right reason").endswith("right reason")


def test_mujoco_explicit_named_arrays_win_over_native_order_qpos():
    source = clip()
    arrays = {"time_s": source.times, "root_pos": source.root,
              "root_quat_wxyz": source.quat, "joint_pos": source.joints,
              "joint_names": np.array(NAMES),
              "qpos": np.full((len(source.times), 15), 999.)}
    loaded = render.load_rollout(arrays)
    np.testing.assert_allclose(loaded.root, source.root)
    np.testing.assert_allclose(loaded.joints, source.joints)
    assert loaded.joint_names == NAMES


def test_batched_comparison_excludes_post_reset_tail():
    source = clip()
    qpos = np.concatenate((source.root, source.quat, source.joints), axis=-1)[:, None]
    valid = np.ones((len(source.times), 1), dtype=bool)
    valid[7:] = False
    qpos[7:] = np.nan  # Invalid later episodes cannot reach renderer interpolation.
    loaded = render.load_rollout({"time_s": source.times, "qpos": qpos,
                                  "valid_mask": valid, "joint_names": np.array(NAMES)})
    assert loaded.times[-1] == source.times[6]
    assert len(loaded.times) == 7
    np.testing.assert_allclose(loaded.sample(2.)[0], source.root[6])


def test_discontinuous_valid_mask_cannot_merge_multiple_episodes():
    source = clip()
    qpos = np.concatenate((source.root, source.quat, source.joints), axis=-1)[:, None]
    valid = np.ones((len(source.times), 1), dtype=bool)
    valid[5] = False
    with pytest.raises(ValueError, match="contiguous first-episode"):
        render.load_rollout({"time_s": source.times, "qpos": qpos,
                             "valid_mask": valid, "joint_names": np.array(NAMES)})


def test_named_joint_mapping_is_independent_of_source_order(monkeypatch):
    source = clip()
    model_names = NAMES[::-1]
    model = SimpleNamespace(nq=15, qpos0=np.zeros(15),
                            jnt_type=np.array([int(render.mujoco.mjtJoint.mjJNT_FREE)]
                                              + [int(render.mujoco.mjtJoint.mjJNT_HINGE)] * 8),
                            jnt_qposadr=np.r_[0, np.arange(7, 15)])
    monkeypatch.setattr(render.mujoco, "mj_name2id", lambda model, kind, name: model_names.index(name) + 1)
    pose = render.named_qpos(model, source, source.times[-1])
    np.testing.assert_allclose(pose[:7], np.r_[source.root[-1], source.quat[-1]])
    np.testing.assert_allclose(pose[7:], source.joints[-1, ::-1])


def test_blank_panel_labels_fail_before_creating_output(tmp_path):
    output = tmp_path / "must_not_exist"
    with pytest.raises(SystemExit) as exc:
        render.main(["--trajectory", "unused.npz", "--motion-file", "unused.npz",
                     "--output-dir", str(output), "--comparison-label", " "])
    assert exc.value.code == 2
    assert not output.exists()
