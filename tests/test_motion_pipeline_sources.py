"""Original mocap crop/clock and assessment identity must survive rendering."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

for dependency in ("mujoco", "matplotlib", "scipy", "imageio_ffmpeg"):
    pytest.importorskip(dependency)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from motion_pipeline_sources import adapted_source_times, with_outcome


def inputs():
    raw = {"frame_numbers": np.arange(1, 9), "time_s": np.arange(8) * .25, "fps": np.array(4.)}
    adaptation = {"calibration_and_adaptation": {"time_scale": 2.,
                  "source_first_frame_inclusive": 3, "source_last_frame_inclusive": 7}}
    robot = {"time_s": np.arange(9) * .25, "source_frame_numbers": 3 + np.arange(9) * .5}
    reference = {"time_s": np.arange(13) * .25, "source_time_s": np.minimum(np.arange(13) * .25, 2.)}
    return raw, adaptation, reference, robot


def test_crop_keeps_original_global_clock_and_terminal_hold():
    args = inputs()
    original = copy.deepcopy(args)
    mapped, ids = adapted_source_times(*args)
    np.testing.assert_array_equal(ids, np.arange(2, 7))
    np.testing.assert_allclose(mapped, .5 + np.minimum(np.arange(13) * .25, 2.) / 2)
    assert mapped[0] == .5
    assert mapped[-1] == mapped[-5] == 1.5
    for current, prior in zip(args, original):
        for key in current:
            if isinstance(current[key], np.ndarray):
                np.testing.assert_array_equal(current[key], prior[key])


@pytest.mark.parametrize("scale", [0, -1, np.inf, np.nan])
def test_reject_invalid_adaptation_scale(scale):
    args = inputs()
    args[1]["calibration_and_adaptation"]["time_scale"] = scale
    with pytest.raises(ValueError, match="time scale"):
        adapted_source_times(*args)


@pytest.mark.parametrize("first,last", [(0, 7), (3, 9), (3, 4), (7, 3)])
def test_reject_missing_or_too_short_source_crop(first, last):
    args = inputs()
    args[1]["calibration_and_adaptation"].update(source_first_frame_inclusive=first, source_last_frame_inclusive=last)
    with pytest.raises(ValueError, match="frame range"):
        adapted_source_times(*args)


def test_reject_wrong_fractional_source_frame_correspondence():
    args = inputs()
    args[3]["source_frame_numbers"][1] += .1
    with pytest.raises(AssertionError):
        adapted_source_times(*args)


def test_reject_export_beyond_robot_endpoint():
    args = inputs()
    args[2]["source_time_s"][-1] += .01
    with pytest.raises(ValueError, match="exceeds adapted"):
        adapted_source_times(*args)


def test_reject_export_clock_phase_warp():
    args = inputs()
    args[2]["source_time_s"][1] += .01
    with pytest.raises(AssertionError):
        adapted_source_times(*args)


def test_reject_robot_clip_longer_than_human_crop():
    args = inputs()
    args[1]["calibration_and_adaptation"]["source_last_frame_inclusive"] = 6
    with pytest.raises(ValueError, match="selected human"):
        adapted_source_times(*args)


@pytest.mark.parametrize("index,field", [(0, "time_s"), (2, "time_s"), (3, "time_s"), (2, "source_time_s")])
def test_reject_nonfinite_clocks(index, field):
    args = inputs()
    args[index][field][1] = np.nan
    with pytest.raises(ValueError):
        adapted_source_times(*args)


@pytest.mark.parametrize("field", ["reference_sha256", "actor_sha256", "isaac_trajectory_sha256", "mujoco_trajectory_sha256"])
def test_outcome_must_describe_exact_displayed_recordings(tmp_path, field):
    provenance = {k: k for k in ("reference_sha256", "actor_sha256", "isaac_trajectory_sha256", "mujoco_trajectory_sha256")}
    assessment = {"task": tmp_path.name, "verdict": "fail", "provenance": dict(provenance)}
    (tmp_path / "trial").mkdir()
    (tmp_path / "trial/assessment.json").write_text(json.dumps(assessment))
    prepared = {"provenance": dict(provenance), "reports": {}, "hashes": {}, "source_times": [0., 1.]}
    prepared["provenance"][field] = "different"
    with pytest.raises(ValueError, match="displayed recordings"):
        with_outcome(prepared, tmp_path)


def test_outcome_preserves_failure_and_original_task_hold(tmp_path):
    provenance = {k: k for k in ("reference_sha256", "actor_sha256", "isaac_trajectory_sha256", "mujoco_trajectory_sha256")}
    (tmp_path / "trial").mkdir()
    (tmp_path / "trial/assessment.json").write_text(json.dumps({"task": tmp_path.name, "verdict": "fail", "provenance": provenance}))
    result = with_outcome({"provenance": provenance, "reports": {}, "hashes": {},
                           "source_times": [1.325, 2.735], "hold_start_task_s": 2.82}, tmp_path)
    assert result["reports"]["assessment"]["verdict"] == "fail"
    assert result["hold_start_task_s"] == 2.82
    assert len(result["hashes"]["assessment"]) == 64
