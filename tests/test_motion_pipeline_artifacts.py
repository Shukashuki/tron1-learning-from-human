"""Published four-panel evidence must match the original experiment results."""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("forward_jump", "turn_jump", "side_jump", "rolling_stop", "crouch", "step_up")


@pytest.mark.parametrize("task", TASKS)
def test_published_fourway_preserves_assessed_recordings_and_artifact_integrity(task):
    folder = ROOT / "results/2026-10-03-motion-pipeline-suite" / task
    report = json.loads((folder / "render_report.json").read_text())
    assessment = json.loads((ROOT / "results/2026-10-03-motion-suite" / task / "assessment.json").read_text())
    assert report["status"] == "rendered"
    assert report["simulation_steps"] == 0
    assert report["full_decode_verified"] is True
    assert report["task"] == task
    assert report["task_verdict"] == assessment["verdict"]
    assert report["failed_checks"] == assessment["failed_checks"]
    for key in ("reference_sha256", "actor_sha256", "isaac_trajectory_sha256", "mujoco_trajectory_sha256"):
        assert report["same_actor_provenance"][key] == assessment["provenance"][key]
    for key, name in (("video_sha256", "motion_pipeline.mp4"), ("overview_sha256", "overview.png")):
        assert report[key] == hashlib.sha256((folder / name).read_bytes()).hexdigest()
    assert report["video_resolution"] == [1920, 1080]
    assert report["video_fps"] == 25
    assert report["playback_speed"] == .5
    assert report["robot_panels_share_camera"] is True
    assert report["task_adapted"] == (task in ("rolling_stop", "crouch"))
    assert report["fixed_shared_camera"] == (not report["task_adapted"])
    assert (report["terrain"] is not None) == (task == "step_up")
    if report["task_adapted"]:
        assert report["human_display_transform"] == "identity_native"
        assert report["original_adaptation"]["time_scale"] == 2
    assert 0 < report["hold_start_task_s"] < report["task_end_s"]
