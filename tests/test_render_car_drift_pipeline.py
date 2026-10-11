"""Captured-car playback provenance and clocks, without GL or new dynamics."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

for dependency in ("mujoco", "matplotlib", "scipy", "imageio_ffmpeg"):
    pytest.importorskip(dependency)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_car_drift_pipeline as render


def source():
    times = np.array([0., .1, .1, .2, .4, .6])
    return {"time_s_estimated": times, "unique_pose_indices": np.array([0, 1, 3, 4, 5]),
            "position_xy_m": np.column_stack((100. + times * 10., 200. + times)),
            "velocity_xy_m_s": np.tile([10., 1.], (len(times), 1)),
            "yaw_rad": np.linspace(0., .2, len(times)), "controls": np.tile([.3, .5, 0., 0.], (len(times), 1))}


def mapping_fixture():
    car = render.CarClip(source())
    times = np.arange(11) * .2
    phases = np.r_[np.zeros(7), np.ones(2), np.full(2, 2)].astype(int)
    source_times = np.minimum(times / 2., .6)
    position = np.array([car.sample(t)["position"] for t in source_times])
    robot = {"time_s": times, "demonstration_time_s_estimated": source_times, "source_phase": phases,
             "qpos": np.zeros((len(times), 15)),
             "root_yaw_rad": np.array([car.sample(t)["yaw"] for t in source_times])}
    robot["qpos"][:, :2] = .08 * (position - car.position[0])
    exported = {key: robot[key].copy() for key in ("time_s", "demonstration_time_s_estimated", "source_phase")}
    adaptation = {"spatial_scale": .08, "time_scale": 2., "heading_alignment_matrix_xy": np.eye(2).tolist(),
                  "recovery_duration_s": .4}
    return car, robot, exported, adaptation


def test_car_retains_native_world_coordinates_and_clamps_terminal_without_extrapolation():
    car = render.CarClip(source())
    sample = car.sample(.3)
    np.testing.assert_allclose(sample["position"], [103., 200.3])
    np.testing.assert_allclose(car.sample(100.)["position"], car.position[-1])
    assert car.sample(100.)["estimated_time_s"] == .6
    np.testing.assert_allclose(sample["controls"], [.3, .5, 0., 0.])


@pytest.mark.parametrize("corruption", ["duplicate_selected_time", "nonfinite", "bad_indices", "bad_position_shape"])
def test_invalid_captured_source_is_rejected(corruption):
    arrays = source()
    if corruption == "duplicate_selected_time":
        arrays["unique_pose_indices"] = np.arange(6)
    elif corruption == "nonfinite":
        arrays["velocity_xy_m_s"][0, 0] = np.nan
    elif corruption == "bad_indices":
        arrays["unique_pose_indices"] = np.array([0, 3, 7])
    else:
        arrays["position_xy_m"] = np.zeros((6, 3))
    with pytest.raises(ValueError):
        render.CarClip(arrays)


def test_original_retiming_source_hold_and_synthetic_phase_labels_are_preserved():
    car, robot, exported, adaptation = mapping_fixture()
    result = render.validate_mapping(car, robot, exported, adaptation)
    assert result["source_end_task_s"] == pytest.approx(1.2)
    assert result["source_hold_estimated_s"] == pytest.approx(.6)
    assert render.source_clock(robot["time_s"], robot["demonstration_time_s_estimated"], .5) == .25
    assert render.source_clock(robot["time_s"], robot["demonstration_time_s_estimated"], 1.8) == .6
    assert "CAPTURED" in render.phase_annotation(robot["time_s"], robot["source_phase"], .8)
    assert "SYNTHETIC STOP" in render.phase_annotation(robot["time_s"], robot["source_phase"], 1.5)
    assert "STATIONARY" in render.phase_annotation(robot["time_s"], robot["source_phase"], 2.)


@pytest.mark.parametrize("corruption", ["hidden_retiming", "hidden_spatial_alignment", "source_advance_during_stop", "unlabeled_phase", "export_mapping"])
def test_hidden_source_changes_cannot_be_rendered_as_captured_motion(corruption):
    car, robot, exported, adaptation = mapping_fixture()
    if corruption == "hidden_retiming":
        adaptation["time_scale"] = 1.
    elif corruption == "hidden_spatial_alignment":
        robot["qpos"][:, 0] += .1
    elif corruption == "source_advance_during_stop":
        robot["demonstration_time_s_estimated"][-1] = .61
    elif corruption == "unlabeled_phase":
        robot["source_phase"][-1] = 0
    else:
        exported["demonstration_time_s_estimated"][-1] = .55
    with pytest.raises((ValueError, AssertionError)):
        render.validate_mapping(car, robot, exported, adaptation)


def test_recorded_prefix_excludes_reset_rows_and_never_extrapolates_terminal_motion():
    names = np.array([f"joint_{i}" for i in range(8)])
    arrays = {"time_s": np.array([0., .02, .04, .06]), "root_position_m": np.array([[[0., 0., .7]], [[.1, 0., .7]], [[.2, 0., .7]], [[99., 0., .7]]]),
              "root_quaternion_wxyz": np.tile([1., 0., 0., 0.], (4, 1, 1)), "joint_position_rad": np.zeros((4, 1, 8)),
              "joint_names": names, "valid_mask": np.array([[True], [True], [True], [False]])}
    clip = render.load_rollout(arrays)
    np.testing.assert_allclose(clip.sample(10.)[0], [.2, 0., .7])
    assert "RECORDING ENDED" in render.terminal_annotation(clip, .06, "early_termination")


def test_telemetry_missing_stays_missing_and_invalid_prefix_nan_is_excluded():
    arrays = {"valid_mask": np.array([[True], [True], [False]]),
              "slip": np.array([[[.1, .2]], [[.3, .4]], [[np.nan, np.nan]]])}
    assert render._telemetry(arrays, "absent", "Isaac", 2, (2,)) is None
    np.testing.assert_allclose(render._telemetry(arrays, "slip", "Isaac", 2, (2,)), [[.1, .2], [.3, .4]])


def test_assessment_comparison_allows_only_numeric_roundoff_and_exact_hashes_gates():
    assert render.assessment_equal({"verdict": "fail", "error": .02}, {"verdict": "fail", "error": .02 + 1e-16})
    assert not render.assessment_equal({"verdict": "fail"}, {"verdict": "pass"})
    assert not render.assessment_equal({"sha256": "abc"}, {"sha256": "def"})
    assert not render.assessment_equal({"passed": True}, {"passed": 1})
    assert not render.assessment_equal({"error": .02}, {"error": .020001})
    assert not render.assessment_equal({"error": np.nan}, {"error": np.nan})


def test_visual_sampling_can_halve_mesh_frames_without_changing_half_speed_clock():
    fifty, fps_fifty = render.render_timeline(5.76, 50., .5)
    twentyfive, fps_twentyfive = render.render_timeline(5.76, 25., .5)
    np.testing.assert_allclose(twentyfive, fifty[::2])
    assert len(fifty) == 289 and len(twentyfive) == 145
    assert fps_fifty == 25. and fps_twentyfive == 12.5
    assert twentyfive[-1] == pytest.approx(5.76)
    assert (twentyfive[1] - twentyfive[0]) * fps_twentyfive == pytest.approx(.5)


@pytest.mark.parametrize("curve", ["reference", "Isaac", "MuJoCo"])
def test_reference_and_actual_beta_display_masks_only_assessor_undefined_speed(curve):
    beta = np.radians([65., 20., -12., 8.])
    speed = np.array([0., .1 - 1e-10, .1, .4])
    original_beta, original_speed = beta.copy(), speed.copy()
    original_thresholds = render.THRESHOLDS.copy()
    result = render.displayed_sideslip_degrees(beta, speed)
    assert np.isnan(result[:2]).all(), curve
    np.testing.assert_allclose(result[2:], [-12., 8.])
    np.testing.assert_array_equal(beta, original_beta)
    np.testing.assert_array_equal(speed, original_speed)
    assert render.THRESHOLDS == original_thresholds
    assert render.THRESHOLDS["sideslip_defined_min_speed_m_s"] == .1
    assert render.THRESHOLDS["drift_min_planar_speed_m_s"] == .4
    assert render.THRESHOLDS["drift_min_contiguous_duration_s"] == .2


def test_source_sideslip_mask_uses_declared_robot_scale_speed_not_native_car_speed():
    samples = [{"velocity": np.array([speed, 0.]), "beta": .2} for speed in [1., 2.5, 10.]]
    speed, beta_deg = render.adapted_car_chart_series(samples, .08, 2.)
    np.testing.assert_allclose(speed, [.04, .1, .4])
    assert np.isnan(beta_deg[0])  # Native car speed 1 m/s is above .1; adapted speed is not.
    np.testing.assert_allclose(beta_deg[1:], np.degrees(.2))
    assert [sample["velocity"][0] for sample in samples] == [1., 2.5, 10.]


@pytest.mark.parametrize("beta,speed", [([0.], [np.nan]), ([np.inf], [1.]), ([0.], [-.1]), ([0.], [1., 2.])])
def test_invalid_sideslip_display_inputs_are_rejected(beta, speed):
    with pytest.raises(ValueError):
        render.displayed_sideslip_degrees(beta, speed)


def test_all_chart_curves_preserve_masked_gaps_and_label_undefined_rest():
    times = np.array([0., .1, .2])
    speed = np.array([.04, .1, .4])
    beta = render.displayed_sideslip_degrees(np.full(3, .2), speed)
    samples = [{"velocity": np.array([value, 0.]), "beta": .2} for value in [1., 2.5, 10.]]
    prepared = {"reference": SimpleNamespace(times=times), "isaac": SimpleNamespace(times=times),
                "mujoco": SimpleNamespace(times=times), "task_times": times, "source_times": times,
                "source_phases": np.zeros(3, int), "reference_beta_deg": beta, "reference_speed": speed,
                "car": SimpleNamespace(sample=lambda at: samples[int(round(at * 10))]),
                "mapping": {"source_end_task_s": .2, "synthetic_stop_end_task_s": .4,
                            "source_spatial_scale": .08, "source_time_scale": 2.},
                "series": {key: {"speed": speed, "beta_deg": beta, "slip": np.full((3, 2), .1),
                                 "contacts": np.ones((3, 2), int)}
                           for key in ("isaac", "mujoco")}}
    fig, axes = render.plt.subplots(3)
    try:
        render._charts(prepared, axes, .5)
        curves = {line.get_label(): line for line in axes[0].lines}
        for label in ("Car source", "Kinematic target", "Isaac", "Mujoco"):
            values = curves[label].get_ydata()
            assert np.isnan(values[0])
            np.testing.assert_allclose(values[1:], np.degrees(.2))
        assert any("Undefined below 0.1 m/s" in text.get_text() for text in axes[0].texts)
    finally:
        render.plt.close(fig)


@pytest.mark.parametrize("fps,speed", [(0., .5), (-25., .5), (np.nan, .5), (np.inf, .5), (25., 0.), (25., np.nan)])
def test_invalid_visual_sampling_and_playback_speed_are_rejected(fps, speed):
    with pytest.raises(ValueError):
        render.render_timeline(5.76, fps, speed)


def test_isaac_behavior_selection_uses_validated_physics_archive_and_full_rate_samples(monkeypatch, tmp_path):
    import assess_drift
    arrays = {"time_s": np.arange(9) * .005, "distinct_physics_signal": np.arange(9)}
    path = tmp_path / "physics_trajectory.npz"
    info = {"file": "physics_trajectory.npz", "sha256": "a" * 64, "sample_dt_s": .005, "kind": "physics_rate"}
    calls = []
    def loader(directory, report):
        calls.append((directory, report))
        return arrays, path, info
    monkeypatch.setattr(assess_drift, "load_isaac_behavior_arrays", loader, raising=False)
    output_arrays, output_path, output_info = render.load_isaac_playback(tmp_path, {"physics_trajectory_file": path.name})
    assert output_arrays is arrays and output_path == path and output_info is info
    assert len(output_arrays["time_s"]) == 9
    assert calls == [(tmp_path, {"physics_trajectory_file": path.name})]


def test_isaac_declared_malformed_physics_archive_never_silently_falls_back(monkeypatch, tmp_path):
    import assess_drift
    def loader(directory, report):
        raise ValueError("Physics archive SHA mismatch")
    monkeypatch.setattr(assess_drift, "load_isaac_behavior_arrays", loader, raising=False)
    with pytest.raises(ValueError, match="SHA mismatch"):
        render.load_isaac_playback(tmp_path, {"physics_trajectory_file": "physics_trajectory.npz"})


def test_renderer_never_integrates_physics_and_does_refresh_forward_kinematics():
    tree = ast.parse(Path(render.__file__).read_text())
    calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert "mj_forward" in calls
    assert "mj_step" not in calls and "mj_step1" not in calls and "mj_step2" not in calls


def test_cli_preserves_existing_artifacts_before_loading_inputs(tmp_path):
    output = tmp_path / "render"
    output.mkdir()
    sentinel = output / "existing.txt"
    sentinel.write_text("preserve")
    with pytest.raises(SystemExit):
        render.main(["--captured-source", "missing.npz", "--robot-reference", "missing.npz", "--isaac-dir", "missing",
                     "--mujoco-dir", "missing", "--output-dir", str(output)])
    assert sentinel.read_text() == "preserve"
