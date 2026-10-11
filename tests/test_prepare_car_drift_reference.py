"""Car-to-TRON1 adaptation preserves source intent and allows contact sideslip."""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("scipy")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_car_drift_reference as drift


def source_fixture():
    times = np.array([0., .1, .1, .2, .3, .4])
    return {
        "time_s_estimated": times,
        "unique_pose_indices": np.array([0, 1, 3, 4, 5]),
        "position_xy_m": np.c_[3. + 2. * times, -5. + .4 * times],
        "velocity_xy_m_s": np.repeat([[2., .4]], len(times), axis=0),
        "yaw_rad": .7 + .2 * times,
    }


def test_source_xy_and_yaw_retain_profile_with_only_declared_uniform_scales():
    source = source_fixture()
    intent = drift.adapt_source(source, spatial_scale=.08, time_scale=2.)
    n = intent["metadata"]["source_frames"]
    query = intent["demonstration_time_s_estimated"][:n]
    heading = np.asarray(intent["metadata"]["heading_alignment_matrix_xy"])
    expected_xy = .08 * np.c_[2. * query, .4 * query] @ heading.T
    np.testing.assert_allclose(intent["root_position"][:n, :2], expected_xy, atol=1e-12)
    np.testing.assert_allclose(intent["root_yaw_rad"][:n], .2 * query, atol=1e-12)
    np.testing.assert_allclose(intent["root_position"][:, 2], .75)
    np.testing.assert_allclose(np.linalg.norm(intent["root_quaternion"], axis=1), 1.)
    assert intent["metadata"]["smoothing_source_s"] == 0.
    assert intent["metadata"]["source_clock_status"] == "reconstructed_estimate_not_original_timestamps"
    assert intent["metadata"]["scaled_logged_velocity_vs_pose_difference_max_m_s"] < 1e-12


def test_source_clock_mapping_is_nonuniform_estimate_and_keeps_unmodified_inputs():
    source = source_fixture()
    copies = {key: value.copy() for key, value in source.items()}
    intent = drift.adapt_source(source)
    n = intent["metadata"]["source_frames"]
    np.testing.assert_allclose(intent["demonstration_time_s_estimated"][:n], intent["time_s"][:n] / 2.)
    np.testing.assert_allclose(intent["source_row_numbers"][:n],
                               np.interp(intent["time_s"][:n] / 2., source["time_s_estimated"][source["unique_pose_indices"]],
                                         source["unique_pose_indices"]))
    for key in source:
        np.testing.assert_array_equal(source[key], copies[key])


def test_source_subframe_omission_is_reported_without_stretching_the_clip_to_fit():
    source = source_fixture()
    source["time_s_estimated"][-1] += .003
    intent = drift.adapt_source(source)
    assert intent["metadata"]["omitted_source_subframe_tail_estimated_s"] == pytest.approx(.003)
    n = intent["metadata"]["source_frames"]
    assert intent["time_s"][n - 1] == pytest.approx(.8)
    assert intent["demonstration_time_s_estimated"][n - 1] == pytest.approx(.4)


def test_recovery_is_distinct_from_captured_motion_and_ends_in_a_true_stationary_tail():
    intent = drift.adapt_source(source_fixture())
    phase = intent["source_phase"]
    n = intent["metadata"]["source_frames"]
    np.testing.assert_array_equal(phase[:n], 0)
    np.testing.assert_array_equal(phase[n:n + 75], 1)
    np.testing.assert_array_equal(phase[n + 75:], 2)
    np.testing.assert_allclose(intent["demonstration_time_s_estimated"][n:], .4)
    assert not np.allclose(intent["root_position"][n], intent["root_position"][n - 1])
    np.testing.assert_allclose(np.diff(intent["root_position"][-25:], axis=0), 0., atol=1e-12)
    np.testing.assert_allclose(np.diff(intent["root_yaw_rad"][-25:]), 0., atol=1e-12)
    assert intent["metadata"]["recovery_initial_speed_m_s"] > 0.
    assert intent["metadata"]["recovery_initial_yaw_rate_rad_s"] == pytest.approx(.1)


@pytest.mark.parametrize("option,value", [
    ("spatial_scale", 0.), ("spatial_scale", -1.), ("spatial_scale", np.nan),
    ("time_scale", 0.), ("time_scale", np.inf),
    ("root_height", -.5), ("root_height", np.nan),
    ("recovery_s", 0.), ("recovery_s", np.inf),
    ("stationary_tail_s", -.1), ("stationary_tail_s", np.nan),
    ("output_fps", 0.), ("output_fps", np.inf),
])
def test_adaptation_rejects_invalid_physical_and_clock_parameters(option, value):
    with pytest.raises(ValueError):
        drift.adapt_source(source_fixture(), **{option: value})


@pytest.mark.parametrize("option", ["recovery_s", "stationary_tail_s"])
def test_endpoint_durations_must_match_the_declared_output_clock(option):
    with pytest.raises(ValueError):
        drift.adapt_source(source_fixture(), **{option: .031})


@pytest.mark.parametrize("field,value", [
    ("unique_pose_indices", np.array([0., 1., 3.])),
    ("unique_pose_indices", np.array([0, 1, 1, 3])),
    ("unique_pose_indices", np.array([1, 3, 4])),
    ("unique_pose_indices", np.array([0, 1, 99])),
    ("unique_pose_indices", np.array([0, 1])),
    ("position_xy_m", np.zeros((6, 3))),
    ("velocity_xy_m_s", np.full((6, 2), np.nan)),
    ("yaw_rad", np.full(6, np.inf)),
    ("time_s_estimated", np.array([0., .1, .05, .2, .3, .4])),
])
def test_adaptation_rejects_broken_capture_arrays(field, value):
    source = source_fixture()
    source[field] = value
    with pytest.raises(ValueError):
        drift.adapt_source(source)


def test_quintic_taper_integral_has_stationary_endpoint_and_known_total_distance():
    np.testing.assert_allclose(drift.smoothstep(np.array([0., .5, 1.])), [0., .5, 1.])
    assert drift.integral_taper(0.) == 0.
    assert drift.integral_taper(1.) == pytest.approx(.5)
    u = np.linspace(0., 1., 10001)
    np.testing.assert_allclose(np.gradient(drift.integral_taper(u), u)[1:-1],
                               1. - drift.smoothstep(u)[1:-1], atol=2e-8)


def test_arbitrary_yaw_wheel_spin_cancels_rolling_but_retains_lateral_sliding():
    intent = drift.adapt_source(source_fixture(), spatial_scale=.8)
    # Rotate the complete robot reference 90 degrees in world space. The old
    # fixed world-X contact equation would become singular in this case.
    rotation = np.array([[0., -1.], [1., 0.]])
    intent["root_position"][:, :2] = intent["root_position"][:, :2] @ rotation.T
    intent["root_yaw_rad"] += np.pi / 2.
    from scipy.spatial.transform import Rotation
    intent["root_quaternion"] = Rotation.from_euler("z", intent["root_yaw_rad"][:, None]).as_quat()[:, [3, 0, 1, 2]]
    arrays, report = drift.solve_reference(intent)
    assert arrays["qpos"].shape == (len(intent["time_s"]), 15)
    assert arrays["qvel"].shape == (len(intent["time_s"]), 14)
    assert report["max_joint_limit_violation_rad"] == 0.
    assert report["max_ground_penetration_m"] < 1e-9
    assert report["max_self_penetration_m"] == 0.
    assert report["max_contact_rolling_velocity_residual_m_s"] < .002
    assert report["max_reference_contact_lateral_slip_m_s"] > .2
    assert report["lateral_slip_constraint_imposed"] is False
    assert report["final_root_speed_m_s"] == 0.
    assert report["final_root_angular_speed_rad_s"] == 0.
    assert report["final_wheel_speed_rad_s"] == 0.
    assert report["simulation_steps"] == 0
    assert report["max_static_com_foreaft_support_error_m"] < 1e-5


def test_geometry_rejects_an_unreachable_root_height():
    intent = drift.adapt_source(source_fixture(), root_height=2.)
    with pytest.raises(ValueError):
        drift.solve_reference(intent)


@pytest.fixture
def verified_capture(tmp_path, monkeypatch):
    from capture_car_drift import decode_csv
    monkeypatch.setattr(drift, "ROOT", tmp_path)
    (tmp_path / "config").mkdir()
    capture_root = tmp_path / "captured"
    raw = capture_root / "raw"
    raw.mkdir(parents=True)
    source = capture_root / "sa_test" / "captured_source.npz"
    source.parent.mkdir()
    csv_bytes = (
        "world_x,world_y,world_heading,world_vx,world_vy,slip_angle,yaw_rate,steer,throttle,hand_brake,brake,world_z,world_vz,pitch\n"
        "0,0,90,0,20,0,0,0,0,False,0,0,0,0\n"
        "0,.1,90,0,20,0,0,.1,0,False,0,0,0,0\n"
        "0,.2,90,0,20,0,0,.2,0,False,0,0,0,0\n"
    ).encode()
    (raw / "example.csv").write_bytes(csv_bytes)
    np.savez_compressed(source, **decode_csv(csv_bytes))
    config = {"repository": "https://example.invalid/source", "commit": "offline-fixture",
              "files": [{"id": "sa_test", "path": "expert_traj/example.csv", "sha256": drift.sha256(raw / "example.csv")}]}
    config_path = tmp_path / "config/drift_sources.json"
    config_path.write_text(json.dumps(config))
    audit = {"source_commit": config["commit"], "configuration_sha256": drift.sha256(config_path),
             "captures": {"sa_test": {"source_sha256": config["files"][0]["sha256"]}}}
    (capture_root / "source_audit.json").write_text(json.dumps(audit))
    return source


def test_source_loader_verifies_the_full_csv_audit_npz_chain_without_network(verified_capture):
    arrays, provenance = drift.load_verified_source(verified_capture)
    assert len(arrays["source_row"]) == 3
    assert provenance["source_npz_sha256"] == drift.sha256(verified_capture)
    assert provenance["source_commit"] == "offline-fixture"
    assert provenance["source_is_physical_car_capture"] is False


@pytest.mark.parametrize("corruption", ["raw_csv", "captured_array", "audit_commit", "configuration"])
def test_source_loader_rejects_a_broken_provenance_link(verified_capture, corruption):
    source_root = verified_capture.parent.parent
    if corruption == "raw_csv":
        path = source_root / "raw/example.csv"
        path.write_bytes(path.read_bytes() + b"\n")
    elif corruption == "captured_array":
        with np.load(verified_capture, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        arrays["position_xy_m"][0, 0] += .01
        np.savez_compressed(verified_capture, **arrays)
    elif corruption == "audit_commit":
        path = source_root / "source_audit.json"
        document = json.loads(path.read_text())
        document["source_commit"] = "unverified-other-commit"
        path.write_text(json.dumps(document))
    else:
        path = drift.ROOT / "config/drift_sources.json"
        document = json.loads(path.read_text())
        document["commit"] = "unverified-other-commit"
        path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        drift.load_verified_source(verified_capture)


def test_generated_reference_exports_in_existing_isaac_contract_with_masked_wheel_spin(tmp_path):
    from export_tracking_motion import convert_motion
    arrays, _ = drift.solve_reference(drift.adapt_source(source_fixture()))
    path = tmp_path / "robot_reference.npz"
    np.savez_compressed(path, **arrays)
    motion, metadata = convert_motion(path, output_fps=50., append_s=0., z_offset=0.)
    assert motion["joint_pos"].shape == (len(arrays["time_s"]), 8)
    np.testing.assert_allclose(motion["qpos_mujoco"], arrays["qpos"], atol=1e-12)
    for index, name in enumerate(motion["joint_names"]):
        assert bool(motion["joint_tracking_mask"][index]) == (not name.startswith("wheel_"))
    assert metadata["root_body_index"] == 0
    assert metadata["independent_urdf_fk_max_position_error_m"] < 1e-9
    assert metadata["independent_urdf_fk_max_rotation_matrix_error"] < 1e-9
    assert metadata["physics_validated"] is False
    np.testing.assert_allclose(motion["qvel_mujoco"][-1], 0.)
