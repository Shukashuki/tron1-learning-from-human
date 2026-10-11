"""Recorded-evidence drift gates: synthetic tests never simulate or load actors."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from assess_drift import (THRESHOLDS, assess_episode, assess_runs, duration_mean,
                          intervals, load_isaac_behavior_arrays, main, sideslip)
from compare_tracking import sha256

LEGS = [f"{joint}_{side}_Joint" for joint in ("abad", "hip", "knee") for side in ("L", "R")]
NAMES = LEGS + ["wheel_L_Joint", "wheel_R_Joint"]


def fixture():
    n = 151
    times = np.arange(n + 1) / 50.
    turn = np.deg2rad(26.) * np.clip((times - .4) / 1.2, 0., 1.)
    beta = np.deg2rad(12.) * ((times >= .6) & (times < 1.))
    speed = .8 * (times < 2.)
    velocity = np.column_stack((speed * np.cos(turn + beta), speed * np.sin(turn + beta), np.zeros(len(times))))
    root = np.column_stack((np.zeros((len(times), 2)), np.full(len(times), .6)))
    root[1:, :2] = np.cumsum(velocity[:-1, :2] / 50., axis=0)
    quat = np.column_stack((np.cos(turn / 2), np.zeros((len(times), 2)), np.sin(turn / 2)))
    data = {"time_s": times, "root_pos": root, "root_quat_wxyz": quat,
            "root_com_lin_vel_w": velocity, "joint_pos": np.zeros((len(times), 8)),
            "joint_names": np.array(NAMES), "action_observation": np.zeros((n, 51)),
            "reference_frame": np.minimum(np.arange(len(times)), n - 1),
            "wheel_ground_contact_count": np.ones((len(times), 2), dtype=int),
            "wheel_lateral_contact_slip_m_s": np.repeat((.1 * ((times >= .6) & (times < 1.)))[:, None], 2, axis=1),
            "nonwheel_ground_contact_count": np.zeros(len(times), dtype=int)}
    reference = {"fps": np.array([50.]), "joint_names": np.array(NAMES), "body_names": np.array(["base_Link"]),
                 "body_pos_w": root[:-1, None].copy(), "body_quat_w": quat[:-1, None].copy(),
                 "body_lin_vel_w": velocity[:-1, None].copy(), "joint_pos": np.zeros((n, 8))}
    contract = {"root_prescribed_during_steps": False, "motion_end_hidden_teleport": False,
                "actor_dim": 51, "action_dim": 8, "leg_joint_names": LEGS,
                "wheel_joint_names": NAMES[-2:], "leg_kp": 80., "leg_kd": 2., "leg_position_scale": 1.,
                "wheel_torque_scale_nm": 12., "leg_torque_limit_nm": 80., "wheel_torque_limit_nm": 12.,
                "physics_dt": .005, "decimation": 4, "policy_fps": 50., "raw_action_clip": [-1., 1.],
                "actuator_model": "dc_motor"}
    report = {"contract": contract, "reference_frames": n, "start_reference_frame": 0, "final_reference_frame": n - 1,
              "completed_full_reference": True, "early_terminated": False, "termination": "motion_end",
              "termination_terms": ["motion_end"], "physics_stepped": True, "root_state_writes": 1, "hidden_resets": 0}
    return data, reference, report


def isaac_fixture(data, report, *, invalid_tail=0):
    n = len(data["time_s"])
    mapping = {"root_pos": "root_position_m", "root_quat_wxyz": "root_quaternion_wxyz",
               "root_com_lin_vel_w": "root_linear_velocity_w_m_s", "joint_pos": "joint_position_rad"}
    arrays = {"time_s": data["time_s"], "joint_names": data["joint_names"],
              "valid_mask": np.ones((n, 1), dtype=bool), "action_observation": np.zeros((n, 1, 51))}
    for key in ("root_pos", "root_quat_wxyz", "root_com_lin_vel_w", "joint_pos", "reference_frame",
                "wheel_ground_contact_count", "wheel_lateral_contact_slip_m_s", "nonwheel_ground_contact_count"):
        arrays[mapping.get(key, key)] = data[key][:, None].copy()
    outcome = {**report, "environment": 0, "end_reason": "timeout"}
    result = {"policy_export": copy.deepcopy(report["contract"]), "reference_frames": report["reference_frames"],
              "summary": {"episodes": [outcome]}, "terminal_states_captured_before_auto_reset": True,
              "reference_state_initialization_during_evaluation": False}
    if invalid_tail:
        arrays["time_s"] = np.r_[arrays["time_s"], arrays["time_s"][-1] + np.arange(1, invalid_tail + 1) / 50.]
        for key, value in list(arrays.items()):
            if key in ("time_s", "joint_names"):
                continue
            pad_shape = (invalid_tail, *value.shape[1:])
            fill = False if value.dtype.kind == "b" else -1 if value.dtype.kind in "iu" else np.nan
            arrays[key] = np.concatenate((value, np.full(pad_shape, fill, dtype=value.dtype)))
    return arrays, result


def synthetic_physics_fixture(data, dt=.005):
    """Synthetic schema-only physical-clock fixture, not simulated evidence."""
    old_times = data["time_s"]
    decimation = round(.02 / dt)
    times = np.arange((len(old_times) - 1) * decimation + 1) * dt
    held = np.minimum(np.arange(len(times)) // decimation, len(old_times) - 1)
    result = {}
    for key, value in data.items():
        if key in ("time_s", "action_observation"):
            continue
        if value.shape and value.shape[0] == len(old_times):
            if value.dtype.kind in "biu":
                result[key] = value[held].copy()
            else:
                flat = value.reshape(len(old_times), -1)
                result[key] = np.column_stack([np.interp(times, old_times, column) for column in flat.T]).reshape(
                    len(times), *value.shape[1:])
        else:
            result[key] = value.copy()
    result["root_quat_wxyz"] /= np.linalg.norm(result["root_quat_wxyz"], axis=-1, keepdims=True)
    result["time_s"] = times
    result["physics_dt_s"] = np.array(dt)
    result["action_observation"] = np.zeros((len(old_times) - 1, 51))
    return result


def run(data, reference, report, engine="MuJoCo"):
    return assess_episode(data, reference, engine=engine, report=report)


@pytest.mark.parametrize("engine", ["MuJoCo", "Isaac"])
def test_complete_grounded_drift_and_stable_exit_pass(engine):
    data, reference, report = fixture()
    if engine == "Isaac":
        data, report = isaac_fixture(data, report, invalid_tail=5)
    result = run(data, reference, report, engine)
    assert result["verdict"] == "pass", result["failed_checks"]
    assert result["metrics"]["maximum_contiguous_drift_s"] == pytest.approx(.4)
    assert result["metrics"]["reference_yaw_change_deg"] == pytest.approx(26.)
    assert result["metrics"]["reference_tracking"]["base_position_rmse_m"] == pytest.approx(0.)
    assert result["complete_contact_evidence"]
    assert not result["safety_verified"]


def test_sideslip_uses_heading_plane_and_wraps_world_yaw():
    angle = np.deg2rad([179., -179.])
    quat = np.column_stack((np.cos(angle / 2), np.zeros((2, 2)), np.sin(angle / 2)))
    beta = np.deg2rad([12., -12.])
    velocity = np.column_stack((np.cos(angle + beta), np.sin(angle + beta), [0., 1.]))
    speed, observed = sideslip(quat, velocity)
    np.testing.assert_allclose(speed, [1., 1.])
    np.testing.assert_allclose(observed, beta, atol=1e-12)


def test_irregular_clock_and_terminal_sample_do_not_manufacture_duration():
    times = np.array([0., .05, .24, .25, .9, 1.])
    runs = intervals(times, np.array([False, True, True, False, False, True]))
    assert runs == [{"start_s": .05, "end_s": .25, "duration_s": .2}]
    assert intervals(times, np.array([False, False, False, False, False, True])) == []
    assert duration_mean(times, np.array([1., 0., 1., 0., 0., 1.]), start=.5, end=1.) == 0.
    assert duration_mean(times, np.ones(len(times)), start=-.1, end=1.) is None


@pytest.mark.parametrize("key", ["wheel_ground_contact_count", "wheel_lateral_contact_slip_m_s"])
def test_missing_contact_patch_evidence_never_passes(key):
    data, reference, report = fixture()
    del data[key]
    result = run(data, reference, report)
    assert result["verdict"] == "fail"
    assert "sustained_grounded_contact_patch_drift" in result["failed_checks"]
    assert not result["complete_contact_evidence"]


def test_legacy_policy_rate_can_be_inspected_but_cannot_pass_paired_physics_gate(paired):
    report = paired["isaac_report"]
    for key in ("physics_trajectory_file", "physics_trajectory_sha256", "physics_dt_s", "physics_recording"):
        del report[key]
    (paired["isaac"] / "report.json").write_text(json.dumps(report))
    arrays, path, info = load_isaac_behavior_arrays(paired["isaac"], report)
    assert path.name == "trajectory.npz"
    assert info["kind"] == "legacy_policy_rate"
    assert info["sample_dt_s"] == pytest.approx(.02)
    assert len(arrays["time_s"]) == 152
    result = assess_runs(paired["isaac"], paired["mujoco"], paired["reference"])
    assert result["isaac"]["verdict"] == "pass"
    assert result["verdict"] == "fail"
    assert "provenance.same_physics_contact_sampling_clock" in result["failed_checks"]
    assert not result["complete_contact_evidence"]


def test_physics_contact_gaps_cannot_be_hidden_by_clean_50hz_samples(paired):
    path = paired["isaac"] / "physics_trajectory.npz"
    with np.load(path, allow_pickle=False) as d:
        arrays = {key: d[key].copy() for key in d.files}
    indexes = np.arange(len(arrays["time_s"]))
    gaps = (indexes % 4 == 1) & (arrays["time_s"] >= .6) & (arrays["time_s"] < 1.)
    arrays["wheel_ground_contact_count"][gaps, 0, 0] = 0
    arrays["wheel_lateral_contact_slip_m_s"][gaps, 0, 0] = 0
    np.savez(path, **arrays)
    paired["isaac_report"]["physics_trajectory_sha256"] = sha256(path)
    (paired["isaac"] / "report.json").write_text(json.dumps(paired["isaac_report"]))
    result = assess_runs(paired["isaac"], paired["mujoco"], paired["reference"])
    assert result["provenance_checks"]["same_physics_contact_sampling_clock"]["passed"]
    assert "sustained_grounded_contact_patch_drift" in result["isaac"]["failed_checks"]
    assert result["isaac"]["metrics"]["maximum_contiguous_drift_s"] < .02


@pytest.mark.parametrize("corruption", ["sampling_dt", "physics_claim", "point_claim"])
def test_mujoco_clock_or_pose_proxy_claim_cannot_pass_matched_physics_gate(paired, corruption):
    if corruption == "sampling_dt":
        path = paired["mujoco"] / "rollout.npz"
        with np.load(path, allow_pickle=False) as d:
            arrays = {key: d[key].copy() for key in d.files}
        arrays["physics_dt_s"] = np.array(.02)
        np.savez(path, **arrays)
    elif corruption == "physics_claim":
        paired["mujoco_report"]["physics_stepped"] = False
    else:
        paired["mujoco_report"]["wheel_lateral_contact_slip_evidence_kind"] = "wheel_bottom_pose_proxy"
    (paired["mujoco"] / "report.json").write_text(json.dumps(paired["mujoco_report"]))
    result = assess_runs(paired["isaac"], paired["mujoco"], paired["reference"])
    assert result["verdict"] == "fail"
    assert "provenance.same_physics_contact_sampling_clock" in result["failed_checks"]


@pytest.mark.parametrize("kind", ["no_side_slip", "too_slow", "reverse", "center_slip_only", "one_wheel", "flight", "split_runs", "short_run"])
def test_actual_drift_requires_simultaneous_speed_beta_two_contacts_and_patch_slip(kind):
    data, reference, report = fixture()
    t = data["time_s"]
    if kind == "no_side_slip":
        heading = 2 * np.arctan2(data["root_quat_wxyz"][:, 3], data["root_quat_wxyz"][:, 0])
        data["root_com_lin_vel_w"][:, :2] = .8 * np.column_stack((np.cos(heading), np.sin(heading)))
    elif kind == "too_slow":
        data["root_com_lin_vel_w"] *= .4
    elif kind == "reverse":
        data["root_com_lin_vel_w"] *= -1
    elif kind == "center_slip_only":
        data["wheel_lateral_contact_slip_m_s"][:] = 0.
    elif kind == "one_wheel":
        data["wheel_ground_contact_count"][(t >= .6) & (t < 1.), 0] = 0
    elif kind == "flight":
        data["wheel_ground_contact_count"][(t >= .6) & (t < 1.)] = 0
    elif kind == "split_runs":
        data["wheel_lateral_contact_slip_m_s"][(t >= .72) & (t < .88)] = 0.
    elif kind == "short_run":
        data["wheel_lateral_contact_slip_m_s"][(t >= .78)] = 0.
    result = run(data, reference, report)
    assert "sustained_grounded_contact_patch_drift" in result["failed_checks"]
    if kind == "flight":
        assert "no_sustained_both_wheel_flight" in result["failed_checks"]


def test_per_wheel_signed_patch_slip_is_compared_by_absolute_magnitude():
    data, reference, report = fixture()
    data["wheel_lateral_contact_slip_m_s"][:, 1] *= -1
    result = run(data, reference, report)
    assert result["verdict"] == "pass"
    assert result["metrics"]["drift_mean_abs_contact_slip_m_s"] == pytest.approx(.1)
    assert result["metrics"]["full_mean_abs_contact_slip_m_s"] < .02


def test_threshold_boundaries_are_fixed_and_inclusive():
    data, reference, report = fixture()
    t = data["time_s"]
    heading = 2 * np.arctan2(data["root_quat_wxyz"][:, 3], data["root_quat_wxyz"][:, 0])
    selected = (t >= .6) & (t < .8)
    direction = heading[selected] + np.deg2rad(8.00001)
    data["root_com_lin_vel_w"][selected, :2] = .4000001 * np.column_stack((np.cos(direction), np.sin(direction)))
    data["wheel_lateral_contact_slip_m_s"][:] = 0.
    data["wheel_lateral_contact_slip_m_s"][selected] = .04
    result = run(data, reference, report)
    assert result["checks"]["sustained_grounded_contact_patch_drift"]["passed"]
    assert result["metrics"]["maximum_contiguous_drift_s"] == pytest.approx(.2)
    assert THRESHOLDS["drift_min_abs_sideslip_deg"] == 8.
    assert THRESHOLDS["drift_min_planar_speed_m_s"] == .4
    assert THRESHOLDS["drift_min_mean_abs_contact_slip_m_s"] == .04


def test_initial_lateral_velocity_injection_fails_even_with_later_grounded_drift():
    data, reference, report = fixture()
    data["root_com_lin_vel_w"][0, 1] = .3
    result = run(data, reference, report)
    assert result["checks"]["sustained_grounded_contact_patch_drift"]["passed"]
    assert "initial_actual_sideslip_not_injected" in result["failed_checks"]
    assert "initial_com_velocity_matches_reference" in result["failed_checks"]


def test_small_injected_lateral_velocity_fails_even_below_initial_angle_limit():
    data, reference, report = fixture()
    data["root_com_lin_vel_w"][0, 1] = .02
    result = run(data, reference, report)
    assert result["checks"]["initial_actual_sideslip_not_injected"]["passed"]
    assert "initial_com_velocity_matches_reference" in result["failed_checks"]


def test_weak_nonwheel_points_cannot_hide_summed_ground_support():
    data, reference, report = fixture()
    data["nonwheel_ground_contact_count"][:] = 0
    data["nonwheel_ground_contact_force_norm_n"] = np.zeros(len(data["time_s"]))
    data["nonwheel_ground_contact_force_norm_n"][25] = 1.1
    result = run(data, reference, report)
    assert "no_nonwheel_ground_impact" in result["failed_checks"]


def test_hopping_then_landing_in_drift_is_failure():
    data, reference, report = fixture()
    data["wheel_ground_contact_count"][(data["time_s"] >= .4) & (data["time_s"] < .5)] = 0
    result = run(data, reference, report)
    assert result["checks"]["sustained_grounded_contact_patch_drift"]["passed"]
    assert "no_sustained_both_wheel_flight" in result["failed_checks"]


@pytest.mark.parametrize("failure", ["speed", "sideslip", "tilt", "support", "nonwheel"])
def test_drift_must_recover_stably(failure):
    data, reference, report = fixture()
    tail = data["time_s"] >= 2.52
    if failure == "speed":
        data["root_com_lin_vel_w"][tail] = [.3, 0, 0]
    elif failure == "sideslip":
        heading = np.deg2rad(26 + 6)
        data["root_com_lin_vel_w"][tail] = [.15 * np.cos(heading), .15 * np.sin(heading), 0]
    elif failure == "tilt":
        angle = np.deg2rad(16 / 2)
        data["root_quat_wxyz"][tail] = [np.cos(angle), np.sin(angle), 0., 0.]
    elif failure == "support":
        data["wheel_ground_contact_count"][tail, 1] = 0
    elif failure == "nonwheel":
        data["nonwheel_ground_contact_count"][-1] = 1
    expected = {"speed": "stable_exit_planar_speed", "sideslip": "stable_exit_sideslip", "tilt": "stable_exit_tilt",
                "support": "stable_exit_two_wheel_support", "nonwheel": "no_nonwheel_ground_impact"}[failure]
    assert expected in run(data, reference, report)["failed_checks"]


def test_rest_has_no_defined_slip_angle():
    data, reference, report = fixture()
    data["root_com_lin_vel_w"][-25:] = [0., .001, 0.]
    result = run(data, reference, report)
    assert result["checks"]["stable_exit_sideslip"]["passed"]
    assert result["metrics"]["tail_sideslip_moving_samples"] == 0


def test_nonwheel_missing_is_disclosed_without_safety_claim():
    data, reference, report = fixture()
    del data["nonwheel_ground_contact_count"]
    result = run(data, reference, report)
    assert result["verdict"] == "pass"
    assert not result["complete_contact_evidence"]
    assert not result["safety_verified"]


@pytest.mark.parametrize("key,value", [("root_state_writes", 2), ("completed_full_reference", False),
                                       ("termination", "early_termination"), ("start_reference_frame", 20)])
def test_survival_claim_or_prescribed_pose_cannot_substitute_for_complete_policy_episode(key, value):
    data, reference, report = fixture()
    report[key] = value
    assert run(data, reference, report)["verdict"] == "fail"


@pytest.mark.parametrize("channel,corruption", [(channel, corruption)
    for channel in ("wheel_ground_contact_count", "wheel_lateral_contact_slip_m_s")
    for corruption in ("nan", "wrong_shape", "negative_count", "fractional_count")
    if channel.endswith("count") or corruption in ("nan", "wrong_shape")])
def test_malformed_contact_evidence_fails(channel, corruption):
    data, reference, report = fixture()
    values = data[channel].astype(float)
    if corruption == "nan":
        values[20, 0] = np.nan
    elif corruption == "wrong_shape":
        values = values[:, 0]
    elif corruption == "negative_count":
        values[20, 0] = -1
    else:
        values[20, 0] = .5
    data[channel] = values
    assert run(data, reference, report)["verdict"] == "fail"


def test_zero_reference_turn_or_displacement_cannot_hide_missing_intent():
    data, reference, report = fixture()
    reference["body_quat_w"][:] = [1., 0., 0., 0.]
    reference["body_pos_w"][:, 0, :2] = 0
    result = run(data, reference, report)
    assert {"nontrivial_reference_turn", "nontrivial_reference_translation"} <= set(result["failed_checks"])


@pytest.fixture
def paired(tmp_path, request):
    data, reference, report = fixture()
    dt = getattr(request, "param", .005)
    report["contract"].update(physics_dt=dt, decimation=round(.02 / dt))
    ia, ir = isaac_fixture(data, report)
    idir, mdir = tmp_path / "isaac", tmp_path / "mujoco"
    idir.mkdir()
    mdir.mkdir()
    motion = tmp_path / "motion.npz"
    np.savez(motion, **reference)
    (idir / "actor_normalized.pt").write_bytes(b"synthetic identity only, never loaded")
    np.savez(idir / "trajectory.npz", **ia)
    physical = synthetic_physics_fixture(data, dt)
    pa, _ = isaac_fixture(physical, report)
    pa["physics_dt_s"] = np.array(dt)
    pa["policy_dt_s"] = np.array(.02)
    pa["simulation_step_counter"] = np.arange(len(pa["time_s"]), dtype=int)
    pa["terminal_sample"] = np.zeros_like(pa["valid_mask"])
    pa["terminal_sample"][-1] = True
    np.savez(idir / "physics_trajectory.npz", **pa)
    np.savez(mdir / "rollout.npz", **physical)
    ir["simulator"] = "IsaacLab/PhysX"
    ir["motion_file_sha256"] = sha256(motion)
    ir["policy_export"].update(actor_file="actor_normalized.pt", actor_includes_observation_normalizer=True,
                               reference_file_sha256=sha256(motion))
    ir.update(physics_trajectory_file="physics_trajectory.npz",
              physics_trajectory_sha256=sha256(idir / "physics_trajectory.npz"), physics_dt_s=dt,
              wheel_lateral_contact_slip_evidence_kind="actual_ground_contact_points",
              physics_recording={"sample_dt_s": dt, "samples": len(pa["time_s"]), "initial_state_samples": 1,
                                 "terminal_states_before_reset": True,
                                 "same_counter_reset_and_forward_updates_excluded": True,
                                 "first_episode_only_valid_mask": True})
    mr = {**report, "engine": "MuJoCo", "motion_sha256": sha256(motion),
          "policy_sha256": sha256(idir / "actor_normalized.pt"), "policy_kind": "normalized_torchscript_actor",
          "wheel_lateral_contact_slip_evidence_kind": "actual_ground_contact_points",
          "isaac_observation_validation": {"status": "passed", "observations_checked": 151, "max_abs_error": 1e-7,
              "trajectory_sha256": sha256(idir / "trajectory.npz")}}
    for directory, metadata in ((idir, ir), (mdir, mr)):
        (directory / "report.json").write_text(json.dumps(metadata))
    return {"isaac": idir, "mujoco": mdir, "reference": motion, "isaac_report": ir, "mujoco_report": mr}


def test_paired_same_actor_grounded_drift_has_common_clock_rmse(paired):
    result = assess_runs(paired["isaac"], paired["mujoco"], paired["reference"])
    assert result["verdict"] == "pass", result["failed_checks"]
    assert result["common_window"]["base_position_rmse_m"] == 0.
    assert result["common_window"]["leg_joint_rmse_rad"] == 0.
    assert result["common_window"]["wheel_angles_compared"] is False
    assert result["provenance_checks"]["same_physics_contact_sampling_clock"]["passed"]
    assert result["behavior_evidence"]["isaac"]["file"] == "physics_trajectory.npz"
    assert result["behavior_evidence"]["isaac"]["kind"] == "physics_rate"
    assert result["behavior_evidence"]["mujoco"]["sample_dt_s"] == .005


@pytest.mark.parametrize("paired", [.0025, .00125], indirect=True)
def test_finer_physical_clocks_require_matching_full_rate_evidence(paired):
    result = assess_runs(paired["isaac"], paired["mujoco"], paired["reference"])
    assert result["verdict"] == "pass", result["failed_checks"]
    assert result["provenance_checks"]["same_physics_contact_sampling_clock"]["passed"]
    assert result["behavior_evidence"]["isaac"]["sample_dt_s"] == paired["isaac_report"]["physics_dt_s"]
    assert result["thresholds"] == THRESHOLDS


@pytest.mark.parametrize("paired", [.0025, .00125], indirect=True)
def test_finer_clocks_do_not_accept_differently_timed_deployed_contract(paired):
    paired["mujoco_report"]["contract"].update(physics_dt=.005, decimation=4)
    (paired["mujoco"] / "report.json").write_text(json.dumps(paired["mujoco_report"]))
    result = assess_runs(paired["isaac"], paired["mujoco"], paired["reference"])
    assert "provenance.same_physics_contact_sampling_clock" in result["failed_checks"]
    assert "provenance.matching_actuator_and_clock_contract" in result["failed_checks"]


@pytest.mark.parametrize("paired", [.004], indirect=True)
def test_undeclared_physical_clock_fails_closed(paired):
    result = assess_runs(paired["isaac"], paired["mujoco"], paired["reference"])
    assert "provenance.same_physics_contact_sampling_clock" in result["failed_checks"]


@pytest.mark.parametrize("dt", [.005, .0025, .00125])
@pytest.mark.parametrize("duration,passed", [(.195, False), (.2, True)])
def test_contiguous_drift_threshold_remains_point_two_seconds_at_all_clocks(dt, duration, passed):
    data, reference, report = fixture()
    report["contract"].update(physics_dt=dt, decimation=round(.02 / dt))
    data = synthetic_physics_fixture(data, dt)
    data["wheel_ground_contact_count"][:] = 0
    start = round(.7 / dt)
    end = start + round(duration / dt)
    data["wheel_ground_contact_count"][start:end] = 1
    result = run(data, reference, report)
    assert result["checks"]["sustained_grounded_contact_patch_drift"]["passed"] is passed
    assert result["metrics"]["maximum_contiguous_drift_s"] == pytest.approx(duration)


@pytest.mark.parametrize("failure", ["actor", "reference", "observations", "wheel_order", "actuator"])
def test_same_actor_reference_observations_and_actions_are_verified(paired, failure):
    if failure == "actor":
        (paired["isaac"] / "actor_normalized.pt").write_bytes(b"different actor")
    elif failure == "reference":
        paired["mujoco_report"]["motion_sha256"] = "0" * 64
    elif failure == "observations":
        paired["mujoco_report"]["isaac_observation_validation"]["trajectory_sha256"] = "0" * 64
    elif failure == "wheel_order":
        paired["mujoco_report"]["contract"]["wheel_joint_names"] = NAMES[-2:][::-1]
    else:
        paired["mujoco_report"]["contract"]["wheel_torque_scale_nm"] = 6.
    (paired["mujoco"] / "report.json").write_text(json.dumps(paired["mujoco_report"]))
    result = assess_runs(paired["isaac"], paired["mujoco"], paired["reference"])
    assert result["verdict"] == "fail"
    assert any(key.startswith("provenance.") for key in result["failed_checks"])


def test_cli_preserves_existing_assessment_and_serializes_all_evidence(paired, tmp_path):
    output = tmp_path / "assessment.json"
    args = ["--isaac-dir", str(paired["isaac"]), "--mujoco-dir", str(paired["mujoco"]),
            "--reference", str(paired["reference"]), "--output", str(output)]
    main(args)
    assert json.loads(output.read_text())["verdict"] == "pass"
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        main(args)
    assert output.read_bytes() == before


def test_behavior_loader_selects_physics_and_returns_verified_metadata(paired):
    arrays, path, info = load_isaac_behavior_arrays(paired["isaac"], paired["isaac_report"])
    assert path == paired["isaac"] / "physics_trajectory.npz"
    assert info == {"file": "physics_trajectory.npz", "sha256": sha256(path),
                    "sample_dt_s": .005, "kind": "physics_rate"}
    assert len(arrays["time_s"]) == 605
    np.testing.assert_allclose(np.diff(arrays["time_s"]), .005)
    result = assess_runs(paired["isaac"], paired["mujoco"], paired["reference"])
    assert result["provenance"]["isaac_trajectory_sha256"] == sha256(paired["isaac"] / "trajectory.npz")
    assert result["behavior_evidence"]["isaac"]["sha256"] != result["provenance"]["isaac_trajectory_sha256"]


@pytest.mark.parametrize("corruption", ["file", "sha", "report_dt", "export_dt", "archive_dt", "metadata_dt",
                                        "samples", "initial_states", "no_metadata", "reset_flag", "terminal_flag",
                                        "valid_mask_flag", "irregular_time", "counter_gap", "counter_duplicate",
                                        "missing_terminal", "early_terminal", "revalidated_episode", "policy_pose"])
def test_bad_declared_physics_evidence_never_falls_back_to_policy_rate(paired, corruption):
    report = paired["isaac_report"]
    path = paired["isaac"] / "physics_trajectory.npz"
    with np.load(path, allow_pickle=False) as d:
        arrays = {key: d[key].copy() for key in d.files}
    archive_changed = False
    if corruption == "file":
        report["physics_trajectory_file"] = "../trajectory.npz"
    elif corruption == "sha":
        report["physics_trajectory_sha256"] = "0" * 64
    elif corruption == "report_dt":
        report["physics_dt_s"] = .02
    elif corruption == "export_dt":
        report["policy_export"]["physics_dt"] = .01
    elif corruption == "archive_dt":
        arrays["physics_dt_s"] = np.array(.02)
        archive_changed = True
    elif corruption == "metadata_dt":
        report["physics_recording"]["sample_dt_s"] = .02
    elif corruption == "samples":
        report["physics_recording"]["samples"] -= 1
    elif corruption == "initial_states":
        report["physics_recording"]["initial_state_samples"] = 2
    elif corruption == "no_metadata":
        del report["physics_recording"]
    elif corruption in ("reset_flag", "terminal_flag", "valid_mask_flag"):
        field = {"reset_flag": "same_counter_reset_and_forward_updates_excluded",
                 "terminal_flag": "terminal_states_before_reset", "valid_mask_flag": "first_episode_only_valid_mask"}[corruption]
        report["physics_recording"][field] = False
    elif corruption == "irregular_time":
        arrays["time_s"][10] += .001
        archive_changed = True
    elif corruption == "counter_gap":
        arrays["simulation_step_counter"][10:] += 1
        archive_changed = True
    elif corruption == "counter_duplicate":
        arrays["simulation_step_counter"][10] -= 1
        archive_changed = True
    elif corruption == "missing_terminal":
        arrays["terminal_sample"][:] = False
        archive_changed = True
    elif corruption == "early_terminal":
        arrays["terminal_sample"][:] = False
        arrays["terminal_sample"][100] = True
        archive_changed = True
    elif corruption == "revalidated_episode":
        arrays["valid_mask"][100] = False
        archive_changed = True
    else:
        arrays["root_position_m"][::4, 0, 0] += .01
        archive_changed = True
    if archive_changed:
        np.savez(path, **arrays)
        report["physics_trajectory_sha256"] = sha256(path)
    (paired["isaac"] / "report.json").write_text(json.dumps(report))
    with pytest.raises((ValueError, KeyError)):
        load_isaac_behavior_arrays(paired["isaac"], report)
    result = assess_runs(paired["isaac"], paired["mujoco"], paired["reference"])
    assert result["verdict"] == "fail"
    assert not result["provenance_checks"]["same_physics_contact_sampling_clock"]["passed"]
    assert result["behavior_evidence"]["isaac"]["kind"] == "invalid_physics_evidence"
    assert "valid_recorded_episode_evidence" in result["isaac"]["failed_checks"]
    assert not result["complete_contact_evidence"]
