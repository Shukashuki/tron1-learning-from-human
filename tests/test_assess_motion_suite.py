"""Synthetic offline evidence tests: no policy loading or physics stepping."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from assess_motion_suite import (THRESHOLDS, assess_episode, duration_fraction,
                                 flight_intervals, reference_episode, terrain_comparison,
                                 assess_runs, main)
from compare_tracking import sha256
from eval_tracking_mujoco import read_contract

NAMES = [f"{part}_{side}_Joint" for part in ("abad", "hip", "knee", "wheel") for side in ("L", "R")]


def fixture(task="forward_jump"):
    n = 101
    t = np.arange(n + 1) / 50
    phase = np.clip((t - .4) / .8, 0, 1)
    ramp = phase * phase * (3 - 2 * phase)
    hump = np.sin(np.pi * phase) ** 2
    p = np.column_stack((.4 * ramp, np.zeros_like(t), .9 + .2 * hump))
    angle = np.zeros_like(t)
    if task == "turn_jump":
        p[:, 0] = 0
        angle = np.pi / 2 * ramp
    elif task == "side_jump":
        p[:, 0], p[:, 1] = .05 * ramp, .4 * hump
    elif task == "rolling_stop":
        p[:, 2] = .9
    elif task == "crouch":
        p[:, 0], p[:, 2] = 0, .9 - .2 * hump
    elif task == "step_up":
        p[:, 0], p[:, 2] = .6 * ramp, .9 + .2 * ramp
    q = np.column_stack((np.cos(angle / 2), np.zeros_like(t), np.zeros_like(t), np.sin(angle / 2)))
    wheel = np.repeat(p[:, None], 2, axis=1)
    wheel[:, :, 2] -= .9 - .127
    wheel[:, 0, 1] += .1485
    wheel[:, 1, 1] -= .1485
    reference = {"fps": np.array([50.]), "joint_names": np.array(NAMES),
                 "body_names": np.array(["base_Link", "wheel_L_Link", "wheel_R_Link"]),
                 "body_pos_w": np.concatenate((p[:-1, None], wheel[:-1]), axis=1),
                 "body_quat_w": np.repeat(q[:-1, None], 3, axis=1),
                 "joint_pos": np.zeros((n, 8))}
    force = np.zeros((len(t), 2, 3))
    force[:, :, 2] = 30
    if task.endswith("_jump"):
        force[(t >= .5) & (t < 1.1)] = 0
    data = {"time_s": t, "joint_names": np.array(NAMES), "root_pos": p,
            "root_quat_wxyz": q, "joint_pos": np.zeros((len(t), 8)),
            "action_observation": np.zeros((n, 51)), "reference_frame": np.minimum(np.arange(len(t)), n - 1),
            "wheel_ground_contact_force_w_n": force, "wheel_pos": wheel,
            "nonwheel_ground_contact_count": np.zeros(len(t), dtype=int)}
    contract = {"root_prescribed_during_steps": False, "motion_end_hidden_teleport": False}
    if task == "step_up":
        contract["terrain"] = {"schema_version": 1, "frame": "world_z_up_m",
                               "ground": {"z": 0, "friction": 1.},
                               "boxes": [{"name": "step", "center": [.6, 0., .1],
                                          "size": [.8, 1., .2], "friction": 1.}]}
    report = {"contract": contract, "reference_frames": n, "start_reference_frame": 0,
              "final_reference_frame": n - 1, "completed_full_reference": True,
              "early_terminated": False, "termination": "motion_end", "termination_terms": ["motion_end"],
              "physics_stepped": True, "root_state_writes": 1, "hidden_resets": 0}
    return data, reference, report


def run(data, reference, report, task="forward_jump", engine="MuJoCo"):
    return assess_episode(data, reference, task, report=report, engine=engine)


@pytest.mark.parametrize("task", ["forward_jump", "turn_jump", "side_jump", "rolling_stop", "crouch", "step_up"])
def test_complete_synthetic_task_meets_fixed_gates(task):
    data, reference, report = fixture(task)
    result = run(data, reference, report, task)
    assert result["verdict"] == "pass", result["failed_checks"]
    assert result["safety_verified"] is False


def test_support_is_duration_weighted_not_sample_count():
    times = np.array([0., .49, .5, .51, 1.])
    assert duration_fraction(times, np.array([1., 0., 1., 0., 0.]), .5) == pytest.approx(.02)
    assert duration_fraction(times, np.ones(5), 2.) is None


def test_flight_uses_elapsed_time_with_irregular_sampling():
    t = np.array([0., .1, .2, .25, .4, .5, 1.])
    f = np.zeros((len(t), 2, 3))
    f[:, :, 2] = 30
    f[2:4] = 0
    events = flight_intervals(t, f)
    assert len(events) == 1
    assert events[0]["no_contact_duration_s"] == pytest.approx(.2)
    f[:2] = 0
    assert flight_intervals(t, f) == []


def test_survival_without_jump_is_failure():
    data, reference, report = fixture()
    data["root_pos"][:, 2] = .9
    data["wheel_ground_contact_force_w_n"][:, :, 2] = 30
    result = run(data, reference, report)
    assert {"minimum_actual_rise", "sustained_flight_and_landing"} <= set(result["failed_checks"])


def test_frozen_reference_cannot_define_forward_success():
    data, reference, report = fixture()
    reference["body_pos_w"][:, 0] = [0, 0, .9]
    result = run(data, reference, report)
    assert "nontrivial_reference_translation" in result["failed_checks"]


def test_early_stop_and_root_teleport_fail_even_if_trajectory_matches():
    data, reference, report = fixture()
    report.update(completed_full_reference=False, early_terminated=True, root_state_writes=2)
    result = run(data, reference, report)
    assert {"completed_full_reference", "no_early_termination",
            "physics_only_no_hidden_reset_or_root_teleport"} <= set(result["failed_checks"])


def test_missing_contact_channels_cannot_pass():
    data, reference, report = fixture()
    del data["wheel_ground_contact_force_w_n"]
    del data["nonwheel_ground_contact_count"]
    result = run(data, reference, report)
    assert {"wheel_contact_evidence_available", "tail_two_wheel_support",
            "no_nonwheel_ground_impact"} <= set(result["failed_checks"])
    assert result["complete_contact_evidence"] is False


def test_reverse_progress_is_rejected():
    data, reference, report = fixture()
    data["root_pos"][:, 0] *= -1
    result = run(data, reference, report)
    assert "intended_net_displacement_progress" in result["failed_checks"]


def test_side_out_and_back_needs_excursion_not_just_final_position():
    data, reference, report = fixture("side_jump")
    data["root_pos"][:, 1] = 0
    result = run(data, reference, report, "side_jump")
    assert result["checks"]["intended_net_displacement_progress"]["passed"]
    assert "intended_side_excursion_progress" in result["failed_checks"]


def test_crouch_requires_real_depth_and_recovery():
    data, reference, report = fixture("crouch")
    data["root_pos"][:, 2] = .9
    assert "crouch_depth_progress" in run(data, reference, report, "crouch")["failed_checks"]
    data["root_pos"][-30:, 2] = .7
    assert "crouch_recovered" in run(data, reference, report, "crouch")["failed_checks"]


def test_final_height_error_fails_despite_xyz_tolerance():
    data, reference, report = fixture()
    data["root_pos"][-30:, 2] += .07
    result = run(data, reference, report)
    assert result["checks"]["final_root_xyz"]["passed"]
    assert "final_root_height" in result["failed_checks"]


def test_step_rejects_wrong_final_wheel_height_or_outside_footprint():
    data, reference, report = fixture("step_up")
    data["wheel_pos"][-1, :, 2] -= .1
    assert "both_final_wheels_on_target_ledge" in run(data, reference, report, "step_up")["failed_checks"]
    data, reference, report = fixture("step_up")
    data["wheel_pos"][-1, 0, 0] = 2.
    assert "both_final_wheels_on_target_ledge" in run(data, reference, report, "step_up")["failed_checks"]


def test_step_terrain_contract_mismatch_or_missing_is_rejected():
    _, _, report = fixture("step_up")
    first = report["contract"]
    assert terrain_comparison(first, first, "step_up")["matched"]
    other = copy.deepcopy(first)
    other["terrain"]["boxes"][0]["size"][0] += .1
    assert not terrain_comparison(first, other, "step_up")["matched"]
    assert not terrain_comparison({}, {}, "step_up")["matched"]


def test_nonwheel_one_newton_impact_is_rejected():
    data, reference, report = fixture()
    del data["nonwheel_ground_contact_count"]
    data["nonwheel_ground_contact_force_norm_n"] = np.zeros(len(data["time_s"]))
    data["nonwheel_ground_contact_force_norm_n"][50] = 1.
    assert "no_nonwheel_ground_impact" in run(data, reference, report)["failed_checks"]


def test_reference_root_is_link_not_silently_com():
    _, reference, _ = fixture()
    reference["qpos_mujoco"] = np.zeros((101, 15))
    with pytest.raises(AssertionError):
        reference_episode(reference)


def test_isaac_net_force_proxy_never_claims_complete_contact_evidence():
    data, reference, report = fixture()
    raw = {"time_s": data["time_s"], "joint_names": data["joint_names"],
           "root_position_m": data["root_pos"][:, None],
           "root_quaternion_wxyz": data["root_quat_wxyz"][:, None],
           "joint_position_rad": data["joint_pos"][:, None],
           "valid_mask": np.ones((len(data["time_s"]), 1), dtype=bool),
           "action_observation": np.zeros((len(data["time_s"]), 1, 51)),
           "reference_frame": data["reference_frame"][:, None],
           "wheel_contact_force_w_n": data["wheel_ground_contact_force_w_n"][:, None]}
    outcome = {**report, "environment": 0, "end_reason": "timeout"}
    isaac = {"policy_export": report["contract"], "reference_frames": report["reference_frames"],
             "summary": {"episodes": [outcome]}, "terminal_states_captured_before_auto_reset": True,
             "reference_state_initialization_during_evaluation": False}
    result = run(raw, reference, isaac, engine="Isaac")
    assert result["verdict"] == "pass", result["failed_checks"]
    assert result["complete_contact_evidence"] is False
    assert result["safety_verified"] is False


def test_thresholds_are_fixed_low_jump_not_old_high_jump_gate():
    assert THRESHOLDS["jump_min_rise_m"] == .08
    assert THRESHOLDS["jump_min_flight_s"] == .08
    assert THRESHOLDS["tail_max_tilt_deg"] == 15.


@pytest.fixture
def paired(tmp_path):
    data, reference, report = fixture()
    idir, mdir = tmp_path / "isaac", tmp_path / "mujoco"
    idir.mkdir()
    mdir.mkdir()
    motion = tmp_path / "motion.npz"
    np.savez(motion, **reference)
    (idir / "actor_normalized.pt").write_bytes(b"synthetic identity, never loaded or executed")
    raw = {"time_s": data["time_s"], "joint_names": data["joint_names"],
           "root_position_m": data["root_pos"][:, None],
           "root_quaternion_wxyz": data["root_quat_wxyz"][:, None],
           "joint_position_rad": data["joint_pos"][:, None],
           "valid_mask": np.ones((len(data["time_s"]), 1), dtype=bool),
           "action_observation": np.zeros((len(data["time_s"]), 1, 51)),
           "reference_frame": data["reference_frame"][:, None],
           "wheel_contact_force_w_n": data["wheel_ground_contact_force_w_n"][:, None]}
    np.savez(idir / "trajectory.npz", **raw)
    np.savez(mdir / "rollout.npz", **data)
    contract = read_contract()
    contract.update(report["contract"])
    contract.update(actor_file="actor_normalized.pt", actor_includes_observation_normalizer=True,
                    reference_file_sha256=sha256(motion))
    outcome = {**report, "environment": 0, "end_reason": "timeout"}
    isaac = {"simulator": "IsaacLab/PhysX", "policy_export": copy.deepcopy(contract),
             "motion_file_sha256": sha256(motion), "reference_frames": report["reference_frames"],
             "summary": {"episodes": [outcome]}, "terminal_states_captured_before_auto_reset": True,
             "reference_state_initialization_during_evaluation": False,
             "legacy_joint_friction_audit": {"all_environments_verified_zero": True,
                 "joint_names": NAMES, "after_first_environment": [0.] * 8}}
    mujoco = {**report, "engine": "MuJoCo", "contract": copy.deepcopy(contract),
              "motion_sha256": sha256(motion), "policy_sha256": sha256(idir / "actor_normalized.pt"),
              "policy_kind": "normalized_torchscript_actor", "actual_joint_velocity_hard_clipped": False,
              "isaac_observation_validation": {"status": "passed", "observations_checked": 101,
                  "max_abs_error": 1e-7, "trajectory_sha256": sha256(idir / "trajectory.npz")}}
    return motion, idir, mdir, isaac, mujoco


def assess_pair(pair):
    motion, idir, mdir, isaac, mujoco = pair
    (idir / "report.json").write_text(json.dumps(isaac))
    (mdir / "report.json").write_text(json.dumps(mujoco))
    return assess_runs("forward_jump", motion, idir, mdir)


def test_integration_accepts_actual_same_actor_hashes_and_reports_separate_common_rmse(paired):
    result = assess_pair(paired)
    assert result["verdict"] == "pass", result["failed_checks"]
    assert result["common_window"]["base_position_rmse_m"] == pytest.approx(0.)
    assert result["complete_contact_evidence"] is False


@pytest.mark.parametrize("missing", [None, "isaac", "mujoco", "both", "failed"])
def test_step_requires_both_actual_terrain_runtime_audits(paired, missing):
    motion, idir, mdir, isaac, mujoco = paired
    terrain = fixture("step_up")[2]["contract"]["terrain"]
    isaac["policy_export"]["terrain"] = copy.deepcopy(terrain)
    mujoco["contract"]["terrain"] = copy.deepcopy(terrain)
    for engine, report in (("isaac", isaac), ("mujoco", mujoco)):
        if missing not in (engine, "both"):
            report["terrain_runtime_audit"] = {"status": "failed" if missing == "failed" else "passed"}
    (idir / "report.json").write_text(json.dumps(isaac))
    (mdir / "report.json").write_text(json.dumps(mujoco))
    result = assess_runs("step_up", motion, idir, mdir)
    for engine in ("isaac", "mujoco"):
        check = result["provenance_checks"][engine + "_terrain_runtime_audit_passed"]
        assert check["passed"] == (missing not in (engine, "both", "failed"))
    assert result["provenance_checks"]["matching_terrain_contract"]["passed"]
    assert result["thresholds"] == THRESHOLDS


@pytest.mark.parametrize("change", ["actor", "motion", "actuator", "friction", "terrain"])
def test_integration_rejects_provenance_mismatch_even_with_passing_motion(paired, change):
    _, _, _, isaac, mujoco = paired
    if change == "actor":
        mujoco["policy_sha256"] = "0" * 64
    elif change == "motion":
        mujoco["motion_sha256"] = "0" * 64
    elif change == "actuator":
        mujoco["contract"]["leg_kp"] = 400.
    elif change == "friction":
        isaac["legacy_joint_friction_audit"]["after_first_environment"][7] = .01
    else:
        mujoco["contract"]["terrain"] = {"schema_version": 1, "frame": "world_z_up_m",
                                          "ground": {"z": 0, "friction": 1.}, "boxes": []}
    result = assess_pair(paired)
    assert result["verdict"] == "fail"
    assert any(name.startswith("provenance.") for name in result["failed_checks"])
    assert result["isaac"]["verdict"] == result["mujoco"]["verdict"] == "pass"


def test_cli_preserves_existing_output(tmp_path):
    output = tmp_path / "report.json"
    output.write_text("user-owned")
    with pytest.raises(FileExistsError):
        main(["--task", "crouch", "--motion-file", "missing.npz", "--isaac-dir", "missing",
              "--mujoco-dir", "missing", "--output", str(output)])
    assert output.read_text() == "user-owned"


def test_actual_cli_serializes_complete_assessment_json(paired, tmp_path):
    assess_pair(paired)
    motion, idir, mdir, _, _ = paired
    output = tmp_path / "complete_assessment.json"
    main(["--task", "forward_jump", "--motion-file", str(motion), "--isaac-dir", str(idir),
          "--mujoco-dir", str(mdir), "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["verdict"] == "pass"
    assert report["mujoco"]["complete_contact_evidence"] is True
    assert report["isaac"]["complete_contact_evidence"] is False


def test_cli_normalizes_numpy_boolean_and_scalar_payload(monkeypatch, tmp_path):
    import assess_motion_suite as module
    monkeypatch.setattr(module, "assess_runs", lambda *a: {
        "verdict": "pass", "failed_checks": [], "nested": {
            "boolean": np.bool_(True), "scalar": np.float64(.25), "array": np.array([1, 2])}})
    output = tmp_path / "numpy_report.json"
    main(["--task", "crouch", "--motion-file", "unused", "--isaac-dir", "unused",
          "--mujoco-dir", "unused", "--output", str(output)])
    assert json.loads(output.read_text())["nested"] == {"boolean": True, "scalar": .25, "array": [1, 2]}


def test_cli_serialization_error_does_not_create_partial_output(monkeypatch, tmp_path):
    import assess_motion_suite as module
    monkeypatch.setattr(module, "assess_runs", lambda *a: {
        "verdict": "fail", "failed_checks": [], "invalid": float("nan")})
    output = tmp_path / "absent_dir" / "report.json"
    with pytest.raises(ValueError):
        main(["--task", "crouch", "--motion-file", "unused", "--isaac-dir", "unused",
              "--mujoco-dir", "unused", "--output", str(output)])
    assert not output.exists()
    assert not output.parent.exists()
