"""Strict same-policy acceptance must not mistake a short bounce for a jump."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from assess_sim2sim import assess_runs, main, tail_support_fraction
from compare_tracking import sha256

LEGS = [f"{joint}_{side}_Joint" for joint in ("abad", "hip", "knee") for side in ("L", "R")]
NAMES = LEGS + ["wheel_L_Joint", "wheel_R_Joint"]


def write_pair(pair):
    np.savez(pair["isaac_dir"] / "trajectory.npz", **pair["i_arrays"])
    np.savez(pair["mujoco_dir"] / "rollout.npz", **pair["m_arrays"])
    pair["mujoco"]["isaac_observation_validation"]["trajectory_sha256"] = sha256(pair["isaac_dir"] / "trajectory.npz")
    for key in ("isaac", "mujoco"):
        (pair[key + "_dir"] / "report.json").write_text(json.dumps(pair[key]))


@pytest.fixture
def paired(tmp_path):
    idir, mdir = tmp_path / "isaac", tmp_path / "mujoco"
    idir.mkdir(); mdir.mkdir()
    (idir / "actor_normalized.pt").write_bytes(b"synthetic actor bytes, never executed")
    ti, tm = np.arange(101) * .02, np.arange(401) * .005

    def state(times):
        root = np.zeros((len(times), 3))
        root[:, 2] = .9 + .35 * np.maximum(0., 1. - np.abs(times - .8) / .25)
        quat = np.tile([1., 0., 0., 0.], (len(times), 1))
        forces = np.zeros((len(times), 2, 3))
        forces[:, :, 2] = 100.
        forces[(times >= .6) & (times < 1.)] = 0.
        refs = np.minimum(np.round(times / .02).astype(int), 99)
        return root, quat, forces, refs

    ri, qi, fi, refi = state(ti)
    rm, qm, fm, refm = state(tm)
    iarrays = {"time_s": ti, "root_position_m": ri[:, None], "root_quaternion_wxyz": qi[:, None],
               "joint_position_rad": np.zeros((len(ti), 1, 8)), "joint_names": np.array(NAMES),
               "valid_mask": np.ones((len(ti), 1), dtype=bool), "reference_frame": refi[:, None],
               "action_observation": np.zeros((len(ti), 1, 51)), "wheel_contact_force_w_n": fi[:, None]}
    marrays = {"time_s": tm, "root_pos": rm, "root_quat_wxyz": qm,
               "joint_pos": np.zeros((len(tm), 8)), "joint_names": np.array(NAMES),
               "reference_frame": refm, "action_observation": np.zeros((len(tm), 51)),
               "wheel_ground_contact_force_w_n": fm}
    contract = {"actor_file": "actor_normalized.pt", "actor_dim": 51, "action_dim": 8,
                "leg_joint_names": LEGS, "policy_fps": 50., "reference_file_sha256": "a" * 64,
                "wheel_joint_names": NAMES[6:], "physics_dt": .005, "decimation": 4,
                "leg_kp": 500., "leg_kd": 10., "leg_position_scale": 1., "wheel_torque_scale_nm": 12.,
                "leg_torque_limit_nm": 80., "wheel_torque_limit_nm": 12., "raw_action_clip": [-1., 1.],
                "actuator_limits_provenance": "Project simulation assumptions; not verified hardware ratings",
                "hardware_ready": False,
                "actor_includes_observation_normalizer": True,
                "root_prescribed_during_steps": False, "motion_end_hidden_teleport": False}
    episode = {"environment": 0, "completed_full_reference": True, "start_reference_frame": 0,
               "final_reference_frame": 99, "end_reason": "timeout", "termination_terms": ["motion_end"],
               "early_terminated": False}
    isaac = {"motion_file_sha256": "a" * 64, "policy_export": contract, "reference_frames": 100,
             "reference_state_initialization_during_evaluation": False,
             "terminal_states_captured_before_auto_reset": True, "summary": {"episodes": [episode]}}
    mujoco = {"motion_sha256": "a" * 64, "policy_sha256": sha256(idir / "actor_normalized.pt"),
              "policy_kind": "normalized_torchscript_actor", "contract": dict(contract), "reference_frames": 100,
              "completed_full_reference": True, "start_reference_frame": 0, "final_reference_frame": 99,
              "termination": "motion_end", "termination_terms": [], "physics_stepped": True,
              "root_state_writes": 1, "hidden_resets": 0,
              "isaac_observation_validation": {"status": "passed", "observations_checked": 100,
                  "max_abs_error": 1e-7, "trajectory_sha256": "filled below"}}
    result = {"isaac_dir": idir, "mujoco_dir": mdir, "isaac": isaac, "mujoco": mujoco,
              "i_arrays": iarrays, "m_arrays": marrays}
    write_pair(result)
    return result


def assess(pair, **kwargs):
    write_pair(pair)
    return assess_runs(pair["isaac_dir"], pair["mujoco_dir"], **kwargs)


def test_complete_jump_passes_dynamic_criteria_but_discloses_contact_gap(paired):
    result = assess(paired)
    assert result["verdict"] == "pass"
    assert result["mujoco"]["selected_flight"]["no_contact_duration_s"] == pytest.approx(.4)
    assert result["mujoco"]["base_height_gain_m"] == pytest.approx(.35)
    assert result["mujoco"]["tail_two_wheel_support_fraction"] == pytest.approx(1.)
    assert result["complete_contact_evidence"] is False
    assert result["isaac"]["contact_force_source"].startswith("net_force_proxy")
    assert result["mujoco"]["nonwheel_ground_contact"]["status"] == "unavailable"
    assert result["hardware_ready"] is False


def test_strict_contact_evidence_is_optional_and_fail_closed(paired):
    result = assess(paired, require_complete_contact_evidence=True)
    assert result["verdict"] == "fail"
    assert "complete_contact_evidence_required" in result["fail_reasons"]
    paired["i_arrays"]["wheel_ground_contact_force_w_n"] = paired["i_arrays"]["wheel_contact_force_w_n"].copy()
    paired["i_arrays"]["nonwheel_ground_contact_count"] = np.zeros((101, 1), dtype=int)
    paired["m_arrays"]["nonwheel_ground_contact_count"] = np.zeros(401, dtype=int)
    result = assess(paired, require_complete_contact_evidence=True)
    assert result["verdict"] == "pass"
    assert result["complete_contact_evidence"] is True


def test_short_bounce_cannot_pass_even_if_old_report_claims_jump(paired):
    forces = paired["m_arrays"]["wheel_ground_contact_force_w_n"]
    forces[:, :, 2] = 100.
    forces[120:132] = 0.  # 60 ms bounce, far below required 200 ms.
    paired["mujoco"]["jump_and_landing_detected"] = True
    paired["mujoco"]["full_clip_with_detected_jump_and_landing"] = True
    result = assess(paired)
    assert "mujoco.sustained_flight_followed_by_landing" in result["fail_reasons"]


def test_earlier_bounce_does_not_hide_later_real_jump(paired):
    paired["m_arrays"]["wheel_ground_contact_force_w_n"][40:52] = 0.
    result = assess(paired)
    assert result["verdict"] == "pass"
    assert result["mujoco"]["selected_flight"]["takeoff_s"] == pytest.approx(.6)


def test_airborne_reset_is_not_a_takeoff(paired):
    paired["m_arrays"]["wheel_ground_contact_force_w_n"][:200] = 0.
    result = assess(paired)
    assert "mujoco.sustained_flight_followed_by_landing" in result["fail_reasons"]


def test_no_landing_fails_even_with_enough_height_and_flight(paired):
    paired["m_arrays"]["wheel_ground_contact_force_w_n"][120:] = 0.
    result = assess(paired)
    assert "mujoco.sustained_flight_followed_by_landing" in result["fail_reasons"]
    assert "mujoco.stable_two_wheel_tail_support" in result["fail_reasons"]


def test_measured_height_not_reference_or_report_controls_gate(paired):
    paired["m_arrays"]["root_pos"][:, 2] = .9
    paired["mujoco"]["base_link_height_gain_m"] = 9.
    result = assess(paired)
    assert "mujoco.minimum_base_height_gain" in result["fail_reasons"]


def test_unstable_tail_support_fails(paired):
    paired["m_arrays"]["wheel_ground_contact_force_w_n"][350:, 0] = 0.
    result = assess(paired)
    assert "mujoco.stable_two_wheel_tail_support" in result["fail_reasons"]


def test_tail_tilt_fails_despite_final_upright_sample(paired):
    angle = np.radians(40.)
    paired["m_arrays"]["root_quat_wxyz"][350] = [np.cos(angle / 2), np.sin(angle / 2), 0, 0]
    result = assess(paired)
    assert "mujoco.stable_tail_tilt" in result["fail_reasons"]
    assert result["mujoco"]["max_tail_tilt_deg"] == pytest.approx(40.)


@pytest.mark.parametrize("field", ["nonwheel_ground_contact_count", "nonwheel_ground_contact_force_norm_n"])
def test_nonwheel_ground_hit_is_a_failure_when_recorded(paired, field):
    paired["m_arrays"][field] = np.zeros(401)
    paired["m_arrays"][field][210] = 1.
    result = assess(paired)
    assert "mujoco.no_nonwheel_ground_contact" in result["fail_reasons"]


def test_early_termination_remains_failure_despite_forged_completion(paired):
    paired["mujoco"]["termination"] = "early_termination"
    paired["mujoco"]["termination_terms"] = ["ee_body_pos"]
    result = assess(paired)
    assert "mujoco.no_early_termination" in result["fail_reasons"]


def test_shortened_recording_cannot_pass_from_completion_flag(paired):
    for key, values in list(paired["m_arrays"].items()):
        if values.shape[0] == 401:
            paired["m_arrays"][key] = values[:360]
    result = assess(paired)
    assert "mujoco.full_recorded_duration" in result["fail_reasons"]
    assert "mujoco.completed_full_reference" in result["fail_reasons"]


@pytest.mark.parametrize("field,value,reason", [
    ("policy_sha256", "b" * 64, "provenance.same_actor"),
    ("motion_sha256", "b" * 64, "provenance.same_reference"),
    ("root_state_writes", 2, "mujoco.physics_only_episode_no_hidden_resets"),
    ("hidden_resets", 1, "mujoco.physics_only_episode_no_hidden_resets"),
    ("start_reference_frame", 5, "mujoco.frame_zero_start"),
])
def test_provenance_and_episode_integrity(paired, field, value, reason):
    paired["mujoco"][field] = value
    assert reason in assess(paired)["fail_reasons"]


def test_post_reset_tail_cannot_enter_metrics(paired):
    paired["i_arrays"]["valid_mask"][80:, 0] = False
    result = assess(paired)
    assert "isaac.full_recorded_duration" in result["fail_reasons"]
    assert "isaac.completed_full_reference" in result["fail_reasons"]


def test_reference_phase_reset_is_rejected(paired):
    paired["m_arrays"]["reference_frame"][200] = 0
    assert "mujoco.monotonic_reference_without_reset" in assess(paired)["fail_reasons"]


def test_irregular_contact_samples_cannot_pass_count_based_flight_detector(paired):
    paired["m_arrays"]["time_s"][100] += .001
    assert "mujoco.uniform_contact_sampling" in assess(paired)["fail_reasons"]


def test_missing_ground_forces_not_replaced_by_net_forces_in_mujoco(paired):
    paired["m_arrays"]["wheel_contact_force_w_n"] = paired["m_arrays"].pop("wheel_ground_contact_force_w_n")
    result = assess(paired)
    assert result["mujoco"]["contact_force_source"] == "unavailable"
    assert result["verdict"] == "fail"


def test_support_fraction_is_time_weighted_not_sample_weighted():
    times = np.array([0., .5, .9, .99, 1.])
    supported = np.array([False, True, False, False, False])
    assert tail_support_fraction(times, supported, .5) == pytest.approx(.8)
    assert tail_support_fraction(times, supported, 2.) is None


def test_cli_writes_new_report_and_refuses_overwrite(paired, tmp_path):
    output = tmp_path / "assessment.json"
    args = ["--isaac-dir", str(paired["isaac_dir"]), "--mujoco-dir", str(paired["mujoco_dir"]),
            "--output", str(output)]
    assert main(args) == 0
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        main(args)
    assert output.read_bytes() == original


@pytest.mark.parametrize("kwargs", [{"min_flight": 0.}, {"min_height_gain": float("nan")},
                                    {"min_tail_support": 1.1}, {"max_tail_tilt_deg": 181.}])
def test_invalid_thresholds_rejected(paired, kwargs):
    with pytest.raises(ValueError):
        assess(paired, **kwargs)


def shared_dc(pair):
    motor = {"actuator_model": "dc_motor", "leg_saturation_effort_nm": 80., "wheel_saturation_effort_nm": 12.,
             "leg_motor_velocity_limit_rad_s": 15., "wheel_motor_velocity_limit_rad_s": 100.,
             "solver_joint_velocity_limit_rad_s": 1000.,
             "motor_velocity_limit_semantics": "no_load_speed_for_torque_speed_curve_not_hard_qvel_clip"}
    pair["isaac"]["policy_export"].update(motor)
    pair["mujoco"]["contract"].update(motor)
    pair["isaac"]["legacy_joint_friction_audit"] = {
        "joint_names": list(NAMES), "before_first_environment": [0.] * 6 + [.01, .01],
        "after_first_environment": [0.] * 8, "all_environments_verified_zero": True,
    }
    pair["mujoco"]["actual_joint_velocity_hard_clipped"] = False
    return pair


def test_historical_missing_model_defaults_to_ideal_pd(paired):
    result = assess(paired)
    model = result["provenance"]["actuator_contract"]["fields"]["actuator_model"]
    assert model == {"isaac": "ideal_pd", "mujoco": "ideal_pd", "matched": True}
    assert result["verdict"] == "pass"
    paired["mujoco"]["contract"]["actuator_model"] = "ideal_pd"
    assert assess(paired)["verdict"] == "pass"


@pytest.mark.parametrize("field", ["leg_kp", "leg_kd", "leg_position_scale", "wheel_torque_scale_nm",
                                  "leg_torque_limit_nm", "wheel_torque_limit_nm", "raw_action_clip",
                                  "physics_dt", "decimation", "policy_fps"])
def test_common_actuator_parameter_mismatch_fails(paired, field):
    contract = paired["mujoco"]["contract"]
    contract[field] = [-.5, .5] if field == "raw_action_clip" else contract[field] * 2
    result = assess(paired)
    assert "provenance.matching_actuator_contract" in result["fail_reasons"]
    assert field in result["provenance"]["actuator_contract"]["mismatched_fields"]


@pytest.mark.parametrize("side", ["isaac", "mujoco", "both"])
@pytest.mark.parametrize("field", ["leg_kp", "leg_kd", "leg_position_scale", "wheel_torque_scale_nm",
                                  "leg_torque_limit_nm", "wheel_torque_limit_nm", "raw_action_clip",
                                  "physics_dt", "decimation", "policy_fps"])
def test_missing_common_parameters_never_match_by_two_defaults(paired, side, field):
    if side in ("isaac", "both"):
        del paired["isaac"]["policy_export"][field]
    if side in ("mujoco", "both"):
        del paired["mujoco"]["contract"][field]
    result = assess(paired)
    assert "provenance.matching_actuator_contract" in result["fail_reasons"]
    assert field in result["provenance"]["actuator_contract"]["mismatched_fields"]


def test_matching_but_inconsistent_control_clock_fails(paired):
    paired["isaac"]["policy_export"]["decimation"] = 8
    paired["mujoco"]["contract"]["decimation"] = 8
    result = assess(paired)
    assert "consistent_control_clock" in result["provenance"]["actuator_contract"]["mismatched_fields"]


def test_shared_dc_contract_and_actual_friction_audit_pass(paired):
    result = assess(shared_dc(paired))
    assert result["verdict"] == "pass"
    assert result["provenance"]["checks"]["legacy_joint_friction_cleared_and_audited"]["passed"] is True
    assert result["provenance"]["checks"]["actual_joint_velocities_not_hard_clipped"]["passed"] is True
    motor = result["provenance"]["actuator_contract"]
    assert motor["hardware_ready"] is False
    assert motor["parameter_assumptions"]["isaac"]
    assert motor["fields"]["solver_joint_velocity_limit_rad_s"]["isaac"] == 1000.


@pytest.mark.parametrize("field", ["leg_saturation_effort_nm", "wheel_saturation_effort_nm",
                                  "leg_motor_velocity_limit_rad_s", "wheel_motor_velocity_limit_rad_s",
                                  "solver_joint_velocity_limit_rad_s", "motor_velocity_limit_semantics"])
def test_dc_specific_parameters_are_required_on_both_sides(paired, field):
    shared_dc(paired)
    del paired["mujoco"]["contract"][field]
    result = assess(paired)
    assert "provenance.matching_actuator_contract" in result["fail_reasons"]
    assert field in result["provenance"]["actuator_contract"]["mismatched_fields"]


@pytest.mark.parametrize("field", ["leg_saturation_effort_nm", "wheel_saturation_effort_nm",
                                  "leg_motor_velocity_limit_rad_s", "wheel_motor_velocity_limit_rad_s",
                                  "solver_joint_velocity_limit_rad_s"])
def test_dc_specific_parameter_disagreement_fails(paired, field):
    shared_dc(paired)
    paired["mujoco"]["contract"][field] *= 2
    result = assess(paired)
    assert field in result["provenance"]["actuator_contract"]["mismatched_fields"]


def test_policy_bytes_do_not_hide_different_motor_models(paired):
    shared_dc(paired)
    paired["mujoco"]["contract"]["actuator_model"] = "ideal_pd"
    result = assess(paired)
    assert "provenance.matching_actuator_contract" in result["fail_reasons"]
    assert result["provenance"]["checks"]["same_actor"]["passed"] is True


@pytest.mark.parametrize("change", ["missing", "not_verified", "nonzero", "nan", "missing_after",
                                   "too_short", "missing_joint", "duplicate_joint", "unexpected_joint"])
def test_dc_friction_audit_fails_closed(paired, change):
    shared_dc(paired)
    audit = paired["isaac"]["legacy_joint_friction_audit"]
    if change == "missing":
        del paired["isaac"]["legacy_joint_friction_audit"]
    elif change == "not_verified":
        audit["all_environments_verified_zero"] = False
    elif change in ("nonzero", "nan"):
        audit["after_first_environment"][-1] = .01 if change == "nonzero" else float("nan")
    elif change == "missing_after":
        del audit["after_first_environment"]
    elif change == "too_short":
        audit["after_first_environment"].pop()
    elif change == "missing_joint":
        audit["joint_names"].pop()
    elif change == "duplicate_joint":
        audit["joint_names"][-1] = audit["joint_names"][0]
    else:
        audit["joint_names"][-1] = "not_a_TRON1_joint"
    result = assess(paired)
    assert "provenance.legacy_joint_friction_cleared_and_audited" in result["fail_reasons"]


@pytest.mark.parametrize("value", [True, None, 0])
def test_dc_qvel_hard_clip_evidence_must_explicitly_be_false(paired, value):
    shared_dc(paired)
    if value is None:
        del paired["mujoco"]["actual_joint_velocity_hard_clipped"]
    else:
        paired["mujoco"]["actual_joint_velocity_hard_clipped"] = value
    assert "provenance.actual_joint_velocities_not_hard_clipped" in assess(paired)["fail_reasons"]


@pytest.mark.parametrize("field,value", [("hardware_ready", True), ("actuator_limits_provenance", "")])
def test_simulation_assumptions_cannot_be_promoted_to_hardware_claims(paired, field, value):
    paired["mujoco"]["contract"][field] = value
    assert "provenance.simulation_parameter_assumptions_disclosed" in assess(paired)["fail_reasons"]


@pytest.mark.parametrize("field,value", [("leg_kp", float("nan")), ("leg_kd", float("inf")),
    ("raw_action_clip", [float("nan"), 1.]), ("raw_action_clip", [[1.], [1., 2.]]),
    ("raw_action_clip", [1., -1.]), ("decimation", 2.5)])
def test_invalid_actuator_values_fail_with_json_serializable_report(paired, field, value):
    paired["mujoco"]["contract"][field] = value
    result = assess(paired)
    assert "provenance.matching_actuator_contract" in result["fail_reasons"]
    assert field in result["provenance"]["actuator_contract"]["mismatched_fields"]
    json.dumps(result, allow_nan=False)


def test_unknown_actuator_model_does_not_pass_when_both_sides_use_same_name(paired):
    paired["isaac"]["policy_export"]["actuator_model"] = "unknown_motor"
    paired["mujoco"]["contract"]["actuator_model"] = "unknown_motor"
    assert "provenance.matching_actuator_contract" in assess(paired)["fail_reasons"]


def test_dc_audit_matches_declared_actual_joint_order(paired):
    shared_dc(paired)
    paired["isaac"]["policy_export"]["actual_joint_order"] = list(reversed(NAMES))
    assert "provenance.legacy_joint_friction_cleared_and_audited" in assess(paired)["fail_reasons"]


def test_changed_shared_dc_profile_is_not_silently_accepted(paired):
    shared_dc(paired)
    paired["isaac"]["policy_export"]["leg_motor_velocity_limit_rad_s"] = 30.
    paired["mujoco"]["contract"]["leg_motor_velocity_limit_rad_s"] = 30.
    result = assess(paired)
    assert "leg_motor_velocity_limit_rad_s" in result["provenance"]["actuator_contract"]["mismatched_fields"]
