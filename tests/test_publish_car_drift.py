"""Offline publication tests; generated fixture states are not experimental evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from assess_drift import assess_runs
from eval_tracking_mujoco import read_contract
import publish_car_drift as publish


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def isaac_material():
    return {"expected_friction": .15, "robot_runtime_friction_range": [.15, .15],
            "ground_colliders": [{"collider": "/World/ground/terrain/CollisionPlane",
                                  "material": "/World/materials/ground", "static_friction": .15,
                                  "dynamic_friction": .15}]}


@pytest.fixture
def trial(tmp_path):
    trial = tmp_path / "trial"
    reference = tmp_path / "reference"
    render = tmp_path / "render"
    output = tmp_path / "public_snapshot"
    for directory in (trial / "training", trial / "isaac", trial / "mujoco", reference, render):
        directory.mkdir(parents=True)
    frame_t = np.arange(50) / 50.
    yaw = np.minimum(frame_t / .4, 1.) * .3
    beta = np.where((frame_t >= .1) & (frame_t < .36), np.deg2rad(12.), 0.)
    speed = .6 * np.clip((.48 - frame_t) / .08, 0., 1.)
    velocity = np.c_[speed * np.cos(yaw + beta), speed * np.sin(yaw + beta), np.zeros(50)]
    root = np.zeros((50, 3))
    root[:, 2] = .75
    root[1:, :2] = np.cumsum(.5 * (velocity[1:, :2] + velocity[:-1, :2]) / 50., axis=0)
    quat = np.c_[np.cos(yaw / 2), np.zeros((50, 2)), np.sin(yaw / 2)]
    names = np.array(read_contract()["observed_joint_velocity_order"])
    motion = reference / "motion.npz"
    np.savez(motion, fps=np.array([50.]), body_pos_w=root[:, None], body_quat_w=quat[:, None],
             body_lin_vel_w=velocity[:, None], body_ang_vel_w=np.zeros((50, 1, 3)),
             joint_pos=np.zeros((50, 8)), joint_vel=np.zeros((50, 8)), joint_names=names,
             body_names=np.array(["base_Link"]))
    np.savez(reference / "robot_reference.npz", qpos=np.zeros((50, 15)))
    actor = trial / "isaac/actor_normalized.pt"
    actor.write_bytes(b"synthetic actor fixture; never executed")
    times = np.arange(51) / 50.
    extend = lambda values: np.concatenate((values, values[-1:]), axis=0)
    frame_ids = np.minimum(np.arange(51), 49)
    np.savez(trial / "isaac/trajectory.npz", time_s=times, joint_names=names,
             root_position_m=extend(root)[:, None], root_quaternion_wxyz=extend(quat)[:, None],
             root_linear_velocity_w_m_s=extend(velocity)[:, None], joint_position_rad=np.zeros((51, 1, 8)),
             action_observation=np.zeros((51, 1, 51)), valid_mask=np.ones((51, 1), dtype=bool),
             reference_frame=frame_ids[:, None], wheel_ground_contact_count=np.ones((51, 1, 2), dtype=int),
             wheel_lateral_contact_slip_m_s=np.full((51, 1, 2), .1),
             nonwheel_ground_contact_count=np.zeros((51, 1), dtype=int))
    np.savez(trial / "mujoco/rollout.npz", time_s=times, joint_names=names,
             root_pos=extend(root), root_quat_wxyz=extend(quat), root_com_lin_vel_w=extend(velocity),
             joint_pos=np.zeros((51, 8)), action_observation=np.zeros((50, 51)), reference_frame=frame_ids,
             wheel_ground_contact_count=np.ones((51, 2), dtype=int),
             wheel_lateral_contact_slip_m_s=np.full((51, 2), .1), nonwheel_ground_contact_count=np.zeros(51, dtype=int))
    profile = {"schema_version": 1, "task": "car_drift", "contact_friction": .15,
               "env_spacing_m": 8., "reward_weights": {"sideslip": 2., "yaw_rate": 1., "speed": 1., "grounded": .4, "tilt": -.2}}
    contract = {**read_contract(), "task_profile": profile,
                "reference_file_sha256": publish.sha256(motion), "actor_file": actor.name,
                "actor_includes_observation_normalizer": True, "checkpoint_sha256": "d" * 64}
    write_json(trial / "isaac/policy_contract.json", contract)
    outcome = {"start_reference_frame": 0, "final_reference_frame": 49, "completed_full_reference": True,
               "termination_terms": ["motion_end"], "early_terminated": False}
    isaac = {"simulator": "IsaacLab/PhysX", "motion_file_sha256": publish.sha256(motion),
             "policy_export": contract, "task_profile": profile, "reference_frames": 50,
             "wheel_lateral_contact_slip_evidence_kind": "actual_ground_contact_points",
             "contact_material_audit": isaac_material(), "terminal_states_captured_before_auto_reset": True,
             "reference_state_initialization_during_evaluation": False,
             "summary": {"episodes": [{**outcome, "environment": 0, "end_reason": "timeout"}]},
             "host": "private-gpu-node", "note": "ssh researcher@10.20.30.40 and /home/researcher/run"}
    mujoco = {"engine": "MuJoCo", "motion_sha256": publish.sha256(motion), "policy_sha256": publish.sha256(actor),
              "wheel_lateral_contact_slip_evidence_kind": "actual_ground_contact_points",
              "policy_kind": "normalized_torchscript_actor", "contract": copy.deepcopy(contract),
              "reference_frames": 50, "termination": "motion_end", "physics_stepped": True,
              "root_state_writes": 1, "hidden_resets": 0, **outcome,
              "contact_material_audit": {"expected_friction": .15, "colliding_geom_count": 4,
                                         "compiled_friction_range": [.15, .15]},
              "isaac_observation_validation": {"status": "passed", "observations_checked": 1,
                                               "trajectory_sha256": publish.sha256(trial / "isaac/trajectory.npz"),
                                               "max_abs_error": 0.}}
    write_json(trial / "isaac/report.json", isaac)
    write_json(trial / "mujoco/report.json", mujoco)
    source = {"source_repository": "https://github.com/ustcly/DOA", "source_commit": "a" * 40,
              "source_csv_path": "expert_traj/example.csv", "source_csv_sha256": "b" * 64,
              "source_npz_sha256": "c" * 64, "source_audit_sha256": "e" * 64,
              "source_configuration_sha256": "f" * 64, "source_is_physical_car_capture": False,
              "source_clock_status": "reconstructed_estimate_not_original_timestamps",
              "source_note": "Host private-gpu-node, user researcher, /data/researcher/private"}
    recipe = {"spatial_scale": .08, "time_scale": 2., "recovery_duration_s": 1.5, "smoothing_source_s": 0.}
    adaptation = {"provenance": source, "calibration_and_adaptation": recipe, "diagnostics": {"simulation_steps": 0},
                  "motion_sha256": publish.sha256(motion), "robot_reference_sha256": publish.sha256(reference / "robot_reference.npz"),
                  "source_modified": False, "physics_validated": False}
    write_json(reference / "adaptation.json", adaptation)
    write_json(reference / "motion.json", {"source_sha256": publish.sha256(reference / "robot_reference.npz"),
               "output_sha256": publish.sha256(motion), "upstream_task_adaptation": recipe,
               "source_car_provenance": source, "frames": 50, "fps": 50., "sample_span_s": .98,
               "positions": "link origins", "linear_velocity": "world COM velocity", "source_time_mapping": "explicit adaptation"})
    write_json(trial / "training/manifest.json", {"motion_sha256": publish.sha256(motion), "task_profile": profile,
               "contact_material_audit": isaac_material(), "num_envs": 2, "iterations_requested": 3,
               "seed": 42, "resume_sha256": "9" * 64, "host": "private-gpu-node",
               "python": "/home/researcher/conda/python", "source_sha256": {"scripts/train_tracking.py": "8" * 64}})
    write_json(trial / "training/run_status.json", {"status": "training_completed", "num_envs": 2,
               "iterations_requested": 3, "last_iteration": 102, "training_environment_steps": 144})
    write_json(trial / "training/runner_config.json", {"num_steps_per_env": 24})
    write_json(trial / "assessment.json", assess_runs(trial / "isaac", trial / "mujoco", motion))
    return trial, reference, render, output


def add_render(fixture):
    trial, reference, render, _ = fixture
    for filename in ("motion_pipeline.mp4", "overview.png", "final.png", "contact_overview.png"):
        (render / filename).write_bytes((filename + " synthetic media fixture").encode())
    report = {"status": "rendered", "simulation_steps": 0, "full_decode_verified": True,
              "source_hashes": {key: publish.sha256(path) for key, path in {
                  "motion": reference / "motion.npz", "robot_reference": reference / "robot_reference.npz",
                  "isaac": trial / "isaac/trajectory.npz", "mujoco": trial / "mujoco/rollout.npz",
                  "isaac_report": trial / "isaac/report.json", "mujoco_report": trial / "mujoco/report.json",
                  "adaptation": reference / "adaptation.json", "export_report": reference / "motion.json",
                  "assessment": trial / "assessment.json"}.items()},
              "video_sha256": publish.sha256(render / "motion_pipeline.mp4"),
              "overview_sha256": publish.sha256(render / "overview.png"),
              "final_sha256": publish.sha256(render / "final.png"),
              "contact_overview_sha256": publish.sha256(render / "contact_overview.png")}
    report["source_hashes"]["captured_source"] = "c" * 64
    isaac = publish.read_json(trial / "isaac/report.json")
    if "physics_trajectory_file" in isaac:
        report["source_hashes"]["isaac_physics"] = publish.sha256(trial / "isaac" / isaac["physics_trajectory_file"])
    write_json(render / "render_report.json", report)


def add_physics_recording(fixture):
    """Generated clock-valid telemetry; these arrays are never physics evidence."""
    trial, reference, _, _ = fixture
    physical_times = np.arange(201) / 200.

    def resample(path):
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        old_times = arrays["time_s"]
        indices = np.searchsorted(old_times, physical_times, side="right") - 1
        for key, value in list(arrays.items()):
            if key == "time_s" or value.shape[:1] != old_times.shape or key == "action_observation":
                continue
            if value.dtype.kind == "f" and "slip" not in key:
                columns = value.reshape((len(old_times), -1))
                arrays[key] = np.column_stack([np.interp(physical_times, old_times, column)
                                              for column in columns.T]).reshape((201, *value.shape[1:]))
            else:
                arrays[key] = value[indices]
        arrays["time_s"] = physical_times
        quaternion_key = "root_quaternion_wxyz" if "root_quaternion_wxyz" in arrays else "root_quat_wxyz"
        quaternion = arrays[quaternion_key]
        quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)
        return arrays

    physical = resample(trial / "isaac/trajectory.npz")
    physical.update({
        "physics_dt_s": np.array(.005), "policy_dt_s": np.array(.02),
        "simulation_step_counter": np.arange(201, dtype=np.int64) + 900,
        "action_observation": np.zeros((201, 1, 51)),
        "actions_raw": np.zeros((201, 1, 8)), "actions_clipped": np.zeros((201, 1, 8)),
        "action_was_applied": np.r_[False, np.ones(200, dtype=bool)][:, None],
        "reference_frame_for_action": np.clip((np.arange(201) - 1) // 4, 0, 49)[:, None],
        "terminal_sample": np.r_[np.zeros(200, dtype=bool), True][:, None],
    })
    physics_path = trial / "isaac/physics_trajectory.npz"
    np.savez(physics_path, **physical)
    report_path = trial / "isaac/report.json"
    report = publish.read_json(report_path)
    report.update({
        "physics_trajectory_file": physics_path.name,
        "physics_trajectory_sha256": publish.sha256(physics_path), "physics_dt_s": .005,
        "physics_recording": {
            "sample_dt_s": .005, "samples": 201, "time_source": "relative simulator physics-step counter",
            "hook": "after original scene.update at each incremented simulator counter during wrapped.step",
            "initial_state_samples": 1, "terminal_states_before_reset": True,
            "same_counter_reset_and_forward_updates_excluded": True, "first_episode_only_valid_mask": True,
            "observations": "actual policy pre-action observations held across four physics substeps; validate actor only with trajectory.npz",
            "primary_policy_trajectory_file": "trajectory.npz", "policy_sample_dt_s": .02,
        },
    })
    write_json(report_path, report)
    mujoco = resample(trial / "mujoco/rollout.npz")
    mujoco["physics_dt_s"] = np.array(.005)
    mujoco["policy_dt_s"] = np.array(.02)
    np.savez(trial / "mujoco/rollout.npz", **mujoco)
    reassess(fixture)


def publish_fixture(fixture, include_render=False):
    trial, reference, render, output = fixture
    return publish.publish_trial(trial, reference, output, render if include_render else None)


def reassess(fixture):
    trial, reference, _, _ = fixture
    write_json(trial / "assessment.json", assess_runs(trial / "isaac", trial / "mujoco", reference / "motion.npz"))


def test_publisher_exports_only_sanitized_hash_linked_evidence(trial):
    result = publish_fixture(trial)
    output = trial[-1]
    assert sorted(path.name for path in output.iterdir()) == ["assessment.json", "reference_audit.json", "summary.json"]
    assert not list(output.rglob("*.pt")) and not list(output.rglob("*.npz"))
    assert result["same_actor_reference_verified"]
    assert result["actual_physics_rollouts_verified"]
    assert result["training"]["requested_budget_completion_verified"]
    assert result["training"]["completed_requested_iterations"] == 3
    assert result["training"]["resume_sha256"] == "9" * 64
    assert result["hardware_ready"] is False
    for name, expected in result["published_artifact_sha256"].items():
        assert publish.sha256(output / name) == expected
    text = "\n".join(path.read_text() for path in output.glob("*.json"))
    for forbidden in ("private-gpu-node", "researcher", "10.20.30.40", "/home/", "/data/", "actor_normalized.pt"):
        assert forbidden not in text
    assert "https://github.com/ustcly/DOA" in text
    assert "reconstructed_estimate_not_original_timestamps" in text


def test_behavior_failures_are_publishable_and_remain_failures(trial):
    path = trial[0] / "mujoco/rollout.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays["wheel_lateral_contact_slip_m_s"][:] = 0.
    np.savez(path, **arrays)
    reassess(trial)
    result = publish_fixture(trial)
    assert result["verdict"] == "fail" and result["behavior_pass"] is False
    assert "mujoco.sustained_grounded_contact_patch_drift" in result["failed_checks"]
    assert result["engines"]["mujoco"]["verdict"] == "fail"
    assert publish.read_json(trial[-1] / "assessment.json")["verdict"] == "fail"


def test_render_publication_checks_input_and_media_hashes(trial):
    add_render(trial)
    result = publish_fixture(trial, include_render=True)
    assert result["rendered_comparison_included"]
    for filename in ("motion_pipeline.mp4", "overview.png", "final.png", "contact_overview.png"):
        assert publish.sha256(trial[-1] / filename) == publish.sha256(trial[2] / filename)
    assert (trial[-1] / "render_report.json").exists()


@pytest.mark.parametrize("key", ["motion", "robot_reference", "isaac", "mujoco", "isaac_report", "mujoco_report", "adaptation", "export_report", "assessment", "captured_source"])
def test_stale_render_inputs_leave_no_public_output(trial, key):
    add_render(trial)
    path = trial[2] / "render_report.json"
    value = publish.read_json(path)
    value["source_hashes"][key] = "0" * 64
    write_json(path, value)
    with pytest.raises(ValueError, match="Render.*SHA"):
        publish_fixture(trial, include_render=True)
    assert not trial[-1].exists()


@pytest.mark.parametrize("filename", ["motion_pipeline.mp4", "overview.png", "final.png", "contact_overview.png"])
def test_tampered_media_cannot_be_published(trial, filename):
    add_render(trial)
    path = trial[2] / filename
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Render.*SHA"):
        publish_fixture(trial, include_render=True)
    assert not trial[-1].exists()


def test_assessment_is_recomputed_and_stale_pass_claim_is_rejected(trial):
    path = trial[0] / "assessment.json"
    assessment = publish.read_json(path)
    assessment["isaac"]["metrics"]["max_planar_com_speed_m_s"] = 123.
    write_json(path, assessment)
    with pytest.raises(ValueError, match="freshly assessed"):
        publish_fixture(trial)
    assert not trial[-1].exists()


def test_existing_even_empty_snapshot_directory_is_never_reused(trial):
    trial[-1].mkdir()
    with pytest.raises(ValueError, match="NEW"):
        publish_fixture(trial)
    assert list(trial[-1].iterdir()) == []


def test_manifest_alone_does_not_claim_requested_iterations_completed(trial):
    (trial[0] / "training/run_status.json").unlink()
    result = publish_fixture(trial)
    assert result["training"]["completion_status"] == "not_supplied"
    assert result["training"]["completed_requested_iterations"] is None
    assert result["training"]["requested_budget_completion_verified"] is False


@pytest.mark.parametrize("engine", ["training", "isaac", "mujoco"])
def test_matching_profile_numbers_do_not_replace_actual_material_audits(trial, engine):
    path = trial[0] / {"training": "training/manifest.json", "isaac": "isaac/report.json", "mujoco": "mujoco/report.json"}[engine]
    value = publish.read_json(path)
    value.pop("contact_material_audit")
    write_json(path, value)
    reassess(trial)
    with pytest.raises(ValueError, match="contact-material audit"):
        publish_fixture(trial)
    assert not trial[-1].exists()


def test_training_material_audit_mismatch_leaves_no_publication(trial):
    path = trial[0] / "training/manifest.json"
    value = publish.read_json(path)
    value["contact_material_audit"]["ground_colliders"][0]["dynamic_friction"] = .6
    write_json(path, value)
    with pytest.raises(ValueError, match="ground friction"):
        publish_fixture(trial)
    assert not trial[-1].exists()


def test_hidden_root_teleport_cannot_be_published_as_physics_evidence(trial):
    path = trial[0] / "mujoco/report.json"
    value = publish.read_json(path)
    value["root_state_writes"] = 2
    write_json(path, value)
    reassess(trial)
    with pytest.raises(ValueError, match="physics-only"):
        publish_fixture(trial)
    assert not trial[-1].exists()


@pytest.mark.parametrize("relative", ["isaac/actor_normalized.pt", "mujoco/rollout.npz", "isaac/policy_contract.json", "training/manifest.json"])
def test_required_missing_inputs_are_rejected_before_any_output(trial, relative):
    (trial[0] / relative).unlink()
    with pytest.raises(ValueError, match="Missing or empty"):
        publish_fixture(trial)
    assert not trial[-1].exists()


@pytest.mark.parametrize("target", ["motion", "robot_reference"])
def test_reference_hash_tampering_is_rejected_before_output(trial, target):
    path = trial[1] / "adaptation.json"
    value = publish.read_json(path)
    value[target + "_sha256"] = "0" * 64
    write_json(path, value)
    with pytest.raises(ValueError, match="not linked"):
        publish_fixture(trial)
    assert not trial[-1].exists()


def test_nonhex_source_hash_does_not_count_as_pinned_provenance(trial):
    adaptation_path = trial[1] / "adaptation.json"
    export_path = trial[1] / "motion.json"
    adaptation, export = publish.read_json(adaptation_path), publish.read_json(export_path)
    adaptation["provenance"]["source_csv_sha256"] = "g" * 64
    export["source_car_provenance"]["source_csv_sha256"] = "g" * 64
    write_json(adaptation_path, adaptation)
    write_json(export_path, export)
    with pytest.raises(ValueError, match="pinned car-source"):
        publish_fixture(trial)
    assert not trial[-1].exists()


def test_reference_model_mismatch_is_rejected(trial):
    path = trial[1] / "adaptation.json"
    value = publish.read_json(path)
    value["model_sha256"] = "a" * 64
    write_json(path, value)
    path = trial[1] / "motion.json"
    value = publish.read_json(path)
    value["model_sha256"] = "b" * 64
    write_json(path, value)
    with pytest.raises(ValueError, match="different robot model"):
        publish_fixture(trial)
    assert not trial[-1].exists()


def test_training_profile_cannot_differ_from_both_deployments(trial):
    path = trial[0] / "training/manifest.json"
    value = publish.read_json(path)
    value["task_profile"]["contact_friction"] = .2
    write_json(path, value)
    with pytest.raises(ValueError, match="same drift task"):
        publish_fixture(trial)
    assert not trial[-1].exists()


def test_source_renderer_requires_completed_decode_verification(trial):
    add_render(trial)
    path = trial[2] / "render_report.json"
    value = publish.read_json(path)
    value["full_decode_verified"] = False
    write_json(path, value)
    with pytest.raises(ValueError, match="decoded playback"):
        publish_fixture(trial, include_render=True)
    assert not trial[-1].exists()


def test_standalone_cli_imports_training_package_when_cwd_is_outside_repository(trial, tmp_path):
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/publish_car_drift.py"),
        "--trial-dir", str(trial[0]), "--reference-dir", str(trial[1]),
        "--output-dir", str(trial[-1]),
    ], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    message = json.loads(result.stdout)
    assert set(message) == {"status", "verdict", "output_dir"}
    assert message["status"] == "assessed_" + message["verdict"]
    assert message["output_dir"] == str(trial[-1])
    assert len(result.stdout.splitlines()) == 1
    summary = publish.read_json(trial[-1] / "summary.json")
    assert summary["same_actor_reference_verified"] is True
    assert summary["actual_physics_rollouts_verified"] is True
    assert (trial[-1] / "summary.json").is_file()


def test_physics_rate_evidence_is_hash_linked_but_never_copied(trial):
    add_physics_recording(trial)
    result = publish_fixture(trial)
    physics_path = trial[0] / "isaac/physics_trajectory.npz"
    assert result["input_sha256"]["isaac_physics_trajectory"] == publish.sha256(physics_path)
    assert result["input_sha256"]["isaac_trajectory"] == publish.sha256(trial[0] / "isaac/trajectory.npz")
    assert result["same_actor_provenance"]["isaac_trajectory_sha256"] == result["input_sha256"]["isaac_trajectory"]
    behavior = result["behavior_evidence"]["isaac"]
    assert behavior == {"file": "physics_trajectory.npz", "sha256": publish.sha256(physics_path),
                        "sample_dt_s": .005, "kind": "physics_rate"}
    assert result["evaluation"]["isaac"]["physics_trajectory_sha256"] == publish.sha256(physics_path)
    assert result["evaluation"]["isaac"]["physics_recording"]["terminal_states_before_reset"] is True
    assert result["evaluation"]["isaac"]["physics_dt_s"] == .005
    assert result["provenance_checks"] == publish.read_json(trial[0] / "assessment.json")["provenance_checks"]
    assert not list(trial[-1].rglob("*.npz"))


def test_physics_rate_render_hash_matches_selected_behavior_archive(trial):
    add_physics_recording(trial)
    add_render(trial)
    result = publish_fixture(trial, include_render=True)
    render = publish.read_json(trial[-1] / "render_report.json")
    assert render["source_hashes"]["isaac_physics"] == result["input_sha256"]["isaac_physics_trajectory"]
    assert render["source_hashes"]["isaac"] == result["input_sha256"]["isaac_trajectory"]


def test_summary_preserves_actual_evaluator_contact_source_field_names(trial):
    add_physics_recording(trial)
    for engine in ("isaac", "mujoco"):
        path = trial[0] / engine / "report.json"
        report = publish.read_json(path)
        report["contact_slip_helper_sha256"] = "1" * 64
        report["contact_slip_measurement"] = {"ground_is_stationary": True, "sample_dt_s": .005,
                                             "aggregation": "normal-force-weighted RMS over active actual ground contact points"}
        if engine == "isaac":
            report["drift_module_sha256"] = "2" * 64
        write_json(path, report)
    reassess(trial)
    result = publish_fixture(trial)
    for engine in ("isaac", "mujoco"):
        evaluation = result["evaluation"][engine]
        assert evaluation["contact_slip_helper_sha256"] == "1" * 64
        assert evaluation["wheel_lateral_contact_slip_evidence_kind"] == "actual_ground_contact_points"
        assert evaluation["contact_slip_measurement"]["sample_dt_s"] == .005
    assert result["evaluation"]["isaac"]["drift_module_sha256"] == "2" * 64


@pytest.mark.parametrize("change", ["missing", "tampered"])
def test_missing_or_tampered_render_physics_hash_leaves_no_output(trial, change):
    add_physics_recording(trial)
    add_render(trial)
    path = trial[2] / "render_report.json"
    render = publish.read_json(path)
    if change == "missing":
        del render["source_hashes"]["isaac_physics"]
    else:
        render["source_hashes"]["isaac_physics"] = "0" * 64
    write_json(path, render)
    with pytest.raises(ValueError, match="Render isaac_physics.*SHA"):
        publish_fixture(trial, include_render=True)
    assert not trial[-1].exists()


def test_renderer_cannot_select_an_unreported_physics_archive(trial):
    add_render(trial)
    path = trial[2] / "render_report.json"
    render = publish.read_json(path)
    render["source_hashes"]["isaac_physics"] = "a" * 64
    write_json(path, render)
    with pytest.raises(ValueError, match="absent from this trial report"):
        publish_fixture(trial, include_render=True)
    assert not trial[-1].exists()


@pytest.mark.parametrize("filename", ["../physics_trajectory.npz", "/tmp/physics_trajectory.npz", "renamed.npz", None])
def test_physics_evidence_filename_must_be_fixed_and_local(trial, filename):
    add_physics_recording(trial)
    path = trial[0] / "isaac/report.json"
    report = publish.read_json(path)
    report["physics_trajectory_file"] = filename
    write_json(path, report)
    with pytest.raises(ValueError, match="local physics_trajectory"):
        publish_fixture(trial)
    assert not trial[-1].exists()


def test_missing_selected_physics_evidence_leaves_no_output(trial):
    add_physics_recording(trial)
    (trial[0] / "isaac/physics_trajectory.npz").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        publish_fixture(trial)
    assert not trial[-1].exists()


def test_changed_physics_archive_cannot_use_old_report_or_assessment(trial):
    add_physics_recording(trial)
    path = trial[0] / "isaac/physics_trajectory.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays["wheel_ground_contact_count"][30, 0, 0] = 0
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        publish_fixture(trial)
    assert not trial[-1].exists()


def test_refreshed_physics_hash_does_not_hide_a_stale_assessment(trial):
    add_physics_recording(trial)
    path = trial[0] / "isaac/physics_trajectory.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays["wheel_lateral_contact_slip_m_s"][:] = 0.
    np.savez(path, **arrays)
    report_path = trial[0] / "isaac/report.json"
    report = publish.read_json(report_path)
    report["physics_trajectory_sha256"] = publish.sha256(path)
    write_json(report_path, report)
    with pytest.raises(ValueError, match="freshly assessed"):
        publish_fixture(trial)
    assert not trial[-1].exists()


@pytest.mark.parametrize("field,value", [("samples", 200), ("sample_dt_s", .02),
                                        ("initial_state_samples", 2), ("terminal_states_before_reset", False),
                                        ("same_counter_reset_and_forward_updates_excluded", False),
                                        ("first_episode_only_valid_mask", False)])
def test_invalid_physics_recording_audit_cannot_be_published(trial, field, value):
    add_physics_recording(trial)
    report_path = trial[0] / "isaac/report.json"
    report = publish.read_json(report_path)
    report["physics_recording"][field] = value
    write_json(report_path, report)
    with pytest.raises(ValueError):
        publish_fixture(trial)
    assert not trial[-1].exists()


def test_physics_archive_is_rehashed_after_preflight_before_output_creation(trial, monkeypatch):
    add_physics_recording(trial)
    original = publish.prepare_publication

    def changed_after_preflight(*args, **kwargs):
        result = original(*args, **kwargs)
        path = trial[0] / "isaac/physics_trajectory.npz"
        path.write_bytes(path.read_bytes() + b"changed after successful preflight")
        return result

    monkeypatch.setattr(publish, "prepare_publication", changed_after_preflight)
    with pytest.raises(ValueError, match="Input changed.*isaac_physics_trajectory"):
        publish_fixture(trial)
    assert not trial[-1].exists()
