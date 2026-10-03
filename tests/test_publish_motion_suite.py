"""CPU-only publishing tests; synthetic artifacts are not physical success evidence."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from publish_motion_suite import TASKS, publish_suite, read_json, sanitize, sha256


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def suite(tmp_path):
    project = tmp_path / "project"
    raw = project / "outputs/suite"
    config_path = project / "config/suite/motion_suite.json"
    profile = {"mode": "random", "seed": 42, "torque_scale_range": [.85, 1.]}
    source_hashes = {name: "a" * 64 for name in (
        "training/tron1_tracking.py", "training/tron1_domain_randomization.py", "scripts/train_tracking.py")}
    budget = {"num_envs": 2, "iterations_per_motion": 3, "seed": 42,
              "env_steps_per_motion": 24, "resume_sha256": "b" * 64,
              "resume_iteration": 10, "physics_hz": 200, "policy_hz": 50}
    config = {"date": "2000-01-01", "budget": budget, "tasks": [], "limitations": ["No hardware proof."]}
    catalog = {}
    names = np.array([f"joint_{i}" for i in range(8)])
    for task in TASKS:
        trial = raw / task / "trial"
        for directory in ("training", "isaac", "mujoco", "render"):
            (trial / directory).mkdir(parents=True)
        reference = raw / task / "motion.npz"
        np.savez(reference, fps=[50.], joint_pos=np.zeros((10, 8)), joint_vel=np.ones((10, 8)),
                 body_lin_vel_w=np.ones((10, 1, 3)) * .3, body_ang_vel_w=np.zeros((10, 1, 3)))
        time = np.arange(11) / 50
        angles = np.linspace(0, np.pi / 2, len(time))
        quat = np.column_stack((np.cos(angles / 2), np.zeros((len(time), 2)), np.sin(angles / 2)))
        np.savez(trial / "isaac/trajectory.npz", time_s=time, joint_names=names,
                 root_position_m=np.zeros((11, 1, 3)), root_quaternion_wxyz=quat[:, None],
                 joint_position_rad=np.zeros((11, 1, 8)), valid_mask=np.ones((11, 1), bool),
                 action_observation=np.zeros((11, 1, 51)))
        np.savez(trial / "mujoco/rollout.npz", time_s=time, joint_names=names, root_pos=np.zeros((11, 3)),
                 root_quat_wxyz=quat, joint_pos=np.zeros((11, 8)), action_observation=np.zeros((10, 51)))
        (trial / "training/model_final.pt").write_bytes(b"synthetic checkpoint")
        (trial / "isaac/actor_normalized.pt").write_bytes(b"synthetic actor")
        (trial / "render/tracking_comparison.mp4").write_bytes(b"synthetic media; not decoded in publisher")
        (trial / "render/overview.png").write_bytes(b"synthetic overview")
        write_json(trial / "training/runner_config.json", {"num_steps_per_env": 4})
        contract = {"physics_dt": .005, "policy_fps": 50.,
                    "reference_file_sha256": sha256(reference),
                    "checkpoint_sha256": sha256(trial / "training/model_final.pt"),
                    "training_domain_randomization": profile}
        write_json(trial / "isaac/policy_contract.json", contract)
        manifest = {"host": "research-node-private", "python": "/data/researcher/env/bin/python",
                    "num_envs": 2, "iterations_requested": 3, "seed": 42,
                    "resume_sha256": budget["resume_sha256"], "motion_sha256": sha256(reference),
                    "domain_randomization_profile": profile, "source_sha256": source_hashes}
        write_json(trial / "training/manifest.json", manifest)
        write_json(trial / "training/run_status.json", {"status": "training_completed", "num_envs": 2,
                   "iterations_requested": 3, "training_environment_steps": 24, "last_iteration": 12})
        write_json(trial / "training/domain_randomization_final_audit.json", {
            "enabled": True, "mode": "random", "profile": profile, "reset_counts": [2, 3],
            "all_sampled_parameters_readback_verified": True, "parameters_hidden_from_actor": True,
            "dc_overspeed_cache_verified": True})
        isaac = {"motion_file_sha256": sha256(reference), "policy_export": contract,
                 "checkpoint_sha256": contract["checkpoint_sha256"],
                 "runner_config_sha256": sha256(trial / "training/runner_config.json")}
        mujoco = {"motion_sha256": sha256(reference), "policy_sha256": sha256(trial / "isaac/actor_normalized.pt")}
        write_json(trial / "isaac/report.json", isaac)
        write_json(trial / "mujoco/report.json", mujoco)
        provenance = {key + "_sha256": sha256(path) for key, path in (
            ("reference", reference), ("actor", trial / "isaac/actor_normalized.pt"),
            ("isaac_trajectory", trial / "isaac/trajectory.npz"), ("mujoco_trajectory", trial / "mujoco/rollout.npz"),
            ("isaac_report", trial / "isaac/report.json"), ("mujoco_report", trial / "mujoco/report.json"))}
        provenance["isaac_actor_file"] = "/home/researcher/private/model.pt"
        verdict = "fail" if task in {"side_jump", "step_up"} else "pass"
        episode = {"verdict": verdict, "failed_checks": ["root_error"] if verdict == "fail" else [],
                   "metrics": {"root_rmse_m": .3 if verdict == "fail" else .03,
                               "flight_intervals": [{"takeoff_s": .04, "landing_s": .14, "no_contact_duration_s": .1}]}}
        assessment = {"task": task, "verdict": verdict, "failed_checks": episode["failed_checks"],
                      "thresholds": {"root_rmse_m": .15}, "provenance": provenance, "common_window": {"duration_s": .2},
                      "isaac": copy.deepcopy(episode), "mujoco": copy.deepcopy(episode),
                      "complete_contact_evidence": False, "safety_verified": False, "hardware_ready": False,
                      "limitations": ["Host research-node-private; user researcher; ssh tester@10.20.30.40; /root/private/run"],
                      "raw_log": "excluded entirely"}
        write_json(trial / "assessment.json", assessment)
        write_json(trial / "render/render_report.json", {
            "status": "rendered", "simulation_steps": 0, "right_panel_kind": "recorded_policy",
            "full_decode_verified": True, "trajectory_sha256": provenance["isaac_trajectory_sha256"],
            "comparison_trajectory_sha256": provenance["mujoco_trajectory_sha256"], "motion_sha256": sha256(reference),
            "comparison_source_result": {"secret": "not copied", "host": "research-node-private"}})
        key = f"cmu_{task}"
        config["tasks"].append({"task": task, "source": {"clip": task, "manifest_key": key},
                                "reference_path": str(reference.relative_to(project)), "caveats": ["Not original human task proof"],
                                "raw_skill_intent": "Synthetic source intent"})
        catalog[key] = {"fps": 120, "source_page": "http://mocap.cs.cmu.edu/search.php?subjectnumber=83",
                        "files": [{"path": "private/raw.amc", "url": "http://mocap.cs.cmu.edu/raw.amc", "sha256": "c" * 64}]}
    write_json(config_path, config)
    write_json(project / "config/mocap_sources.json", catalog)
    return project, raw, config_path, tmp_path / "published"


def publish(fixture):
    project, raw, config, output = fixture
    return publish_suite(raw, output, config, project_root=project)


def test_publish_preserves_failures_and_actual_media(suite):
    result = publish(suite)
    project, raw, config, output = suite
    assert result["task_count"] == 6
    assert result["behavior_pass_count"] == 4
    assert result["budget_pass_count"] == 6
    assert not result["hardware_ready"]
    assert not result["statistical_robustness_benchmark"]
    for item in result["tasks"]:
        assert item["raw_skill_intent_certified"] is False
        assert item["budget"]["final_iteration_offset_verified"] is True
        assert item["reference"]["action_duration_s"] == pytest.approx(.2)
        assert item["reference"]["sample_span_s"] == pytest.approx(.18)
        for key in ("video", "overview"):
            assert sha256(output / item["artifacts"][key]) == item["hashes"][key + "_sha256"]
        assert sha256(output / item["artifacts"]["assessment"]) == item["hashes"]["published_assessment_sha256"]
        if item["task"] == "side_jump":
            assert item["verdict"] == "fail" and item["failed_checks"] == ["root_error"]
    assert not list(output.rglob("*.pt")) and not list(output.rglob("*.npz"))
    assert len(list(output.rglob("*.*"))) == 19
    text = "\n".join(path.read_text() for path in output.rglob("*.json"))
    for forbidden in ("research-node-private", "researcher", "tester@", "10.20.30.40", "/root/private", "/home/", "/data/", "raw_log"):
        assert forbidden not in text
    assert "http://mocap.cs.cmu.edu/" in text


def test_turn_airborne_and_total_are_separate_descriptive_quantities(suite):
    turn = next(item for item in publish(suite)["tasks"] if item["task"] == "turn_jump")
    for engine in ("isaac", "mujoco"):
        data = turn["supplemental_yaw_diagnostics"][engine]
        assert data["total_yaw_change_deg"] == pytest.approx(90.)
        assert data["flights"][0]["airborne_yaw_change_deg"] == pytest.approx(45.)
        assert data["flights"][0]["yaw_before_takeoff_deg"] == pytest.approx(18.)
        assert "posthoc descriptive only" in data["status"]
        assert data["does_not_certify_90_degree_airborne_turn"] is True


@pytest.mark.parametrize("relative", ["step_up/trial/render/overview.png", "crouch/trial/training/run_status.json"])
def test_any_missing_task_input_leaves_no_output(suite, relative):
    (suite[1] / relative).unlink()
    with pytest.raises(ValueError, match="no output written"):
        publish(suite)
    assert not suite[3].exists()


def test_collects_all_preflight_errors_without_partial_write(suite):
    for task in ("forward_jump", "step_up"):
        (suite[1] / task / "trial/assessment.json").unlink()
    with pytest.raises(ValueError) as error:
        publish(suite)
    assert "forward_jump" in str(error.value) and "step_up" in str(error.value)
    assert not suite[3].exists()


@pytest.mark.parametrize("relative", ["isaac/actor_normalized.pt", "training/model_final.pt", "mujoco/rollout.npz", "isaac/report.json"])
def test_tampered_artifact_hash_rejected(suite, relative):
    path = suite[1] / "forward_jump/trial" / relative
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash|match"):
        publish(suite)
    assert not suite[3].exists()


def test_nonempty_destination_not_overwritten(suite):
    suite[3].mkdir()
    existing = suite[3] / "precious.txt"
    existing.write_text("keep")
    with pytest.raises(ValueError, match="refusing overwrite"):
        publish(suite)
    assert existing.read_text() == "keep" and len(list(suite[3].iterdir())) == 1


def test_empty_destination_allowed(suite):
    suite[3].mkdir()
    assert publish(suite)["task_count"] == 6


def test_budget_failure_is_distinct_from_behavior_success(suite):
    status = suite[1] / "forward_jump/trial/training/run_status.json"
    value = read_json(status)
    value["last_iteration"] = 11
    write_json(status, value)
    result = publish(suite)
    forward = result["tasks"][0]
    assert forward["behavior_pass"] and not forward["budget"]["pass"]
    assert result["budget_pass_count"] == 5 and result["behavior_pass_count"] == 4


def test_incomplete_training_not_publishable(suite):
    path = suite[1] / "crouch/trial/training/run_status.json"
    value = read_json(path)
    value["status"] = "running"
    write_json(path, value)
    with pytest.raises(ValueError, match="training incomplete"):
        publish(suite)
    assert not suite[3].exists()


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_json_rejected_before_writing(suite, literal):
    path = suite[1] / "forward_jump/trial/assessment.json"
    path.write_text(path.read_text().replace('"root_rmse_m": 0.03', '"root_rmse_m": ' + literal))
    with pytest.raises(ValueError):
        publish(suite)
    assert not suite[3].exists()


def test_sanitizer_retains_scientific_values_and_official_links():
    value = {"password": "do not expose", "api_key": "do not expose", "host": "private-node",
             "note": "node private-node; ssh alpha@172.16.1.2; /home/alpha/env/bin/python; C:\\Users\\alpha\\asset",
             "usd_prim": "/World/envs/env_0/Terrain", "source": "http://mocap.cs.cmu.edu/subjects/83.asf",
             "version": "5.1.0.0", "sha256": "f" * 64, "value": .1163959}
    clean = sanitize(value, {"private-node", "alpha"})
    assert not {"password", "api_key", "host"} & clean.keys()
    assert "private-node" not in clean["note"] and "alpha" not in clean["note"]
    assert "172.16.1.2" not in clean["note"]
    assert clean["usd_prim"] == value["usd_prim"]
    assert clean["source"] == value["source"] and clean["version"] == value["version"]
    assert clean["sha256"] == value["sha256"] and clean["value"] == value["value"]
    with pytest.raises(ValueError, match="finite"):
        sanitize(float("nan"))


def test_optional_terrain_retrospective_capture_is_labeled_and_hashed(suite):
    path = suite[1] / "step_up/trial/terrain_implementation_provenance.json"
    write_json(path, {"capture_kind": "retrospective_capture", "module_sha256": "d" * 64,
                      "source_path": "/data/researcher/workspace/training/tron1_terrain.py"})
    step = next(item for item in publish(suite)["tasks"] if item["task"] == "step_up")
    assert step["terrain"]["implementation_provenance"]["capture_kind"] == "retrospective_capture"
    assert step["terrain"]["implementation_provenance"]["source_path"] == "<private-path>"
    assert step["hashes"]["terrain_implementation_provenance_sha256"] == sha256(path)


def test_missing_or_duplicate_task_configuration_fails(suite):
    config = read_json(suite[2])
    config["tasks"][-1] = config["tasks"][0]
    write_json(suite[2], config)
    with pytest.raises(ValueError, match="six tasks exactly once"):
        publish(suite)
    assert not suite[3].exists()


def test_contradictory_pass_does_not_hide_engine_failure(suite):
    path = suite[1] / "forward_jump/trial/assessment.json"
    value = read_json(path)
    value["isaac"]["verdict"] = "fail"
    value["isaac"]["failed_checks"] = ["root_error"]
    write_json(path, value)
    with pytest.raises(ValueError, match="suite pass contradicts"):
        publish(suite)
    assert not suite[3].exists()


def test_ip_redaction_and_version_field_context():
    assert sanitize("ssh host 1.2.3.4") == "ssh host <private-ip>"
    assert sanitize({"versions": {"isaacsim": "5.1.0.0"}}) == {"versions": {"isaacsim": "5.1.0.0"}}
