"""CPU-only orchestration tests; no Torch, Isaac or simulator execution."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_tracking_robustness as bench

NAMES = ["abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint",
         "knee_L_Joint", "knee_R_Joint", "wheel_L_Joint", "wheel_R_Joint"]


def test_fixed_draws_are_reproducible_and_independent_per_joint():
    manifest = bench.make_draws(NAMES)
    assert manifest == bench.make_draws(NAMES)
    assert manifest != bench.make_draws(NAMES, seed=20261005)
    assert len(manifest["draws"]) == 17
    assert manifest["draws"][0]["domain_parameters"]["is_nominal"] is True
    for draw in manifest["draws"][1:]:
        p = draw["domain_parameters"]
        assert not p["is_nominal"]
        assert len(set(p["torque_scale"])) == len(set(p["velocity_scale"])) == 8
        assert len(set(p["wheel_friction_nm"])) == 2
        assert all(.85 <= v < 1 for v in p["torque_scale"])
        assert all(.85 <= v < 1 for v in p["velocity_scale"])
        assert all(0 <= v < .3 for v in p["wheel_friction_nm"])
        assert p["friction_smoothing_rad_s"] == .5
        assert draw["domain_parameters_sha256"] == bench.canonical_hash(p)


@pytest.mark.parametrize("kwargs", [{"seed": -1}, {"seed": True}, {"random_draws": 0},
                                     {"random_draws": 1.5}, {"random_draws": False}])
def test_invalid_draw_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        bench.make_draws(NAMES, **kwargs)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    contract = {
        "actor_dim": 51, "action_dim": 8, "physics_dt": .005, "decimation": 4, "policy_fps": 50.,
        "leg_kp": 500., "leg_kd": 10., "leg_position_scale": 1., "wheel_torque_scale_nm": 12.,
        "leg_torque_limit_nm": 80., "wheel_torque_limit_nm": 12., "raw_action_clip": [-1., 1.],
        "actuator_model": "dc_motor", "leg_saturation_effort_nm": 80., "wheel_saturation_effort_nm": 12.,
        "leg_motor_velocity_limit_rad_s": 15., "wheel_motor_velocity_limit_rad_s": 100.,
        "solver_joint_velocity_limit_rad_s": 1000.,
        "motor_velocity_limit_semantics": "no_load_speed_for_torque_speed_curve_not_hard_qvel_clip",
        "leg_joint_names": NAMES[:6], "wheel_joint_names": NAMES[6:], "observed_joint_velocity_order": NAMES,
        "actor_includes_observation_normalizer": True,
        "root_prescribed_during_steps": False, "motion_end_hidden_teleport": False,
    }
    args = {"output_dir": tmp_path / "benchmark", "random_draws": 2}
    for name in ("baseline", "candidate"):
        args[name + "_policy"] = tmp_path / (name + ".pt")
        args[name + "_policy"].write_bytes(name.encode())
        args[name + "_contract"] = tmp_path / (name + ".json")
        args[name + "_contract"].write_text(json.dumps(contract))
    for name in ("motion_file", "model"):
        args[name] = tmp_path / name
        args[name].write_bytes(name.encode())
    calls = []
    mutation = {"change": lambda report, arrays, name, draw: None}

    def evaluate(policy, motion_file, model, output, *, policy_path, contract_path,
                 domain_parameters, contact_threshold_n, no_visual_mesh):
        calls.append((Path(policy_path).stem, copy.deepcopy(domain_parameters)))
        assert contact_threshold_n == 5 and no_visual_mesh is True
        output.mkdir(parents=True)
        times = np.arange(401) * .005
        root = np.zeros((401, 3))
        root[:, 2] = .9 + .35 * np.maximum(0., 1 - np.abs(times - .8) / .25)
        forces = np.zeros((401, 2, 3)); forces[:, :, 2] = 100.
        forces[(times >= .6) & (times < 1.)] = 0.
        arrays = {"time_s": times, "root_pos": root, "root_quat_wxyz": np.tile([1., 0., 0., 0.], (401, 1)),
                  "joint_pos": np.zeros((401, 8)), "joint_names": np.array(NAMES),
                  "reference_frame": np.minimum(np.round(times / .02).astype(int), 99),
                  "action_observation": np.zeros((100, 51)), "wheel_ground_contact_force_w_n": forces,
                  "nonwheel_ground_contact_count": np.zeros(401, dtype=int)}
        report = {"contract": {**contract, "domain_parameters": copy.deepcopy(domain_parameters)},
                  "domain_parameters": copy.deepcopy(domain_parameters),
                  "domain_parameters_resampled_during_episode": False, "actual_joint_velocity_hard_clipped": False,
                  "policy_sha256": bench.sha256(policy_path), "contract_source_sha256": bench.sha256(contract_path),
                  "motion_sha256": bench.sha256(motion_file), "model_sha256": bench.sha256(model),
                  "reference_frames": 100, "completed_full_reference": True, "start_reference_frame": 0,
                  "final_reference_frame": 99, "termination": "motion_end", "termination_terms": [],
                  "physics_stepped": True, "root_state_writes": 1, "hidden_resets": 0, "engine_version": "stub"}
        mutation["change"](report, arrays, Path(policy_path).stem, domain_parameters)
        np.savez(output / "rollout.npz", **arrays)
        (output / "report.json").write_text(json.dumps(report))
        return report

    monkeypatch.setattr(bench, "load_torchscript", lambda path: str(path))
    monkeypatch.setattr(bench, "run_evaluation", evaluate)
    return args, calls, mutation


def test_paired_benchmark_uses_real_gates_identical_draws_and_separate_nominal(inputs):
    args, calls, _ = inputs
    report = bench.run_benchmark(**args)
    assert len(calls) == 6
    for i in range(0, 6, 2):
        assert calls[i][0] == "baseline" and calls[i + 1][0] == "candidate"
        assert calls[i][1] == calls[i + 1][1]
    assert report["summary"]["nominal"]["both_pass"] == 1
    assert report["summary"]["random_only"]["both_pass"] == 2
    assert report["population_success_rate_estimate"] is False
    assert report["hardware_ready"] is False
    assert report["thresholds"] == bench.THRESHOLDS
    assert (args["output_dir"] / "draws.json").is_file()
    assert report["pairs"][0]["candidate"]["metrics"]["flight_s"] == pytest.approx(.4)
    assert report["pairs"][0]["candidate"]["metrics"]["base_height_gain_m"] == pytest.approx(.35)
    assert report["pairs"][0]["metric_delta_candidate_minus_baseline"]["base_height_gain_m"] == 0


@pytest.mark.parametrize("failure", ["height", "bounce", "early", "nonwheel", "contact_missing",
                                      "wrong_params", "missing_params", "contract_params", "resampled",
                                      "hard_clipped", "wrong_hash"])
def test_failed_candidate_is_retained_and_cannot_inflate_success(inputs, failure):
    args, _, mutation = inputs
    def change(report, arrays, name, draw):
        if name != "candidate" or draw["is_nominal"]:
            return
        if failure == "height": arrays["root_pos"][:, 2] = .9
        elif failure == "bounce": arrays["wheel_ground_contact_force_w_n"][:180, :, 2] = 100.
        elif failure == "early": report["termination_terms"] = ["ee_body_pos"]
        elif failure == "nonwheel": arrays["nonwheel_ground_contact_count"][10] = 1
        elif failure == "contact_missing": arrays.pop("nonwheel_ground_contact_count")
        elif failure == "wrong_params": report["domain_parameters"]["torque_scale"] = [.81] * 8
        elif failure == "missing_params": report.pop("domain_parameters")
        elif failure == "contract_params": report["contract"].pop("domain_parameters")
        elif failure == "resampled": report["domain_parameters_resampled_during_episode"] = True
        elif failure == "hard_clipped": report["actual_joint_velocity_hard_clipped"] = True
        elif failure == "wrong_hash": report["policy_sha256"] = "0" * 64
    mutation["change"] = change
    report = bench.run_benchmark(**args)
    assert report["summary"]["nominal"]["both_pass"] == 1
    assert report["summary"]["random_only"]["baseline_only_pass"] == 2
    assert report["summary"]["random_only"]["candidate_pass_count"] == 0
    assert all(pair["candidate"]["fail_reasons"] for pair in report["pairs"][1:])


def test_refuses_existing_output_even_empty(inputs):
    args, calls, _ = inputs
    args["output_dir"].mkdir()
    with pytest.raises(FileExistsError):
        bench.run_benchmark(**args)
    assert calls == []


def test_mismatched_nominal_contract_fails_before_episode(inputs):
    args, calls, _ = inputs
    contract = json.loads(args["candidate_contract"].read_text())
    contract["leg_kp"] = 499.
    args["candidate_contract"].write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="same nominal DC"):
        bench.run_benchmark(**args)
    assert calls == []


def test_cli_requires_explicit_contracts_and_preserves_defaults(monkeypatch, tmp_path):
    got = {}
    monkeypatch.setattr(bench, "run_benchmark", lambda **kwargs: got.update(kwargs) or {"summary": {}})
    cli = [argument for name in ("baseline-policy", "baseline-contract", "candidate-policy", "candidate-contract",
                                "motion-file", "model", "output-dir") for argument in ("--" + name, str(tmp_path / name))]
    assert bench.main(cli) == 0
    assert got["seed"] == 20261004 and got["random_draws"] == 16
    with pytest.raises(SystemExit):
        bench.main([])
