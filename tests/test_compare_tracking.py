"""Synthetic same-policy sim2sim comparisons; no Torch or simulator needed."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_tracking import common_window_rmse, compare, sha256

LEGS = [f"{joint}_{side}_Joint" for joint in ("abad", "hip", "knee") for side in ("L", "R")]
NAMES = LEGS + ["wheel_L_Joint", "wheel_R_Joint"]


def save_json(path, value):
    path.write_text(json.dumps(value))


def contact(dt):
    return {"available": True, "flight_intervals": [], "first_takeoff_s": None,
            "first_landing_s": None, "max_flight_duration_s": 0., "sampling_interval_s": dt}


@pytest.fixture
def paired(tmp_path):
    isaac_dir, mujoco_dir = tmp_path / "isaac", tmp_path / "mujoco"
    isaac_dir.mkdir()
    mujoco_dir.mkdir()
    (isaac_dir / "actor_normalized.pt").write_bytes(b"synthetic actor identity only; never loaded")
    ti, tm = np.linspace(0., 1., 6), np.linspace(0., 1.2, 25)
    ri = np.column_stack((ti, ti * .1, 1. + ti * .2))
    rm = np.column_stack((tm, tm * .1, 1. + tm * .2))
    qi = ti[:, None] * np.arange(1, 9)[None] / 10
    qm = tm[:, None] * np.arange(1, 9)[None] / 10
    quaternion_i = np.tile([1., 0, 0, 0], (len(ti), 2, 1))
    isaac_arrays = {"time_s": ti, "root_position_m": np.repeat(ri[:, None], 2, axis=1),
                    "root_quaternion_wxyz": quaternion_i,
                    "joint_position_rad": np.repeat(qi[:, None], 2, axis=1),
                    "valid_mask": np.ones((len(ti), 2), dtype=bool),
                    "action_observation": np.zeros((len(ti), 2, 51)), "joint_names": np.array(NAMES)}
    mujoco_arrays = {"time_s": tm, "root_pos": rm, "root_quat_wxyz": np.tile([1., 0, 0, 0], (len(tm), 1)),
                     "joint_pos": qm, "action_observation": np.zeros((len(tm), 51)), "joint_names": np.array(NAMES)}
    np.savez(isaac_dir / "trajectory.npz", **isaac_arrays)
    np.savez(mujoco_dir / "rollout.npz", **mujoco_arrays)
    contract = {"actor_dim": 51, "action_dim": 8, "leg_joint_names": LEGS,
                "actor_file": "actor_normalized.pt", "actor_includes_observation_normalizer": True,
                "reference_file_sha256": "a" * 64}
    episode = {"environment": 0, "completed_full_reference": True, "end_reason": "timeout",
               "termination_terms": ["motion_end"], "jump_and_landing_detected": False,
               "reference_height_rmse_m": 0., "contact": contact(.2)}
    isaac = {"simulator": "IsaacLab/PhysX", "motion_file_sha256": "a" * 64,
             "policy_export": contract, "summary": {"episodes": [episode, {**episode, "environment": 1}]}}
    mujoco = {"engine": "MuJoCo", "motion_sha256": "a" * 64,
              "policy_sha256": sha256(isaac_dir / "actor_normalized.pt"),
              "policy_kind": "normalized_torchscript_actor", "contract": contract,
              "isaac_observation_validation": {"status": "passed", "observations_checked": 10,
                  "max_abs_error": 1e-6, "trajectory_sha256": sha256(isaac_dir / "trajectory.npz")},
              "completed_full_reference": True, "termination": "motion_end", "termination_terms": [],
              "jump_and_landing_detected": False, "reference_height_rmse_m": 0., "contact": contact(.05)}
    save_json(isaac_dir / "report.json", isaac)
    save_json(mujoco_dir / "report.json", mujoco)
    return {"isaac_dir": isaac_dir, "mujoco_dir": mujoco_dir, "out": tmp_path / "comparison",
            "isaac": isaac, "mujoco": mujoco, "i_arrays": isaac_arrays, "m_arrays": mujoco_arrays}


def write_arrays(pair):
    np.savez(pair["isaac_dir"] / "trajectory.npz", **pair["i_arrays"])
    np.savez(pair["mujoco_dir"] / "rollout.npz", **pair["m_arrays"])
    pair["mujoco"]["isaac_observation_validation"]["trajectory_sha256"] = sha256(pair["isaac_dir"] / "trajectory.npz")
    save_json(pair["mujoco_dir"] / "report.json", pair["mujoco"])


def test_exact_common_window_and_name_reordering(paired):
    p = paired
    p["m_arrays"]["root_pos"] += [.1, .2, .3]
    error = np.arange(1, 7) * .01
    p["m_arrays"]["joint_pos"][:, :6] += error
    # Huge wheel-angle differences MUST not contribute to leg RMSE.
    p["m_arrays"]["joint_pos"][:, 6:] += 1000
    permutation = [7, 2, 5, 0, 3, 6, 1, 4]
    p["m_arrays"]["joint_pos"] = p["m_arrays"]["joint_pos"][:, permutation]
    p["m_arrays"]["joint_names"] = p["m_arrays"]["joint_names"][permutation]
    write_arrays(p)
    result = compare(p["isaac_dir"], p["mujoco_dir"], p["out"])
    common = result["common_window"]
    assert common["end_s"] == 1.
    assert common["base_position_rmse_m"] == pytest.approx(np.sqrt(.14))
    assert common["base_z_rmse_m"] == pytest.approx(.3)
    assert common["leg_joint_rmse_rad"] == pytest.approx(np.sqrt(np.mean(error ** 2)))
    assert list(common["leg_joint_rmse_rad_by_name"].keys()) == LEGS
    assert result["isaac_recorded_environments"] == 2
    assert result["statistical_robustness_benchmark"] is False
    assert result["hardware_ready"] is False
    assert result["isaac_env0"]["completed_full_reference"] is True
    assert result["isaac_env0"]["jump_and_landing_detected"] is False
    assert (p["out"] / "report.json").is_file()


def test_early_terminal_prefix_excludes_reset_and_long_tail(paired):
    p = paired
    p["i_arrays"]["valid_mask"][4:, 0] = False
    for key in ("root_position_m", "root_quaternion_wxyz", "joint_position_rad", "action_observation"):
        p["i_arrays"][key][4:, 0] = np.nan
    p["m_arrays"]["root_pos"][p["m_arrays"]["time_s"] > .600001] += 100
    p["isaac"]["summary"]["episodes"][0]["completed_full_reference"] = False
    p["isaac"]["summary"]["episodes"][0]["end_reason"] = "early_termination"
    save_json(p["isaac_dir"] / "report.json", p["isaac"])
    write_arrays(p)
    result = compare(p["isaac_dir"], p["mujoco_dir"], p["out"])
    assert result["common_window"]["end_s"] == pytest.approx(.6)
    assert result["common_window"]["base_position_rmse_m"] < 1e-11
    assert result["isaac_env0"]["completed_full_reference"] is False
    assert result["mujoco"]["completed_full_reference"] is True
    assert result["mujoco"]["base_link_height_gain_m"] > 100


def test_time_weighted_integral_is_independent_of_sampling_density():
    # Delta z = t over [0,1], whose exact time RMS is sqrt(integral(t^2))=sqrt(1/3).
    t1 = np.array([0., .1, .6, 1.])
    t2 = np.array([0., .001, .002, .9, 1.])
    a = {"time": t1, "root": np.zeros((len(t1), 3)), "joints": np.zeros((len(t1), 8)), "names": NAMES}
    b = {"time": t2, "root": np.c_[t2 * 0, t2 * 0, t2], "joints": np.zeros((len(t2), 8)), "names": NAMES}
    actual = common_window_rmse(a, b, LEGS)
    assert actual["base_z_rmse_m"] == pytest.approx(np.sqrt(1 / 3), abs=1e-14)
    assert actual["base_position_rmse_m"] == pytest.approx(np.sqrt(1 / 3), abs=1e-14)


@pytest.mark.parametrize("mismatch,pattern", [
    ("motion", "Reference motion SHA256 mismatch"),
    ("actor", "Actor policy SHA256 mismatch"),
    ("verification", "verification did not pass"),
    ("verification_trajectory", "verification trajectory SHA256 mismatch"),
    ("actor_dim", "51 observations"),
])
def test_provenance_mismatch_is_fatal_and_writes_nothing(paired, mismatch, pattern):
    p = paired
    if mismatch == "motion":
        p["mujoco"]["motion_sha256"] = "b" * 64
    elif mismatch == "actor":
        p["mujoco"]["policy_sha256"] = "b" * 64
    elif mismatch == "verification":
        p["mujoco"]["isaac_observation_validation"]["status"] = "failed"
    elif mismatch == "verification_trajectory":
        p["mujoco"]["isaac_observation_validation"]["trajectory_sha256"] = "b" * 64
    else:
        p["mujoco"]["contract"]["actor_dim"] = 50
    save_json(p["mujoco_dir"] / "report.json", p["mujoco"])
    with pytest.raises(ValueError, match=pattern):
        compare(p["isaac_dir"], p["mujoco_dir"], p["out"])
    assert not p["out"].exists()


def test_missing_contact_data_and_reopened_episode_fail(paired):
    p = paired
    p["mujoco"].pop("contact")
    save_json(p["mujoco_dir"] / "report.json", p["mujoco"])
    with pytest.raises(ValueError, match="Missing MuJoCo episode report.contact"):
        compare(p["isaac_dir"], p["mujoco_dir"], p["out"])
    assert not p["out"].exists()
    p["mujoco"]["contact"] = contact(.05)
    p["i_arrays"]["valid_mask"][2, 0] = False  # Later True means an unsafe restarted episode.
    write_arrays(p)
    with pytest.raises(ValueError, match="contiguous valid first-episode prefix"):
        compare(p["isaac_dir"], p["mujoco_dir"], p["out"])


def test_existing_comparison_is_preserved(paired):
    p = paired
    p["out"].mkdir()
    sentinel = p["out"] / "report.json"
    sentinel.write_text("user-owned output")
    with pytest.raises(FileExistsError, match="Preserving"):
        compare(p["isaac_dir"], p["mujoco_dir"], p["out"])
    assert sentinel.read_text() == "user-owned output"
