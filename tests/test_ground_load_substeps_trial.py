"""CPU/fake-config single-factor checks; no environment, actor or physics."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from training import tron1_ground_load_substeps_trial as trial


ROOT = Path(__file__).resolve().parents[1]


def baseline():
    return json.loads((ROOT / trial.BASELINE013_PROFILE_FILE).read_text())


def parent_contract():
    # Portable schema fixture transcribed from the frozen004 nominal contract;
    # unit tests do not depend on private result directories or checkpoints.
    legs = ["abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint", "knee_L_Joint", "knee_R_Joint"]
    wheels = ["wheel_L_Joint", "wheel_R_Joint"]
    bodies = ["base_Link", "abad_L_Link", "abad_R_Link", "hip_L_Link", "hip_R_Link", "knee_L_Link", "knee_R_Link"]
    return {
        "policy_fps": 50.0, "physics_dt": .005, "decimation": 4, "actor_dim": 51, "critic_dim": 120, "action_dim": 8,
        "leg_joint_names": legs, "wheel_joint_names": wheels, "observed_joint_velocity_order": legs + wheels,
        "body_names": bodies + ["wheel_L_Link", "wheel_R_Link"], "orientation_body_names": bodies,
        "actor_terms": [["ref_leg_q_ref_leg_dq_phase_sin_cos", 14], ["motion_anchor_pos_b", 3],
            ["motion_anchor_ori_b_6d_first_two_columns_row_major", 6], ["base_lin_vel_b", 3],
            ["base_ang_vel_b", 3], ["leg_joint_pos_rel", 6], ["joint_velocities", 8], ["previous_clipped_actions", 8]],
        "raw_action_clip": [-1.0, 1.0], "leg_position_scale": 1.0,
        "leg_position_offset": "reference_frame_0_joint_pos", "wheel_torque_scale_nm": 12.0,
        "leg_kp": 500.0, "leg_kd": 10.0, "leg_torque_limit_nm": 80.0, "wheel_torque_limit_nm": 12.0,
        "actuator_model": "dc_motor", "leg_saturation_effort_nm": 80.0, "wheel_saturation_effort_nm": 12.0,
        "leg_motor_velocity_limit_rad_s": 15.0, "wheel_motor_velocity_limit_rad_s": 100.0,
        "solver_joint_velocity_limit_rad_s": 1000.0,
        "motor_velocity_limit_semantics": "no_load_speed_for_torque_speed_curve_not_hard_qvel_clip",
        "legacy_joint_friction_coefficient": 0.0, "legacy_joint_friction_startup_readback_required": True,
        "rigid_body_max_angular_speed_rad_s": 100.0, "rigid_body_angular_speed_config_units": "degrees_per_second",
        "quaternion_order": "wxyz", "root_prescribed_during_steps": False,
        "wheel_orientation_tracking": False, "wheel_angle_observation": False,
        "reference_state_initialization_training_only": True, "training_full_clip_start_fraction": .2,
        "deterministic_evaluation_start_frame": 0, "motion_end_hidden_teleport": False,
        "domain_randomization": {"schema_version": 1, "mode": "random", "seed": 42,
            "torque_scale_range": [.85, 1.0], "velocity_scale_range": [.85, 1.0],
            "wheel_friction_nm_range": [0.0, .3], "nominal_probability": .2, "friction_smoothing_rad_s": .5},
        "actuator_limits_provenance": "project sim2sim baseline settings; NOT verified manufacturer hardware ratings",
        "hardware_ready": False, "task_profile": json.loads((ROOT / trial.PARENT004_PROFILE_FILE).read_text()),
    }


def new_profile():
    value = baseline()
    value[trial.SAMPLING_KEY] = trial.SAMPLING_MODE
    value["description"] = "Prospective016: mean of four reserve costs; all other013 terms held."
    return value


def new_contract(original=None):
    value = copy.deepcopy(parent_contract() if original is None else original)
    value["task_profile"] = new_profile()
    return value


def fake_cfg():
    from training import tron1_drift as drift
    from training import tron1_ground_load_reserve as reserve
    profile = baseline()
    functions = {"sideslip": drift.sideslip_tracking, "yaw_rate": drift.yaw_rate_tracking,
                 "speed": drift.speed_tracking, "grounded": drift.grounded_reward, "tilt": drift.tilt_penalty}
    values = {"upstream_unchanged": SimpleNamespace(func=object(), weight=7.0, params={"nested": [1, 2]})}
    for name, function in functions.items():
        values["drift_" + name] = SimpleNamespace(func=function, weight=profile["reward_weights"][name],
            params={"std": profile["speed_std_m_s"]} if name == "speed" else {})
    values["drift_wheel_vertical_velocity"] = SimpleNamespace(func=drift.wheel_vertical_velocity_penalty,
        weight=-5.0, params={"std": .02})
    values["drift_leg_target_rate"] = SimpleNamespace(func=drift.leg_target_rate_penalty, weight=-1.0, params={})
    values["drift_wheel_ground_load_reserve"] = SimpleNamespace(func=reserve.wheel_ground_load_reserve_penalty,
        weight=-1.0, params={"scale_fraction": .20})
    return SimpleNamespace(rewards=SimpleNamespace(**values), scene=object(), sim=object(), events=object())


def fake_dense_module():
    def getter(env, scale_fraction):
        raise AssertionError("Configuration must never compute a reward")
    return SimpleNamespace(MODE=trial.SAMPLING_MODE, substep_wheel_ground_load_reserve_penalty=getter)


def test_actual_frozen_profile_and_historical_sources():
    for name, expected in trial.HISTORICAL_SOURCE_SHA256.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    assert hashlib.sha256((ROOT / trial.BASELINE013_PROFILE_FILE).read_bytes()).hexdigest() == trial.BASELINE013_PROFILE_SHA256
    assert hashlib.sha256((ROOT / trial.PARENT004_PROFILE_FILE).read_bytes()).hexdigest() == trial.PARENT004_PROFILE_SHA256


def test_profile_is_exact_one_factor_private_copy():
    value, old = new_profile(), baseline()
    before, old_before = copy.deepcopy(value), copy.deepcopy(old)
    checked = trial.validate_dense_load_profile(value, old)
    assert checked == value and checked is not value
    assert value == before and old == old_before
    assert checked["reward_weights"]["sideslip"] == 2.0
    assert checked["wheel_ground_load_reserve"] == {"weight": -1.0, "scale_fraction": .20}
    assert "wheel_vertical_velocity_sampling" not in checked
    checked["wheel_ground_load_reserve"]["weight"] = -2.0
    assert value == before


@pytest.mark.parametrize("key", sorted(set(new_profile()) - {"description", trial.SAMPLING_KEY}))
def test_every_held_top_level_field_cannot_disappear(key):
    value = new_profile(); value.pop(key)
    with pytest.raises(ValueError):
        trial.validate_dense_load_profile(value, baseline())


@pytest.mark.parametrize("key", ["unknown", "wheel_ground_load_sampling", "wheel_vertical_velocity_sampling",
                                 "wheel_ground_load_reserve_module_sha256", "terrain"])
def test_unknown_or_alias_profile_fields_rejected(key):
    value = new_profile(); value[key] = "all_substeps_mean"
    with pytest.raises(ValueError):
        trial.validate_dense_load_profile(value, baseline())


@pytest.mark.parametrize("mode", [None, True, 1, 1.0, "last_substep", "all_substeps_mean", "mean_force", "max", ""])
def test_sampling_aliases_rejected(mode):
    value = new_profile(); value[trial.SAMPLING_KEY] = mode
    with pytest.raises(ValueError):
        trial.validate_dense_load_profile(value, baseline())


@pytest.mark.parametrize("description", [None, True, 1, "", "   ", []])
def test_invalid_description_rejected(description):
    value = new_profile(); value["description"] = description
    with pytest.raises(ValueError):
        trial.validate_dense_load_profile(value, baseline())


@pytest.mark.parametrize("path,replacement", [
    (("schema_version",), True), (("schema_version",), 1.0), (("leg_kd",), 10),
    (("env_spacing_m",), 8), (("contact_friction",), .16), (("speed_std_m_s",), .11),
    (("reward_weights", "sideslip"), 4.0), (("reward_weights", "sideslip"), 2),
    (("reward_weights", "grounded"), True), (("wheel_ground_load_reserve", "weight"), -2.0),
    (("wheel_ground_load_reserve", "weight"), -1), (("wheel_ground_load_reserve", "scale_fraction"), .21),
    (("wheel_vertical_velocity_weight",), -4.0), (("wheel_vertical_velocity_std_m_s",), .03),
    (("mujoco_contact_solref",), (.04, 2.0)), (("mujoco_contact_solref",), [.04, 2]),
    (("action_rate_weight",), float("nan")), (("leg_target_rate_weight",), float("inf")),
])
def test_held_values_and_numeric_types_rejected(path, replacement):
    value = new_profile(); target = value
    for key in path[:-1]: target = target[key]
    target[path[-1]] = replacement
    with pytest.raises(ValueError):
        trial.validate_dense_load_profile(value, baseline())


def test_fake_baseline_cannot_redefine_factor():
    value, old = new_profile(), baseline()
    value["reward_weights"]["sideslip"] = old["reward_weights"]["sideslip"] = 4.0
    with pytest.raises(ValueError):
        trial.validate_dense_load_profile(value, old)


@pytest.mark.parametrize("bad_source", sorted(trial.HISTORICAL_SOURCE_SHA256))
def test_changed_historical_source_fails_closed(bad_source):
    original = trial._sha256
    with patch.object(trial, "_sha256", side_effect=lambda p: "0" * 64 if Path(p) == ROOT / bad_source else original(p)):
        with pytest.raises(RuntimeError, match="Historical source changed"):
            trial.validate_dense_load_profile(new_profile(), baseline())


def test_changed_historical_baseline_file_fails_closed():
    original = trial._sha256
    with patch.object(trial, "_sha256", side_effect=lambda p: "0" * 64 if Path(p) == ROOT / trial.BASELINE013_PROFILE_FILE else original(p)):
        with pytest.raises(RuntimeError, match="Historical profile changed"):
            trial.validate_dense_load_profile(new_profile(), baseline())


def test_loader_preserves_input_and_rejects_duplicate_or_nonfinite_json(tmp_path):
    path = tmp_path / "prospective.json"
    path.write_text(json.dumps(new_profile()))
    assert trial.load_dense_load_profile(path) == new_profile()
    for payload in ('{"schema_version":1,"schema_version":1}', '{"value":NaN}', '{"value":Infinity}'):
        path.write_text(payload)
        with pytest.raises(ValueError): trial.load_dense_load_profile(path)


@pytest.mark.parametrize("reference", ["004", "013"])
def test_typed_nominal_contract_only_task_profile_changes(reference):
    original = parent_contract()
    if reference == "013": original["task_profile"] = baseline()
    value = new_contract(original)
    before, original_before = copy.deepcopy(value), copy.deepcopy(original)
    checked = trial.validate_dense_load_contract(value, original, new_profile())
    assert checked == value and checked is not value
    assert value == before and original == original_before


@pytest.mark.parametrize("key", sorted(trial.NOMINAL_CONTRACT_KEYS - {"task_profile"}))
def test_every_nominal_field_is_held_typed(key):
    original = parent_contract(); value = new_contract(original)
    value[key] = "unexpected changed field"
    with pytest.raises(ValueError): trial.validate_dense_load_contract(value, original, new_profile())


@pytest.mark.parametrize("key,value", [("decimation", True), ("actor_dim", 51.0), ("leg_kp", 500),
    ("leg_kd", 12.0), ("physics_dt", .0025), ("policy_fps", 50), ("hardware_ready", 0)])
def test_bad_reference_cannot_authorize_changed_nominal_controls(key, value):
    original = parent_contract(); original[key] = value
    with pytest.raises(ValueError): trial.validate_dense_load_contract(new_contract(original), original, new_profile())


@pytest.mark.parametrize("key", ["actual_body_order", "actor_file", "sampler_audit", "reserve_sampling"])
def test_nominal_contract_rejects_unknown_and_export_fields(key):
    original = parent_contract(); value = new_contract(original)
    original[key] = value[key] = "not a nominal field"
    with pytest.raises(ValueError): trial.validate_dense_load_contract(value, original, new_profile())


def test_cfg_replaces_one_private_function_only_and_no_execution():
    cfg = fake_cfg(); before = dict(vars(cfg.rewards)); metadata = {k: (v.func, v.weight, copy.deepcopy(v.params)) for k, v in before.items()}
    old_cfg_refs = (cfg.scene, cfg.sim, cfg.events); profile = new_profile(); profile_before = copy.deepcopy(profile)
    real_import = trial.importlib.import_module; dense = fake_dense_module()
    with patch.object(trial.importlib, "import_module", side_effect=lambda name: dense if name == "training.tron1_ground_load_substeps" else real_import(name)):
        assert trial.configure_dense_ground_load(cfg, profile) is cfg
    replacement = cfg.rewards.drift_wheel_ground_load_reserve
    assert replacement is not before["drift_wheel_ground_load_reserve"]
    assert replacement.func is dense.substep_wheel_ground_load_reserve_penalty
    assert replacement.weight == -1.0 and replacement.params == {"scale_fraction": .20}
    assert replacement.params is not before["drift_wheel_ground_load_reserve"].params
    for name, old in before.items():
        assert (old.func, old.weight, old.params) == metadata[name]
        if name != "drift_wheel_ground_load_reserve": assert getattr(cfg.rewards, name) is old
    assert profile == profile_before and (cfg.scene, cfg.sim, cfg.events) == old_cfg_refs


@pytest.mark.parametrize("term_name,field,value", [
    ("drift_sideslip", "weight", 4.0), ("drift_sideslip", "params", {"unused": 0}),
    ("drift_speed", "params", {"std": .2}), ("drift_grounded", "weight", .5),
    ("drift_wheel_vertical_velocity", "params", {"std_m_s": .02}),
    ("drift_wheel_vertical_velocity", "weight", -5),
    ("drift_wheel_ground_load_reserve", "weight", -1),
    ("drift_wheel_ground_load_reserve", "params", {"scale_fraction": .2, "sampling": "mean"}),
    ("drift_leg_target_rate", "params", {"extra": True}),
])
def test_cfg_rejects_wrong_existing_terms_before_mutation(term_name, field, value):
    cfg = fake_cfg(); setattr(getattr(cfg.rewards, term_name), field, value)
    before = dict(vars(cfg.rewards))
    with pytest.raises(ValueError): trial.configure_dense_ground_load(cfg, new_profile())
    assert all(getattr(cfg.rewards, name) is old for name, old in before.items())


def test_cfg_rejects_function_alias_and_duplicate_configuration():
    cfg = fake_cfg(); original = cfg.rewards.drift_wheel_ground_load_reserve.func
    def alias(env, scale_fraction): return original(env, scale_fraction)
    cfg.rewards.drift_wheel_ground_load_reserve.func = alias
    with pytest.raises(ValueError): trial.configure_dense_ground_load(cfg, new_profile())
    cfg = fake_cfg(); real_import = trial.importlib.import_module; dense = fake_dense_module()
    with patch.object(trial.importlib, "import_module", side_effect=lambda name: dense if name == "training.tron1_ground_load_substeps" else real_import(name)):
        trial.configure_dense_ground_load(cfg, new_profile())
        with pytest.raises(ValueError): trial.configure_dense_ground_load(cfg, new_profile())


def test_imports_and_configuration_never_construct_or_step_env():
    tree = ast.parse(Path(trial.__file__).read_text())
    top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = [alias.name for node in top_imports if isinstance(node, ast.Import) for alias in node.names]
    assert not any(name.startswith(("torch", "isaac", "omni", "mujoco")) for name in names)
    body = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "configure_dense_ground_load")
    called = [node.func.attr for node in ast.walk(body) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert not set(called) & {"step", "reset", "compute", "learn", "forward", "initialize_ground_load_reserve"}
