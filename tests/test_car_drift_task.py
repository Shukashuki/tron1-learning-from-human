"""Optional drift-profile validation and engine material integration, without Isaac."""
from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
import sys
import types
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from training import tron1_drift as drift
from eval_tracking_mujoco import read_contract, validate_contract


def task_profile():
    return {"schema_version": 1, "task": "car_drift", "contact_friction": .15,
            "env_spacing_m": 8., "reward_weights": {
                "sideslip": 2., "yaw_rate": 1., "speed": 1., "grounded": .4, "tilt": -.2}}


def test_profile_validation_returns_an_independent_copy():
    original = task_profile()
    result = drift.validate_task_profile(original)
    assert result == original
    result["reward_weights"]["speed"] = 3.
    assert original["reward_weights"]["speed"] == 1.


def test_repository_profile_is_loadable_and_uses_declared_low_grip_material():
    profile = drift.load_task_profile(ROOT / "config/car_drift_task.json")
    assert profile["task"] == "car_drift"
    assert profile["contact_friction"] == .15
    assert profile["env_spacing_m"] == 8.


@pytest.mark.parametrize("invalid", [None, [], "car_drift", 1, True])
def test_profile_requires_an_object(invalid):
    with pytest.raises(ValueError):
        drift.validate_task_profile(invalid)


@pytest.mark.parametrize("invalid", [None, 0, 2, True, 1., "1"])
def test_profile_requires_an_integer_nonboolean_schema_version(invalid):
    profile = task_profile()
    profile["schema_version"] = invalid
    with pytest.raises(ValueError):
        drift.validate_task_profile(profile)


@pytest.mark.parametrize("invalid", [None, "drift", "jump", 1, True])
def test_profile_requires_the_named_car_drift_task(invalid):
    profile = task_profile()
    profile["task"] = invalid
    with pytest.raises(ValueError):
        drift.validate_task_profile(profile)


@pytest.mark.parametrize("key,invalid", [
    ("contact_friction", 0.), ("contact_friction", .009), ("contact_friction", 1.01),
    ("env_spacing_m", 2.99), ("env_spacing_m", 20.01),
    ("contact_friction", True), ("env_spacing_m", False),
    ("contact_friction", np.nan), ("env_spacing_m", np.inf),
    ("contact_friction", ".15"), ("env_spacing_m", None),
])
def test_profile_rejects_nonfinite_or_out_of_bounds_physical_parameters(key, invalid):
    profile = task_profile()
    profile[key] = invalid
    with pytest.raises(ValueError):
        drift.validate_task_profile(profile)


@pytest.mark.parametrize("weights", [None, [], {}, "sideslip", ["sideslip", "yaw_rate", "speed", "grounded", "tilt"]])
def test_reward_weights_require_a_mapping_of_all_five_named_terms(weights):
    profile = task_profile()
    profile["reward_weights"] = weights
    with pytest.raises(ValueError):
        drift.validate_task_profile(profile)


@pytest.mark.parametrize("key", ["sideslip", "yaw_rate", "speed", "grounded", "tilt"])
@pytest.mark.parametrize("value", [True, None, "1", np.nan, np.inf, 10.01, -10.01])
def test_profile_rejects_bad_reward_values(key, value):
    profile = task_profile()
    profile["reward_weights"][key] = value
    with pytest.raises(ValueError):
        drift.validate_task_profile(profile)


@pytest.mark.parametrize("key,value", [("sideslip", -1.), ("yaw_rate", -1.), ("speed", -1.), ("grounded", -1.), ("tilt", 1.)])
def test_reward_signs_cannot_reward_tilt_or_penalize_correct_tracking(key, value):
    profile = task_profile()
    profile["reward_weights"][key] = value
    with pytest.raises(ValueError):
        drift.validate_task_profile(profile)


def test_drift_profile_does_not_change_historical_motor_or_observation_contract():
    historical = read_contract()
    assert "task_profile" not in historical
    assert (historical["actor_dim"], historical["critic_dim"], historical["action_dim"]) == (51, 120, 8)
    assert historical["physics_dt"] == .005
    assert historical["decimation"] == 4
    assert historical["raw_action_clip"] == [-1., 1.]
    assert historical["leg_position_scale"] == 1.
    assert historical["wheel_torque_scale_nm"] == 12.
    assert historical["leg_kp"] == 500.
    assert historical["leg_kd"] == 10.
    before = copy.deepcopy(historical)
    combined = validate_contract({**historical, "task_profile": task_profile()})
    assert {key: value for key, value in combined.items() if key != "task_profile"} == historical
    assert historical == before
    combined["task_profile"]["reward_weights"]["speed"] = 4.
    assert historical == before


def test_saved_contract_revalidates_drift_profile_without_executing_isaac(tmp_path):
    contract = {**read_contract(), "task_profile": task_profile()}
    path = tmp_path / "policy_contract.json"
    path.write_text(json.dumps(contract))
    assert read_contract(path) == contract
    contract["task_profile"]["contact_friction"] = 0.
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError):
        read_contract(path)


def test_mujoco_both_contacting_materials_have_the_same_effective_friction():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody>
          <geom name="floor" type="plane" size="5 5 .1" friction=".9 .003 .0004"/>
          <body name="wheel" pos="0 0 .04"><freejoint/>
            <geom name="wheel_collision" type="sphere" size=".05" friction=".6 .004 .0005"/>
            <geom name="visual_only" type="sphere" size=".01" contype="0" conaffinity="0" mass="0" friction=".77 .002 .0002"/>
          </body>
        </worldbody></mujoco>""")
    original = model.geom_friction.copy()
    audit = drift.configure_mujoco_materials(model, task_profile())
    floor, wheel, visual = [model.geom(name).id for name in ("floor", "wheel_collision", "visual_only")]
    np.testing.assert_allclose(model.geom_friction[[floor, wheel], 0], .15)
    np.testing.assert_allclose(model.geom_friction[:, 1:], original[:, 1:])
    np.testing.assert_allclose(model.geom_friction[visual], original[visual])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon > 0
    for contact in data.contact:
        np.testing.assert_allclose(contact.friction[:2], .15)
    assert audit["compiled_friction_range"] == [.15, .15]
    assert audit["colliding_geom_count"] == 2


def fake_configuration():
    rewards = NS(**{name: NS(weight=.5, params={"std": .3}) for name in (
        "motion_global_anchor_pos", "motion_global_anchor_ori", "motion_body_lin_vel", "motion_body_ang_vel",
        "motion_body_ori", "joint_limit", "undesired_contacts")})
    rewards.action_rate_l2 = NS(weight=-.01, params={})
    return NS(scene=NS(terrain=NS(visual_material=None, physics_material=None),
                       contact_forces=NS(debug_vis=True, update_period=0.)),
              sim=NS(), commands=NS(), rewards=rewards,
              events=NS(physics_material=NS(params={})),
              terminations=NS(**{key: NS(params={}) for key in ("anchor_pos", "anchor_ori", "ee_body_pos")}))


def config_constructor(*args, **kwargs):
    return NS(**kwargs)


def extract_configuration_factory():
    """Execute only the factory AST with lightweight Isaac configuration types."""
    source = ast.parse((ROOT / "training/tron1_tracking.py").read_text())
    factory = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "make_env_cfg")
    contract = read_contract()
    class Articulation:
        InitialStateCfg = staticmethod(config_constructor)
        def __new__(cls, **kwargs):
            return NS(**kwargs)
    namespace = {
        "Path": Path, "np": np, "math": math, "POLICY_FPS": 50.,
        "LEG_JOINT_NAMES": contract["leg_joint_names"], "WHEEL_JOINT_NAMES": contract["wheel_joint_names"],
        "ACTION_JOINT_NAMES": contract["observed_joint_velocity_order"], "BODY_NAMES": contract["body_names"],
        "NON_WHEEL_BODY_NAMES": contract["orientation_body_names"],
        "WHEEL_BODY_NAMES": ["wheel_L_Link", "wheel_R_Link"],
        "TronTrackingEnvCfg": fake_configuration, "ArticulationCfg": Articulation,
        "DCMotorCfg": config_constructor, "TronMotionCommandCfg": config_constructor,
        "SceneEntityCfg": config_constructor, "TerminationTermCfg": config_constructor,
        "motion_ended": lambda *args: False,
        "sim_utils": NS(**{key: config_constructor for key in (
            "RigidBodyMaterialCfg", "UsdFileCfg", "RigidBodyPropertiesCfg", "ArticulationRootPropertiesCfg")}),
    }
    exec(compile(ast.Module(body=[factory], type_ignores=[]), "isolated_tracking_factory", "exec"), namespace)
    return namespace["make_env_cfg"]


@pytest.fixture
def configuration_inputs(tmp_path, monkeypatch):
    module = types.ModuleType("isaaclab.managers")
    module.RewardTermCfg = config_constructor
    monkeypatch.setitem(sys.modules, "isaaclab.managers", module)
    contract = read_contract()
    motion = tmp_path / "motion.npz"
    np.savez_compressed(motion, fps=np.array([50.]), joint_names=np.array(contract["observed_joint_velocity_order"]),
                        joint_pos=np.zeros((3, 8)), body_pos_w=np.zeros((3, len(contract["body_names"]), 3)))
    asset = tmp_path / "placeholder.usd"
    asset.write_text("# Only path existence is used by the offline factory test.\n")
    return motion, asset


def test_historical_default_configuration_stays_at_point_six_friction(configuration_inputs):
    cfg = extract_configuration_factory()(*configuration_inputs, num_envs=4)
    assert cfg.scene.env_spacing == 3.
    assert cfg.scene.terrain.physics_material.static_friction == .6
    assert cfg.scene.terrain.physics_material.dynamic_friction == .6
    assert cfg.sim.physics_material is cfg.scene.terrain.physics_material
    assert not hasattr(cfg.rewards, "drift_sideslip")
    assert cfg.sim.dt == .005
    assert cfg.decimation == 4


def test_profile_is_applied_only_when_requested_and_robot_ground_mu_agree(configuration_inputs):
    cfg = extract_configuration_factory()(*configuration_inputs, num_envs=4, task_profile=task_profile())
    assert cfg.scene.env_spacing == 8.
    assert cfg.scene.terrain.physics_material.static_friction == .15
    assert cfg.scene.terrain.physics_material.dynamic_friction == .15
    assert cfg.events.physics_material.params["static_friction_range"] == (.15, .15)
    assert cfg.events.physics_material.params["dynamic_friction_range"] == (.15, .15)
    assert cfg.sim.physics_material is cfg.scene.terrain.physics_material
    assert cfg.rewards.drift_sideslip.func is drift.sideslip_tracking
    assert cfg.rewards.drift_sideslip.weight == 2.
    assert cfg.rewards.drift_tilt.weight == -.2
    assert cfg.rewards.motion_global_anchor_pos.weight == 2.
    assert cfg.rewards.motion_global_anchor_ori.weight == 2.


@pytest.mark.parametrize("key,value", [
    ("speed_std_m_s", .0299), ("speed_std_m_s", 1.001), ("speed_std_m_s", 0.),
    ("speed_std_m_s", True), ("speed_std_m_s", None), ("speed_std_m_s", ".1"),
    ("speed_std_m_s", np.nan), ("speed_std_m_s", np.inf),
    ("action_rate_weight", -2.001), ("action_rate_weight", .001),
    ("action_rate_weight", False), ("action_rate_weight", None),
    ("action_rate_weight", np.nan), ("action_rate_weight", np.inf),
])
def test_optional_stop_reward_parameters_are_finite_and_bounded(key, value):
    profile = task_profile()
    profile[key] = value
    with pytest.raises(ValueError):
        drift.validate_task_profile(profile)


@pytest.mark.parametrize("value", [
    None, (), [.02], [.02, 1., 1.], "0.02,1", [.02, True],
    [.0099, 1.], [.0501, 1.], [.02, .999], [.02, 2.001],
    [np.nan, 1.], [.02, np.inf], [-.02, 1.], [True, 1.],
])
def test_optional_mujoco_contact_solref_requires_explicit_safe_timeconstant_and_damping(value):
    profile = task_profile()
    profile["mujoco_contact_solref"] = value
    with pytest.raises(ValueError):
        drift.validate_task_profile(profile)


@pytest.mark.parametrize("timeconstant,damping", [(.01, 1.), (.05, 2.), (.02, 1.)])
def test_contact_solref_accepts_declared_boundary_values(timeconstant, damping):
    profile = task_profile()
    profile.update(speed_std_m_s=.03, action_rate_weight=-2., mujoco_contact_solref=[timeconstant, damping])
    assert drift.validate_task_profile(profile) == profile


def test_stop_task_changes_declared_speed_accuracy_and_action_smoothness_only(configuration_inputs):
    profile = drift.load_task_profile(ROOT / "config/car_drift_stop_task.json")
    assert profile["speed_std_m_s"] == .1
    assert profile["action_rate_weight"] == -.5
    assert profile["mujoco_contact_solref"] == [.02, 1.]
    cfg = extract_configuration_factory()(*configuration_inputs, num_envs=4, task_profile=profile)
    assert cfg.rewards.drift_speed.params["std"] == .1
    assert cfg.rewards.drift_speed.weight == 3.
    assert cfg.rewards.action_rate_l2.weight == -.5
    assert cfg.scene.terrain.physics_material.static_friction == .15
    assert cfg.scene.terrain.physics_material.dynamic_friction == .15
    assert cfg.sim.dt == .005 and cfg.decimation == 4
    assert cfg.scene.robot.actuators["legs"].stiffness == 500.
    assert cfg.scene.robot.actuators["legs"].damping == 30.
    assert cfg.scene.robot.actuators["wheels"].damping == 0.


def test_optional_stop_parameters_leave_historical_drift_defaults_unchanged(configuration_inputs):
    cfg = extract_configuration_factory()(*configuration_inputs, num_envs=4, task_profile=task_profile())
    assert cfg.rewards.drift_speed.params["std"] == .35
    assert cfg.rewards.action_rate_l2.weight == -.01
    assert cfg.scene.robot.actuators["legs"].damping == 10.
    assert not hasattr(cfg.rewards, "drift_leg_target_rate")
    assert not hasattr(cfg.rewards, "drift_wheel_vertical_velocity")


@pytest.mark.parametrize("mode", [None, True, False, 0, 1, [], {}, "mean", "", "ALL_SUBSTEPS_MEAN"])
def test_vertical_velocity_sampling_rejects_invalid_modes(mode):
    profile = task_profile()
    profile.update(wheel_vertical_velocity_weight=-5., wheel_vertical_velocity_std_m_s=.02,
                   wheel_vertical_velocity_sampling=mode)
    with pytest.raises(ValueError, match="sampling mode"):
        drift.validate_task_profile(profile)


@pytest.mark.parametrize("mode", ["last_substep", "all_substeps_mean"])
def test_sampling_mode_requires_a_declared_reward_weight_and_scale(mode):
    profile = task_profile()
    profile["wheel_vertical_velocity_sampling"] = mode
    with pytest.raises(ValueError, match="explicit reward weight"):
        drift.validate_task_profile(profile)
    profile["wheel_vertical_velocity_weight"] = -5.
    with pytest.raises(ValueError, match="explicit wheel_vertical_velocity_std"):
        drift.validate_task_profile(profile)


def test_default_sampling_installer_does_not_touch_environment_or_import_helper(monkeypatch):
    monkeypatch.setitem(sys.modules, "training.tron1_substep_rewards", None)
    untouched_env = object()
    assert drift.install_drift_reward_sampling(untouched_env, task_profile()) is None
    profile = task_profile()
    profile.update(wheel_vertical_velocity_weight=-5., wheel_vertical_velocity_std_m_s=.02,
                   wheel_vertical_velocity_sampling="last_substep")
    assert drift.install_drift_reward_sampling(untouched_env, profile) is None


def test_explicit_all_substep_sampling_selects_reward_and_installer(configuration_inputs, monkeypatch):
    calls = []
    handler = object()
    helper = types.ModuleType("training.tron1_substep_rewards")
    def install(env):
        calls.append(env)
        return handler
    def reward(env, std_m_s):
        return (env, std_m_s)
    helper.install_wheel_vertical_velocity_accumulator = install
    helper.substep_wheel_vertical_velocity_penalty = reward
    monkeypatch.setitem(sys.modules, "training.tron1_substep_rewards", helper)
    profile = task_profile()
    profile.update(wheel_vertical_velocity_weight=-5., wheel_vertical_velocity_std_m_s=.02,
                   wheel_vertical_velocity_sampling="all_substeps_mean")
    cfg = extract_configuration_factory()(*configuration_inputs, num_envs=4, task_profile=profile)
    term = cfg.rewards.drift_wheel_vertical_velocity
    assert term.func is reward and term.params == {"std_m_s": .02} and term.weight == -5.
    env = object()
    assert drift.install_drift_reward_sampling(env, profile) is handler
    assert calls == [env]
    contract = drift.apply_task_profile_to_contract(read_contract(), profile)
    assert validate_contract(contract)["task_profile"] == profile
    assert (cfg.sim.dt, cfg.decimation, contract["actor_dim"], contract["action_dim"]) == (.005, 4, 51, 8)


def test_last_substep_sampling_keeps_existing_reward_function_and_parameters(configuration_inputs):
    profile = task_profile()
    profile.update(wheel_vertical_velocity_weight=-5., wheel_vertical_velocity_std_m_s=.02)
    for explicit in (False, True):
        if explicit:
            profile["wheel_vertical_velocity_sampling"] = "last_substep"
        cfg = extract_configuration_factory()(*configuration_inputs, num_envs=4, task_profile=profile)
        term = cfg.rewards.drift_wheel_vertical_velocity
        assert term.func is drift.wheel_vertical_velocity_penalty
        assert term.params == {"std": .02} and term.weight == -5.


@pytest.mark.parametrize("key", ["leg_target_rate_weight", "wheel_vertical_velocity_weight"])
@pytest.mark.parametrize("value", [-5.001, .001, True, False, None, "-.5", [], np.nan, np.inf, -np.inf])
def test_optional_contact_robust_penalties_reject_invalid_weights(key, value):
    profile = task_profile()
    profile.update({key: value, "wheel_vertical_velocity_std_m_s": .02})
    with pytest.raises(ValueError):
        drift.validate_task_profile(profile)


@pytest.mark.parametrize("value", [.004999, .100001, 0., -.01, True, False, None, ".02", [], np.nan, np.inf, -np.inf])
def test_optional_vertical_speed_std_is_finite_positive_and_bounded(value):
    profile = task_profile()
    profile["wheel_vertical_velocity_std_m_s"] = value
    with pytest.raises(ValueError):
        drift.validate_task_profile(profile)


@pytest.mark.parametrize("weight,std", [(-5., .005), (0., .1), (-.5, .02)])
def test_contact_robust_profile_accepts_explicit_boundary_values_without_mutation(weight, std):
    profile = task_profile()
    profile.update(leg_target_rate_weight=weight, wheel_vertical_velocity_weight=weight,
                   wheel_vertical_velocity_std_m_s=std)
    original = copy.deepcopy(profile)
    validated = drift.validate_task_profile(profile)
    assert profile == original and validated == original and validated is not profile


def test_vertical_speed_weight_requires_explicit_scale_even_when_weight_is_zero():
    for weight in (-.5, 0.):
        profile = {**task_profile(), "wheel_vertical_velocity_weight": weight}
        with pytest.raises(ValueError, match="explicit wheel_vertical_velocity_std_m_s"):
            drift.validate_task_profile(profile)


def test_vertical_speed_std_alone_adds_no_reward_or_default_behavior(configuration_inputs):
    profile = {**task_profile(), "wheel_vertical_velocity_std_m_s": .02}
    cfg = extract_configuration_factory()(*configuration_inputs, num_envs=4, task_profile=profile)
    assert not hasattr(cfg.rewards, "drift_leg_target_rate")
    assert not hasattr(cfg.rewards, "drift_wheel_vertical_velocity")
    assert cfg.rewards.action_rate_l2.weight == -.01


@pytest.mark.parametrize("enabled", ["leg", "vertical", "both"])
def test_optional_contact_robust_terms_activate_only_for_their_weight_keys(configuration_inputs, enabled):
    profile = task_profile()
    if enabled in ("leg", "both"):
        profile["leg_target_rate_weight"] = -.5
    if enabled in ("vertical", "both"):
        profile.update(wheel_vertical_velocity_weight=-2., wheel_vertical_velocity_std_m_s=.02)
    cfg = extract_configuration_factory()(*configuration_inputs, num_envs=4, task_profile=profile)
    assert hasattr(cfg.rewards, "drift_leg_target_rate") == (enabled in ("leg", "both"))
    assert hasattr(cfg.rewards, "drift_wheel_vertical_velocity") == (enabled in ("vertical", "both"))
    if enabled in ("leg", "both"):
        term = cfg.rewards.drift_leg_target_rate
        assert term.func is drift.leg_target_rate_penalty and term.weight == -.5 and term.params == {}
    if enabled in ("vertical", "both"):
        term = cfg.rewards.drift_wheel_vertical_velocity
        assert term.func is drift.wheel_vertical_velocity_penalty and term.weight == -2.
        assert term.params == {"std": .02}
    assert cfg.rewards.action_rate_l2.weight == -.01
    assert cfg.scene.robot.actuators["legs"].stiffness == 500.
    assert cfg.scene.robot.actuators["legs"].damping == 10.
    assert cfg.scene.terrain.physics_material.static_friction == .15
    assert cfg.sim.dt == .005 and cfg.decimation == 4


def test_leg_target_rate_is_physical_leg_only_per_environment_and_has_no_history():
    # NumPy arrays exercise the same device-neutral tensor slicing/arithmetic.
    previous = np.array([[.1] * 6 + [-1., 1.], [0.] * 8])
    action = previous.copy()
    action[0, :6] += .01
    action[1, :6] += np.array([.02, -.02, 0., 0., 0., 0.])
    action[:, 6:] *= -1.  # Arbitrarily large wheel-action changes must not enter this cost.
    env = NS(action_manager=NS(action=action, prev_action=previous), step_dt=.02)
    action_before, previous_before = action.copy(), previous.copy()
    np.testing.assert_allclose(drift.leg_target_rate_penalty(env), [1.5, 2.])
    env.step_dt = .01
    np.testing.assert_allclose(drift.leg_target_rate_penalty(env), [6., 8.])
    np.testing.assert_array_equal(action, action_before)
    np.testing.assert_array_equal(previous, previous_before)
    env.action_manager.prev_action = action.copy()
    np.testing.assert_array_equal(drift.leg_target_rate_penalty(env), np.zeros(2))
    assert not any("history" in key or "cache" in key for key in vars(env))


def test_wheel_vertical_speed_uses_named_link_origins_not_com_or_other_body_motion():
    class LinkData:
        body_link_lin_vel_w = np.zeros((2, 4, 3))
        @property
        def body_lin_vel_w(self):
            raise AssertionError("Legacy velocity may be COM; do not read it")
        @property
        def body_com_lin_vel_w(self):
            raise AssertionError("COM velocity is not wheel-link origin velocity")
    data = LinkData()
    data.body_link_lin_vel_w[:, :, 0:2] = 500.  # Planar movement is intentionally irrelevant.
    data.body_link_lin_vel_w[:, 0, 2] = 100.   # Nonwheel vertical motion is irrelevant.
    data.body_link_lin_vel_w[0, [3, 1], 2] = [.02, -.04]
    data.body_link_lin_vel_w[1, [3, 1], 2] = [0., .01]
    robot = NS(body_names=["base_Link", "wheel_R_Link", "hip_L_Link", "wheel_L_Link"], data=data)
    env = NS(scene={"robot": robot})
    before = data.body_link_lin_vel_w.copy()
    np.testing.assert_allclose(drift.wheel_vertical_velocity_penalty(env, std=.02), [5., .25])
    np.testing.assert_allclose(drift.wheel_vertical_velocity_penalty(env, std=.04), [1.25, .0625])
    np.testing.assert_array_equal(data.body_link_lin_vel_w, before)
    assert not any("history" in key or "cache" in key for key in vars(env))


def test_vertical_speed_never_falls_back_when_link_velocity_api_is_missing():
    robot = NS(body_names=["wheel_L_Link", "wheel_R_Link"],
               data=NS(body_lin_vel_w=np.zeros((1, 2, 3))))
    with pytest.raises(AttributeError, match="body_link_lin_vel_w"):
        drift.wheel_vertical_velocity_penalty(NS(scene={"robot": robot}), std=.02)


@pytest.mark.parametrize("explicit", [False, True])
def test_compiled_contact_solref_and_real_contact_match_without_margin_or_gap_changes(explicit):
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string("""
        <mujoco><option timestep=".005"/><worldbody>
          <geom name="floor" type="plane" size="5 5 .1" solref=".004 1" margin=".002" gap=".001"/>
          <body name="wheel" pos="0 0 .04"><freejoint/>
            <geom name="wheel_collision" type="sphere" size=".05" solref=".004 1" margin=".002" gap=".001"/>
            <geom name="visual_only" type="sphere" size=".01" contype="0" conaffinity="0" mass="0" solref=".03 1.5"/>
          </body>
        </worldbody></mujoco>""")
    solref, margin, gap = model.geom_solref.copy(), model.geom_margin.copy(), model.geom_gap.copy()
    profile = task_profile()
    if explicit:
        profile["mujoco_contact_solref"] = [.02, 1.]
    audit = drift.configure_mujoco_materials(model, profile)
    colliding = np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    expected = np.array([.02, 1.]) if explicit else np.array([.004, 1.])
    np.testing.assert_allclose(model.geom_solref[colliding], np.repeat(expected[None], len(colliding), axis=0))
    np.testing.assert_allclose(model.geom_solref[model.geom("visual_only").id], solref[model.geom("visual_only").id])
    np.testing.assert_array_equal(model.geom_margin, margin)
    np.testing.assert_array_equal(model.geom_gap, gap)
    assert audit["contact_margin_and_gap_changed"] is False
    assert audit["explicit_contact_solref"] == ([.02, 1.] if explicit else None)
    assert audit["refsafe_minimum_timeconst_s"] == .01
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon > 0
    for contact in data.contact:
        np.testing.assert_allclose(contact.solref, expected)


@pytest.mark.parametrize("invalid", [4.999, 50.001, 0., -1., True, False, None, "30", [30.], np.nan, np.inf, -np.inf])
def test_optional_leg_damping_rejects_invalid_types_nonfinite_and_out_of_range_values(invalid):
    profile = task_profile()
    profile["leg_kd"] = invalid
    with pytest.raises(ValueError):
        drift.validate_task_profile(profile)


@pytest.mark.parametrize("value", [5., 30., 50.])
def test_optional_leg_damping_accepts_declared_safe_bounds(value):
    profile = task_profile()
    profile["leg_kd"] = value
    assert drift.validate_task_profile(profile)["leg_kd"] == value


def test_damping_profile_is_applied_to_actual_factory_configuration_without_changing_stiffness_or_wheels(configuration_inputs):
    profile = task_profile()
    profile["leg_kd"] = 30.
    cfg = extract_configuration_factory()(*configuration_inputs, num_envs=4, task_profile=profile)
    assert cfg.scene.robot.actuators["legs"].stiffness == 500.
    assert cfg.scene.robot.actuators["legs"].damping == 30.
    assert cfg.scene.robot.actuators["wheels"].stiffness == 0.
    assert cfg.scene.robot.actuators["wheels"].damping == 0.
    assert cfg.scene.robot.actuators["legs"].effort_limit == 80.
    assert cfg.scene.robot.actuators["wheels"].effort_limit == 12.
    assert cfg.sim.dt == .005 and cfg.decimation == 4


def test_shared_damping_contract_override_changes_only_task_profile_and_leg_kd():
    base = read_contract()
    profile = task_profile()
    profile["leg_kd"] = 30.
    base_before, profile_before = copy.deepcopy(base), copy.deepcopy(profile)
    contract = drift.apply_task_profile_to_contract(base, profile)
    assert contract["leg_kp"] == 500. and contract["leg_kd"] == 30.
    assert (contract["actor_dim"], contract["critic_dim"], contract["action_dim"]) == (51, 120, 8)
    assert contract["leg_torque_limit_nm"] == 80.
    assert contract["wheel_torque_limit_nm"] == 12.
    assert contract["leg_motor_velocity_limit_rad_s"] == 15.
    assert contract["wheel_motor_velocity_limit_rad_s"] == 100.
    assert {key: value for key, value in contract.items() if key not in ("leg_kd", "task_profile")} == {
        key: value for key, value in base.items() if key != "leg_kd"}
    assert base == base_before and profile == profile_before
    contract["leg_joint_names"][0] = "changed-only-in-private-copy"
    contract["task_profile"]["reward_weights"]["speed"] = 8.
    assert base == base_before and profile == profile_before


def test_unspecified_damping_keeps_historical_contract_at_five_hundred_ten():
    original = read_contract()
    contract = drift.apply_task_profile_to_contract(original, task_profile())
    assert contract["leg_kp"] == 500. and contract["leg_kd"] == 10.
    assert {key: value for key, value in contract.items() if key != "task_profile"} == original


def test_overridden_damping_contract_roundtrips_and_controls_mujoco_pd(tmp_path):
    from eval_tracking_mujoco import action_torques
    profile = task_profile()
    profile["leg_kd"] = 30.
    contract = drift.apply_task_profile_to_contract(read_contract(), profile)
    path = tmp_path / "policy_contract.json"
    path.write_text(json.dumps(contract))
    deployed = read_contract(path)
    assert deployed == contract
    torque = action_torques(np.zeros(8), np.zeros(8), np.r_[np.full(6, .5), np.zeros(2)], np.zeros(8), deployed)
    np.testing.assert_allclose(torque, np.r_[np.full(6, -15.), np.zeros(2)])
