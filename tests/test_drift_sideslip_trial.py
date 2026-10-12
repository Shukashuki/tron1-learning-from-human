"""Essential CPU-only trial015 single-factor checks; no Isaac/app/actor needed."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import tron1_drift as drift
from training import tron1_drift_sideslip_trial as trial
from training import tron1_substep_rewards as sampler


def load(name):
    return json.loads((ROOT / "config" / name).read_text())


@pytest.fixture
def profiles():
    return (load("car_drift_lr_sync_sideslip_task.json"), load("car_drift_substep_task.json"),
            load("car_drift_contact_task.json"))


def test_actual_sealed_profile_and_protocol_single_factor(profiles):
    new, old, parent = profiles
    assert hashlib.sha256((ROOT / "config/car_drift_lr_sync_sideslip_task.json").read_bytes()).hexdigest() == "a067a61a6b817639e26e41a8c7e20b0f3cff503ffda4f459cef43fd605bbd2ff"
    protocol = load("car_drift_lr_sync_sideslip_trial.json")
    assert protocol["single_factor"]["changed_path"] == "reward_weights.sideslip"
    assert protocol["single_factor"]["before"] == 2 and protocol["single_factor"]["after"] == 4
    assert trial.validate_sideslip_profile(new, old) == new
    assert trial.validate_original_parent_profile(new, parent) == new


def test_validators_return_copies_without_mutating_any_input(profiles):
    new, old, parent = profiles
    before = copy.deepcopy(profiles)
    result = trial.validate_sideslip_profile(new, old)
    result["reward_weights"]["speed"] = 9.
    normalized = trial.normalize_sideslip_profile_for_sampling(new)
    assert normalized["reward_weights"]["sideslip"] == 2.0
    assert normalized["description"] == new["description"]
    normalized["mujoco_contact_solref"][0] = .01
    assert profiles == before
    assert trial.validate_original_parent_profile(new, parent) is not new


@pytest.mark.parametrize("which", [0, 1, 2])
@pytest.mark.parametrize("value", [None, [], True, "profile", 3])
def test_nonobject_profiles_fail_closed(profiles, which, value):
    values = list(profiles); values[which] = value
    if which == 2:
        call = lambda: trial.validate_original_parent_profile(values[0], values[2])
    else:
        call = lambda: trial.validate_sideslip_profile(values[0], values[1])
    with pytest.raises(ValueError): call()


@pytest.mark.parametrize("value", [2., 3., 5., 0., -4., True, False, None, "4", np.nan, np.inf, -np.inf])
def test_new_sideslip_must_be_exactly_finite_nonboolean_four(profiles, value):
    profiles[0]["reward_weights"]["sideslip"] = value
    with pytest.raises(ValueError): trial.validate_sideslip_profile(*profiles[:2])


@pytest.mark.parametrize("path,value", [
    (("contact_friction",), .2), (("env_spacing_m",), 9.),
    (("schema_version",), 1.), (("task",), "jump"),
    (("reward_weights", "yaw_rate"), 2.), (("reward_weights", "speed"), 4.),
    (("reward_weights", "grounded"), .5), (("reward_weights", "tilt"), -.1),
    (("speed_std_m_s",), .2), (("action_rate_weight",), -.4), (("leg_kd",), 12.),
    (("leg_target_rate_weight",), -.5), (("wheel_vertical_velocity_weight",), -4.),
    (("wheel_vertical_velocity_std_m_s",), .03),
    (("wheel_vertical_velocity_sampling",), "last_substep"),
    (("mujoco_contact_solref",), [.03, 2.]),
    (("wheel_ground_load_reserve",), {"weight": -1., "scale_fraction": .2}),
    (("new_teacher",), "wheel-data"), (("beta_error_std_rad",), .1),
    (("beta_speed_mask_m_s",), .2), (("reward_weights", "grounded"), False),
])
def test_every_other_profile_factor_is_rejected(profiles, path, value):
    target = profiles[0]
    for key in path[:-1]: target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError): trial.validate_sideslip_profile(*profiles[:2])
    with pytest.raises(ValueError): trial.validate_original_parent_profile(profiles[0], profiles[2])


@pytest.mark.parametrize("key", ["schema_version", "description", "reward_weights", "speed_std_m_s"])
def test_missing_profile_fields_fail(profiles, key):
    profiles[0].pop(key)
    with pytest.raises(ValueError): trial.validate_sideslip_profile(*profiles[:2])


@pytest.mark.parametrize("key", ["leg_kd", "wheel_vertical_velocity_weight", "reward_weights.speed", "solref"])
def test_unchanged_numeric_type_substitution_is_not_silent(profiles, key):
    if key == "reward_weights.speed": profiles[0]["reward_weights"]["speed"] = 3
    elif key == "solref": profiles[0]["mujoco_contact_solref"][1] = 2
    else: profiles[0][key] = int(profiles[0][key])
    with pytest.raises(ValueError): trial.validate_sideslip_profile(*profiles[:2])


@pytest.mark.parametrize("value", [None, "", "  ", 15, False, {}, []])
def test_description_metadata_must_be_nonempty_string(profiles, value):
    profiles[0]["description"] = value
    with pytest.raises(ValueError): trial.validate_sideslip_profile(*profiles[:2])


def test_description_can_change_or_stay_but_never_changes_other_values(profiles):
    new, old, _ = profiles
    new["description"] = old["description"]
    assert trial.validate_sideslip_profile(new, old) == new


def test_old014_pure_sampling_validator_accepts_only_normalized_copy(profiles):
    path = ROOT / "outputs/drift-trial014-deployment-20261012/guard_sampling.py"
    spec = importlib.util.spec_from_file_location("old014_pure_sampling_for015_test", path)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    original_function = old.validate_sampling_profile
    new, _, parent = profiles
    before = copy.deepcopy(new)
    with pytest.raises(ValueError): old.validate_sampling_profile(new, parent)
    normalized = trial.normalize_sideslip_profile_for_sampling(new)
    assert old.validate_sampling_profile(normalized, parent) == normalized
    assert new == before and old.validate_sampling_profile is original_function


@pytest.mark.parametrize("key", ["wheel_vertical_velocity_sampling", "new_key"])
def test_parent_must_be_original_implicit_sampling_profile(profiles, key):
    profiles[2][key] = "all_substeps_mean"
    with pytest.raises(ValueError): trial.validate_original_parent_profile(profiles[0], profiles[2])


@pytest.fixture
def contract_pair(profiles):
    original = json.loads((ROOT / "outputs/drift-sim2sim-20261011/trial004/training/manifest.json").read_text())["contract"]
    original["task_profile"] = copy.deepcopy(profiles[1])
    changed = copy.deepcopy(original); changed["task_profile"] = copy.deepcopy(profiles[0])
    return changed, original, profiles[0]


def test_complete_nominal_contract_only_explicit_profile_changes(contract_pair, profiles):
    before = copy.deepcopy(contract_pair)
    result = trial.validate_sideslip_contract(*contract_pair)
    assert result == contract_pair[0] and result is not contract_pair[0]
    result["leg_joint_names"][0] = "private-change"
    assert contract_pair == before
    parent = copy.deepcopy(contract_pair[1]); parent["task_profile"] = profiles[2]
    assert trial.validate_sideslip_contract(contract_pair[0], parent, profiles[0]) == contract_pair[0]


@pytest.mark.parametrize("key,value", [("leg_kp", 400.), ("leg_kd", 12.), ("physics_dt", .0025),
    ("decimation", 8), ("actor_dim", 52), ("action_dim", 9), ("policy_fps", 100.),
    ("controller_gain_diagnostic", {}), ("physics_clock_refinement", {}), ("new_teacher", True),
    ("domain_randomization", "disabled"), ("leg_action_scale_rad", .5)])
def test_any_nominal_contract_change_fails(contract_pair, key, value):
    contract_pair[0][key] = value
    with pytest.raises(ValueError): trial.validate_sideslip_contract(*contract_pair)


@pytest.mark.parametrize("change", ["missing_key", "bool_alias", "numeric_type", "profile_mismatch", "extra_profile_key"])
def test_contract_missing_unknown_typed_or_unvalidated_profile_changes_fail(contract_pair, change):
    contract, original, profile = contract_pair
    if change == "missing_key": contract.pop("actor_dim")
    elif change == "bool_alias": contract["action_dim"] = True
    elif change == "numeric_type": contract["actor_dim"] = float(contract["actor_dim"])
    elif change == "profile_mismatch": contract["task_profile"]["description"] = "different description"
    else: contract["task_profile"]["hidden_teacher"] = True
    with pytest.raises(ValueError): trial.validate_sideslip_contract(contract, original, profile)


def _factory_helpers():
    path = ROOT / "tests/test_car_drift_task.py"
    spec = importlib.util.spec_from_file_location("isolated_cfg_helpers_for015", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _snapshot(value):
    if isinstance(value, NS): return {k: _snapshot(v) for k,v in vars(value).items()}
    if type(value) is dict: return {k: _snapshot(v) for k,v in value.items()}
    if type(value) in (list, tuple): return (type(value).__name__, [_snapshot(v) for v in value])
    if isinstance(value, np.ndarray): return (str(value.dtype), value.shape, value.tolist())
    if callable(value): return value
    return value


def test_full_isolated_factory_configuration_diff_is_exactly_sideslip_weight(profiles, tmp_path, monkeypatch):
    helper = _factory_helpers()
    managers = types.ModuleType("isaaclab.managers"); managers.RewardTermCfg = helper.config_constructor
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    motion = tmp_path / "fixture.npz"
    contract = helper.read_contract()
    np.savez_compressed(motion, fps=np.array([50.]), joint_names=np.asarray(contract["observed_joint_velocity_order"]),
        joint_pos=np.zeros((3,8)),body_pos_w=np.zeros((3,len(contract["body_names"]),3)))
    asset = tmp_path / "fixture.usd"; asset.write_text("# Offline path-only fixture\n")
    factory = helper.extract_configuration_factory()
    old = factory(motion, asset, num_envs=2048, device="cuda:5", task_profile=profiles[1])
    new = factory(motion, asset, num_envs=2048, device="cuda:5", task_profile=profiles[0])
    assert new.rewards.drift_sideslip.func is old.rewards.drift_sideslip.func is drift.sideslip_tracking
    assert new.rewards.drift_sideslip.params == old.rewards.drift_sideslip.params == {}
    assert new.rewards.drift_sideslip.weight == 4. and old.rewards.drift_sideslip.weight == 2.
    assert new.rewards.drift_wheel_vertical_velocity.func is sampler.substep_wheel_vertical_velocity_penalty
    assert new.rewards.drift_wheel_vertical_velocity.params == {"std_m_s": .02}
    assert new.rewards.drift_wheel_vertical_velocity.weight == -5.
    new.rewards.drift_sideslip.weight = old.rewards.drift_sideslip.weight
    assert _snapshot(new) == _snapshot(old)


@pytest.fixture
def math_stubs(monkeypatch):
    torch = types.ModuleType("torch")
    torch.atan2 = np.arctan2; torch.sin = np.sin; torch.cos = np.cos; torch.exp = np.exp
    torch.linalg = NS(vector_norm=lambda x,dim:np.linalg.norm(x,axis=dim))
    math_module = types.ModuleType("isaaclab.utils.math")
    def inverse(q):
        result = q.copy(); result[:,1:] *= -1; return result
    def apply(q,v):
        xyz=q[:,1:]; t=2*np.cross(xyz,v)
        return v + q[:,:1]*t + np.cross(xyz,t)
    math_module.quat_inv = inverse; math_module.quat_apply = apply
    monkeypatch.setitem(sys.modules,"torch",torch)
    monkeypatch.setitem(sys.modules,"isaaclab.utils.math",math_module)
    return apply


def reward_env(actual_beta_deg, reference_beta_deg, actual_speed=.5, reference_speed=.5):
    actual_beta, ref_beta = np.radians(np.atleast_1d(actual_beta_deg)), np.radians(np.atleast_1d(reference_beta_deg))
    actual_speed=np.broadcast_to(actual_speed,actual_beta.shape); reference_speed=np.broadcast_to(reference_speed,ref_beta.shape)
    actual=np.c_[actual_speed*np.cos(actual_beta),actual_speed*np.sin(actual_beta),np.zeros(len(actual_beta))]
    reference=np.c_[reference_speed*np.cos(ref_beta),reference_speed*np.sin(ref_beta),np.zeros(len(ref_beta))]
    quat=np.tile([1.,0.,0.,0.],(len(actual_beta),1))
    robot=NS(data=NS(root_quat_w=quat.copy(),root_lin_vel_w=actual))
    command=NS(anchor_quat_w=quat.copy(),body_lin_vel_w=reference[:,None,:])
    return NS(scene={"robot":robot},command_manager=NS(get_term=lambda name:command))


@pytest.mark.parametrize("actual,reference,error_deg", [(8,8,0),(0,8,-8),(-8,8,-16),
    (18,8,10),(-10,-10,0),(10,-10,20),(179,-179,-2),(-179,179,2)])
def test_actual_original_signed_wrapped_beta_formula_without_torch(actual,reference,error_deg,math_stubs):
    env=reward_env(actual,reference)
    error=drift.drift_motion_terms(env)[-1]
    np.testing.assert_allclose(error,np.radians([error_deg]),atol=1e-14)
    np.testing.assert_allclose(drift.sideslip_tracking(env),np.exp(-(np.radians([error_deg])/.12)**2),atol=1e-14)


@pytest.mark.parametrize("actual_speed,ref_speed,expected", [(.25,.5,0),(.25-1e-12,.5,0),
    (.25+1e-12,.5,1),(.5,.3,0),(.5,.3-1e-12,0),(.5,.3+1e-12,1),(.5,0,0),(0,.5,0),(0,0,0)])
def test_strict_actual_and_reference_world_speed_masks_unchanged(actual_speed,ref_speed,expected,math_stubs):
    np.testing.assert_allclose(drift.sideslip_tracking(reward_env(0,0,actual_speed,ref_speed)),[expected],atol=0)


def test_weight_amplitude_doubles_only_unmasked_reward_and_stopped_target_stays_zero(math_stubs):
    env=reward_env([0,8,-8],[8,8,8],actual_speed=[.5,.5,.2],reference_speed=.5)
    raw=drift.sideslip_tracking(env)
    np.testing.assert_array_equal(4*raw,2*(2*raw))
    assert raw[2] == 0 and drift.sideslip_tracking(reward_env(30,0,.5,0))[0] == 0


def test_full_quaternion_reward_vs_yaw_plane_difference_is_preserved(math_stubs):
    env=reward_env(45,45,1.,1.)
    angle=np.radians(40)
    env.scene["robot"].data.root_quat_w[:]=[np.cos(angle/2),0,np.sin(angle/2),0]
    expected=np.arctan2(np.sin(np.pi/4),np.cos(angle)*np.cos(np.pi/4))-np.pi/4
    np.testing.assert_allclose(drift.drift_motion_terms(env)[-1],[expected],atol=1e-14)
    assert expected>0 and drift.sideslip_tracking(env)[0]<1


def test_common_world_yaw_rotation_preserves_signed_beta_error(math_stubs):
    env=reward_env(12,-9)
    expected=drift.sideslip_tracking(env).copy()
    angle=.71; quat=np.array([[np.cos(angle/2),0,0,np.sin(angle/2)]])
    command=env.command_manager.get_term("motion"); robot=env.scene["robot"]
    robot.data.root_lin_vel_w=math_stubs(quat,robot.data.root_lin_vel_w)
    command.body_lin_vel_w[:,0]=math_stubs(quat,command.body_lin_vel_w[:,0])
    robot.data.root_quat_w=quat.copy(); command.anchor_quat_w=quat.copy()
    np.testing.assert_allclose(drift.sideslip_tracking(env),expected,atol=1e-14)


@pytest.fixture
def live_env():
    functions={"sideslip":drift.sideslip_tracking,"yaw_rate":drift.yaw_rate_tracking,"speed":drift.speed_tracking,
               "grounded":drift.grounded_reward,"tilt":drift.tilt_penalty}
    configs={"drift_"+name:NS(func=functions[name],weight=weight,params={"std":.1} if name=="speed" else {})
             for name,weight in trial.HELD_DRIFT_WEIGHTS.items()}
    configs["drift_wheel_vertical_velocity"]=NS(func=sampler.substep_wheel_vertical_velocity_penalty,weight=-5.,params={"std_m_s":.02})
    def other_reward(*args,**kwargs):raise AssertionError("Do not compute rewards in readback")
    configs["other_original_reward"]=NS(func=other_reward,weight=.5,params={"list":[1,2.],"tuple":(True,"x"),"slice":slice(None,4,2)})
    calls=[]
    class Manager:
        active_terms=list(configs)
        def get_term_cfg(self,name):calls.append(name);return configs[name]
        def compute(self,*args):raise AssertionError("Do not compute")
        def reset(self,*args):raise AssertionError("Do not reset")
    env=NS(reward_manager=Manager(),reset=lambda:(_ for _ in ()).throw(AssertionError("Do not reset")))
    return env,configs,calls


def test_live_readback_is_read_only_complete_and_stable_json(live_env):
    env,configs,calls=live_env; before=copy.deepcopy(configs)
    result=trial.live_reward_configuration(env)
    assert result["status"]=="live_reward_configuration_verified"
    assert calls==env.reward_manager.active_terms and len(calls)==len(set(calls))
    assert configs==before and result["reward_compute_calls"]==result["environment_reset_calls"]==0
    assert result["live_full_training_readback"] is False
    assert trial.validate_live_reward_audit(result,wheel_vz_capture_expected=False)==result
    other=next(x for x in result["active_reward_terms"] if x["name"]=="other_original_reward")
    assert other["params"]["tuple"]=={"__type__":"tuple","items":[True,"x"]}
    assert other["params"]["slice"]=={"__type__":"slice","start":None,"stop":4,"step":2}
    assert "0x" not in json.dumps(result,allow_nan=False)
    other["params"]["list"][0]=99
    assert configs["other_original_reward"].params["list"][0]==1


def test_explicit_same_object_vz_fixture_wrapper_is_marked_not_execution_proof(live_env):
    env,configs,_=live_env
    def capture_reward(active_env,std_m_s):raise AssertionError("Readback must not evaluate wrapper")
    configs["drift_wheel_vertical_velocity"].func=capture_reward
    with pytest.raises(ValueError):trial.live_reward_configuration(env)
    result=trial.live_reward_configuration(env,wheel_vz_capture_function=capture_reward)
    exception=result["wheel_vz_capture_exception"]
    assert exception["enabled"] is True and exception["identity_matches_explicit_capture_function"] is True
    assert exception["original_call_count_proved_by_configuration"] is False
    assert exception["production_function"]=="training.tron1_substep_rewards.substep_wheel_vertical_velocity_penalty"
    assert trial.validate_live_reward_audit(result)==result
    with pytest.raises(ValueError):trial.validate_live_reward_audit(result,wheel_vz_capture_expected=False)


@pytest.mark.parametrize("function", [None,1,"capture",drift.sideslip_tracking,sampler.substep_wheel_vertical_velocity_penalty])
def test_wrong_or_implicit_capture_identity_rejected(live_env,function):
    env,configs,_=live_env
    configs["drift_wheel_vertical_velocity"].func=lambda *args:None
    with pytest.raises(ValueError):trial.live_reward_configuration(env,wheel_vz_capture_function=function)


@pytest.mark.parametrize("name", ["sideslip","yaw_rate","speed","grounded","tilt"])
@pytest.mark.parametrize("change", ["weight","function","params","missing"])
def test_held_named_live_drift_terms_cannot_change(live_env,name,change):
    env,configs,_=live_env; key="drift_"+name
    if change=="weight":configs[key].weight+=.1
    elif change=="function":configs[key].func=lambda *args:None
    elif change=="params":configs[key].params["new_mask"]=.2
    else:env.reward_manager.active_terms.remove(key)
    with pytest.raises(ValueError):trial.live_reward_configuration(env)


@pytest.mark.parametrize("change", ["weight","std","parameter_name","bool_weight","missing","reserve","duplicate_name","unknown_param"])
def test_malformed_live_sampling_configuration_fails_closed(live_env,change):
    env,configs,_=live_env; cfg=configs["drift_wheel_vertical_velocity"]
    if change=="weight":cfg.weight=-4.
    elif change=="std":cfg.params["std_m_s"]=.03
    elif change=="parameter_name":cfg.params={"std":.02}
    elif change=="bool_weight":cfg.weight=True
    elif change=="missing":env.reward_manager.active_terms.remove("drift_wheel_vertical_velocity")
    elif change=="reserve":
        configs["drift_wheel_ground_load_reserve"]=copy.deepcopy(cfg)
        env.reward_manager.active_terms.append("drift_wheel_ground_load_reserve")
    elif change=="duplicate_name":env.reward_manager.active_terms.append("drift_sideslip")
    else:configs["other_original_reward"].params={"unsupported":object()}
    with pytest.raises(ValueError):trial.live_reward_configuration(env)


def test_scene_entity_config_uses_stored_fields_without_resolve_repr_or_properties(live_env):
    env,configs,_=live_env
    def fail(self,*args):raise AssertionError("Never call repr/resolve")
    cls=type("SceneEntityCfg",(),{"__module__":"isaaclab.managers.scene_entity_cfg","__repr__":fail,"resolve":fail})
    entity=cls(); entity.name="robot"; entity.joint_ids=slice(None);entity.body_ids=[8,9];entity.preserve_order=True
    configs["other_original_reward"].params={"asset_cfg":entity}
    report=trial.live_reward_configuration(env)
    encoded=next(t for t in report["active_reward_terms"] if t["name"]=="other_original_reward")["params"]["asset_cfg"]
    assert encoded["__config_type__"]=="isaaclab.managers.scene_entity_cfg.SceneEntityCfg"
    assert encoded["fields"]=={"name":"robot","joint_ids":{"__type__":"slice","start":None,"stop":None,"step":None},"body_ids":[8,9],"preserve_order":True}


@pytest.mark.parametrize("change", ["unknown_field","schema_bool","source","compute","reset","fullrun","duplicate","weight","params","function","capture_flag","capture_claim","term_extra","nonfinite","limitations"])
def test_pure_live_audit_validator_rejects_forged_or_expansive_claims(live_env,change):
    audit=trial.live_reward_configuration(live_env[0])
    terms={t["name"]:t for t in audit["active_reward_terms"]}
    if change=="unknown_field":audit["hardware_ready"]=True
    elif change=="schema_bool":audit["schema_version"]=True
    elif change=="source":audit["source_sha256"]["training/tron1_drift.py"]="0"*64
    elif change=="compute":audit["reward_compute_calls"]=False
    elif change=="reset":audit["environment_reset_calls"]=1
    elif change=="fullrun":audit["live_full_training_readback"]=True
    elif change=="duplicate":audit["active_reward_terms"].append(copy.deepcopy(audit["active_reward_terms"][0]))
    elif change=="weight":terms["drift_sideslip"]["weight"]=2.
    elif change=="params":terms["drift_sideslip"]["params"]={"std":.2}
    elif change=="function":terms["drift_sideslip"]["function"]="forged.sideslip"
    elif change=="capture_flag":audit["wheel_vz_capture_exception"]["enabled"]=0
    elif change=="capture_claim":audit["wheel_vz_capture_exception"]["original_call_count_proved_by_configuration"]=True
    elif change=="term_extra":terms["drift_sideslip"]["raw_reward_values"]=[1.]
    elif change=="nonfinite":terms["other_original_reward"]["weight"]=np.nan
    else:audit["limitations"]=[]
    with pytest.raises(ValueError):trial.validate_live_reward_audit(audit,wheel_vz_capture_expected=False)


def test_source_mismatch_blocks_live_readback_before_manager_access(live_env,monkeypatch):
    env,_,calls=live_env
    monkeypatch.setitem(trial.REWARD_SOURCE_SHA256,"training/tron1_drift.py","0"*64)
    with pytest.raises(ValueError):trial.live_reward_configuration(env)
    assert calls==[]


def test_import_is_inert_without_torch_isaac_or_mujoco(tmp_path):
    code=f"import sys;sys.path.insert(0,{str(ROOT)!r});import training.tron1_drift_sideslip_trial;assert not any(x in sys.modules for x in ('torch','isaaclab','mujoco','carb','omni'))"
    result=subprocess.run([sys.executable,"-B","-c",code],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_live_reader_has_no_compute_reset_rng_or_mutation_calls():
    source=ast.parse((ROOT/"training/tron1_drift_sideslip_trial.py").read_text())
    func=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=="live_reward_configuration")
    names={n.func.attr for n in ast.walk(func) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert names.isdisjoint({"compute","reset","step","seed","manual_seed","set_term_cfg","resolve"})
