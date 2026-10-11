"""Pure evaluation-clock contracts; no actors, Isaac imports, or training."""
from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.tron1_physics_clock import (
    PHYSICS_CLOCKS, apply_physics_clock_to_contract, configure_physics_clock, validate_physics_clock,
)

SPEC = importlib.util.spec_from_file_location("clock_evaluator", ROOT / "scripts/eval_tracking.py")
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def contract(dt=.005, decimation=4):
    return {"actor_dim": 51, "action_dim": 8, "policy_fps": 50., "physics_dt": dt,
            "decimation": decimation, "leg_kp": 500., "leg_kd": 10.,
            "leg_torque_limit_nm": 80., "wheel_torque_limit_nm": 12.,
            "leg_joint_names": [f"leg{i}" for i in range(6)],
            "task_profile": {"task": "car_drift", "reward_weights": {"speed": 3.},
                             "mujoco_contact_solref": [.04, 2.]}}


def configuration():
    return NS(sim=NS(dt=.005, render_interval=4, solver="preserve"), decimation=4,
              scene=NS(contact_forces=NS(update_period=.005),
                       sensors={"wheel": NS(update_period=0.)}, robot=NS(gains=[500., 10.])),
              rewards={"weights": "unchanged"}, episode_length_s=5.8)


def environment(cfg):
    return NS(sim=NS(get_physics_dt=lambda: cfg.sim.dt), physics_dt=cfg.sim.dt,
              step_dt=cfg.sim.dt * cfg.decimation,
              scene=NS(sensors={"contact_forces": NS(cfg=NS(update_period=cfg.scene.contact_forces.update_period))}))


@pytest.mark.parametrize("dt,decimation", PHYSICS_CLOCKS.items())
def test_exact_supported_clocks_preserve_50hz_interface(dt, decimation):
    assert validate_physics_clock(contract(dt, decimation)) == (dt, decimation)
    assert dt * decimation == .02


def test_legacy_minimal_contract_without_policy_fps_keeps_50hz():
    value = contract()
    value.pop("policy_fps")
    assert validate_physics_clock(value) == (.005, 4)


@pytest.mark.parametrize("value", [None, [], True, "clock"])
def test_noncontract_inputs_fail(value):
    with pytest.raises(ValueError):
        validate_physics_clock(value)


@pytest.mark.parametrize("field,value", [
    ("physics_dt", None), ("physics_dt", True), ("physics_dt", ".0025"),
    ("physics_dt", float("nan")), ("physics_dt", float("inf")), ("physics_dt", 10 ** 500),
    ("physics_dt", 0.), ("physics_dt", -.005), ("physics_dt", .01), ("physics_dt", .004),
    ("physics_dt", .0025), ("decimation", 8), ("decimation", 4.), ("decimation", True),
    ("decimation", 0), ("decimation", "4"), ("decimation", None),
    ("actor_dim", 50), ("actor_dim", 51.), ("actor_dim", True),
    ("action_dim", 7), ("action_dim", 8.), ("action_dim", False),
    ("policy_fps", 100.), ("policy_fps", 0.), ("policy_fps", True),
    ("policy_fps", "50"), ("policy_fps", float("nan")),
])
def test_malformed_or_incompatible_clock_contracts_fail(field, value):
    with pytest.raises(ValueError):
        validate_physics_clock({**contract(), field: value})


@pytest.mark.parametrize("field", ["actor_dim", "action_dim", "physics_dt", "decimation"])
def test_missing_required_clock_fields_fail(field):
    value = contract()
    del value[field]
    with pytest.raises(ValueError):
        validate_physics_clock(value)


@pytest.mark.parametrize("dt,decimation", PHYSICS_CLOCKS.items())
def test_configuration_override_changes_only_numerical_timing_and_contact_sampling(dt, decimation):
    cfg = configuration()
    before = copy.deepcopy(cfg)
    assert configure_physics_clock(cfg, dt) == (dt, decimation)
    assert cfg.sim.dt == dt and cfg.decimation == decimation
    assert cfg.sim.render_interval == decimation
    assert cfg.scene.contact_forces.update_period == dt
    assert cfg.scene.sensors["wheel"].update_period == 0.
    assert cfg.scene.robot == before.scene.robot
    assert cfg.rewards == before.rewards and cfg.episode_length_s == before.episode_length_s
    assert cfg.sim.solver == before.sim.solver


@pytest.mark.parametrize("dt", [None, True, ".0025", .004, float("nan"), float("inf")])
def test_invalid_configuration_request_leaves_original_clock_unchanged(dt):
    cfg = configuration()
    before = copy.deepcopy(cfg)
    with pytest.raises(ValueError):
        configure_physics_clock(cfg, dt)
    assert cfg == before


@pytest.mark.parametrize("dt,decimation", PHYSICS_CLOCKS.items())
def test_refined_contract_preserves_trained_clock_and_all_nonclock_fields(dt, decimation):
    original = contract()
    before = copy.deepcopy(original)
    out = apply_physics_clock_to_contract(original, dt, trained_contract=original)
    assert original == before
    assert out["physics_dt"] == dt and out["decimation"] == decimation
    metadata = out["physics_clock_refinement"]
    assert metadata == {
        "schema_version": 1, "scope": "evaluation_only_numerical_refinement",
        "trained_physics_dt_s": .005, "trained_decimation": 4,
        "actual_physics_dt_s": dt, "actual_decimation": decimation,
        "policy_dt_s": .02, "policy_fps": 50.,
    }
    for key in original.keys() - {"physics_dt", "decimation"}:
        assert out[key] == original[key]
    assert validate_physics_clock(out) == (dt, decimation)
    out["task_profile"]["mujoco_contact_solref"][0] = .05
    assert original == before


def test_no_explicit_override_is_deep_copy_identical_default_contract():
    original = contract()
    out = apply_physics_clock_to_contract(original, None)
    assert out == original and out is not original
    assert "physics_clock_refinement" not in out
    out["leg_joint_names"][0] = "changed"
    assert original["leg_joint_names"][0] == "leg0"


def test_refinement_cannot_change_checkpoint_task_profile():
    trained = contract()
    actual = contract()
    actual["task_profile"]["reward_weights"]["speed"] = 1.
    with pytest.raises(ValueError, match="trained task profile"):
        apply_physics_clock_to_contract(actual, .0025, trained)


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("schema_version", 2), ("scope", "training"),
    ("trained_physics_dt_s", .004), ("trained_decimation", 8),
    ("actual_physics_dt_s", .005), ("actual_decimation", 4),
    ("policy_dt_s", .04), ("policy_fps", 100.),
])
def test_inconsistent_refinement_metadata_is_rejected(field, value):
    out = apply_physics_clock_to_contract(contract(), .0025)
    out["physics_clock_refinement"][field] = value
    with pytest.raises(ValueError):
        validate_physics_clock(out)


@pytest.mark.parametrize("dt,decimation", PHYSICS_CLOCKS.items())
def test_live_clock_audit_reads_simulator_environment_and_contact_sensor(dt, decimation):
    cfg = configuration()
    configure_physics_clock(cfg, dt)
    value = apply_physics_clock_to_contract(contract(), dt)
    report = EVALUATOR.physics_clock_readback(environment(cfg), cfg, value)
    assert report == {"status": "verified", "decimation": decimation,
                      "simulator_physics_dt_s": dt, "environment_physics_dt_s": dt,
                      "policy_dt_s": .02, "net_contact_sensor_update_period_s": dt}


@pytest.mark.parametrize("corruption", ["simulator", "environment", "policy", "sensor", "decimation", "nonfinite"])
def test_bad_live_clock_readback_fails_before_rollout(corruption):
    cfg = configuration()
    configure_physics_clock(cfg, .0025)
    env = environment(cfg)
    if corruption == "simulator":
        env.sim.get_physics_dt = lambda: .005
    elif corruption == "environment":
        env.physics_dt = .005
    elif corruption == "policy":
        env.step_dt = .04
    elif corruption == "sensor":
        env.scene.sensors["contact_forces"].cfg.update_period = .005
    elif corruption == "decimation":
        cfg.decimation = 4
    else:
        env.physics_dt = float("nan")
    with pytest.raises(RuntimeError, match="Live evaluation physics clock"):
        EVALUATOR.physics_clock_readback(env, cfg, apply_physics_clock_to_contract(contract(), .0025))


def test_evaluator_cli_is_explicit_and_default_is_not_an_override():
    tree = ast.parse((ROOT / "scripts/eval_tracking.py").read_text())
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--physics-dt-s")
    kwargs = {item.arg: item.value for item in call.keywords}
    assert ast.literal_eval(kwargs["choices"]) == (.005, .0025, .00125)
    assert ast.literal_eval(kwargs["default"]) is None


def test_evaluator_preserves_task_profile_guard_and_applies_override_before_environment():
    tree = ast.parse((ROOT / "scripts/eval_tracking.py").read_text())
    comparisons = [node for node in ast.walk(tree) if isinstance(node, ast.Compare)]
    assert any(ast.unparse(node) == "trained_manifest.get('task_profile') != task_profile" for node in comparisons)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    configure = next(node for node in calls if node.func.id == "configure_physics_clock")
    construct = next(node for node in calls if node.func.id == "TerminalCaptureEnv")
    assert configure.lineno < construct.lineno
    assert sum(node.func.id == "apply_physics_clock_to_contract" for node in calls) == 2
