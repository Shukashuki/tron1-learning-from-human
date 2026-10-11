"""Isolated real evaluator Kd branches; no app, tensors or physics."""
import ast
import copy
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from test_eval_controller_gain_entrypoint import digest, execute, main_nodes, state


@pytest.fixture
def damping_state(state):
    state['deployment_leg_kp'] = None
    state['deployment_leg_kd'] = 12.
    return state


def branches():
    _, protected = main_nodes()
    trained = next(n for n in protected.body if isinstance(n, ast.If)
                   and ast.unparse(n.test) == 'training_manifest_path.is_file()')
    damping = next(n for n in protected.body if isinstance(n, ast.If)
                   and ast.unparse(n.test) == 'deployment_leg_kd is not None')
    return [trained, damping]


def test_explicit_kd_changes_only_damping_and_seals_original_contract(damping_state):
    original = copy.deepcopy(damping_state['task_profile'])
    execute(branches(), damping_state)
    assert damping_state['task_profile'] == {**original, 'leg_kd': 12.}
    assert 'leg_kp' not in damping_state['task_profile']
    audit = damping_state['controller_gain_diagnostic']
    assert audit['scope'] == 'zero_shot_common_leg_kd'
    assert audit['trained_leg_kd'] == 10. and audit['deployment_leg_kd'] == 12.
    assert audit['unchanged_leg_kp'] == 500.
    assert audit['additional_training_updates'] == 0
    assert audit['original_training_task_profile'] == original
    assert audit['original_manifest_sha256'] == digest(damping_state['training_manifest_path'])
    assert audit['checkpoint_sha256'] == digest(damping_state['args'].checkpoint)


def test_default_does_not_import_damping_helper_or_add_metadata(damping_state, monkeypatch):
    monkeypatch.setitem(sys.modules, 'training.tron1_controller_damping_diagnostic', NS())
    damping_state['deployment_leg_kd'] = None
    original = copy.deepcopy(damping_state['task_profile'])
    execute(branches(), damping_state)
    assert damping_state['task_profile'] == original
    assert damping_state['controller_gain_diagnostic'] is None


def test_original_mismatch_is_not_bypassed_by_damping_flag(damping_state, monkeypatch):
    monkeypatch.setitem(sys.modules, 'training.tron1_controller_damping_diagnostic', NS())
    damping_state['task_profile']['contact_friction'] = .2
    with pytest.raises(ValueError, match='differs from training'):
        execute(branches(), damping_state)


@pytest.mark.parametrize('missing', ['manifest', 'asset'])
def test_original_manifest_or_asset_mismatch_aborts(damping_state, missing):
    if missing == 'manifest':
        damping_state['training_manifest_path'] = damping_state['training_manifest_path'].parent / 'absent.json'
    else:
        damping_state['args'].asset_path.write_bytes(b'different asset')
    with pytest.raises(ValueError):
        execute(branches(), damping_state)


@pytest.mark.parametrize('kd,kp,physics,domain', [(float('nan'), None, None, None),
    (float('inf'), None, None, None), (4.999, None, None, None), (13.001, None, None, None),
    (12., 400., None, None), (12., None, .005, None), (12., None, None, Path('draw.json'))])
def test_cli_rejects_invalid_or_multifactor_kd_requests_before_app(kd, kp, physics, domain):
    main, _ = main_nodes()
    guard = next(n for n in main.body if isinstance(n, ast.If)
                 and ast.unparse(n.test) == 'deployment_leg_kd is not None')
    def error(message):
        raise ValueError(message)
    with pytest.raises(ValueError):
        execute([guard], {'deployment_leg_kd': kd, 'deployment_leg_kp': kp,
                'args': NS(physics_dt_s=physics, domain_parameters=domain),
                'parser': NS(error=error), 'math': __import__('math')})


def test_kp_and_kd_are_mutually_exclusive_and_preparation_precedes_environment():
    main, protected = main_nodes()
    group = next(n for n in main.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'gain_diagnostics' for t in n.targets))
    assert ast.unparse(group.value) == 'parser.add_mutually_exclusive_group()'
    flags = {n.args[0].value for n in ast.walk(main) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
             and n.func.value.id == 'gain_diagnostics' and n.func.attr == 'add_argument'}
    assert flags == {'--deployment-leg-kp', '--deployment-leg-kd'}
    config_line = min(n.lineno for n in ast.walk(protected) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Name) and n.func.id == 'make_env_cfg')
    assert branches()[1].end_lineno < config_line


def test_frozen_kp_helper_is_unchanged_for_trial007_republication():
    assert digest(ROOT / 'training/tron1_controller_gain_diagnostic.py') == '96ef3cba1761248062f12912b110ec3a000819f644a0030a5a89e28551197b4a'
