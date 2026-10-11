"""Execute isolated evaluator validation branches; no app, actor or physics."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from training.tron1_drift import apply_task_profile_to_contract
from eval_tracking_mujoco import read_contract as observation_action_contract

SOURCE = ROOT / 'scripts/eval_tracking.py'


def main_nodes():
    main = next(n for n in ast.parse(SOURCE.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'main')
    protected = next(n for n in main.body if isinstance(n, ast.Try))
    return main, protected


def execute(nodes, namespace):
    module = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(module, str(SOURCE), 'exec'), namespace)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def state(tmp_path):
    profile = json.loads((ROOT / 'config/car_drift_contact_task.json').read_text())
    nominal = apply_task_profile_to_contract(observation_action_contract(), profile)
    asset, motion, checkpoint = [tmp_path / name for name in ('asset.usda', 'motion.npz', 'model_final.pt')]
    for path in (asset, motion, checkpoint):
        path.write_bytes(b'isolated validation fixture; never executed')
    manifest_path = tmp_path / 'manifest.json'
    manifest = {'task_profile': copy.deepcopy(profile), 'contract': copy.deepcopy(nominal),
                'terrain': None, 'motion_sha256': digest(motion), 'asset_sha256': digest(asset)}
    manifest_path.write_text(json.dumps(manifest))
    return {'args': NS(asset_path=asset, motion_file=motion, checkpoint=checkpoint),
            'training_manifest_path': manifest_path, 'trained_manifest': None,
            'task_profile': copy.deepcopy(profile), 'terrain_spec': None,
            'controller_gain_diagnostic': None, 'deployment_leg_kp': 400.,
            'observation_action_contract': observation_action_contract,
            'sha256': digest, 'json': json}


def validation_nodes():
    _, protected = main_nodes()
    trained = next(n for n in protected.body if isinstance(n, ast.If)
                   and ast.unparse(n.test) == 'training_manifest_path.is_file()')
    gain = next(n for n in protected.body if isinstance(n, ast.If)
                and ast.unparse(n.test) == 'deployment_leg_kp is not None')
    return [trained, gain]


def test_explicit_gain_records_original_contract_and_changes_only_effective_kp(state):
    original = copy.deepcopy(state['task_profile'])
    execute(validation_nodes(), state)
    assert state['task_profile'] == {**original, 'leg_kp': 400.}
    audit = state['controller_gain_diagnostic']
    assert audit['original_training_task_profile'] == original
    assert audit['original_nominal_contract']['leg_kp'] == 500.
    assert audit['deployment_leg_kp'] == 400.
    assert audit['additional_training_updates'] == 0
    assert audit['original_manifest_sha256'] == digest(state['training_manifest_path'])
    assert audit['checkpoint_sha256'] == digest(state['args'].checkpoint)


def test_default_never_enters_diagnostic_or_changes_profile(state, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'training.tron1_controller_gain_diagnostic', NS())
    state['deployment_leg_kp'] = None
    original = copy.deepcopy(state['task_profile'])
    execute(validation_nodes(), state)
    assert state['task_profile'] == original and state['controller_gain_diagnostic'] is None


def test_original_profile_mismatch_rejected_before_diagnostic(state, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'training.tron1_controller_gain_diagnostic', NS())
    state['task_profile']['contact_friction'] = .2
    with pytest.raises(ValueError, match='differs from training'):
        execute(validation_nodes(), state)
    assert state['controller_gain_diagnostic'] is None


def test_missing_original_manifest_aborts(state):
    state['training_manifest_path'] = state['training_manifest_path'].parent / 'absent.json'
    with pytest.raises(ValueError, match='original checkpoint training manifest'):
        execute(validation_nodes(), state)


def test_changed_asset_aborts_before_helper_or_environment(state, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'training.tron1_controller_gain_diagnostic', NS())
    state['args'].asset_path.write_bytes(b'different asset')
    with pytest.raises(ValueError, match='retain the trained robot asset'):
        execute(validation_nodes(), state)


def test_gain_preparation_precedes_environment_and_final_validation_precedes_archives():
    _, protected = main_nodes()
    gain = next(n for n in protected.body if isinstance(n, ast.If)
                and ast.unparse(n.test) == 'deployment_leg_kp is not None')
    calls = {n.func.id: n.lineno for n in ast.walk(protected)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert gain.end_lineno < calls['make_env_cfg'] < calls['TerminalCaptureEnv']
    save_lines = [n.lineno for n in ast.walk(protected) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr == 'savez_compressed']
    assert calls['validate_deployment_gain_diagnostic'] < min(save_lines)


@pytest.mark.parametrize('gain,physics,domain', [(float('nan'), None, None),
    (float('inf'), None, None), (99., None, None), (501., None, None),
    (400., .005, None), (400., None, Path('draw.json'))])
def test_cli_rejects_invalid_or_multifactor_requests_before_app(gain, physics, domain):
    main, _ = main_nodes()
    guard = next(n for n in main.body if isinstance(n, ast.If)
                 and ast.unparse(n.test) == 'deployment_leg_kp is not None')
    def error(message):
        raise ValueError(message)
    with pytest.raises(ValueError):
        execute([guard], {'deployment_leg_kp': gain, 'args': NS(physics_dt_s=physics,
                domain_parameters=domain), 'parser': NS(error=error), 'math': __import__('math')})
