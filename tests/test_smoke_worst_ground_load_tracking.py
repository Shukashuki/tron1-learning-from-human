"""Synthetic018 fixture tests; do not claim actual GPU or behavior verification."""
import ast
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_worst_ground_load_tracking.py"
spec = importlib.util.spec_from_file_location("worst_smoke_test", SCRIPT)
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)


def test_import_inert(tmp_path):
    code = f"import importlib.util,sys; q=importlib.util.spec_from_file_location('v',{str(SCRIPT)!r}); m=importlib.util.module_from_spec(q); q.loader.exec_module(m); assert not any(k.startswith(('torch','isaac','omni','mujoco')) for k in sys.modules)"
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("n", [8, 2048])
def test_independent_worst_cost_and_actual_density_units(n):
    f = np.full((64, n, 2, 3), 100., dtype=np.float32); f[::4, :, 1, 2] = 0.
    mg = np.full(n, 200., dtype=np.float32)
    c = np.exp(-np.maximum(f[:, :, :, 2].astype(float), 0)/(.2*mg[None, :, None])).sum(-1).reshape(16,4,n).max(1)
    arrays = {"force_matrix_world_n": f, "mg_n": mg, "cost": c.astype(np.float32),
        "ground_load_weighted_reward_density": (-c).astype(np.float32),
        "ground_load_integrated_reward_contribution": (-.02*c).astype(np.float32)}
    audit = s.validate_saved_worst_cost_evidence(arrays, n)
    assert audit["new_physics_or_actor_executed"] is False
    for key in ("cost", "ground_load_weighted_reward_density", "ground_load_integrated_reward_contribution"):
        bad = {k: v.copy() for k,v in arrays.items()}; bad[key] += .01
        with pytest.raises(ValueError): s.validate_saved_worst_cost_evidence(bad, n)


def ledger():
    return [{"name": "drift_wheel_ground_load_reserve", "function": "training.tron1_ground_load_substeps.substep_wheel_ground_load_reserve_penalty", "weight": -1., "params": {"scale_fraction": .20}},
        {"name": "drift_wheel_axle_orientation", "function": "original_axle", "weight": 1., "params": {"std": .4}}]


def current_ledger():
    value = ledger(); value[0]["function"] = "training.tron1_worst_ground_load_substeps.substep_worst_wheel_ground_load_reserve_penalty"
    return value


def test_only_aggregation_function_changes():
    s.assert_worst_only(ledger(), current_ledger())


@pytest.mark.parametrize("change", ["weight", "params", "axle", "missing", "second", "old_mean"])
def test_any_other_ledger_change_refused(change):
    value = current_ledger()
    if change == "weight": value[0]["weight"] = -2.
    if change == "params": value[0]["params"]["scale_fraction"] = .3
    if change == "axle": value[1]["function"] = "modified"
    if change == "missing": value.pop()
    if change == "second": value.append({"name": "extra", "function": "x", "weight": 1., "params": {}})
    if change == "old_mean": value = ledger()
    with pytest.raises(ValueError): s.assert_worst_only(ledger(), value)


def test_original_observer_and_cleanup_preserved():
    old = ast.parse((ROOT / "scripts/smoke_wheel_axle_tracking.py").read_text())
    new = ast.parse(SCRIPT.read_text())
    def functions(tree):
        return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    before, after = functions(old), functions(new)
    for name in ("observe", "capture", "assert_armed_ledger", "validate_cli", "close_environment_preserving_pending", "numpy_axle_reward", "deterministic_gpu_invariance"):
        assert ast.dump(before[name], include_attributes=False) == ast.dump(after[name], include_attributes=False)
    source = SCRIPT.read_text()
    assert "install_worst_ground_load_accumulator(env)" in source
    assert "configure_worst_ground_load_trial(cfg, profile)" in source
    assert 'expected = total / 4' not in source
