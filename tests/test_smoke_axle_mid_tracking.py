"""Synthetic/source-only021 checks, not GPU fixtures or learned behavior."""
import ast
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_axle_mid_tracking.py"
OLD = ROOT / "scripts/smoke_tight_axle_tracking.py"
spec = importlib.util.spec_from_file_location("weight_smoke_test", SCRIPT)
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)


def test_import_inert(tmp_path):
    code = f"import importlib.util,sys; q=importlib.util.spec_from_file_location('v',{str(SCRIPT)!r}); m=importlib.util.module_from_spec(q); q.loader.exec_module(m); assert not any(k.startswith(('torch','isaac','omni','mujoco')) for k in sys.modules)"
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())


def ledger():
    return [{"name": "drift_wheel_ground_load_reserve", "function": "training.tron1_worst_ground_load_substeps.substep_worst_wheel_ground_load_reserve_penalty", "weight": -1., "params": {"scale_fraction": .20}},
        {"name": "drift_wheel_axle_orientation", "function": "training.tron1_tight_axle_tracking.wheel_axle_orientation_tracking", "weight": 1., "params": {"command_name": "motion", "std": .2}}]


def current_ledger():
    value = ledger(); value[1]["weight"] = 2.0
    return value


def test_only_weight_changes():
    baseline, current = ledger(), current_ledger()
    before = copy.deepcopy(baseline)
    s.assert_weight_only(baseline, current)
    assert baseline == before


@pytest.mark.parametrize("change", ["load_function", "load_weight", "load_scale", "axle_mid", "axle_function", "old_std", "missing", "duplicate", "extra"])
def test_any_other_ledger_change_refused(change):
    value = current_ledger()
    if change == "load_function": value[0]["function"] = "mean"
    if change == "load_weight": value[0]["weight"] = -2.
    if change == "load_scale": value[0]["params"]["scale_fraction"] = .3
    if change == "axle_mid": value[1]["weight"] = 1
    if change == "axle_function": value[1]["function"] = "modified"
    if change == "old_std": value[1]["params"]["std"] = .4
    if change == "missing": value.pop()
    if change == "duplicate": value.append(copy.deepcopy(value[0]))
    if change == "extra": value.append({"name": "extra", "function": "x", "weight": 1., "params": {}})
    with pytest.raises(ValueError): s.assert_weight_only(ledger(), value)


@pytest.mark.parametrize("n", [8, 2048])
def test_independent_numpy_axle_and_rewardmanager_units(n):
    ref = np.zeros((16, n, 2, 4)); ref[..., 0] = 1
    actual = ref.copy(); actual[..., 0, 0] = np.cos(.15); actual[..., 0, 1] = np.sin(.15)
    expected = np.full((16, n), np.exp(-(.3**2/2)/(.2**2)))
    np.testing.assert_allclose(s.numpy_axle_reward(ref, actual), expected, atol=1e-12)
    arrays = {"axle_reference_quaternion_wxyz": ref, "axle_actual_quaternion_wxyz": actual,
              "axle_weighted_reward_density": 2*expected, "axle_integrated_reward_contribution": 2*expected*.02}
    assert not s.validate_saved_axle_evidence(arrays, n)["physics_or_actor_executed"]
    for key in ("axle_weighted_reward_density", "axle_integrated_reward_contribution"):
        bad = {k: v.copy() for k, v in arrays.items()}; bad[key] += .01
        with pytest.raises(ValueError): s.validate_saved_axle_evidence(bad, n)


def test_original_force_observer_reset_and_cleanup_preserved():
    def functions(path):
        return {n.name: n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef)}
    before, after = functions(OLD), functions(SCRIPT)
    for name in ("observe", "capture", "assert_armed_ledger", "validate_cli", "close_environment_preserving_pending",
                 "validate_saved_worst_cost_evidence", "stored_density", "resolve_baseline_parameters",
                 "numpy_axle_reward", "deterministic_gpu_invariance"):
        assert ast.dump(before[name], include_attributes=False) == ast.dump(after[name], include_attributes=False)
    assert "configure_axle_mid_trial(cfg, profile)" in SCRIPT.read_text()
    assert "configure_tight_axle_trial(baseline_cfg, baseline_tight_profile())" in SCRIPT.read_text()


@pytest.mark.parametrize("n", [8, 2048])
def test_old_unweighted_or_double_weighted_density_refused(n):
    ref = np.zeros((16, n, 2, 4)); ref[..., 0] = 1
    for factor in (1., 16.):
        arrays = {"axle_reference_quaternion_wxyz": ref, "axle_actual_quaternion_wxyz": ref.copy(),
                  "axle_weighted_reward_density": np.full((16, n), factor),
                  "axle_integrated_reward_contribution": np.full((16, n), factor*.02)}
        with pytest.raises(ValueError): s.validate_saved_axle_evidence(arrays, n)
