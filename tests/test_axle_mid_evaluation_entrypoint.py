"""Source/CLI021 checks only; no real physics, actor or AppLauncher."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OLD = ROOT / "scripts/eval_tracking_tight_axle.py"
NEW = ROOT / "scripts/eval_tracking_axle_mid.py"


def imported(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPERS = imported(ROOT / "tests/test_wheel_axle_entrypoints.py", "original_cli_test_helpers")


def args(tmp_path):
    return HELPERS.change_arg(HELPERS.evaluation_args(tmp_path), "--task-profile",
                             str(ROOT / "config/car_drift_axle_mid_task.json"))


@pytest.mark.parametrize("extra", [
    ["--deployment-leg-kp", "400"], ["--deployment-leg-kd", "12"],
    ["--physics-dt-s", ".0025"], ["--terrain-file", "other.json"],
    ["--domain-parameters", "other.json"], ["--num-envs", "2"],
    ["--seed", "43"], ["--contact-threshold-n", "1"],
])
def test_original_nominal_scope_and_gates_reject_overrides(monkeypatch, tmp_path, extra):
    calls = HELPERS.application_stub(monkeypatch)
    module = imported(NEW, "eval021_invalid")
    monkeypatch.setattr(sys, "argv", args(tmp_path) + extra)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2 and not calls and not (tmp_path / "new-output").exists()


@pytest.mark.parametrize("flag", ["--record-physics", "--verify-unmasked-device"])
def test_cannot_omit_physics_or_native_device_evidence(monkeypatch, tmp_path, flag):
    calls = HELPERS.application_stub(monkeypatch)
    module = imported(NEW, "eval021_missing_evidence")
    values = args(tmp_path)
    values.remove(flag)
    monkeypatch.setattr(sys, "argv", values)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2 and not calls and not (tmp_path / "new-output").exists()


@pytest.mark.parametrize("profile", ["car_drift_wheel_axle_task.json", "car_drift_dense_ground_load_task.json", "car_drift_worst_ground_load_task.json", "car_drift_tight_axle_task.json", "car_drift_tight_axle_production_task.json"])
def test_mean_profile_cannot_silently_enable_max(monkeypatch, tmp_path, profile):
    calls = HELPERS.application_stub(monkeypatch)
    module = imported(NEW, "eval021_mean_profile")
    monkeypatch.setattr(sys, "argv", HELPERS.change_arg(args(tmp_path), "--task-profile", str(ROOT / "config" / profile)))
    with pytest.raises(ValueError):
        module.main()
    assert not calls and not (tmp_path / "new-output").exists()


def test_original_recorder_and_metrics_are_identical_objects():
    module = imported(NEW, "eval021_original_recorder")
    import eval_tracking_dense_ground_load as baseline
    for name in ("PhysicsTrajectoryRecorder", "physics_clock_readback", "sha256", "summarize_episodes"):
        assert getattr(module, name) is getattr(baseline, name)


def test_no_new_physics_reset_export_action_or_actor_calls():
    names = {"step", "reset", "load", "save", "get_inference_policy", "get_observations",
             "write_root_state_to_sim", "write_root_pose_to_sim", "begin_step", "finish_step",
             "should_capture", "capture", "clamp", "export_policy_as_jit", "policy", "exported"}
    def selected(path):
        return [ast.dump(node, include_attributes=False) for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, ast.Call) and
                ((isinstance(node.func, ast.Attribute) and node.func.attr in names)
                 or (isinstance(node.func, ast.Name) and node.func.id in names))]
    assert selected(OLD) == selected(NEW)
    def observers(path):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in
                ("to_numpy", "state_snapshot", "TerminalCaptureEnv", "physics_scene_update")}
    assert observers(OLD) == observers(NEW)
    def loops(path):
        return [ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "step_index"]
    assert loops(OLD) == loops(NEW)
    source = NEW.read_text()
    assert "configure_axle_mid_trial(cfg, task_profile)" in source
    assert "install_worst_ground_load_accumulator(env)" in source
    assert '"dense_ground_load_sampling": "all_substeps_max_cost"' in source
    assert "c865d498c94ceafee1bf20f72ded62ac7ee417ec91e370c475ecbaa029fc1663" in source


def test_import_does_not_start_application_or_load_learning_libraries(tmp_path):
    code = (f"import importlib.util,sys;sys.path.insert(0,{str(ROOT/'scripts')!r});"
            f"s=importlib.util.spec_from_file_location('e',{str(NEW)!r});"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "assert not any(k.startswith(('torch','isaac','omni','mujoco','rsl_rl')) for k in sys.modules)")
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())
