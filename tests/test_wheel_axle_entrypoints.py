"""Source and rejected-CLI checks: no real AppLauncher, actor or physics."""
import ast
from collections import Counter
import importlib.util
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_entry(filename, label):
    spec = importlib.util.spec_from_file_location(label, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def application_stub(monkeypatch):
    calls = []
    parent = types.ModuleType("isaaclab")
    parent.__path__ = []
    app = types.ModuleType("isaaclab.app")
    class AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--device", default="cuda:1")
            parser.add_argument("--headless", action="store_true")
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("Rejected preflight must not start an application")
    app.AppLauncher = AppLauncher
    monkeypatch.setitem(sys.modules, "isaaclab", parent)
    monkeypatch.setitem(sys.modules, "isaaclab.app", app)
    return calls


def training_args(tmp_path):
    return ["train", "--motion-file", str(tmp_path / "motion.npz"),
            "--resume", str(tmp_path / "parent.pt"), "--output-dir", str(tmp_path / "new-output"),
            "--num-envs", "2048", "--iterations", "600", "--save-interval", "600",
            "--seed", "42", "--smoke-steps", "8", "--domain-randomization", "motor-v1",
            "--verify-unmasked-device", "--task-profile", str(ROOT / "config/car_drift_wheel_axle_task.json")]


def evaluation_args(tmp_path):
    return ["eval", "--motion-file", str(tmp_path / "motion.npz"), "--asset-path", str(tmp_path / "asset.usd"),
            "--checkpoint", str(tmp_path / "final.pt"), "--runner-config", str(tmp_path / "runner.json"),
            "--output-dir", str(tmp_path / "new-output"), "--num-envs", "1", "--seed", "42",
            "--record-physics", "--verify-unmasked-device", "--contact-threshold-n", "5",
            "--task-profile", str(ROOT / "config/car_drift_wheel_axle_task.json")]


def change_arg(args, name, value):
    result = args.copy()
    result[result.index(name) + 1] = value
    return result


@pytest.mark.parametrize("name,value", [("--num-envs", "8"), ("--iterations", "601"),
                                       ("--save-interval", "100"), ("--seed", "43"),
                                       ("--smoke-steps", "9"), ("--domain-randomization", "none")])
def test_training_budget_or_domain_override_rejected_before_app(monkeypatch, tmp_path, name, value):
    calls = application_stub(monkeypatch)
    module = load_entry("train_tracking_wheel_axle.py", "axle_train_rejected")
    monkeypatch.setattr(sys, "argv", change_arg(training_args(tmp_path), name, value))
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2 and not calls and not (tmp_path / "new-output").exists()


def test_training_requires_explicit_resume_before_isaac_import(monkeypatch, tmp_path):
    calls = application_stub(monkeypatch)
    module = load_entry("train_tracking_wheel_axle.py", "axle_train_no_parent")
    args = training_args(tmp_path)
    index = args.index("--resume")
    del args[index:index + 2]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2 and not calls and not (tmp_path / "new-output").exists()


def test_training_rejects_wrong_parent_before_output_or_application(monkeypatch, tmp_path):
    calls = application_stub(monkeypatch)
    module = load_entry("train_tracking_wheel_axle.py", "axle_train_wrong_parent")
    (tmp_path / "parent.pt").write_bytes(b"Not the original004 checkpoint")
    monkeypatch.setattr(sys, "argv", training_args(tmp_path))
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2 and not calls and not (tmp_path / "new-output").exists()


@pytest.mark.parametrize("filename,args_factory", [("train_tracking_wheel_axle.py", training_args),
                                                   ("eval_tracking_wheel_axle.py", evaluation_args)])
def test_historical_profile_cannot_silently_enable_axle_term(monkeypatch, tmp_path, filename, args_factory):
    calls = application_stub(monkeypatch)
    module = load_entry(filename, "axle_wrong_profile_" + filename)
    args = change_arg(args_factory(tmp_path), "--task-profile", str(ROOT / "config/car_drift_dense_ground_load_task.json"))
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(ValueError):
        module.main()
    assert not calls and not (tmp_path / "new-output").exists()


@pytest.mark.parametrize("extra", [["--deployment-leg-kp", "400"], ["--deployment-leg-kd", "12"],
                                   ["--physics-dt-s", ".0025"], ["--terrain-file", "other.json"],
                                   ["--domain-parameters", "other.json"]])
def test_evaluation_diagnostic_overrides_rejected(monkeypatch, tmp_path, extra):
    calls = application_stub(monkeypatch)
    module = load_entry("eval_tracking_wheel_axle.py", "axle_eval_rejected_override")
    monkeypatch.setattr(sys, "argv", evaluation_args(tmp_path) + extra)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2 and not calls and not (tmp_path / "new-output").exists()


@pytest.mark.parametrize("name,value", [("--num-envs", "2"), ("--seed", "43"), ("--contact-threshold-n", "1")])
def test_evaluation_original_scope_and_contact_gate_held(monkeypatch, tmp_path, name, value):
    calls = application_stub(monkeypatch)
    module = load_entry("eval_tracking_wheel_axle.py", "axle_eval_rejected_scope")
    monkeypatch.setattr(sys, "argv", change_arg(evaluation_args(tmp_path), name, value))
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2 and not calls and not (tmp_path / "new-output").exists()


@pytest.mark.parametrize("flag", ["--record-physics", "--verify-unmasked-device"])
def test_evaluation_requires_physics_recording_and_device_audit(monkeypatch, tmp_path, flag):
    calls = application_stub(monkeypatch)
    module = load_entry("eval_tracking_wheel_axle.py", "axle_eval_missing_evidence")
    args = evaluation_args(tmp_path)
    args.remove(flag)
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2 and not calls and not (tmp_path / "new-output").exists()


def test_evaluator_reuses_exact_frozen_recorder_and_metrics():
    module = load_entry("eval_tracking_wheel_axle.py", "axle_eval_helper_identity")
    import eval_tracking_dense_ground_load as baseline
    for name in ("PhysicsTrajectoryRecorder", "physics_clock_readback", "sha256", "summarize_episodes"):
        assert getattr(module, name) is getattr(baseline, name)


@pytest.mark.parametrize("old,new", [("train_tracking_dense_ground_load.py", "train_tracking_wheel_axle.py"),
                                    ("eval_tracking_dense_ground_load.py", "eval_tracking_wheel_axle.py")])
def test_no_added_integration_actor_reset_or_learning_calls_in_main(old, new):
    # Structural regression, not runtime proof: the live fixture must verify clocks.
    names = {"step", "reset", "learn", "load", "save", "get_inference_policy", "compute_observations",
             "compute", "write_root_state_to_sim", "write_root_pose_to_sim", "close"}
    def counts(filename):
        tree = ast.parse((ROOT / "scripts" / filename).read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        return Counter(n.func.attr for n in ast.walk(node) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute) and n.func.attr in names)
    assert counts(new) == counts(old)
