"""Read-only-source lifecycle tests for opt-in substep reward entrypoints.

Execute selected REAL setup/finally AST statements against lightweight stubs.
No Isaac or Torch import, app launch, GPU access, or physics is involved.
Accumulator numerical/native-clock behavior belongs to test_substep_rewards.
"""

from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def main_try(filename):
    tree = ast.parse((ROOT / "scripts" / filename).read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    blocks = [node for node in main.body if isinstance(node, ast.Try) and node.finalbody]
    assert len(blocks) == 1
    return blocks[0]


def execute(nodes, namespace):
    tree = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(tree, "<actual-entrypoint-ast>", "exec"), namespace)


def is_assignment_call(node, target, function):
    return (isinstance(node, ast.Assign) and any(isinstance(item, ast.Name) and item.id == target for item in node.targets)
            and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == function)


def installer_node(nodes):
    matches = [node for node in nodes if isinstance(node, ast.If)
               and any(isinstance(child, ast.ImportFrom) and child.module == "training.tron1_drift"
                       and any(alias.name == "install_drift_reward_sampling" for alias in child.names)
                       for child in ast.walk(node))]
    assert len(matches) == 1
    return matches[0]


class LifecycleError(RuntimeError):
    pass


class FakeScene:
    def __init__(self, env):
        self.env = env

    def update(self, dt):
        self.env.events.append(("scene", self.env._sim_step_counter, dt))
        return "scene-return"


class FakeEnv:
    def __init__(self, cfg=None):
        self.events = []
        self._sim_step_counter = 0
        self.physics_dt = .005
        self.scene = FakeScene(self)
        self.termination_manager = types.SimpleNamespace(active_terms=("finished",),
            terminated=np.array([True]), time_outs=np.array([False]), get_term=lambda name: np.array([True]))
        self.scene.update(dt=self.physics_dt)  # real constructor prepopulates data at counter0
        self.failure_substep = None

    def reset(self):
        self.events.append(("explicit-reset", self._sim_step_counter))
        self.scene.update(dt=self.physics_dt)  # nonphysical/same-counter refresh
        return {}, {}

    def _reset_idx(self, ids):
        self.events.append(("base-reset-idx", self._sim_step_counter))
        self.scene.update(dt=self.physics_dt)

    def step(self, actions):
        self.events.append(("process-action", self._sim_step_counter))
        self.scene.update(dt=self.physics_dt)  # pre-action same-counter refresh
        for substep in range(1, 5):
            self._sim_step_counter += 1
            self.scene.update(dt=self.physics_dt)
            if self.failure_substep == substep:
                raise LifecycleError("step failed")
        self.events.append(("reward", self._sim_step_counter))
        self._reset_idx([0])  # terminal capture must still dispatch on subclass
        return "unchanged-step-return"

    def close(self):
        self.events.append(("env-close", self._sim_step_counter))


class StubHandler:
    """Opaque lifecycle stand-in; does not test numerical accumulator logic."""

    def __init__(self, env):
        self.env = env
        self.events = env.events
        self.previous_step = env.step
        self.previous_update = env.scene.update
        self.inside = False
        self.open = False
        self.count = 0
        self.fail_close = False
        self.last_counter = env._sim_step_counter
        self.events.append(("helper-install", env._sim_step_counter))

        def update(*args, **kwargs):
            result = self.previous_update(*args, **kwargs)
            if self.inside and self.open and env._sim_step_counter != self.last_counter:
                self.count += 1
                self.last_counter = env._sim_step_counter
                self.events.append(("helper-sample", self.last_counter))
                if self.count == 4:
                    self.open = False
            return result

        def step(*args, **kwargs):
            self.count = 0
            self.last_counter = env._sim_step_counter
            self.inside = self.open = True
            try:
                return self.previous_step(*args, **kwargs)
            finally:
                self.inside = self.open = False

        self.update_wrapper, self.step_wrapper = update, step
        env.scene.update = update
        env.step = step

    def close(self):
        self.events.append(("helper-close", self.env._sim_step_counter))
        if self.fail_close:
            raise LifecycleError("helper close failed")
        assert self.env.scene.update is self.update_wrapper, "Incorrect observer restoration order"
        assert self.env.step is self.step_wrapper
        self.env.scene.update = self.previous_update
        self.env.step = self.previous_step


class FakeWrapper:
    def __init__(self, env, clip_actions=None):
        self.env = env
        self.env.reset()  # installed RSL constructor calls self.env.reset()

    def step(self, actions):
        return self.env.step(actions)  # installed RSL delegates dynamically

    def close(self):
        self.env.events.append(("wrapped-close", self.env._sim_step_counter))
        self.env.close()


class FakeRecorder:
    def __init__(self, *args):
        self.last = int(args[2])
        self.inside = False
        self.counters = []

    def should_capture(self, counter):
        return self.inside and counter != self.last

    def capture(self, counter, snapshot):
        self.counters.append(counter)
        self.last = counter


@pytest.fixture
def stub_namespace(monkeypatch):
    handlers = []
    drift = types.ModuleType("training.tron1_drift")

    def install(env, profile):
        if profile is None or profile.get("wheel_vertical_velocity_sampling") != "all_substeps_mean":
            return None
        handler = StubHandler(env)
        handlers.append(handler)
        return handler

    drift.install_drift_reward_sampling = install
    monkeypatch.setitem(sys.modules, "training.tron1_drift", drift)
    namespace = {"ManagerBasedRLEnv": FakeEnv, "RslRlVecEnvWrapper": FakeWrapper, "cfg": object(),
        "task_profile": {"wheel_vertical_velocity_sampling": "all_substeps_mean"},
        "reward_accumulator": None, "PhysicsTrajectoryRecorder": FakeRecorder, "original_scene_update": None,
        "to_numpy": np.asarray, "state_snapshot": lambda env: {"snapshot_counter": np.array([env._sim_step_counter])},
        "torch": types.SimpleNamespace(as_tensor=np.asarray), "obs": {"policy": np.zeros((1, 51))},
        "initial": {}, "args": types.SimpleNamespace(record_physics=True, num_envs=1, headless=True),
        "output": Path("/not-written"), "report": {}, "write_json": lambda *args: None}
    return namespace, handlers


def train_setup(namespace):
    block = main_try("train_tracking.py")
    env = next(node for node in block.body if is_assignment_call(node, "env", "ManagerBasedRLEnv"))
    install = installer_node(block.body)
    wrapped = next(node for node in block.body if is_assignment_call(node, "wrapped", "RslRlVecEnvWrapper"))
    assert env.lineno < install.lineno < wrapped.lineno
    execute([env, install, wrapped], namespace)
    return block


def eval_setup(namespace):
    block = main_try("eval_tracking.py")
    terminal_class = next(node for node in block.body if isinstance(node, ast.ClassDef) and node.name == "TerminalCaptureEnv")
    execute([terminal_class], namespace)
    namespace["env"] = namespace["TerminalCaptureEnv"](cfg=None)
    namespace["wrapped"] = FakeWrapper(namespace["env"])
    observer = next(node for node in block.body if isinstance(node, ast.If)
                    and isinstance(node.test, ast.Attribute) and node.test.attr == "record_physics")
    install = installer_node(block.body)
    assert observer.lineno < install.lineno
    execute([observer, install], namespace)
    return block


def attach_app(namespace):
    events = namespace["env"].events
    namespace["app"] = types.SimpleNamespace(close=lambda **kwargs: events.append(("app-close", kwargs)))


@pytest.mark.parametrize("enabled", [False, True])
def test_train_constructor_rsl_reset_and_step_delegate_without_init_samples(stub_namespace, enabled):
    namespace, handlers = stub_namespace
    if not enabled:
        namespace["task_profile"] = None
    block = train_setup(namespace)
    env = namespace["env"]
    assert env.events[0] == ("scene", 0, .005)
    if enabled:
        assert len(handlers) == 1 and handlers[0].count == 0 and not handlers[0].inside
        assert namespace["wrapped"].step(None) == "unchanged-step-return"
        assert handlers[0].count == 4
    else:
        assert not handlers and namespace["reward_accumulator"] is None
        assert namespace["wrapped"].step(None) == "unchanged-step-return"
    attach_app(namespace)
    execute(block.finalbody, namespace)
    assert [event[0] for event in env.events][-2:] == ["env-close", "app-close"]


@pytest.mark.parametrize("failure_substep", [None, 1, 2, 4])
def test_eval_actual_observer_setup_terminal_capture_and_lifo_cleanup(stub_namespace, failure_substep):
    namespace, handlers = stub_namespace
    block = eval_setup(namespace)
    env, handler, recorder = namespace["env"], handlers[0], namespace["physics_recorder"]
    assert handler.previous_update is namespace["physics_scene_update"]
    assert handler.count == 0
    env.capture_resets = True
    env.failure_substep = failure_substep
    recorder.inside = True
    if failure_substep is None:
        assert namespace["wrapped"].step(None) == "unchanged-step-return"
        assert env.terminal_snapshots[0]["state"]["snapshot_counter"] == 4
        assert env.terminal_snapshots[0]["early_terminated"] is True
    else:
        with pytest.raises(LifecycleError, match="step failed"):
            namespace["wrapped"].step(None)
    assert recorder.counters == list(range(1, (failure_substep or 4) + 1))
    assert handler.count == (failure_substep or 4)
    assert not handler.inside and not handler.open
    attach_app(namespace)
    execute(block.finalbody, namespace)
    assert env.scene.update == namespace["original_scene_update"]
    assert [event[0] for event in env.events][-4:] == ["helper-close", "wrapped-close", "env-close", "app-close"]


@pytest.mark.parametrize("entrypoint", ["train", "eval"])
def test_entrypoint_finally_closes_env_and_app_even_helper_close_fails(stub_namespace, entrypoint):
    namespace, handlers = stub_namespace
    block = train_setup(namespace) if entrypoint == "train" else eval_setup(namespace)
    handlers[0].fail_close = True
    attach_app(namespace)
    with pytest.raises(LifecycleError, match="helper close failed"):
        execute(block.finalbody, namespace)
    assert [event[0] for event in namespace["env"].events][-2:] == ["env-close", "app-close"]


def test_eval_cleanup_never_overwrites_unknown_later_observer(stub_namespace):
    namespace, handlers = stub_namespace
    block = eval_setup(namespace)
    unknown = lambda *args, **kwargs: "unknown"
    namespace["env"].scene.update = unknown
    handlers[0].fail_close = True
    attach_app(namespace)
    with pytest.raises(LifecycleError):
        execute(block.finalbody, namespace)
    assert namespace["env"].scene.update is unknown
    assert [event[0] for event in namespace["env"].events][-2:] == ["env-close", "app-close"]


def test_eval_observer_is_not_restored_early_before_report_and_finally():
    block = main_try("eval_tracking.py")
    early_restores = [node for statement in block.body for node in ast.walk(statement)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and node.value.id == "original_scene_update"
        and any(isinstance(target, ast.Attribute) and target.attr == "update" for target in node.targets)]
    assert not early_restores


def test_train_status_write_failure_still_restores_helper_and_closes_env_app(stub_namespace):
    namespace, handlers = stub_namespace
    block = train_setup(namespace)
    handler = handlers[0]
    attach_app(namespace)

    def fail_write(*args):
        namespace["env"].events.append(("status-write-error", None))
        raise OSError("run status write failed")

    namespace["write_json"] = fail_write
    with pytest.raises(OSError, match="run status write failed"):
        execute(block.finalbody, namespace)
    env = namespace["env"]
    assert env.scene.update == handler.previous_update
    assert env.step == handler.previous_step
    assert [event[0] for event in env.events][-4:] == [
        "status-write-error", "helper-close", "env-close", "app-close",
    ]


def device_audit_nodes(block):
    """Select the actual optional gather/init/provenance statements, not copies."""
    guard = [node for node in block.body if isinstance(node, ast.If)
             and any(isinstance(child, ast.ImportFrom)
                     and child.module == "training.tron1_runtime_device" for child in ast.walk(node))]
    initial = [node for node in block.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "runtime_device_audit"
                       for target in node.targets) and isinstance(node.value, ast.Constant)
               and node.value.value is None]
    artifact = [node for node in block.body if isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "runtime_device_audit"]
    assert len(guard) == len(initial) == len(artifact) == 1
    return initial[0], guard[0], artifact[0]


def device_setup(namespace, entrypoint):
    filename = "train_tracking.py" if entrypoint == "train" else "eval_tracking.py"
    block = main_try(filename)
    function = "ManagerBasedRLEnv" if entrypoint == "train" else "TerminalCaptureEnv"
    if entrypoint == "eval":
        terminal = next(node for node in block.body if isinstance(node, ast.ClassDef)
                        and node.name == "TerminalCaptureEnv")
        execute([terminal], namespace)
    env = next(node for node in block.body if is_assignment_call(node, "env", function))
    initial, guard, artifact = device_audit_nodes(block)
    wrapped = next(node for node in block.body if is_assignment_call(node, "wrapped", "RslRlVecEnvWrapper"))
    runner = next(node for node in block.body if is_assignment_call(node, "runner", "OnPolicyRunner"))
    # A guard failure must stop before wrapper reset, actor loading, or steps.
    assert block.body.index(initial) == block.body.index(env) + 1
    assert block.body.index(guard) == block.body.index(initial) + 1
    assert env.lineno < initial.lineno < guard.lineno < wrapped.lineno < runner.lineno
    execute([env, initial, guard, wrapped], namespace)
    return artifact


def install_device_stub(monkeypatch, *, failure=False):
    calls = []
    audit = {"status": "verified", "index_scope": "unmasked_cuda_ordinal",
             "native_and_tensor_device_consistency_verified": True,
             "physical_mapping_verified": False, "exclusivity_verified": False}
    module = types.ModuleType("training.tron1_runtime_device")

    def validate(env, **kwargs):
        calls.append((env, kwargs))
        env.events.append(("device-audit", env._sim_step_counter))
        if failure:
            raise LifecycleError("native device mismatch")
        return audit

    module.validate_simulation_device = validate
    monkeypatch.setitem(sys.modules, "training.tron1_runtime_device", module)
    return calls, audit


@pytest.mark.parametrize("entrypoint", ["train", "eval"])
@pytest.mark.parametrize("option", ["absent", False])
def test_default_entrypoints_do_not_import_or_call_device_helper(stub_namespace, monkeypatch, entrypoint, option):
    namespace, _ = stub_namespace
    calls, _ = install_device_stub(monkeypatch, failure=True)
    if option != "absent":
        namespace["args"].verify_unmasked_device = option
    # device is deliberately absent: disabled/default must not inspect it.
    assert not hasattr(namespace["args"], "device")
    device_setup(namespace, entrypoint)
    assert not calls and namespace["runtime_device_audit"] is None
    assert "validate_simulation_device" not in namespace
    assert not any(event[0] == "device-audit" for event in namespace["env"].events)


@pytest.mark.parametrize("entrypoint", ["train", "eval"])
def test_opt_in_entrypoint_audits_requested_cli_device_once_before_wrapper_reset(stub_namespace, monkeypatch, entrypoint):
    namespace, _ = stub_namespace
    calls, audit = install_device_stub(monkeypatch)
    namespace["args"].verify_unmasked_device = True
    namespace["args"].device = "cuda:3"
    device_setup(namespace, entrypoint)
    env = namespace["env"]
    assert calls == [(env, {"requested_device": "cuda:3"})]
    assert namespace["runtime_device_audit"] is audit
    events = [event[0] for event in env.events]
    assert events.index("device-audit") < events.index("explicit-reset")
    assert env._sim_step_counter == 0 and "process-action" not in events


@pytest.mark.parametrize("entrypoint", ["train", "eval"])
@pytest.mark.parametrize("enabled", [False, True])
def test_device_artifact_and_exact_module_hash_only_added_when_enabled(stub_namespace, monkeypatch, entrypoint, enabled):
    namespace, _ = stub_namespace
    calls, audit = install_device_stub(monkeypatch)
    namespace["args"].verify_unmasked_device = enabled
    namespace["args"].device = "cuda:3"
    artifact = device_setup(namespace, entrypoint)
    namespace["ROOT"] = ROOT
    hashes = []

    def sha256(path):
        hashes.append(Path(path))
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    namespace["sha256"] = sha256
    namespace["manifest"] = {"source_sha256": {"existing": "unchanged"}, "kept": [1, 2]}
    namespace["report"] = {"existing": "unchanged"}
    before_manifest, before_report = copy.deepcopy(namespace["manifest"]), copy.deepcopy(namespace["report"])
    execute([artifact], namespace)
    path = ROOT / "training/tron1_runtime_device.py"
    if enabled:
        assert calls and hashes == [path]
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        if entrypoint == "train":
            assert namespace["manifest"] == {**before_manifest, "runtime_device_audit": audit,
                "source_sha256": {**before_manifest["source_sha256"], "training/tron1_runtime_device.py": expected}}
            assert namespace["report"] == before_report
        else:
            assert namespace["report"] == {**before_report, "runtime_device_audit": audit,
                "runtime_device_module_sha256": expected}
            assert namespace["manifest"] == before_manifest
    else:
        assert not calls and not hashes
        assert namespace["manifest"] == before_manifest and namespace["report"] == before_report


@pytest.mark.parametrize("entrypoint", ["train", "eval"])
def test_device_validation_failure_stops_before_wrapper_reset_or_policy_steps(stub_namespace, monkeypatch, entrypoint):
    namespace, handlers = stub_namespace
    calls, _ = install_device_stub(monkeypatch, failure=True)
    namespace["args"].verify_unmasked_device = True
    namespace["args"].device = "cuda:3"
    with pytest.raises(LifecycleError, match="native device mismatch"):
        device_setup(namespace, entrypoint)
    assert calls == [(namespace["env"], {"requested_device": "cuda:3"})]
    assert "wrapped" not in namespace and "runner" not in namespace
    assert not handlers and namespace["reward_accumulator"] is None
    assert namespace["env"]._sim_step_counter == 0
    assert [event[0] for event in namespace["env"].events] == ["scene", "device-audit"]
