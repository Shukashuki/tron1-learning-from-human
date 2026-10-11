"""Read-only-source lifecycle tests for opt-in substep reward entrypoints.

Execute selected REAL setup/finally AST statements against lightweight stubs.
No Isaac or Torch import, app launch, GPU access, or physics is involved.
Accumulator numerical/native-clock behavior belongs to test_substep_rewards.
"""

from __future__ import annotations

import ast
import copy
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
