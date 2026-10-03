"""Offline regression for PhysX legacy-friction initialization; no Isaac/Torch.

Execute the production function extracted with AST against a tiny NumPy tensor
protocol. The stub intentionally exposes no robot state-writing API.
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


TASK = Path(__file__).resolve().parents[1] / "training" / "tron1_tracking.py"


class Tensor:
    """Only the tensor operations used by the production startup function."""

    def __init__(self, values, device="cpu"):
        self.values = np.asarray(values)
        self.device = device

    @property
    def dtype(self):
        return self.values.dtype

    def clone(self):
        return Tensor(self.values.copy(), self.device)

    def cpu(self):
        return self if self.device == "cpu" else Tensor(self.values.copy(), "cpu")

    def __getitem__(self, key):
        return Tensor(self.values[key], self.device)

    def tolist(self):
        return self.values.tolist()

    def all(self):
        return bool(self.values.all())

    def item(self):
        return self.values.item()


class TorchStub:
    int32 = np.int32

    @staticmethod
    def arange(stop, *, dtype, device):
        return Tensor(np.arange(stop, dtype=dtype), device)

    @staticmethod
    def zeros_like(source):
        return Tensor(np.zeros_like(source.values), source.device)

    @staticmethod
    def isfinite(source):
        return Tensor(np.isfinite(source.values), source.device)

    @staticmethod
    def count_nonzero(source):
        return Tensor(np.asarray(np.count_nonzero(source.values)), source.device)


class GuardedPhysxView:
    """Returning a live view catches missing clone() in the before audit."""
    __slots__ = ("coefficients", "setter_behavior", "calls")

    def __init__(self, values, setter_behavior="apply"):
        self.coefficients = Tensor(np.array(values, dtype=np.float32), "cuda:0")
        self.setter_behavior = setter_behavior
        self.calls = []

    def get_dof_friction_coefficients(self):
        self.calls.append(("get_friction",))
        return self.coefficients

    def set_dof_friction_coefficients(self, values, indices):
        # PhysX expects CPU int32 environment indices even with GPU simulation.
        assert indices.device == "cpu"
        assert indices.dtype == np.dtype(np.int32)
        assert values.device == "cpu"
        self.calls.append(("set_friction", values.clone(), indices.clone()))
        if self.setter_behavior == "ignore":
            return
        self.coefficients.values[indices.values] = values.values
        if self.setter_behavior == "nan":
            self.coefficients.values[-1, -1] = np.nan
        elif self.setter_behavior == "partial":
            self.coefficients.values[-1, -1] = .01

    def __getattr__(self, name):
        raise AssertionError(f"Startup friction initialization touched forbidden PhysX API: {name}")


class GuardedRobot:
    __slots__ = ("root_physx_view", "joint_names", "_root", "_q", "_dq")

    def __init__(self, view, joint_names):
        self.root_physx_view, self.joint_names = view, list(joint_names)
        self._root = np.arange(13, dtype=float)
        self._q = np.arange(len(joint_names), dtype=float)
        self._dq = -self._q.copy()
        for value in (self._root, self._q, self._dq):
            value.flags.writeable = False

    @property
    def root_state(self):
        return self._root

    @property
    def joint_position(self):
        return self._q

    @property
    def joint_velocity(self):
        return self._dq

    def __getattr__(self, name):
        raise AssertionError(f"Startup friction initialization touched forbidden robot API: {name}")


@pytest.fixture
def clear_friction():
    tree = ast.parse(TASK.read_text())
    functions = [copy.deepcopy(node) for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "clear_legacy_joint_friction"]
    assert len(functions) == 1, "The regression must execute exactly one production startup function"
    namespace = {"torch": TorchStub}
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, str(TASK), "exec"), namespace)
    return namespace["clear_legacy_joint_friction"]


def environment(values=None, setter_behavior="apply"):
    if values is None:
        values = np.arange(32, dtype=np.float32).reshape(4, 8) / 1000
        values[:, 6:] += .01
    view = GuardedPhysxView(values, setter_behavior)
    # Deliberately nonalphabetic names: the audit must preserve the native order.
    names = ["wheel_R_Joint", "hip_L_Joint", "knee_R_Joint", "abad_L_Joint",
             "wheel_L_Joint", "hip_R_Joint", "knee_L_Joint", "abad_R_Joint"]
    robot = GuardedRobot(view, names)
    env = SimpleNamespace(scene={"robot": robot}, num_envs=len(values))
    return env, view, robot


def test_all_environment_coefficients_zero_and_original_first_row_preserved(clear_friction):
    env, view, robot = environment()
    initial = view.coefficients.values.copy()
    clear_friction(env)
    np.testing.assert_array_equal(view.coefficients.values, np.zeros_like(initial))
    audit = env.legacy_joint_friction_audit
    np.testing.assert_array_equal(audit["before_first_environment"], initial[0])
    np.testing.assert_array_equal(audit["after_first_environment"], np.zeros(8))
    assert audit["all_environments_verified_zero"] is True
    assert audit["joint_names"] == robot.joint_names
    assert audit["joint_names"] is not robot.joint_names
    json.dumps(audit, allow_nan=False)
    assert [call[0] for call in view.calls] == ["get_friction", "set_friction", "get_friction"]


@pytest.mark.parametrize("env_ids", [None, [1], [], np.array([2, 3], dtype=np.int64)])
def test_cpu_int32_indices_always_address_every_environment(clear_friction, env_ids):
    env, view, _ = environment()
    clear_friction(env, env_ids)
    setters = [call for call in view.calls if call[0] == "set_friction"]
    assert len(setters) == 1
    _, values, indices = setters[0]
    np.testing.assert_array_equal(indices.values, np.arange(env.num_envs, dtype=np.int32))
    assert indices.device == "cpu" and indices.dtype == np.dtype(np.int32)
    assert values.values.shape == (env.num_envs, 8)
    assert values.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(view.coefficients.values, 0.)


@pytest.mark.parametrize("behavior", ["ignore", "nan", "partial"])
def test_failed_or_nonfinite_readback_cannot_publish_success_audit(clear_friction, behavior):
    env, view, _ = environment(setter_behavior=behavior)
    with pytest.raises(RuntimeError, match="Legacy PhysX joint friction was not cleared"):
        clear_friction(env)
    assert not hasattr(env, "legacy_joint_friction_audit")
    assert [call[0] for call in view.calls] == ["get_friction", "set_friction", "get_friction"]
    # Both 'nan' and 'partial' alter only the last environment: checking only
    # the first environment would incorrectly pass these two regressions.
    if behavior != "ignore":
        np.testing.assert_array_equal(view.coefficients.values[0], 0.)


def test_already_zero_coefficients_are_safe_and_audited(clear_friction):
    env, view, _ = environment(np.zeros((1, 8), dtype=np.float32))
    clear_friction(env)
    assert env.legacy_joint_friction_audit["before_first_environment"] == [0.] * 8
    assert env.legacy_joint_friction_audit["after_first_environment"] == [0.] * 8
    np.testing.assert_array_equal(view.coefficients.values, 0.)


def test_friction_initialization_never_writes_root_or_joint_state(clear_friction):
    env, view, robot = environment()
    before = [robot.root_state.copy(), robot.joint_position.copy(), robot.joint_velocity.copy()]
    clear_friction(env)
    for old, current in zip(before, (robot.root_state, robot.joint_position, robot.joint_velocity)):
        np.testing.assert_array_equal(old, current)
        assert not current.flags.writeable
    assert set(vars(env)) == {"scene", "num_envs", "legacy_joint_friction_audit"}
    assert {call[0] for call in view.calls} == {"get_friction", "set_friction"}


def test_legacy_friction_event_is_registered_once_at_startup_not_episode_reset():
    tree = ast.parse(TASK.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TronEventsCfg")
    events = [node.value for node in cls.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "legacy_joint_friction"
                      for target in node.targets)]
    assert len(events) == 1
    call = events[0]
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "EventTermCfg"
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    assert isinstance(keywords["func"], ast.Name) and keywords["func"].id == "clear_legacy_joint_friction"
    assert ast.literal_eval(keywords["mode"]) == "startup"
