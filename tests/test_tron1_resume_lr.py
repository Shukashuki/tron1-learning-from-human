"""Pure CPU tests of the opt-in scalar-only LR restore; no training evidence."""
from __future__ import annotations

import ast
from collections.abc import Mapping
import copy
from fractions import Fraction
import json
import math
from pathlib import Path
import pickle
import random
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.tron1_resume_lr import AUDIT_SCOPE, restore_resume_learning_rate


class MomentObject:
    """Tensor-like opaque state: reading/converting/updating it is forbidden."""

    def __init__(self, values):
        self.values = tuple(values)

    def cpu(self):
        raise AssertionError("Helper must not touch tensor state")

    def numpy(self):
        raise AssertionError("Helper must not touch tensor state")

    def item(self):
        raise AssertionError("Helper must not touch tensor state")


def algorithm(groups=None):
    params = [MomentObject([1., 2.]), MomentObject([3., 4.])]
    states = {index: {"step": MomentObject([80060]), "exp_avg": MomentObject([.1]),
                      "exp_avg_sq": MomentObject([.2])} for index in range(2)}
    optimizer = SimpleNamespace(param_groups=([{"lr": 1e-5, "params": params,
        "betas": (.9, .999), "eps": 1e-8, "weight_decay": 0.}] if groups is None else groups),
        state=states)
    return SimpleNamespace(schedule="adaptive", is_multi_gpu=False, learning_rate=.001,
        desired_kl=.01, optimizer=optimizer,
        policy=SimpleNamespace(actor_normalizer=MomentObject([145072128, .2]),
                               critic_normalizer=MomentObject([145072128, .3])),
        current_learning_iteration=3996, rollout_buffers=MomentObject([24, 2048]),
        config={"learning_rate": .001, "seed": 42},
        rng_marker={"seed": 42, "state": (1, 2, 3)}, external_gpu_context_count=8)


def snapshot(alg):
    """Preserve both graph contents and object identities except the target field."""
    graph = {key: value for key, value in vars(alg).items() if key != "learning_rate"}
    return {"scalar": alg.learning_rate if hasattr(alg, "learning_rate") else "missing",
            "graph": pickle.dumps(graph, protocol=5),
            "identities": {key: id(value) for key, value in graph.items()}}


def assert_failure_unchanged(alg, **kwargs):
    before = snapshot(alg)
    with pytest.raises(ValueError):
        restore_resume_learning_rate(alg, **kwargs)
    assert snapshot(alg) == before


def test_scalar_only_graph_state_and_rng_are_unchanged():
    alg = algorithm()
    before = snapshot(alg)
    rng = random.getstate()
    group = alg.optimizer.param_groups[0]
    group_items = {key: id(value) for key, value in group.items()}
    result = restore_resume_learning_rate(alg, expected_scalar_before=.001, expected_loaded_lr=1e-5)
    assert alg.learning_rate == 1e-5
    after = snapshot(alg)
    assert after["graph"] == before["graph"]
    assert after["identities"] == before["identities"]
    assert {key: id(value) for key, value in group.items()} == group_items
    assert random.getstate() == rng
    assert result == {
        "schema_version": 1, "scope": AUDIT_SCOPE, "status": "restored_from_loaded_optimizer",
        "scalar_before": .001, "loaded_group_learning_rates": [1e-5], "scalar_after": 1e-5,
        "touched_field": "algorithm.learning_rate", "optimizer_group_learning_rates_unchanged": True,
        "adaptive_schedule_preserved": True, "expected_scalar_before": .001,
        "expected_loaded_lr": 1e-5, "multi_gpu": False,
    }
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("group_count", [1, 2, 4, 17])
def test_exact_uniform_multiple_groups(group_count):
    groups = [{"lr": 1e-5, "name": f"group{index}"} for index in range(group_count)]
    alg = algorithm(groups)
    originals = copy.deepcopy(groups)
    result = restore_resume_learning_rate(alg)
    assert groups == originals
    assert result["loaded_group_learning_rates"] == [1e-5] * group_count
    assert result["expected_scalar_before"] is None and result["expected_loaded_lr"] is None


def test_tuple_of_mapping_groups_and_real_fraction_values():
    groups = (MappingProxyType({"lr": Fraction(1, 100000)}), {"lr": Fraction(1, 100000)})
    alg = algorithm(groups)
    alg.learning_rate = Fraction(1, 1000)
    result = restore_resume_learning_rate(alg, expected_scalar_before=Fraction(1, 1000),
                                         expected_loaded_lr=Fraction(1, 100000))
    assert alg.learning_rate is groups[0]["lr"]
    assert groups[0]["lr"] == Fraction(1, 100000)
    assert result["scalar_before"] == .001
    json.dumps(result, allow_nan=False)


def test_mapping_learning_rate_is_fetched_once_then_validated_and_assigned():
    class OnceMapping(Mapping):
        def __init__(self):
            self.reads = 0

        def __iter__(self):
            return iter(("lr",))

        def __len__(self):
            return 1

        def __getitem__(self, key):
            if key != "lr":
                raise KeyError(key)
            self.reads += 1
            if self.reads > 1:
                raise AssertionError("Do not fetch a different value after validation")
            return 1e-5

    group = OnceMapping()
    alg = algorithm([group])
    result = restore_resume_learning_rate(alg)
    assert group.reads == 1 and alg.learning_rate == 1e-5
    assert result["loaded_group_learning_rates"] == [1e-5]


@pytest.mark.parametrize("bad", [True, False, 0, -1, None, ".001", [.001], complex(.001),
                                 float("nan"), float("inf"), float("-inf"), 10 ** 400])
def test_invalid_scalar_never_mutates_graph(bad):
    alg = algorithm()
    alg.learning_rate = bad
    before = snapshot(alg)
    with pytest.raises(ValueError):
        restore_resume_learning_rate(alg)
    after = snapshot(alg)
    # NaN is unequal to itself; compare its serialized scalar representation.
    assert pickle.dumps(after) == pickle.dumps(before)


@pytest.mark.parametrize("bad", [True, False, 0, -1, None, "1e-5", [1e-5], complex(1e-5),
                                 float("nan"), float("inf"), float("-inf"), 10 ** 400])
def test_invalid_group_learning_rate_never_mutates_graph(bad):
    assert_failure_unchanged(algorithm([{"lr": bad}]))


@pytest.mark.parametrize("groups", [[], (), {}, "", b"", bytearray(), [None], [True], [{}],
                                    [["lr", 1e-5]], [("lr", 1e-5)], [SimpleNamespace(lr=1e-5)]])
def test_malformed_groups_never_mutate_graph(groups):
    assert_failure_unchanged(algorithm(groups))


def test_none_param_groups_is_rejected():
    alg = algorithm()
    alg.optimizer.param_groups = None
    assert_failure_unchanged(alg)


def test_group_generator_is_not_consumed():
    values = [{"lr": 1e-5}]
    groups = (value for value in values)
    alg = algorithm()
    alg.optimizer.param_groups = groups
    with pytest.raises(ValueError):
        restore_resume_learning_rate(alg)
    assert alg.learning_rate == .001
    assert next(groups) is values[0]


@pytest.mark.parametrize("other", [1.5e-5, math.nextafter(1e-5, math.inf)])
def test_nonuniform_groups_rejected_without_tolerance(other):
    assert_failure_unchanged(algorithm([{"lr": 1e-5}, {"lr": other}]))


def test_distinct_real_values_that_round_to_same_float_are_not_uniform():
    first = Fraction(1, 8)
    second = first + Fraction(1, 10 ** 40)
    assert first != second and float(first) == float(second)
    assert_failure_unchanged(algorithm([{"lr": first}, {"lr": second}]))


@pytest.mark.parametrize("field", ["expected_scalar_before", "expected_loaded_lr"])
def test_expected_real_value_does_not_use_float_rounding_as_tolerance(field):
    alg = algorithm([{"lr": Fraction(1, 8)}])
    alg.learning_rate = Fraction(1, 8)
    alias = Fraction(1, 8) + Fraction(1, 10 ** 40)
    assert float(alias) == .125 and alias != Fraction(1, 8)
    assert_failure_unchanged(alg, **{field: alias})


@pytest.mark.parametrize("bad", ["fixed", None, True, False, 42, {}])
def test_unsupported_schedule_fails_before_mutation(bad):
    alg = algorithm()
    alg.schedule = bad
    assert_failure_unchanged(alg)


@pytest.mark.parametrize("bad", [True, 1, 0, None, "False", []])
def test_unsupported_or_ambiguous_distributed_mode_fails_before_mutation(bad):
    alg = algorithm()
    alg.is_multi_gpu = bad
    assert_failure_unchanged(alg)


def test_cross_gpu_contexts_are_not_distributed_ppo():
    alg = algorithm()
    assert alg.external_gpu_context_count == 8 and alg.is_multi_gpu is False
    assert restore_resume_learning_rate(alg)["multi_gpu"] is False
    assert alg.external_gpu_context_count == 8


@pytest.mark.parametrize("name", ["schedule", "is_multi_gpu", "learning_rate", "optimizer"])
def test_missing_algorithm_attributes_fail_before_mutation(name):
    alg = algorithm()
    delattr(alg, name)
    assert_failure_unchanged(alg)


def test_missing_optimizer_param_groups_fails_before_mutation():
    alg = algorithm()
    del alg.optimizer.param_groups
    assert_failure_unchanged(alg)


@pytest.mark.parametrize("value", [None, {}, True])
def test_invalid_algorithm_objects_are_rejected(value):
    with pytest.raises(ValueError):
        restore_resume_learning_rate(value)


@pytest.mark.parametrize("field", ["expected_scalar_before", "expected_loaded_lr"])
@pytest.mark.parametrize("bad", [True, False, 0, -1, "1e-5", [], complex(1e-5),
                                 float("nan"), float("inf"), float("-inf"), 10 ** 400])
def test_invalid_expectations_fail_before_mutation(field, bad):
    assert_failure_unchanged(algorithm(), **{field: bad})


@pytest.mark.parametrize("kwargs", [
    {"expected_scalar_before": .002},
    {"expected_scalar_before": math.nextafter(.001, math.inf)},
    {"expected_loaded_lr": 2e-5},
    {"expected_loaded_lr": math.nextafter(1e-5, math.inf)},
])
def test_expected_value_mismatch_is_exact_and_nonmutating(kwargs):
    assert_failure_unchanged(algorithm(), **kwargs)


def test_strict_expected_before_rejects_duplicate_sync_without_new_mutation():
    alg = algorithm()
    restore_resume_learning_rate(alg, expected_scalar_before=.001, expected_loaded_lr=1e-5)
    assert_failure_unchanged(alg, expected_scalar_before=.001, expected_loaded_lr=1e-5)


def test_generic_already_equal_scalar_is_valid_and_does_not_touch_groups():
    alg = algorithm()
    alg.learning_rate = 1e-5
    before = snapshot(alg)
    result = restore_resume_learning_rate(alg)
    assert snapshot(alg) == before
    assert result["scalar_before"] == result["scalar_after"] == 1e-5


def test_only_algorithm_scalar_attribute_assignment_exists_in_helper():
    path = ROOT / "training/tron1_resume_lr.py"
    tree = ast.parse(path.read_text())
    attributes = [node for node in ast.walk(tree) if isinstance(node, ast.Attribute)
                  and isinstance(node.ctx, ast.Store)]
    assert len(attributes) == 1
    assert ast.unparse(attributes[0]) == "algorithm.learning_rate"
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr in ("load", "load_state_dict", "step", "update", "manual_seed", "seed")
                   for node in ast.walk(tree))


def test_import_is_pure_stdlib_and_has_no_app_or_torch_import(tmp_path):
    script = (
        "import json,sys; "
        f"sys.path.insert(0,{str(ROOT)!r}); "
        "import training.tron1_resume_lr; "
        "assert not any(name in sys.modules for name in "
        "('torch','numpy','isaacsim','isaaclab','rsl_rl','omni')); "
        "print(json.dumps({'pure_stdlib':True}))"
    )
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {"pure_stdlib": True}
