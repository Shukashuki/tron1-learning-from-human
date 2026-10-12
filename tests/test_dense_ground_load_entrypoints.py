"""Offline proofs for the derived dense-load entrypoints, not simulator evidence.

These tests parse the actual sources and execute selected real AST statements
against small stubs. No Isaac/Torch application, actor, optimizer, physical
step, contact fit or production output is created. Exact AST normalization
checks the undeclared remainder against the byte-pinned historical sources.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts/train_tracking_dense_ground_load.py"
EVAL = ROOT / "scripts/eval_tracking_dense_ground_load.py"
BASE_TRAIN = ROOT / "scripts/train_tracking_lr_sync.py"
BASE_EVAL = ROOT / "scripts/eval_tracking.py"
PROFILE = ROOT / "config/car_drift_dense_ground_load_task.json"
FROZEN = {
    "scripts/train_tracking.py": "ea9e4cd14f25ff3ba5202b9ba43ac0f08320418e2311cbae205c8449f3a1da9c",
    "scripts/train_tracking_lr_sync.py": "9bd544b82e211633bb37c3fd8a658448d6d0e9b90f380f01a78dd9e8de02b8cb",
    "scripts/eval_tracking.py": "fd90089810d3d14bfd96210723afdd3a36113a19f667ec60b2df617dd032f992",
    "scripts/assess_drift.py": "a2214c47d0f8b262fb8f5b70a709908ef1d7487fd2b1393becb8ad8dfc672fa5",
}
TRAIN_PREFLIGHT = '''
if (args.num_envs, args.iterations, args.save_interval, args.seed, args.smoke_steps) != (2048, 600, 600, 42, 8):
    parser.error("Trial016 requires the predeclared 2048/600/600/42/8 budget")
if (args.task_profile is None or args.terrain_file is not None or args.domain_profile is not None
        or args.domain_randomization != "motor-v1" or not args.verify_unmasked_device):
    parser.error("Trial016 requires explicit profile, motor-v1, native-device audit and no terrain/domain overrides")
sys.path.insert(0, str(ROOT))
from training.tron1_ground_load_substeps_trial import load_dense_load_profile
load_dense_load_profile(args.task_profile)
'''
EVAL_PREFLIGHT = '''
if sha256(ROOT / "scripts/eval_tracking.py") != "fd90089810d3d14bfd96210723afdd3a36113a19f667ec60b2df617dd032f992":
    parser.error("Frozen evaluator baseline source mismatch")
if (args.task_profile is None or args.num_envs != 1 or args.seed != 42
        or args.terrain_file is not None or args.domain_parameters is not None
        or args.deployment_leg_kp is not None or args.deployment_leg_kd is not None
        or args.physics_dt_s is not None or not args.record_physics
        or not args.verify_unmasked_device or args.contact_threshold_n != 5.0):
    parser.error("Trial016 requires one nominal seed42 200Hz episode, original gains and contact gates")
sys.path.insert(0, str(ROOT))
from training.tron1_ground_load_substeps_trial import load_dense_load_profile
load_dense_load_profile(args.task_profile)
'''
ORIGINAL_SAMPLER = '''
if task_profile is not None:
    from training.tron1_drift import install_drift_reward_sampling
    reward_accumulator = install_drift_reward_sampling(env, task_profile)
'''
MATERIAL_AUDIT = '''
contact_material_audit = None
if task_profile is not None:
    from training.tron1_drift import audit_isaac_materials
    contact_material_audit = audit_isaac_materials(env, task_profile)
'''
DENSE_INSTALL = '''
from training.tron1_ground_load_substeps import install_ground_load_accumulator
reward_accumulator = install_ground_load_accumulator(env)
'''
TRAIN_EXTRA_SOURCES = '''
manifest["dense_ground_load_sampling"] = "all_substeps_mean_cost"
manifest["baseline_entry_source_sha256"] = BASE_LR_TRAINING_SOURCE_SHA256
for name in ("training/tron1_ground_load_substeps.py",
             "training/tron1_ground_load_substeps_trial.py",
             "scripts/train_tracking_lr_sync.py"):
    manifest["source_sha256"][name] = sha256(ROOT / name)
'''
EVAL_OLD_AUDIT = '''
report.update({"reward_sampling_audit": reward_accumulator.audit(),
               "substep_reward_module_sha256": sha256(ROOT / "training/tron1_substep_rewards.py")})
'''
EVAL_NEW_AUDIT = '''
report.update({"reward_sampling_audit": reward_accumulator.audit(),
               "substep_reward_module_sha256": sha256(ROOT / "training/tron1_substep_rewards.py"),
               "dense_ground_load_module_sha256": sha256(ROOT / "training/tron1_ground_load_substeps.py"),
               "dense_ground_load_trial_module_sha256": sha256(ROOT / "training/tron1_ground_load_substeps_trial.py"),
               "baseline_evaluator_source_sha256": sha256(ROOT / "scripts/eval_tracking.py"),
               "dense_ground_load_sampling": "all_substeps_mean_cost"})
'''


def tree(path):
    return ast.parse(path.read_text())


def dump(node):
    return ast.dump(node, include_attributes=False)


def main(path):
    return next(n for n in tree(path).body if isinstance(n, ast.FunctionDef) and n.name == "main")


def main_try(path):
    return next(n for n in main(path).body if isinstance(n, ast.Try) and n.finalbody)


def matches(container, snippet):
    expected = ast.parse(snippet).body
    found = []
    for node in ast.walk(container):
        for _, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            for index in range(len(value) - len(expected) + 1):
                if all(isinstance(n, ast.AST) for n in value[index:index + len(expected)]) and all(
                    dump(a) == dump(b) for a, b in zip(value[index:index + len(expected)], expected)
                ):
                    found.append((value, index, len(expected)))
    assert len(found) == 1, (snippet, len(found))
    return found[0]


def replace(container, before, after=""):
    sequence, index, length = matches(container, before)
    sequence[index:index + length] = ast.parse(after).body


def real_nodes(path, snippet):
    sequence, index, length = matches(main(path), snippet)
    return sequence[index:index + length]


def execute(nodes, namespace):
    code = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(code, "<dense-load-entrypoint-real-ast>", "exec"), namespace)


def load_module(path):
    spec = importlib.util.spec_from_file_location("pure_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name,expected", FROZEN.items())
def test_historical_sources_remain_byte_pinned(name, expected):
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize("path", [TRAIN, EVAL])
def test_new_module_import_has_no_simulator_torch_mujoco_side_effect(path, tmp_path):
    code = (
        "import importlib.util,sys; "
        f"s=importlib.util.spec_from_file_location('inert', {str(path)!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "assert 'torch' not in sys.modules; assert 'mujoco' not in sys.modules; "
        "assert not any(n.startswith(('isaaclab','isaacsim')) for n in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("seed,save", [(42, 600), (0, 100), (13, 3)])
def test_ppo_configuration_unchanged(seed, save):
    old, new = load_module(BASE_TRAIN), load_module(TRAIN)
    assert new.ppo_config(seed, save) == old.ppo_config(seed, save)
    if (seed, save) == (42, 600):
        blob = (json.dumps(new.ppo_config(seed, save), indent=2, ensure_ascii=False) + "\n").encode()
        assert hashlib.sha256(blob).hexdigest() == "7c309e49bed37045a29f337e9ed3f418bfaae437483a27d0c947b9685e170738"


def test_training_ast_exact_declared_delta_preserves_ppo_actions_resets_rng_learn_and_cleanup():
    baseline, new = tree(BASE_TRAIN), tree(TRAIN)
    new.body[0] = copy.deepcopy(baseline.body[0])
    replace(new, 'BASE_LR_TRAINING_SOURCE_SHA256 = "9bd544b82e211633bb37c3fd8a658448d6d0e9b90f380f01a78dd9e8de02b8cb"')
    replace(new, '''if sha256(ROOT / "scripts/train_tracking_lr_sync.py") != BASE_LR_TRAINING_SOURCE_SHA256:
    parser.error("Frozen LR-sync baseline SHA mismatch")''')
    replace(new, TRAIN_PREFLIGHT)
    replace(new, '''from training.tron1_ground_load_substeps_trial import load_dense_load_profile, configure_dense_ground_load
if args.task_profile is None:
    raise ValueError("Dense-load entry requires an explicit trial016 task profile")
task_profile = load_dense_load_profile(args.task_profile)''', '''from training.tron1_drift import load_task_profile
task_profile = load_task_profile(args.task_profile) if args.task_profile else None''')
    replace(new, "configure_dense_ground_load(cfg, task_profile)")
    replace(new, MATERIAL_AUDIT + DENSE_INSTALL, ORIGINAL_SAMPLER + MATERIAL_AUDIT)
    replace(new, TRAIN_EXTRA_SOURCES)
    assert dump(new) == dump(baseline)


def test_evaluator_ast_exact_declared_delta_preserves_actor_export_terminal_state_and_recorder():
    baseline, new = tree(BASE_EVAL), tree(EVAL)
    new.body[0] = copy.deepcopy(baseline.body[0])
    replace(new, EVAL_PREFLIGHT)
    replace(new, '''from training.tron1_ground_load_substeps_trial import load_dense_load_profile, configure_dense_ground_load
task_profile = load_dense_load_profile(args.task_profile)''', '''from training.tron1_drift import load_task_profile
task_profile = load_task_profile(args.task_profile) if args.task_profile else None''')
    replace(new, "configure_dense_ground_load(cfg, task_profile)")
    replace(new, DENSE_INSTALL, ORIGINAL_SAMPLER)
    replace(new, EVAL_NEW_AUDIT, EVAL_OLD_AUDIT)
    assert dump(new) == dump(baseline)


def preflight_namespace(path, monkeypatch, profile_failure=False, source_failure=False):
    events = []
    def error(message):
        events.append(("rejected", message))
        raise ValueError(message)
    def validate(profile):
        events.append(("pure_profile", profile))
        if profile_failure:
            raise ValueError("Malformed dense profile")
        return {"validated": True}
    module = types.ModuleType("training.tron1_ground_load_substeps_trial")
    module.load_dense_load_profile = validate
    monkeypatch.setitem(sys.modules, module.__name__, module)
    values = dict(task_profile=PROFILE, terrain_file=None, num_envs=2048 if path == TRAIN else 1,
                  seed=42, device="cuda:6", verify_unmasked_device=True, iterations=600,
                  save_interval=600, smoke_steps=8, domain_profile=None, domain_randomization="motor-v1",
                  domain_parameters=None, deployment_leg_kp=None, deployment_leg_kd=None,
                  physics_dt_s=None, record_physics=True, contact_threshold_n=5.0)
    namespace = {"args": types.SimpleNamespace(**values), "parser": types.SimpleNamespace(error=error),
                 "sys": types.SimpleNamespace(path=types.SimpleNamespace(insert=lambda *a: events.append(("path",)))),
                 "ROOT": ROOT, "sha256": lambda p: "wrong" if source_failure else FROZEN[str(p.relative_to(ROOT))]}
    return namespace, events


@pytest.mark.parametrize("path,snippet", [(TRAIN, TRAIN_PREFLIGHT), (EVAL, EVAL_PREFLIGHT)])
@pytest.mark.parametrize("device", ["cuda:2", "cuda:5", "cuda:6", "cuda:7"])
def test_valid_preflight_allows_explicit_ordinal_not_automatic_fallback(path, snippet, device, monkeypatch):
    namespace, events = preflight_namespace(path, monkeypatch)
    namespace["args"].device = device
    execute(real_nodes(path, snippet), namespace)
    assert events == [("path",), ("pure_profile", PROFILE)]
    assert namespace["args"].device == device


@pytest.mark.parametrize("key,value", [
    ("num_envs", 512), ("num_envs", 2049), ("iterations", 599), ("iterations", 601),
    ("save_interval", 100), ("seed", 43), ("smoke_steps", 0), ("smoke_steps", 9),
    ("task_profile", None), ("terrain_file", Path("terrain.json")),
    ("domain_profile", Path("domain.json")), ("domain_randomization", "none"),
    ("verify_unmasked_device", False),
])
def test_training_budget_and_override_rejection_before_profile_output_or_app(key, value, monkeypatch):
    namespace, events = preflight_namespace(TRAIN, monkeypatch)
    setattr(namespace["args"], key, value)
    with pytest.raises(ValueError, match="Trial016"):
        execute(real_nodes(TRAIN, TRAIN_PREFLIGHT), namespace)
    assert len(events) == 1 and events[0][0] == "rejected"


@pytest.mark.parametrize("key,value", [
    ("num_envs", 2), ("seed", 0), ("task_profile", None),
    ("terrain_file", Path("terrain.json")), ("domain_parameters", Path("draw.json")),
    ("deployment_leg_kp", 500.0), ("deployment_leg_kd", 10.0),
    ("physics_dt_s", .005), ("physics_dt_s", .0025), ("record_physics", False),
    ("verify_unmasked_device", False), ("contact_threshold_n", 0.0),
    ("contact_threshold_n", 4.0), ("contact_threshold_n", 6.0), ("contact_threshold_n", float("nan")),
])
def test_eval_strict_nominal_episode_and_original_contact_gate_preflight(key, value, monkeypatch):
    namespace, events = preflight_namespace(EVAL, monkeypatch)
    setattr(namespace["args"], key, value)
    with pytest.raises(ValueError, match="Trial016"):
        execute(real_nodes(EVAL, EVAL_PREFLIGHT), namespace)
    assert len(events) == 1 and events[0][0] == "rejected"


@pytest.mark.parametrize("path,snippet", [(TRAIN, TRAIN_PREFLIGHT), (EVAL, EVAL_PREFLIGHT)])
def test_profile_failure_prevents_app_output_and_config_creation(path, snippet, monkeypatch):
    namespace, events = preflight_namespace(path, monkeypatch, profile_failure=True)
    with pytest.raises(ValueError, match="Malformed dense profile"):
        execute(real_nodes(path, snippet), namespace)
    assert [event[0] for event in events] == ["path", "pure_profile"]


def test_evaluator_stale_original_hash_rejected_before_profile_or_app(monkeypatch):
    namespace, events = preflight_namespace(EVAL, monkeypatch, source_failure=True)
    with pytest.raises(ValueError, match="baseline source mismatch"):
        execute(real_nodes(EVAL, EVAL_PREFLIGHT), namespace)
    assert len(events) == 1 and events[0][0] == "rejected"


@pytest.mark.parametrize("broken", ["base", "lr"])
def test_training_original_sources_verified_before_app_import(broken):
    nodes = main(TRAIN).body
    index = next(i for i, n in enumerate(nodes) if isinstance(n, ast.ImportFrom) and n.module == "isaaclab.app")
    checks = nodes[index - 2:index]
    assert len(checks) == 2 and all(isinstance(n, ast.If) for n in checks)
    reads = []
    def digest(path):
        relative = str(path.relative_to(ROOT)); reads.append(relative)
        fail = "scripts/train_tracking.py" if broken == "base" else "scripts/train_tracking_lr_sync.py"
        return "wrong" if relative == fail else FROZEN[relative]
    namespace = {"ROOT": ROOT, "sha256": digest,
                 "BASE_TRAINING_SOURCE_SHA256": FROZEN["scripts/train_tracking.py"],
                 "BASE_LR_TRAINING_SOURCE_SHA256": FROZEN["scripts/train_tracking_lr_sync.py"],
                 "parser": types.SimpleNamespace(error=lambda message: (_ for _ in ()).throw(ValueError(message)))}
    with pytest.raises(ValueError, match="SHA mismatch"):
        execute(checks, namespace)
    assert reads == ["scripts/train_tracking.py"] + ([] if broken == "base" else ["scripts/train_tracking_lr_sync.py"])


@pytest.mark.parametrize("argv", [[], ["--motion-file", "unused.npz", "--output-dir", "unused"]])
def test_missing_resume_rejects_without_isaac_import_or_output(argv, tmp_path):
    result = subprocess.run([sys.executable, "-B", str(TRAIN), *argv], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2
    assert "requires --resume before AppLauncher" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr and "Traceback" not in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_preflight_occurs_before_output_mkdir_and_application_in_both_sources():
    for path, snippet in ((TRAIN, TRAIN_PREFLIGHT), (EVAL, EVAL_PREFLIGHT)):
        nodes = real_nodes(path, snippet)
        calls = [n for n in ast.walk(main(path)) if isinstance(n, ast.Call)]
        application = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "AppLauncher")
        mkdir = next(n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "mkdir")
        assert nodes[-1].lineno < mkdir.lineno < application.lineno


@pytest.mark.parametrize("path", [TRAIN, EVAL])
def test_configuration_precedes_constructor_and_native_audit_precedes_cache_install(path):
    block = main_try(path)
    calls = [n for n in ast.walk(block) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    by_name = {n.func.id: n for n in calls}
    constructor = "ManagerBasedRLEnv" if path == TRAIN else "TerminalCaptureEnv"
    assert by_name["make_env_cfg"].lineno < by_name["configure_dense_ground_load"].lineno < by_name[constructor].lineno
    assert by_name["validate_simulation_device"].lineno < by_name["audit_isaac_materials"].lineno < by_name["install_ground_load_accumulator"].lineno
    assert "install_drift_reward_sampling" not in by_name


@pytest.mark.parametrize("path", [TRAIN, EVAL])
@pytest.mark.parametrize("configuration_failure", [False, True])
def test_actual_private_configuration_applied_once_before_environment_construction(path, configuration_failure):
    block = main_try(path)
    start = next(i for i, node in enumerate(block.body) if isinstance(node, ast.Assign)
                 and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                 and node.value.func.id == "make_env_cfg")
    stop = next(i for i, node in enumerate(block.body[start:], start) if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "env" for t in node.targets))
    events = []
    config = types.SimpleNamespace(original_rewards=object(), seed=None)
    profile = {"only_dense_profile": True}
    def factory(*positional, **keywords):
        assert positional == ("motion", "asset", 2048 if path == TRAIN else 1, "cuda:6")
        assert keywords["task_profile"] is profile
        assert keywords["terrain_spec"] is None
        assert keywords["dr_profile"] == ("motor-v1" if path == TRAIN else None)
        assert keywords.get("eval_mode", False) is (path == EVAL)
        events.append("original_factory")
        return config
    def configure(active, checked_profile):
        assert active is config and checked_profile is profile
        events.append("single_private_reward_configuration")
        if configuration_failure:
            raise ValueError("dense reward identity failure")
    def construct(*, cfg):
        assert cfg is config and cfg.seed == 42
        events.append("environment_constructor")
        return object()
    namespace = {"args": types.SimpleNamespace(motion_file="motion", asset_path="asset", seed=42,
                    num_envs=2048 if path == TRAIN else 1, device="cuda:6", physics_dt_s=None),
                 "profile": "motor-v1", "domain_parameters": None, "terrain_spec": None,
                 "task_profile": profile, "make_env_cfg": factory, "configure_dense_ground_load": configure,
                 "ManagerBasedRLEnv": construct, "TerminalCaptureEnv": construct}
    if configuration_failure:
        with pytest.raises(ValueError, match="dense reward identity failure"):
            execute(block.body[start:stop + 1], namespace)
        assert events == ["original_factory", "single_private_reward_configuration"]
        assert "env" not in namespace and config.seed is None
    else:
        execute(block.body[start:stop + 1], namespace)
        assert events == ["original_factory", "single_private_reward_configuration", "environment_constructor"]


def test_real_pure_profile_preflight_accepts_only_mean_instantaneous_cost_delta_and_copies():
    helper = load_module(ROOT / "training/tron1_ground_load_substeps_trial.py")
    before = PROFILE.read_bytes()
    checked = helper.load_dense_load_profile(PROFILE)
    assert checked == json.loads(before)
    assert checked["wheel_ground_load_reserve_sampling"] == "all_substeps_mean_cost"
    assert checked["wheel_ground_load_reserve"] == {"weight": -1.0, "scale_fraction": .2}
    assert "wheel_vertical_velocity_sampling" not in checked
    checked["reward_weights"]["sideslip"] = 999
    assert helper.load_dense_load_profile(PROFILE)["reward_weights"]["sideslip"] == 2.0
    assert PROFILE.read_bytes() == before


@pytest.mark.parametrize("key,value", [
    ("wheel_ground_load_reserve_sampling", "last_substep"),
    ("wheel_ground_load_reserve_sampling", "all_substeps_mean_force"),
    ("wheel_ground_load_reserve_sampling", True),
    ("wheel_ground_load_reserve", {"weight": -.5, "scale_fraction": .2}),
    ("wheel_ground_load_reserve", {"weight": -1.0, "scale_fraction": .3}),
    ("wheel_ground_load_reserve", {"weight": True, "scale_fraction": .2}),
    ("wheel_ground_load_reserve", {"weight": -1.0, "scale_fraction": .2, "extra": 0}),
    ("wheel_vertical_velocity_sampling", "all_substeps_mean"),
    ("wheel_vertical_velocity_weight", -4.0), ("wheel_vertical_velocity_std_m_s", .03),
    ("leg_kd", 12.0), ("leg_kp", 400.0), ("contact_friction", .2),
    ("mujoco_contact_solref", [.02, 2.0]), ("action_rate_weight", -.4),
    ("reward_weights", {"sideslip": 4.0, "yaw_rate": 1.0, "speed": 3.0, "grounded": .4, "tilt": -.2}),
    ("description", ""), ("description", False), ("future_silent_field", 0),
])
def test_real_profile_loader_rejects_silent_second_factor_before_application(key, value, tmp_path):
    helper = load_module(ROOT / "training/tron1_ground_load_substeps_trial.py")
    profile = json.loads(PROFILE.read_text())
    profile[key] = value
    target = tmp_path / "malformed-profile.json"
    target.write_text(json.dumps(profile))
    with pytest.raises(ValueError):
        helper.load_dense_load_profile(target)


@pytest.mark.parametrize("blob", [
    '{"schema_version": 1, "schema_version": 1}',
    '{"contact_friction": NaN}', '{"contact_friction": Infinity}',
])
def test_real_profile_loader_rejects_duplicate_keys_and_nonfinite_json(blob, tmp_path):
    helper = load_module(ROOT / "training/tron1_ground_load_substeps_trial.py")
    target = tmp_path / "invalid-json-profile.json"
    target.write_text(blob)
    with pytest.raises(ValueError):
        helper.load_dense_load_profile(target)


@pytest.mark.parametrize("path", [TRAIN, EVAL])
@pytest.mark.parametrize("fail_cache", [False, True])
def test_actual_setup_material_cache_initialized_before_accumulator(monkeypatch, path, fail_cache):
    events = []
    env = types.SimpleNamespace(cache=None)
    handler = object()
    def audit(active, profile):
        assert active is env and profile == {"profile": True}
        events.append("material_and_cache")
        if fail_cache:
            raise ValueError("cache validation failed")
        env.cache = {"mg_from_live_mass_gravity": True}
        return {"ground_load_reserve": env.cache}
    def install(active):
        assert active is env and active.cache is not None
        events.append("dense_install")
        return handler
    drift = types.ModuleType("training.tron1_drift"); drift.audit_isaac_materials = audit
    dense = types.ModuleType("training.tron1_ground_load_substeps"); dense.install_ground_load_accumulator = install
    monkeypatch.setitem(sys.modules, drift.__name__, drift)
    monkeypatch.setitem(sys.modules, dense.__name__, dense)
    namespace = {"env": env, "task_profile": {"profile": True}}
    nodes = real_nodes(path, MATERIAL_AUDIT) + real_nodes(path, DENSE_INSTALL)
    if fail_cache:
        with pytest.raises(ValueError, match="cache validation failed"):
            execute(nodes, namespace)
        assert events == ["material_and_cache"]
        assert "reward_accumulator" not in namespace
    else:
        execute(nodes, namespace)
        assert events == ["material_and_cache", "dense_install"]
        assert namespace["reward_accumulator"] is handler


def test_eval_actual_recorder_is_installed_before_dense_wrapper(monkeypatch):
    events = []
    def original_update(*args, **kwargs):
        events.append("original_scene_update")
        return "original-result"
    env = types.SimpleNamespace(scene=types.SimpleNamespace(update=original_update), _sim_step_counter=0,
                                num_envs=1, physics_dt=.005)
    recorder = types.SimpleNamespace(should_capture=lambda count: count > 0,
                                    capture=lambda count, state: events.append(("capture", count, state)))
    def install(active):
        observer = active.scene.update
        assert observer is namespace["physics_scene_update"]
        events.append("dense_install_after_recorder")
        return object()
    dense = types.ModuleType("training.tron1_ground_load_substeps"); dense.install_ground_load_accumulator = install
    monkeypatch.setitem(sys.modules, dense.__name__, dense)
    namespace = {"env": env, "args": types.SimpleNamespace(record_physics=True, num_envs=1), "initial": {}, "obs": {"policy": "obs"},
                 "to_numpy": lambda value: value, "PhysicsTrajectoryRecorder": lambda *args: recorder,
                 "state_snapshot": lambda active: "pre-reset-state"}
    block = main_try(EVAL)
    recorder_if = next(n for n in block.body if isinstance(n, ast.If) and dump(n.test) == dump(ast.parse("args.record_physics", mode="eval").body)
                       and any(isinstance(m, ast.FunctionDef) and m.name == "physics_scene_update" for m in n.body))
    execute([recorder_if] + real_nodes(EVAL, DENSE_INSTALL), namespace)
    assert events == ["dense_install_after_recorder"]
    # A reset/forward with no counter change chains but must not fetch/capture.
    assert env.scene.update() == "original-result"
    assert events == ["dense_install_after_recorder", "original_scene_update"]
    env._sim_step_counter = 1
    assert env.scene.update() == "original-result"
    assert events[-2:] == ["original_scene_update", ("capture", 1, "pre-reset-state")]


@pytest.mark.parametrize("path", [TRAIN, EVAL])
def test_new_provenance_uses_actual_dense_sources_not_fake_baseline_identity(path):
    namespace = {"ROOT": ROOT, "sha256": lambda value: "ACTUAL:" + str(value.relative_to(ROOT)),
                 "manifest": {"source_sha256": {"original": "kept"}}, "report": {"original": "kept"},
                 "reward_accumulator": types.SimpleNamespace(audit=lambda: {"actual_sampler": True}),
                 "BASE_LR_TRAINING_SOURCE_SHA256": FROZEN["scripts/train_tracking_lr_sync.py"]}
    if path == TRAIN:
        execute(real_nodes(path, TRAIN_EXTRA_SOURCES), namespace)
        result = namespace["manifest"]
        assert result["source_sha256"] == {"original": "kept", **{
            name: "ACTUAL:" + name for name in ("training/tron1_ground_load_substeps.py",
                "training/tron1_ground_load_substeps_trial.py", "scripts/train_tracking_lr_sync.py")}}
        assert result["baseline_entry_source_sha256"] == FROZEN["scripts/train_tracking_lr_sync.py"]
    else:
        execute(real_nodes(path, EVAL_NEW_AUDIT), namespace)
        result = namespace["report"]
        assert result["reward_sampling_audit"] == {"actual_sampler": True}
        for key, name in (("substep_reward_module_sha256", "training/tron1_substep_rewards.py"),
                          ("dense_ground_load_module_sha256", "training/tron1_ground_load_substeps.py"),
                          ("dense_ground_load_trial_module_sha256", "training/tron1_ground_load_substeps_trial.py"),
                          ("baseline_evaluator_source_sha256", "scripts/eval_tracking.py")):
            assert result[key] == "ACTUAL:" + name
    assert result["dense_ground_load_sampling"] == "all_substeps_mean_cost"


@pytest.mark.parametrize("path,failure", [(path, failure) for path in (TRAIN, EVAL)
                         for failure in (None, "sampler", "environment", "status_write")
                         if path == TRAIN or failure != "status_write"])
def test_actual_finally_cleanup_lifo_even_when_stage_raises(path, failure):
    events = []
    def close(name):
        def run(*args, **kwargs):
            events.append(name)
            if failure == name:
                raise OSError(name + " failed")
        return run
    def original(*args, **kwargs):
        pass
    def recorder(*args, **kwargs):
        pass
    env = types.SimpleNamespace(scene=types.SimpleNamespace(update=recorder), close=close("environment"))
    def sampler_close():
        events.append("sampler")
        env.scene.update = recorder
        if failure == "sampler":
            raise OSError("sampler failed")
    def write(*args):
        events.append("status_write")
        if failure == "status_write":
            raise OSError("status_write failed")
    namespace = {"reward_accumulator": types.SimpleNamespace(close=sampler_close), "env": env,
                 "wrapped": types.SimpleNamespace(close=close("environment")), "app": types.SimpleNamespace(close=close("app")),
                 "args": types.SimpleNamespace(headless=True), "original_scene_update": original,
                 "physics_scene_update": recorder, "output": Path("/not-written"), "report": {}, "write_json": write}
    nodes = main_try(path).finalbody
    if failure is None:
        execute(nodes, namespace)
    else:
        with pytest.raises(OSError, match=failure + " failed"):
            execute(nodes, namespace)
    expected = (["status_write"] if path == TRAIN else []) + ["sampler", "environment", "app"]
    assert events == expected
    if path == EVAL:
        assert env.scene.update is original


def test_training_final_audit_no_extra_steps_actor_calls_or_cost_reads():
    snippet = '''if reward_accumulator is not None:
    write_json(output / "reward_sampling_final_audit.json", reward_accumulator.audit())'''
    events = []
    audit = {"source_backed_only": True}
    namespace = {"reward_accumulator": types.SimpleNamespace(audit=lambda: audit), "output": Path("/not-written"),
                 "write_json": lambda path, value: events.append((path.name, value))}
    execute(real_nodes(TRAIN, snippet), namespace)
    assert events == [("reward_sampling_final_audit.json", audit)]

