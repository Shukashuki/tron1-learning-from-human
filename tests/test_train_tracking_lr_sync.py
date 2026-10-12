"""Pure static/AST lifecycle proof for the new explicit LR-sync entrypoint.

No Isaac, Torch, GPU, environment, optimizer update or app is used. Selected
REAL entrypoint nodes execute against small stubs; the frozen base is unchanged.
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
BASE = ROOT / "scripts/train_tracking.py"
NEW = ROOT / "scripts/train_tracking_lr_sync.py"
BASE_SHA = "ea9e4cd14f25ff3ba5202b9ba43ac0f08320418e2311cbae205c8449f3a1da9c"
RUNNER_SHA = "7c309e49bed37045a29f337e9ed3f418bfaae437483a27d0c947b9685e170738"
EARLY = '''resume_parser = argparse.ArgumentParser(add_help=False)
resume_parser.add_argument("--resume", type=Path)
resume_preflight, _ = resume_parser.parse_known_args()
if resume_preflight.resume is None:
    parser.error("This explicit LR-sync entrypoint requires --resume before AppLauncher")
if sha256(ROOT / "scripts/train_tracking.py") != BASE_TRAINING_SOURCE_SHA256:
    parser.error("Frozen base training entrypoint SHA mismatch")
'''
AFTER_LOAD = '''resume_learning_rate_audit = restore_resume_learning_rate(
    runner.alg, expected_scalar_before=0.001, expected_loaded_lr=1e-5,
)
manifest["source_sha256"]["scripts/train_tracking.py"] = sha256(ROOT / "scripts/train_tracking.py")
manifest["source_sha256"]["training/tron1_resume_lr.py"] = sha256(ROOT / "training/tron1_resume_lr.py")
manifest["resume_source_audit"] = resume_source_audit
manifest["resume_learning_rate_audit"] = resume_learning_rate_audit
write_json(output / "manifest.json", manifest)
write_json(output / "resume_learning_rate_audit.json", resume_learning_rate_audit)
print("TRON1_RESUME_LEARNING_RATE_AUDIT", json.dumps(resume_learning_rate_audit), flush=True)
'''


def tree(path=NEW):
    return ast.parse(path.read_text())


def dump(node):
    return ast.dump(node, include_attributes=False)


def load_module(path):
    spec = importlib.util.spec_from_file_location("pure_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main_parts():
    main = next(node for node in tree().body if isinstance(node, ast.FunctionDef) and node.name == "main")
    block = next(node for node in main.body if isinstance(node, ast.Try) and node.finalbody)
    return main, block


def execute(nodes, namespace):
    code = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(code, "<real-lr-sync-entrypoint-ast>", "exec"), namespace)


def source_nodes(block):
    imported = next(node for node in block.body if isinstance(node, ast.ImportFrom) and node.module == "training.tron1_resume_lr")
    guard = next(node for node in block.body if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "audit_installed_resume_sources")
    return imported, guard


def post_load_nodes(block):
    start = next(i for i,node in enumerate(block.body) if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "restore_resume_learning_rate")
    result = block.body[start:start+len(ast.parse(AFTER_LOAD).body)]
    assert dump(ast.Module(body=result,type_ignores=[])) == dump(ast.parse(AFTER_LOAD))
    return result


def test_frozen_base_hash_and_inert_new_import():
    assert hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA
    module = load_module(NEW)
    assert module.BASE_TRAINING_SOURCE_SHA256 == BASE_SHA
    assert module.RSL_RESUME_SOURCE_SHA256 == {
        "runners/on_policy_runner.py": "e51e4dd49664ccc8743a16b71d2971d392f40a6988136f10fe8c5f03ab6a83b5",
        "algorithms/ppo.py": "deafc8c947eba4df3e91b393869426cdab8d7b71e05974c3734125d2331d7d1c",
        "networks/normalization.py": "7db82f8021d62a660b4229782b52ac47e32d781206dba709d6bc436bf13f6672",
    }
    assert "torch" not in sys.modules and not any(name.startswith("isaaclab") for name in sys.modules)


@pytest.mark.parametrize("seed,save", [(42,600),(0,100),(13,3)])
def test_ppo_configuration_exactly_original(seed,save):
    base,new=load_module(BASE),load_module(NEW)
    assert base.ppo_config(seed,save) == new.ppo_config(seed,save)
    if (seed,save)==(42,600):
        blob=(json.dumps(new.ppo_config(seed,save),indent=2,ensure_ascii=False)+"\n").encode()
        assert hashlib.sha256(blob).hexdigest() == RUNNER_SHA


def test_ast_exact_allowlist_no_environment_rng_smoke_reset_norm_cleanup_changes():
    base,modified=tree(BASE),copy.deepcopy(tree())
    # Exact allowlist: documentation/constants, pure installed-source guard,
    # required resume preflight, one early source check, one post-load audit span.
    assert isinstance(base.body[0],ast.Expr) and isinstance(modified.body[0],ast.Expr)
    modified.body[0]=copy.deepcopy(base.body[0])
    modified.body=[node for node in modified.body if not
        (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in
            {"BASE_TRAINING_SOURCE_SHA256","RSL_RESUME_SOURCE_SHA256"} for t in node.targets))
        and not (isinstance(node,ast.FunctionDef) and node.name=="audit_installed_resume_sources")]
    main=next(n for n in modified.body if isinstance(n,ast.FunctionDef) and n.name=="main")
    resume_call=next(n for n in ast.walk(main) if isinstance(n,ast.Call) and n.args
        and isinstance(n.args[0],ast.Constant) and n.args[0].value=="--resume"
        and any(k.arg=="required" for k in n.keywords))
    assert any(k.arg=="required" and isinstance(k.value,ast.Constant) and k.value.value is True for k in resume_call.keywords)
    resume_call.keywords=[k for k in resume_call.keywords if k.arg!="required"]
    launch_import=next(i for i,n in enumerate(main.body) if isinstance(n,ast.ImportFrom) and n.module=="isaaclab.app")
    early=main.body[launch_import-5:launch_import]
    assert dump(ast.Module(body=early,type_ignores=[]))==dump(ast.parse(EARLY))
    del main.body[launch_import-5:launch_import]
    block=next(n for n in main.body if isinstance(n,ast.Try) and n.finalbody)
    imported,guard=source_nodes(block)
    assert imported.lineno<guard.lineno
    block.body.remove(imported);block.body.remove(guard)
    audit_span=post_load_nodes(block)
    start=block.body.index(audit_span[0]);del block.body[start:start+len(audit_span)]
    assert dump(modified)==dump(base)


@pytest.mark.parametrize("argv", [[],["--motion-file","unused.npz","--output-dir","unused"],
    ["--motion-file","unused.npz","--output-dir","unused","--headless","--device","cuda:5"]])
def test_missing_resume_fails_before_isaac_import_or_output_creation(argv,tmp_path):
    call=subprocess.run([sys.executable,str(NEW),*argv],cwd=tmp_path,capture_output=True,text=True)
    assert call.returncode==2
    assert "requires --resume before AppLauncher" in call.stderr
    assert "ModuleNotFoundError" not in call.stderr and "Traceback" not in call.stderr
    assert not (tmp_path/"unused").exists()


def test_preflight_base_hash_failure_is_before_app_import(monkeypatch):
    main,_=main_parts();index=next(i for i,n in enumerate(main.body) if isinstance(n,ast.ImportFrom) and n.module=="isaaclab.app")
    events=[]
    class Parser:
        def error(self,message):
            events.append(message);raise ValueError(message)
    namespace={"parser":Parser(),"argparse":types.SimpleNamespace(ArgumentParser=lambda **kw:types.SimpleNamespace(
        add_argument=lambda *a,**kw:None,parse_known_args=lambda:(types.SimpleNamespace(resume=Path("some.pt")),[]))),
        "Path":Path,"ROOT":ROOT,"sha256":lambda path:"wrong","BASE_TRAINING_SOURCE_SHA256":BASE_SHA}
    with pytest.raises(ValueError,match="base training entrypoint SHA mismatch"):
        execute(main.body[index-5:index],namespace)
    assert events and "isaaclab.app" not in sys.modules


@pytest.mark.parametrize("bad", [None,"version","runner","ppo","normalizer","no_package_location"])
def test_installed_source_guard_exact_pins_readonly_before_load(monkeypatch,bad):
    module=load_module(NEW);read=[]
    fake=types.ModuleType("rsl_rl");fake.__file__="/declared/rsl_rl/__init__.py"
    if bad=="no_package_location":fake.__file__=None
    monkeypatch.setitem(sys.modules,"rsl_rl",fake)
    monkeypatch.setattr(module.importlib.metadata,"version",lambda name:"3.0.0" if bad=="version" else "3.0.1")
    changed={"runner":"runners/on_policy_runner.py","ppo":"algorithms/ppo.py","normalizer":"networks/normalization.py"}.get(bad)
    def digest(path):
        name=str(path.relative_to(Path("/declared/rsl_rl")));read.append(name)
        return "wrong" if name==changed else module.RSL_RESUME_SOURCE_SHA256[name]
    monkeypatch.setattr(module,"sha256",digest)
    if bad is None:
        result=module.audit_installed_resume_sources()
        assert result=={"schema_version":1,"rsl_rl_version":"3.0.1","source_root":"/declared/rsl_rl",
            "expected_source_sha256":module.RSL_RESUME_SOURCE_SHA256,"actual_source_sha256":module.RSL_RESUME_SOURCE_SHA256,
            "verified_before_runner_load":True}
        assert len(read)==3
    else:
        with pytest.raises(RuntimeError):module.audit_installed_resume_sources()


def lifecycle_namespace(monkeypatch,source_failure=False,load_failure=False,sync_failure=False,write_failure=False):
    from training.tron1_resume_lr import restore_resume_learning_rate
    events=[];writes={}
    alg=types.SimpleNamespace(schedule="adaptive",is_multi_gpu=False,learning_rate=.001,
        optimizer=types.SimpleNamespace(param_groups=[{"lr":.001,"moments":[1,2]}]),
        normalizer={"count":145072128},iteration=3996)
    def loaded(path,**kwargs):
        events.append(("load",path,kwargs))
        if load_failure:raise RuntimeError("load failed")
        alg.optimizer.param_groups[0]["lr"]=1e-5
    runner=types.SimpleNamespace(alg=alg,load=loaded)
    fake=types.ModuleType("training.tron1_resume_lr")
    def sync(value,**kwargs):
        events.append(("sync",value,kwargs))
        if sync_failure:raise ValueError("sync failed")
        return restore_resume_learning_rate(value,**kwargs)
    fake.restore_resume_learning_rate=sync;monkeypatch.setitem(sys.modules,"training.tron1_resume_lr",fake)
    def sources():
        events.append(("source",))
        if source_failure:raise RuntimeError("source failed")
        return {"rsl_rl_version":"3.0.1","verified_before_runner_load":True}
    def write(path,value):
        events.append(("write",Path(path).name))
        if write_failure:raise OSError("audit write failed")
        writes[Path(path).name]=json.loads(json.dumps(value))
    namespace={"runner":runner,"args":types.SimpleNamespace(resume=Path("parent.pt"),device="cuda:5",num_envs=2048,iterations=600),
        "audit_installed_resume_sources":sources,"manifest":{"source_sha256":{"scripts/train_tracking_lr_sync.py":"actual-new","old_module":"kept"},"contract":{"untouched":True}},
        "sha256":lambda path:"SHA:"+str(path.relative_to(ROOT)),"ROOT":ROOT,"output":Path("/not-written"),
        "write_json":write,"print":lambda *args,**kw:events.append(("print",args[0])),"json":json,"report":{}}
    return namespace,events,writes


def run_load_sync_artifacts(namespace):
    _,block=main_parts();imported,sources=source_nodes(block)
    load_node=next(node for node in block.body if isinstance(node,ast.If) and isinstance(node.test,ast.Attribute) and node.test.attr=="resume")
    span=post_load_nodes(block)
    training=next(node for node in block.body if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
        and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name)
        and node.value.func.value.id=="report" and any(k.arg=="status" and isinstance(k.value,ast.Constant) and k.value.value=="training" for k in node.value.keywords))
    marker=next(node for node in block.body if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and node.value.args
        and isinstance(node.value.args[0],ast.Constant) and node.value.args[0].value=="TRON1_TRAINING_STARTED")
    learn=next(node for node in block.body if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
        and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=="learn")
    assert sources.lineno<load_node.lineno<span[0].lineno<training.lineno<marker.lineno<learn.lineno
    execute([imported,sources,load_node,*span,training,marker],namespace)


def test_real_postload_scalar_only_and_json_artifacts_before_training_marker(monkeypatch):
    namespace,events,writes=lifecycle_namespace(monkeypatch)
    before=copy.deepcopy(namespace["runner"].alg)
    run_load_sync_artifacts(namespace)
    alg=namespace["runner"].alg
    assert alg.learning_rate==1e-5 and alg.normalizer==before.normalizer and alg.iteration==before.iteration
    assert alg.optimizer.param_groups==[{"lr":1e-5,"moments":[1,2]}]
    assert writes["manifest.json"]["resume_learning_rate_audit"]==writes["resume_learning_rate_audit.json"]
    assert writes["manifest.json"]["resume_source_audit"]==namespace["resume_source_audit"]
    assert writes["manifest.json"]["contract"]=={"untouched":True}
    assert writes["manifest.json"]["source_sha256"]=={"scripts/train_tracking_lr_sync.py":"actual-new","old_module":"kept",
        "scripts/train_tracking.py":"SHA:scripts/train_tracking.py","training/tron1_resume_lr.py":"SHA:training/tron1_resume_lr.py"}
    names=[event[0] if event[0]!="print" else event[1] for event in events]
    assert names==["source","load","sync","write","write","TRON1_RESUME_LEARNING_RATE_AUDIT","TRON1_TRAINING_STARTED"]


@pytest.mark.parametrize("failure",["source","load","sync","write"])
def test_any_source_load_restore_persist_failure_stops_before_learn_started(monkeypatch,failure):
    kwargs={"source_failure":False,"load_failure":False,"sync_failure":False,"write_failure":False};kwargs[failure+"_failure"]=True
    namespace,events,_=lifecycle_namespace(monkeypatch,**kwargs)
    with pytest.raises((RuntimeError,ValueError,OSError)):run_load_sync_artifacts(namespace)
    assert not any(event[0]=="print" and event[1]=="TRON1_TRAINING_STARTED" for event in events)
    if failure=="source":assert not any(event[0] in {"load","sync"} for event in events)
    if failure in {"source","load","sync"}:assert namespace["runner"].alg.learning_rate==.001


def test_actual_manifest_source_comprehension_names_new_file_not_base(monkeypatch):
    _,block=main_parts()
    manifest=next(node for node in block.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="manifest" for t in node.targets))
    source=next(value for key,value in zip(manifest.value.keys,manifest.value.values) if isinstance(key,ast.Constant) and key.value=="source_sha256")
    expression=ast.fix_missing_locations(ast.Expression(body=source))
    result=eval(compile(expression,"<actual-new-source-map>","eval"),{"ROOT":ROOT,"Path":Path,"__file__":str(NEW),"sha256":lambda p:hashlib.sha256(p.read_bytes()).hexdigest()})
    assert "scripts/train_tracking_lr_sync.py" in result and "scripts/train_tracking.py" not in result
    assert result["scripts/train_tracking_lr_sync.py"]==hashlib.sha256(NEW.read_bytes()).hexdigest()
    assert len(result)==6
    namespace={"manifest":{"source_sha256":result},"runtime_device_audit":{"status":"verified"},
        "ROOT":ROOT,"sha256":lambda p:hashlib.sha256(p.read_bytes()).hexdigest()}
    runtime=next(node for node in block.body if isinstance(node,ast.If) and isinstance(node.test,ast.Compare)
        and isinstance(node.test.left,ast.Name) and node.test.left.id=="runtime_device_audit")
    source_additions=[node for node in post_load_nodes(block) if isinstance(node,ast.Assign)
        and isinstance(node.targets[0],ast.Subscript) and isinstance(node.targets[0].value,ast.Subscript)]
    execute([runtime,*source_additions],namespace)
    assert len(result)==9
    assert set(result)=={"training/tron1_tracking.py","training/tron1_domain_randomization.py","training/tron1_terrain.py",
        "training/tron1_drift.py","training/tron1_contact_slip.py","training/tron1_runtime_device.py",
        "scripts/train_tracking_lr_sync.py","scripts/train_tracking.py","training/tron1_resume_lr.py"}
    assert result["scripts/train_tracking.py"]==BASE_SHA
