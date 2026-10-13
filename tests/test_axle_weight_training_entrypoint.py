"""Source-only020 trainer checks, not PPO/GPU execution."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/"scripts/train_tracking_tight_axle.py"
NEW=ROOT/"scripts/train_tracking_axle_weight.py"


def functions(path):
    return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}


def imported(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_original_ppo_architecture_schedule_and_source_audit_held():
    old,new=functions(OLD),functions(NEW)
    for name in ("ppo_config","write_json","sha256","audit_installed_resume_sources"):
        assert ast.dump(old[name],include_attributes=False)==ast.dump(new[name],include_attributes=False)
    a,b=imported("trainer017",OLD),imported("trainer020",NEW)
    assert a.ppo_config(42,600)==b.ppo_config(42,600)


def test_inert_import_no_simulator_or_torch(tmp_path):
    code=f"import importlib.util,sys; s=importlib.util.spec_from_file_location('v',{str(NEW)!r}); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert not any(k.startswith(('torch','isaac','omni','mujoco','rsl_rl')) for k in sys.modules)"
    result=subprocess.run([sys.executable,"-B","-c",code],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert not list(tmp_path.iterdir())


def test_learning_resume_and_final_save_calls_unchanged():
    def selected(path):
        return [ast.dump(n,include_attributes=False) for n in ast.walk(functions(path)["main"])
            if isinstance(n,ast.Call) and ((isinstance(n.func,ast.Attribute) and n.func.attr in ("load","learn","save","reset","step"))
                or (isinstance(n.func,ast.Name) and n.func.id in ("OnPolicyRunner","restore_resume_learning_rate")))]
    assert selected(OLD)==selected(NEW)
    source=NEW.read_text()
    assert "(2048, 600, 600, 42, 8)" in source
    assert "configure_axle_weight_trial(cfg, task_profile)" in source
    assert "install_worst_ground_load_accumulator(env)" in source
    assert 'manifest["dense_ground_load_sampling"] = "all_substeps_max_cost"' in source
    assert '"654acc37ccd8ff41f7b10d0b71abe9c85051cafe58be7f3b5d893755906c7bc3"' in source
