"""Analytic cylinder-support checks only; no simulator/actor execution."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_recorded_wheel_pose.py"
spec = importlib.util.spec_from_file_location("wheel_geometry", SCRIPT)
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def test_cylinder_support_horizontal_vertical_and_tilted():
    axes = np.array([0., 1., -1., 1. / np.sqrt(2)])
    actual = M.cylinder_bottom_z(1., axes, .127, .025)
    np.testing.assert_allclose(actual, [1.-.127, 1.-.025, 1.-.025, 1.-(.127+.025)/np.sqrt(2)], atol=1e-15)


@pytest.mark.parametrize("center,axis,radius,width", [(float('nan'), 0., .127, .025),
    (1., float('nan'), .127, .025), (1., 1.01, .127, .025), (1., 0., 0., .025), (1., 0., .127, -.025)])
def test_invalid_geometry_rejected(center, axis, radius, width):
    with pytest.raises(ValueError): M.cylinder_bottom_z(center, axis, radius, width)


def test_inert_import_and_no_force_or_integration_methods(tmp_path):
    source = SCRIPT.read_text()
    assert "mujoco.mj_kinematics(model, data)" in source
    assert not any(x in source for x in ("mujoco.mj_step(", "mujoco.mj_forward(", "mujoco.mj_fwdPosition(", "mujoco.mj_contactForce("))
    code = (f"import importlib.util,sys;s=importlib.util.spec_from_file_location('m',{str(SCRIPT)!r});"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "assert not any(k.startswith(('torch','mujoco','isaac','omni')) for k in sys.modules)")
    r = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert not list(tmp_path.iterdir())
