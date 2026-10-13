"""Synthetic publication rejection/copy/decode tests, not new policy evidence."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/publish_car_drift_worst_ground_load.py"
spec = importlib.util.spec_from_file_location("public018_tests", SCRIPT)
PUB = importlib.util.module_from_spec(spec); spec.loader.exec_module(PUB)


def test_pinned_actual_pair_and_frozen_assessor_sources():
    assert PUB.sha(ROOT / PUB.PAIR_FILE) == PUB.PAIR_SHA
    assert PUB.sha(ROOT / "scripts/publish_car_drift.py") == PUB.BASE_SHA
    assert PUB.sha(ROOT / "scripts/render_car_drift_pipeline.py") == PUB.RENDERER_SHA
    source = SCRIPT.read_text()
    assert "fresh_proof = pair.audit(trial, completion_sha)" in source
    assert "proof == fresh_proof" in source
    assert '"status": "assessed_" + assessment["verdict"]' in source


def test_preexisting_snapshot_rejected_before_any_validation_or_write(tmp_path, monkeypatch):
    output = tmp_path / "published"; output.mkdir()
    sentinel = output / "previous-failure.json"; sentinel.write_bytes(b"preserve")
    monkeypatch.setattr(PUB, "prepare", lambda *a: (_ for _ in ()).throw(AssertionError("must not run")))
    with pytest.raises(ValueError): PUB.publish(tmp_path, tmp_path, tmp_path, output, "a" * 64, "b" * 64)
    assert sentinel.read_bytes() == b"preserve" and len(list(output.iterdir())) == 1


def test_preflight_failure_creates_no_output(tmp_path, monkeypatch):
    output = tmp_path / "new-public"
    monkeypatch.setattr(PUB, "prepare", lambda *a: (_ for _ in ()).throw(ValueError("actual source mismatch")))
    with pytest.raises(ValueError): PUB.publish(tmp_path, tmp_path, tmp_path, output, "a" * 64, "b" * 64)
    assert not output.exists()


def test_exact_eight_file_copy_preserves_fail_result(tmp_path, monkeypatch):
    render = tmp_path / "render"; render.mkdir()
    payloads = {name: b'{}\n' for name in ("assessment.json", "reference_audit.json", "render_report.json")}
    payloads["summary.json"] = PUB.encode({"verdict": "fail", "failed_checks": ["mujoco.sustained_grounded_contact_patch_drift"]})
    media = {}
    for name in ("motion_pipeline.mp4", "overview.png", "final.png", "contact_overview.png"):
        p = render / name; p.write_bytes(b"synthetic test bytes only")
        media[name] = {"path": p, "sha256": PUB.sha(p)}
    monkeypatch.setattr(PUB, "prepare", lambda *a: (payloads, media))
    output = tmp_path / "public"
    summary = PUB.publish(tmp_path, tmp_path, render, output, "a" * 64, "b" * 64)
    assert summary["verdict"] == "fail" and {p.name for p in output.iterdir()} == PUB.ARTIFACTS
    assert all(PUB.sha(output / n) == m["sha256"] for n, m in media.items())


@pytest.mark.parametrize("frames,codec,size,bad_frame", [(144, "h264", [1920, 1080], False),
    (145, "other", [1920, 1080], False), (145, "h264", [1280, 720], False), (145, "h264", [1920, 1080], True)])
def test_incomplete_or_wrong_media_cannot_claim_decode(monkeypatch, frames, codec, size, bad_frame):
    report = {"video_resolution": [1920, 1080], "video_fps": 12.5, "video_frames": 145}
    frame = b"short" if bad_frame else bytes(1920 * 1080 * 3)
    def reader(*args):
        yield {"size": size, "codec": codec, "fps": 12.5, "duration": 11.6}
        for _ in range(frames): yield frame
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", SimpleNamespace(read_frames=reader))
    with pytest.raises(ValueError): PUB.decode_video(Path("synthetic.mp4"), report)


def test_complete_decode_counts_every_frame(monkeypatch):
    frame = bytes(1920 * 1080 * 3)
    def reader(*args):
        yield {"size": [1920, 1080], "codec": "h264", "fps": 12.5, "duration": 11.6}
        for _ in range(145): yield frame
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", SimpleNamespace(read_frames=reader))
    assert PUB.decode_video(Path("synthetic.mp4"), {"video_resolution": [1920, 1080], "video_fps": 12.5, "video_frames": 145})["independent_complete_decode"] is True


def test_inert_import_no_actor_physics_or_output(tmp_path):
    code = (f"import importlib.util,sys;s=importlib.util.spec_from_file_location('p',{str(SCRIPT)!r});"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "assert not any(k.startswith(('torch','isaac','omni','mujoco','rsl_rl')) for k in sys.modules)")
    run = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert not list(tmp_path.iterdir())
