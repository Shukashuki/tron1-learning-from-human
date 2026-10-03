"""Four-way playback clocks and skeleton validation, without GL or dynamics."""
from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

for dependency in ("mujoco", "matplotlib", "scipy", "imageio_ffmpeg"):
    pytest.importorskip(dependency)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_motion_pipeline as render


def skeleton_inputs():
    times = np.array([0., 1., 3.])
    positions = np.arange(27, dtype=float).reshape(3, 3, 3) / 10.
    names = ["pelvis", "left_hip", "left_knee"]
    parents = np.array([-1, 0, 1])
    return times, positions, names, parents


def test_keypoints_interpolate_on_original_nonuniform_source_clock():
    times, positions, names, parents = skeleton_inputs()
    clip = render.KeypointClip(times, positions, names, parents)
    np.testing.assert_allclose(clip.sample(.25), .75 * positions[0] + .25 * positions[1])
    np.testing.assert_allclose(clip.sample(2.), .5 * positions[1] + .5 * positions[2])
    np.testing.assert_allclose(clip.sample(1.), positions[1])


def test_keypoints_clamp_before_start_and_after_terminal_without_mutating_source():
    times, positions, names, parents = skeleton_inputs()
    original_positions, original_times = positions.copy(), times.copy()
    clip = render.KeypointClip(times, positions, names, parents)
    np.testing.assert_allclose(clip.sample(-10.), positions[0])
    np.testing.assert_allclose(clip.sample(99.), positions[-1])
    np.testing.assert_array_equal(positions, original_positions)
    np.testing.assert_array_equal(times, original_times)


@pytest.mark.parametrize("times", [
    np.array([]), np.array([0.]), np.array([0., 0., 1.]),
    np.array([0., 2., 1.]), np.array([0., np.nan, 2.]),
    np.array([0., 1., np.inf]), np.array([[0., 1., 2.]]),
])
def test_keypoints_reject_invalid_source_times(times):
    _, positions, names, parents = skeleton_inputs()
    with pytest.raises(ValueError):
        render.KeypointClip(times, positions, names, parents)


@pytest.mark.parametrize("positions", [
    np.zeros((2, 3, 3)), np.zeros((3, 2, 3)), np.zeros((3, 3, 2)),
    np.zeros((3, 3, 3, 1)), np.full((3, 3, 3), np.nan),
    np.full((3, 3, 3), np.inf),
])
def test_keypoints_reject_mismatched_or_nonfinite_positions(positions):
    times, _, names, parents = skeleton_inputs()
    with pytest.raises(ValueError):
        render.KeypointClip(times, positions, names, parents)


@pytest.mark.parametrize("names", [
    ["pelvis", "hip", "hip"], ["pelvis", "hip"],
    ["pelvis", "hip", "knee", "ankle"],
])
def test_keypoints_require_unique_complete_names(names):
    times, positions, _, parents = skeleton_inputs()
    with pytest.raises(ValueError):
        render.KeypointClip(times, positions, names, parents)


@pytest.mark.parametrize("parents", [
    np.array([-1, 0]), np.array([[-1, 0, 1]]),
    np.array([-1, 0, 1.5]), np.array([-1., np.nan, 1.]),
    np.array([-1, -1, 1]), np.array([1, 2, 0]),
    np.array([-2, 0, 1]), np.array([-1, 3, 1]),
    np.array([-1, 0, 2]), np.array([-1, 2, 1]),
])
def test_keypoints_reject_ambiguous_broken_or_cyclic_parent_graphs(parents):
    times, positions, names, _ = skeleton_inputs()
    with pytest.raises(ValueError):
        render.KeypointClip(times, positions, names, parents)


def test_keypoint_parent_graph_need_not_be_topologically_sorted():
    times, positions, names, _ = skeleton_inputs()
    clip = render.KeypointClip(times, positions, names, np.array([2, -1, 1]))
    np.testing.assert_allclose(clip.sample(0.), positions[0])


def test_source_clock_preserves_crop_retiming_and_terminal_hold():
    task = np.array([0., 1., 3., 4.])
    source = np.array([2., 2.5, 3.5, 3.5])
    assert render.source_clock(task, source, .5) == pytest.approx(2.25)
    assert render.source_clock(task, source, 2.) == pytest.approx(3.)
    assert render.source_clock(task, source, 3.5) == pytest.approx(3.5)
    assert render.source_clock(task, source, -10.) == pytest.approx(2.)
    assert render.source_clock(task, source, 99.) == pytest.approx(3.5)


def test_source_clock_accepts_constant_terminal_source():
    assert render.source_clock(np.array([0., 1.]), np.array([2., 2.]), .5) == 2.


@pytest.mark.parametrize("task,source", [
    ([], []), ([0.], [1.]), ([0., 1.], [0., .5, 1.]),
    ([[0., 1.]], [[0., 1.]]), ([0., 1.], [[0., 1.]]),
    ([0., 0.], [0., 1.]), ([1., 0.], [0., 1.]),
    ([0., np.nan], [0., 1.]), ([0., np.inf], [0., 1.]),
    ([0., 1.], [1., 0.]), ([0., 1.], [0., np.nan]),
    ([0., 1.], [0., np.inf]),
])
def test_source_clock_rejects_bad_timeline_correspondences(task, source):
    with pytest.raises(ValueError):
        render.source_clock(np.asarray(task), np.asarray(source), .5)


def test_frame_times_use_exact_policy_grid_and_include_terminal_grid_point():
    np.testing.assert_allclose(render.frame_times(.04), [0., .02, .04], atol=1e-15)
    np.testing.assert_allclose(render.frame_times(.031, fps=50.), [0., .02, .04], atol=1e-15)
    np.testing.assert_allclose(render.frame_times(.1, fps=20.), [0., .05, .1], atol=1e-15)


def test_frame_times_do_not_add_a_frame_for_endpoint_roundoff():
    end = np.nextafter(2.44, np.inf)
    times = render.frame_times(end, fps=50.)
    assert len(times) == 123
    assert times[0] == 0.
    assert times[-1] == pytest.approx(2.44)
    np.testing.assert_allclose(np.diff(times), .02, atol=1e-14)


@pytest.mark.parametrize("end,fps", [
    (0., 50.), (-1., 50.), (np.nan, 50.), (np.inf, 50.),
    (1., 0.), (1., -1.), (1., np.nan), (1., np.inf),
])
def test_frame_times_reject_nonpositive_or_nonfinite_parameters(end, fps):
    with pytest.raises(ValueError):
        render.frame_times(end, fps=fps)


def test_renderer_does_forward_kinematics_but_never_advances_physics():
    tree = ast.parse((ROOT / "scripts" / "render_motion_pipeline.py").read_text())
    calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
             for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, (ast.Attribute, ast.Name))}
    assert "mj_forward" in calls
    assert not calls.intersection({"mj_step", "mj_step1", "mj_step2"})


@pytest.fixture
def hash_chain(tmp_path):
    paths = []
    digests = {}
    for label in ("human", "gmr", "mink", "motion", "model"):
        path = tmp_path / (label + ".bin")
        path.write_bytes((label + " recorded artifact").encode())
        paths.append(path)
        digests[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    gmr_report = {"source_sha256": digests["human"], "baseline_sha256": digests["mink"],
                  "model_sha256": digests["model"]}
    mink_report = {"source_sha256": digests["human"], "model_sha256": digests["model"]}
    export_report = {"source_sha256": digests["gmr"], "output_sha256": digests["motion"],
                     "model_sha256": digests["model"]}
    return paths, [gmr_report, mink_report, export_report]


def test_hash_chain_accepts_same_human_retargeting_and_exported_reference(hash_chain):
    paths, reports = hash_chain
    render.validate_hash_chain(*paths, *reports)


@pytest.mark.parametrize("report_index,key", [
    (0, "source_sha256"), (0, "baseline_sha256"), (1, "source_sha256"),
    (2, "source_sha256"), (2, "output_sha256"), (2, "model_sha256"),
])
def test_hash_chain_rejects_changed_provenance_digest(hash_chain, report_index, key):
    paths, reports = hash_chain
    changed = copy.deepcopy(reports)
    changed[report_index][key] = "0" * 64
    with pytest.raises(ValueError):
        render.validate_hash_chain(*paths, *changed)


@pytest.mark.parametrize("path_index", range(5))
def test_hash_chain_rejects_changed_actual_artifact_bytes(hash_chain, path_index):
    paths, reports = hash_chain
    paths[path_index].write_bytes(b"different artifact")
    with pytest.raises(ValueError):
        render.validate_hash_chain(*paths, *reports)
