"""Shared static-ledge schema and exact compiled MuJoCo geometry tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import types
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.tron1_terrain import (apply_isaac_terrain, augment_mujoco_xml, load_terrain,
                                    mujoco_terrain_runtime_audit, validate_terrain)


@pytest.fixture
def terrain():
    return {"schema_version": 1, "frame": "world_z_up_m", "ground": {"z": 0, "friction": .6},
            "boxes": [{"name": "step_ledge", "center": [.65, -.2, .06], "size": [.6, 1., .12], "friction": .6}],
            "provenance": {"geometry_estimated": True, "ground_truth_geometry": False}}


def test_spec_round_trip_and_no_mutation(terrain, tmp_path):
    before = copy.deepcopy(terrain)
    normalized = validate_terrain(terrain)
    normalized["boxes"][0]["center"][0] = 99
    assert terrain == before
    path = tmp_path / "terrain.json"
    path.write_text(json.dumps(terrain))
    assert load_terrain(path) == terrain


@pytest.mark.parametrize("change,pattern", [
    (lambda s: s.update(frame="world_y_up_cm"), "world_z_up_m"),
    (lambda s: s["ground"].update(z=.1), "z=0"),
    (lambda s: s["boxes"][0].update(size=[.6, 1, -.12]), "positive"),
    (lambda s: s["boxes"][0].update(center=[.65, -.2, .2]), "rest on z=0"),
    (lambda s: s["boxes"][0].update(friction=float("nan")), "finite"),
    (lambda s: s["boxes"][0].update(name="../bad"), "identifiers"),
    (lambda s: s["boxes"].append(copy.deepcopy(s["boxes"][0])), "unique"),
    (lambda s: s["boxes"][0].update(center=[True, -.2, .06]), "finite number"),
])
def test_schema_rejects_ambiguous_or_unsafe_geometry(terrain, change, pattern):
    change(terrain)
    with pytest.raises(ValueError, match=pattern):
        validate_terrain(terrain)


def test_mujoco_box_bounds_friction_and_world_body(terrain):
    mujoco = pytest.importorskip("mujoco")
    original = '<mujoco><worldbody><geom name="floor" type="plane" size="3 3 .1"/><body name="ball" pos="0 0 1"><freejoint/><geom type="sphere" size=".02" mass="1"/></body></worldbody></mujoco>'
    xml = augment_mujoco_xml(original, terrain)
    assert "terrain_step_ledge" not in original
    model = mujoco.MjModel.from_xml_string(xml)
    geom = model.geom("terrain_step_ledge")
    assert model.geom_bodyid[geom.id] == 0  # Counts as ground contact, not an unrelated body.
    assert int(model.geom_type[geom.id]) == int(mujoco.mjtGeom.mjGEOM_BOX)
    np.testing.assert_allclose(geom.pos, [.65, -.2, .06])
    np.testing.assert_allclose(geom.size, [.3, .5, .06])  # MJ half extents.
    np.testing.assert_allclose(geom.pos - geom.size, [.35, -.7, 0])
    np.testing.assert_allclose(geom.pos + geom.size, [.95, .3, .12])
    np.testing.assert_allclose(geom.friction, [.6, 0, 0])
    np.testing.assert_allclose(model.geom("floor").friction, [.6, 0, 0])
    assert model.geom_contype[geom.id] == 2 and model.geom_conaffinity[geom.id] == 1
    audit = mujoco_terrain_runtime_audit(model, terrain)
    assert audit["status"] == "passed" and audit["box_count"] == 1
    assert audit["boxes"][0]["body_id"] == 0
    np.testing.assert_allclose(audit["boxes"][0]["actual_aabb_max_m"], [.95, .3, .12])
    json.dumps(audit, allow_nan=False)
    with pytest.raises(ValueError, match="already exists"):
        augment_mujoco_xml(xml, terrain)


@pytest.mark.parametrize("field,value,pattern", [
    ("geom_pos", [.7, -.2, .06], "world position"),
    ("geom_size", [.6, 1., .12], "half size"),
    ("geom_quat", [.70710678, 0., 0., .70710678], "unrotated"),
    ("geom_bodyid", 1, "static worldbody"),
    ("geom_type", 2, "wrong geom type"),
    ("geom_friction", [.8, 0., 0.], "friction"),
    ("geom_friction", [.6, .01, 0.], "friction"),
    ("geom_contype", 0, "collision masks"),
    ("geom_conaffinity", 0, "collision masks"),
    ("geom_condim", 1, "collision masks"),
    ("geom_pos", [float("nan"), -.2, .06], "world position"),
])
def test_compiled_audit_rejects_runtime_geometry_mutation(terrain, field, value, pattern):
    mujoco = pytest.importorskip("mujoco")
    source = '<mujoco><worldbody><geom name="floor" type="plane" size="3 3 .1"/><body name="ball"><freejoint/><geom type="sphere" size=".02" mass="1"/></body></worldbody></mujoco>'
    model = mujoco.MjModel.from_xml_string(augment_mujoco_xml(source, terrain))
    getattr(model, field)[model.geom("terrain_step_ledge").id] = value
    with pytest.raises(ValueError, match=pattern):
        mujoco_terrain_runtime_audit(model, terrain)


def test_compiled_audit_rejects_missing_extra_boxes_and_changed_ground(terrain):
    mujoco = pytest.importorskip("mujoco")
    source = '<mujoco><worldbody><geom name="floor" type="plane" size="3 3 .1"/></worldbody></mujoco>'
    with pytest.raises(ValueError, match="box names differ"):
        mujoco_terrain_runtime_audit(mujoco.MjModel.from_xml_string(source), terrain)
    xml = augment_mujoco_xml(source, terrain)
    extra = xml.replace('</worldbody>', '<geom name="terrain_unexpected" type="box" size=".1 .1 .1"/></worldbody>')
    with pytest.raises(ValueError, match="box names differ"):
        mujoco_terrain_runtime_audit(mujoco.MjModel.from_xml_string(extra), terrain)
    model = mujoco.MjModel.from_xml_string(xml)
    model.geom_friction[model.geom("floor").id] = [1., 0., 0.]
    with pytest.raises(ValueError, match="floor friction"):
        mujoco_terrain_runtime_audit(model, terrain)


def test_isaac_lazy_helper_authors_static_full_size_per_env_box(terrain, monkeypatch):
    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    class Asset(Config):
        InitialStateCfg = Config
    package, sim, assets = types.ModuleType("isaaclab"), types.ModuleType("isaaclab.sim"), types.ModuleType("isaaclab.assets")
    package.__path__ = []
    package.sim, package.assets = sim, assets
    for name in ("CuboidCfg", "CollisionPropertiesCfg", "RigidBodyMaterialCfg", "PreviewSurfaceCfg"):
        setattr(sim, name, Config)
    assets.AssetBaseCfg = Asset
    for name, module in (("isaaclab", package), ("isaaclab.sim", sim), ("isaaclab.assets", assets)):
        monkeypatch.setitem(sys.modules, name, module)
    scene = Config(terrain=Config(), env_spacing=1.)
    names = apply_isaac_terrain(scene, terrain)
    assert names == ["terrain_box_step_ledge"]
    box = scene.terrain_box_step_ledge
    assert box.prim_path == "{ENV_REGEX_NS}/Terrain_step_ledge"
    assert box.spawn.size == (.6, 1., .12)
    assert box.init_state.pos == (.65, -.2, .06)
    assert box.spawn.collision_props.collision_enabled is True
    assert not hasattr(box.spawn, "rigid_props")
    assert box.spawn.physics_material.static_friction == .6
    assert box.spawn.physics_material.dynamic_friction == .6
    assert box.spawn.physics_material.restitution == 0
    assert scene.env_spacing == pytest.approx(2.9)
    with pytest.raises(ValueError, match="overwrite"):
        apply_isaac_terrain(scene, terrain)


def test_ledge_height_comes_from_foot_support_not_pelvis():
    pytest.importorskip("mujoco")
    pytest.importorskip("mink")
    sys.path.insert(0, str(ROOT / "scripts"))
    from prepare_step_reference import estimate_terrain
    times = np.linspace(0, 4, 81)
    progress = np.clip((times - 1) / 2, 0, 1)
    positions = np.zeros((81, 5, 3))
    positions[:, 0, 2] = 999 * progress  # Deliberately nonsensical pelvis excursion.
    positions[:, 1:, 2] = .2 * progress[:, None]
    human = {"time_s": times, "joint_names": np.array(["root", "lfoot", "ltoes", "rfoot", "rtoes"]),
             "joint_positions_world_m": positions}
    targets = np.zeros((81, 2, 3))
    targets[:, :, 0] = -.02 + .8 * progress[:, None]
    targets[:, :, 1] = [.15, -.15]
    targets[:, :, 2] = .127 + .15 * progress[:, None]
    spec, offset = estimate_terrain(human, targets, .75, .127)
    assert spec["boxes"][0]["size"][2] == pytest.approx(.15)
    assert spec["provenance"]["geometry_estimated"] is True
    assert spec["provenance"]["ground_truth_geometry"] is False
    assert offset == pytest.approx(.001)
    human["joint_positions_world_m"][:, 3:, 2] *= .5
    with pytest.raises(ValueError, match="not identifiable"):
        estimate_terrain(human, targets, .75, .127)
