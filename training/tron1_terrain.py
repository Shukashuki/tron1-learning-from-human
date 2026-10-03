"""Shared, metric, axis-aligned static terrain for Isaac and MuJoCo tasks.

No simulator is imported at module load. A box's size is its FULL XYZ extent;
its center is relative to each Isaac environment origin / MuJoCo world origin.
These helpers never change robot poses or turn reference playback into physics.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def _number(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{context} must be a finite number")
    return float(value)


def validate_terrain(spec):
    """Return a deep copied, validated spec; reject ambiguous units/geometry."""
    if not isinstance(spec, dict) or spec.get("schema_version") != 1:
        raise ValueError("Terrain schema_version must be 1")
    if spec.get("frame") != "world_z_up_m":
        raise ValueError("Terrain frame must explicitly be world_z_up_m")
    required = {"schema_version", "frame", "ground", "boxes"}
    if not required <= spec.keys() or set(spec) - required - {"provenance"}:
        raise ValueError("Terrain requires schema_version/frame/ground/boxes and optional provenance only")
    result = copy.deepcopy(spec)
    ground = result["ground"]
    if not isinstance(ground, dict) or set(ground) != {"z", "friction"}:
        raise ValueError("Terrain ground requires exactly z and friction")
    ground["z"] = _number(ground["z"], "ground.z")
    if abs(ground["z"]) > 1e-9:
        raise ValueError("This shared terrain supports only the existing z=0 ground plane")
    ground["friction"] = _number(ground["friction"], "ground.friction")
    if not 0 <= ground["friction"] <= 2:
        raise ValueError("ground.friction must lie in [0,2]")
    if not isinstance(result["boxes"], list) or len(result["boxes"]) > 16:
        raise ValueError("boxes must be a list of at most 16 static boxes")
    names = set()
    for box in result["boxes"]:
        if not isinstance(box, dict) or set(box) != {"name", "center", "size", "friction"}:
            raise ValueError("Each box requires exactly name, center, full size and friction")
        name = box["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name in names:
            raise ValueError("Box names must be unique simple identifiers")
        names.add(name)
        for key in ("center", "size"):
            if not isinstance(box[key], (list, tuple)) or len(box[key]) != 3:
                raise ValueError(f"Box {name}.{key} must contain exactly three numbers")
            box[key] = [_number(value, f"{name}.{key}") for value in box[key]]
        if any(value <= 0 for value in box["size"]):
            raise ValueError("All full box dimensions must be positive")
        if abs(box["center"][2] - box["size"][2] / 2 - ground["z"]) > 1e-6:
            raise ValueError("Supported ledges must rest on z=0, not float or extend underground")
        box["friction"] = _number(box["friction"], f"{name}.friction")
        if not 0 <= box["friction"] <= 2:
            raise ValueError("Box friction must lie in [0,2]")
    if "provenance" in result and not isinstance(result["provenance"], dict):
        raise ValueError("provenance must be an object")
    # Reject nonportable metadata (NaN, arbitrary objects) before saving/loading.
    json.dumps(result, allow_nan=False)
    return result


def load_terrain(path):
    return validate_terrain(json.loads(Path(path).read_text(encoding="utf-8")))


def apply_isaac_terrain(scene_cfg, spec):
    """Add collision-only STATIC cuboids to every cloned Isaac environment.

    No RigidBodyPropertiesCfg is authored, so boxes are fixed world colliders,
    not free dynamic boxes. InteractiveScene enumerates the added AssetBaseCfg
    attributes. Returns the names of those attributes for diagnostics.
    """
    spec = validate_terrain(spec)
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    def material(friction):
        return sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction,
                                             restitution=0.0, friction_combine_mode="average",
                                             restitution_combine_mode="average")

    if not hasattr(scene_cfg, "terrain") or scene_cfg.terrain is None:
        raise ValueError("Isaac scene must already have the shared z=0 ground terrain")
    scene_cfg.terrain.physics_material = material(spec["ground"]["friction"])
    attributes = []
    largest_extent = 0.
    for box in spec["boxes"]:
        attribute = "terrain_box_" + box["name"]
        if hasattr(scene_cfg, attribute):
            raise ValueError(f"Refusing to overwrite existing Isaac scene attribute {attribute}")
        asset = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Terrain_" + box["name"],
            spawn=sim_utils.CuboidCfg(
                size=tuple(box["size"]),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                physics_material=material(box["friction"]),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.25, .32, .40)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(box["center"])),
        )
        setattr(scene_cfg, attribute, asset)
        attributes.append(attribute)
        largest_extent = max(largest_extent, *(abs(box["center"][i]) + box["size"][i] / 2 for i in (0, 1)))
    if spec["boxes"]:
        scene_cfg.env_spacing = max(float(scene_cfg.env_spacing), 2 * largest_extent + 1.)
    return attributes


def augment_mujoco_xml(xml_text, spec):
    """Return an augmented XML string; originals are never edited.

    Box geoms belong DIRECTLY to worldbody (body id 0), so existing wheel-ground
    contact-force telemetry correctly counts support on either plane or ledge.
    """
    spec = validate_terrain(spec)
    root = ET.fromstring(xml_text)
    world = root.find("worldbody")
    if world is None:
        raise ValueError("MuJoCo terrain requires an existing worldbody")
    planes = [geom for geom in world.findall("geom") if geom.get("type") == "plane"]
    if len(planes) != 1:
        raise ValueError("MuJoCo terrain requires exactly one direct worldbody ground plane")
    plane = planes[0]
    position = [float(value) for value in plane.get("pos", "0 0 0").split()]
    if len(position) != 3 or abs(position[2]) > 1e-9:
        raise ValueError("Existing MuJoCo ground plane must already be at z=0")
    if any(key in plane.attrib for key in ("quat", "euler", "axisangle", "xyaxes", "zaxis")):
        raise ValueError("Ground plane must use its unrotated +Z normal")
    plane.set("friction", f"{spec['ground']['friction']:.12g} 0 0")
    plane.set("contype", "2")
    plane.set("conaffinity", "1")
    existing_names = {geom.get("name") for geom in root.iter("geom")}
    for box in spec["boxes"]:
        name = "terrain_" + box["name"]
        if name in existing_names:
            raise ValueError(f"Terrain geom already exists: {name}")
        ET.SubElement(world, "geom", name=name, type="box",
                      pos=" ".join(f"{value:.12g}" for value in box["center"]),
                      size=" ".join(f"{value / 2:.12g}" for value in box["size"]),
                      friction=f"{box['friction']:.12g} 0 0", contype="2", conaffinity="1", condim="3",
                      rgba="0.25 0.32 0.40 1")
    return ET.tostring(root, encoding="unicode")


def mujoco_terrain_runtime_audit(model, spec, tolerance_m=1e-8):
    """Verify compiled collision geometry, not XML text or copied contract metadata.

    All terrain geoms must belong directly to worldbody, so their compiled
    positions/orientations are world coordinates and cannot move dynamically.
    This reads the post-configuration model without changing simulator state.
    """
    spec = validate_terrain(spec)
    tolerance_m = _number(tolerance_m, "tolerance_m")
    if not 0 < tolerance_m <= 1e-5:
        raise ValueError("Terrain audit tolerance_m must lie in (0,1e-5]")
    import mujoco

    def close(actual, expected, label, tolerance=tolerance_m):
        values = [float(value) for value in actual]
        errors = [abs(value - target) for value, target in zip(values, expected)]
        if (len(values) != len(expected) or not all(math.isfinite(v) for v in values)
                or max(errors, default=0.) > tolerance):
            raise ValueError(f"Compiled terrain {label} differs: actual={values}, expected={expected}")
        return values

    def inspect(index, expected_type, friction):
        name = model.geom(index).name
        if int(model.geom_bodyid[index]) != 0:
            raise ValueError(f"Compiled terrain {name} must be directly attached to static worldbody")
        if int(model.geom_type[index]) != int(expected_type):
            raise ValueError(f"Compiled terrain {name} has the wrong geom type")
        quat = [float(value) for value in model.geom_quat[index]]
        close([abs(quat[0]), *quat[1:]], [1., 0., 0., 0.], f"{name} unrotated quaternion", 1e-9)
        actual_friction = close(model.geom_friction[index], [friction, 0., 0.], f"{name} friction", 1e-9)
        masks = [int(model.geom_contype[index]), int(model.geom_conaffinity[index])]
        if masks != [2, 1] or int(model.geom_condim[index]) != 3:
            raise ValueError(f"Compiled terrain {name} collision masks/condim differ from 2/1/3")
        return {"geom_name": name, "geom_id": int(index), "body_id": 0,
                "static_collider": True, "quaternion_wxyz": quat,
                "friction": actual_friction, "contype": masks[0], "conaffinity": masks[1], "condim": 3}

    expected_names = {"terrain_" + box["name"] for box in spec["boxes"]}
    actual_names = {model.geom(i).name for i in range(model.ngeom)
                    if (model.geom(i).name or "").startswith("terrain_")}
    if actual_names != expected_names:
        raise ValueError(f"Compiled terrain box names differ: actual={actual_names}, expected={expected_names}")
    planes = [i for i in range(model.ngeom)
              if int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_PLANE)]
    if len(planes) != 1:
        raise ValueError("Compiled terrain requires exactly one ground plane")
    ground = inspect(planes[0], mujoco.mjtGeom.mjGEOM_PLANE, spec["ground"]["friction"])
    ground_position = [float(v) for v in model.geom_pos[planes[0]]]
    close(ground_position, [ground_position[0], ground_position[1], spec["ground"]["z"]], "ground plane position")
    ground["position_world_m"] = ground_position
    records = []
    for box in spec["boxes"]:
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_" + box["name"])
        if index < 0:
            raise ValueError(f"Compiled terrain collider missing: {box['name']}")
        record = inspect(index, mujoco.mjtGeom.mjGEOM_BOX, box["friction"])
        position = close(model.geom_pos[index], box["center"], f"{box['name']} world position")
        half = close(model.geom_size[index], [v / 2 for v in box["size"]], f"{box['name']} half size")
        record.update(name=box["name"], position_world_m=position, half_size_m=half,
                      actual_aabb_min_m=[p - s for p, s in zip(position, half)],
                      actual_aabb_max_m=[p + s for p, s in zip(position, half)],
                      expected_aabb_min_m=[p - s / 2 for p, s in zip(box["center"], box["size"])],
                      expected_aabb_max_m=[p + s / 2 for p, s in zip(box["center"], box["size"])])
        records.append(record)
    return {"status": "passed", "box_count": len(records), "tolerance_m": tolerance_m,
            "method": "actual post-configuration compiled MjModel worldbody geoms: type, position, half size, rotation, friction, collision masks and condim",
            "ground": ground, "boxes": records}


def terrain_runtime_audit(env, spec, tolerance_m=1e-5):
    """Verify the ACTUALLY SPAWNED env0 collider bounds, collision and staticness.

    Uses composed USD world bounds minus the simulator's environment origin;
    checking only CuboidCfg values would not establish that spawning succeeded.
    """
    spec = validate_terrain(spec)
    from pxr import Usd, UsdGeom, UsdPhysics
    from isaacsim.core.utils.stage import get_current_stage
    stage = get_current_stage()
    if stage is None or abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.) > 1e-9:
        raise ValueError("Terrain audit requires a live metric USD stage")
    origin = env.scene.env_origins[0]
    if hasattr(origin, "detach"):
        origin = origin.detach().cpu()
    origin = [float(value) for value in origin.tolist()]
    if len(origin) != 3:
        raise ValueError("Expected a three-dimensional environment origin")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
                                                    UsdGeom.Tokens.proxy], useExtentsHint=False)
    records = []
    for box in spec["boxes"]:
        path = "/World/envs/env_0/Terrain_" + box["name"]
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise ValueError(f"Terrain collider was not spawned: {path}")
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        lower = [float(value) - origin[index] for index, value in enumerate(bounds.GetMin())]
        upper = [float(value) - origin[index] for index, value in enumerate(bounds.GetMax())]
        expected_lower = [center - size / 2 for center, size in zip(box["center"], box["size"])]
        expected_upper = [center + size / 2 for center, size in zip(box["center"], box["size"])]
        errors = [abs(actual - expected) for actual, expected in zip(lower + upper, expected_lower + expected_upper)]
        if not all(math.isfinite(value) for value in lower + upper) or max(errors) > tolerance_m:
            raise ValueError(f"Spawned terrain bounds differ from spec: {path}, actual={(lower, upper)}, expected={(expected_lower, expected_upper)}")
        enabled_colliders, dynamic_bodies = [], []
        for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if child.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get():
                enabled_colliders.append(str(child.GetPath()))
            if child.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(child).GetRigidBodyEnabledAttr().Get():
                dynamic_bodies.append(str(child.GetPath()))
        if not enabled_colliders or dynamic_bodies:
            raise ValueError(f"Terrain must have enabled collision and no dynamic rigid body: {path}, colliders={enabled_colliders}, dynamic={dynamic_bodies}")
        records.append({"name": box["name"], "prim_path": path,
                        "actual_environment_relative_aabb_min_m": lower,
                        "actual_environment_relative_aabb_max_m": upper,
                        "expected_aabb_min_m": expected_lower, "expected_aabb_max_m": expected_upper,
                        "max_bound_error_m": max(errors), "enabled_collision_prims": enabled_colliders,
                        "static_collider": True})
    return {"status": "passed", "environment_index": 0, "environment_origin_w_m": origin,
            "box_count": len(records), "tolerance_m": tolerance_m,
            "method": "composed USD env0 world AABB minus actual environment origin, CollisionAPI enabled, no enabled RigidBodyAPI",
            "boxes": records}
