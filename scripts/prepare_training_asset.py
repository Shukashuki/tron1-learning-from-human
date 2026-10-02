"""Make a self-contained physics-only TRON1 USD without changing physics.

Run with .venv-assets/bin/python. Source files remain untouched. The output
removes only checked render-only visual subtrees and non-physics materials,
de-instances retained collision geometry, and flattens all composition arcs.
Every retained prim's schemas, type, attributes, time samples, connections,
relationships, and world transform must match a canonical source snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Sdf.Path):
        return str(value)
    if isinstance(value, Sdf.AssetPath):
        return {"asset_path": value.path}
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        return {"real": float(value.GetReal()), "imaginary": serial(value.GetImaginary())}
    try:
        return [serial(item) for item in value]
    except TypeError:
        return str(value)


def all_prims(stage):
    return list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))


def under(path, roots):
    return any(path == root or path.HasPrefix(root) for root in roots)


def snapshot(stage, excluded):
    result = {}
    for prim in all_prims(stage):
        if under(prim.GetPath(), excluded):
            continue
        attrs = {}
        for attribute in prim.GetAttributes():
            attrs[attribute.GetName()] = {
                "type": str(attribute.GetTypeName()), "value": serial(attribute.Get()),
                "custom": attribute.IsCustom(), "variability": str(attribute.GetVariability()),
                "time_samples": {str(t): serial(attribute.Get(t)) for t in attribute.GetTimeSamples()},
                "connections": [str(path) for path in attribute.GetConnections()],
            }
        item = {"type": prim.GetTypeName(), "schemas": prim.GetAppliedSchemas(),
                "attributes": attrs,
                "relationships": {rel.GetName(): [str(path) for path in rel.GetTargets()]
                                  for rel in prim.GetRelationships()}}
        if prim.IsA(UsdGeom.Xformable):
            item["world_transform"] = serial(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        result[str(prim.GetPath())] = item
    return result


def snapshot_hash(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare(source: Path, output: Path):
    source, output = source.resolve(), output.resolve()
    report_path = output.parent / "report.json"
    if output.exists() or report_path.exists():
        raise FileExistsError("Use a new output directory; existing assets/reports are preserved")
    source_hash = sha256(source)
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise ValueError(f"Cannot open source stage {source}")
    prims = all_prims(stage)
    removed_roots = [prim.GetPath() for prim in prims
                     if prim.GetName() == "visuals" and prim.GetParent().HasAPI(UsdPhysics.RigidBodyAPI)]
    looks = stage.GetPrimAtPath(stage.GetDefaultPrim().GetPath().AppendChild("Looks"))
    if looks:
        removed_roots.append(looks.GetPath())
    if not removed_roots:
        raise ValueError("No standard render-only subtrees found; refusing heuristic removal")
    for prim in prims:
        if under(prim.GetPath(), removed_roots):
            if prim.IsA(UsdPhysics.Joint) or any("Physics" in schema or "Physx" in schema for schema in prim.GetAppliedSchemas()):
                raise ValueError(f"Refusing to remove a physics-bearing visual/material subtree: {prim.GetPath()}")
    before = snapshot(stage, removed_roots)
    # Session-only overrides: nothing is saved to the source asset or layers.
    stage.SetEditTarget(stage.GetSessionLayer())
    for prim in list(stage.Traverse()):
        if prim.IsInstance() and not under(prim.GetPath(), removed_roots):
            prim.SetInstanceable(False)
    for path in removed_roots:
        stage.GetPrimAtPath(path).SetActive(False)
    flat = stage.Flatten()
    trimmed = Usd.Stage.Open(flat)
    for path in removed_roots:
        if trimmed.GetPrimAtPath(path):
            trimmed.RemovePrim(path)
    # Removing visuals must never introduce dangling material/physics targets.
    for prim in all_prims(trimmed):
        for relationship in prim.GetRelationships():
            for target in relationship.GetTargets():
                if under(target.GetPrimPath(), removed_roots):
                    raise ValueError(f"Retained relationship targets a removed subtree: {relationship.GetPath()}")
    after = snapshot(trimmed, [])
    if before != after:
        differences = [path for path in sorted(set(before) | set(after)) if before.get(path) != after.get(path)]
        raise ValueError(f"Physics/retained-prims changed while flattening: {differences}")
    if flat.GetExternalReferences() or flat.subLayerPaths:
        raise ValueError("Flattened physics asset still depends on external layers")
    for prim in all_prims(trimmed):
        if prim.IsInstance() or prim.HasAuthoredReferences() or prim.HasAuthoredPayloads():
            raise ValueError(f"Unexpected remaining composition arc: {prim.GetPath()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not flat.Export(str(output)):
        raise RuntimeError(f"USD export failed: {output}")
    reopened = Usd.Stage.Open(str(output))
    roundtrip = snapshot(reopened, [])
    if roundtrip != before:
        raise ValueError("On-disk USD physics snapshot does not match the original")
    if sha256(source) != source_hash:
        raise ValueError("Original source unexpectedly changed")
    remaining = all_prims(reopened)
    rigid = [str(p.GetPath()) for p in remaining if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [str(p.GetPath()) for p in remaining if p.IsA(UsdPhysics.Joint)]
    collision_roots = [str(p.GetPath()) for p in remaining if p.HasAPI(UsdPhysics.CollisionAPI)]
    colliders = [p for p in remaining if p.IsA(UsdGeom.Gprim)]
    if len(rigid) != 10 or len(joints) != 9 or len(collision_roots) != 9:
        raise ValueError("Expected original 10 rigid bodies (including IMU), 9 joints and 9 collision groups")
    stage_metadata = {"default_prim": str(reopened.GetDefaultPrim().GetPath()),
                      "meters_per_unit": UsdGeom.GetStageMetersPerUnit(reopened),
                      "up_axis": str(UsdGeom.GetStageUpAxis(reopened))}
    if (stage.GetDefaultPrim().GetPath() != reopened.GetDefaultPrim().GetPath()
            or UsdGeom.GetStageMetersPerUnit(stage) != UsdGeom.GetStageMetersPerUnit(reopened)
            or UsdGeom.GetStageUpAxis(stage) != UsdGeom.GetStageUpAxis(reopened)):
        raise ValueError("Stage coordinate metadata changed")
    report = {"status": "physics_snapshot_identical", "source": str(source), "source_sha256": source_hash,
              "output": str(output), "output_sha256": sha256(output), "output_bytes": output.stat().st_size,
              "self_contained": True, "external_references": [], "stage": stage_metadata,
              "removed_render_only_subtrees": [str(path) for path in removed_roots],
              "retained_prims": len(remaining), "rigid_bodies": rigid, "joints": joints,
              "collision_groups": collision_roots,
              "collision_geometry": [{"path": str(p.GetPath()), "type": p.GetTypeName()} for p in colliders],
              "preserved_fixed_imu": any(path.endswith("/limx_imu") for path in rigid),
              "snapshot_covers": ["all retained prim types and applied APIs", "all attributes including mass/COM/inertia",
                                  "joint frames, axes, limits, drives", "collision dimensions and transforms",
                                  "attribute time samples and connections", "relationship targets", "world transforms"],
              "source_physics_snapshot_sha256": snapshot_hash(before),
              "output_physics_snapshot_sha256": snapshot_hash(roundtrip),
              "original_source_unchanged": True, "dynamics_model_changed": False,
              "dynamics_runtime_validated": False,
              "note": "Rendering-only meshes/materials removed; retained collider geometry remains visible as primitives."}
    with report_path.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "assets/robots/WF_TRON1A/WF_TRON1A.usd")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/training-asset/WF_TRON1A.usda")
    args = parser.parse_args()
    prepare(args.source, args.output)


if __name__ == "__main__":
    main()
