#!/usr/bin/env python3
"""Fetch and verify the pinned official TRON1 wheel-foot USD asset.

Downloading uses only the standard library. Optional structural validation uses
OpenUSD (``pip install usd-core==25.5.1``), or Isaac Sim's bundled Python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import urllib.error
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "config" / "assets.lock.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "assets" / "robots" / "WF_TRON1A"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matches(path: Path, entry: dict) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == entry["size"]
        and sha256(path) == entry["sha256"]
    )


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def validate_usd(entrypoint: Path, manifest: dict) -> dict:
    try:
        from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils
    except ImportError as exc:
        raise RuntimeError(
            "--validate requires OpenUSD. Use Isaac Sim's Python or install "
            "usd-core==25.5.1 in a dedicated virtual environment."
        ) from exc

    stage = Usd.Stage.Open(str(entrypoint))
    if stage is None:
        raise RuntimeError(f"Cannot open USD stage: {entrypoint}")
    errors = stage.GetCompositionErrors()
    if errors:
        raise RuntimeError(f"USD composition errors: {errors}")
    default_prim = stage.GetDefaultPrim()
    if str(default_prim.GetPath()) != manifest["default_prim"]:
        raise RuntimeError(f"Unexpected default prim: {default_prim.GetPath()}")
    if UsdGeom.GetStageUpAxis(stage) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise RuntimeError("Expected a Z-up stage with meter units")

    # OmniPBR is provided by Isaac Sim, so standalone usd-core may warn that it
    # cannot resolve the MDL shader. Any other missing dependency is an error.
    _, _, unresolved = UsdUtils.ComputeAllDependencies(str(entrypoint))
    simulator_dependencies = set(manifest["simulator_provided_dependencies"])
    missing = set(unresolved) - simulator_dependencies
    if missing:
        raise RuntimeError(f"Unresolved USD dependencies: {sorted(missing)}")

    prims = list(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
    roots = [str(prim.GetPath()) for prim in prims if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if roots != [manifest["articulation_root"]]:
        raise RuntimeError(f"Unexpected articulation roots: {roots}")
    joint_names = [prim.GetName() for prim in prims if prim.IsA(UsdPhysics.RevoluteJoint)]
    expected = manifest["revolute_joints"]
    if len(joint_names) != len(expected) or set(joint_names) != set(expected):
        raise RuntimeError(f"Unexpected movable joints: {joint_names}")
    bodies = [prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    masses = [UsdPhysics.MassAPI(prim).GetMassAttr().Get() for prim in bodies]
    if not masses or any(mass is None or not math.isfinite(mass) or mass <= 0 for mass in masses):
        raise RuntimeError("Every rigid body must have a positive finite mass")
    meshes = [UsdGeom.Mesh(prim) for prim in prims if prim.IsA(UsdGeom.Mesh)]
    if not meshes or any(not mesh.GetPointsAttr().Get() for mesh in meshes):
        raise RuntimeError("Expected nonempty mesh geometry")
    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    ).ComputeWorldBound(default_prim).ComputeAlignedRange()
    transform = UsdGeom.Xformable(default_prim).GetLocalTransformation()
    return {
        "stage": str(entrypoint),
        "default_prim": str(default_prim.GetPath()),
        "articulation_root": roots[0],
        "revolute_joints": joint_names,
        "rigid_bodies": len(bodies),
        "total_mass_kg": sum(masses),
        "meshes": len(meshes),
        "up_axis": "Z",
        "meters_per_unit": 1.0,
        "root_transform": [list(row) for row in transform],
        "bounds_min": list(bounds.GetMin()),
        "bounds_max": list(bounds.GetMax()),
        "simulator_provided_dependencies": sorted(set(unresolved) & simulator_dependencies),
    }


def prepare(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise RuntimeError("Unsupported asset manifest version")
    output = args.output_dir.resolve()
    provenance = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    provenance_path = output / "SOURCE.json"
    pending = []
    # Check all existing files before changing any of them.
    for entry in manifest["files"]:
        relative = Path(entry["destination"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Unsafe destination in manifest: {relative}")
        target = output / relative
        if matches(target, entry):
            continue
        if target.exists() and not args.force:
            raise RuntimeError(f"Existing asset differs from pinned checksum: {target}. Use --force to replace it.")
        if args.check_only:
            raise RuntimeError(f"Missing or modified pinned asset: {target}")
        pending.append((entry, target))
    if provenance_path.exists() and provenance_path.read_bytes() != provenance and not args.force:
        raise RuntimeError(f"Existing provenance differs: {provenance_path}. Use --force to replace it.")
    if args.check_only and not provenance_path.is_file():
        raise RuntimeError(f"Missing provenance: {provenance_path}")

    for entry, target in pending:
        if args.source_dir is not None:
            content = (args.source_dir / entry["source"]).read_bytes()
        else:
            url = manifest["raw_base_url"] + entry["source"]
            request = urllib.request.Request(url, headers={"User-Agent": "tron1-asset-preparation/1"})
            with urllib.request.urlopen(request, timeout=120) as response:
                content = response.read()
        if len(content) != entry["size"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Downloaded/source asset failed checksum: {entry['source']}")
        atomic_write(target, content)
        print(f"Prepared {target}")
    if not args.check_only and (not provenance_path.exists() or provenance_path.read_bytes() != provenance):
        atomic_write(provenance_path, provenance)

    entrypoint = output / manifest["entrypoint"]
    print(f"Verified {len(manifest['files'])} pinned files at {manifest['revision']}")
    print(f"Model: {entrypoint}")
    if args.validate:
        print(json.dumps(validate_usd(entrypoint, manifest), indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-dir", type=Path, help="Copy from an existing official repository checkout instead of downloading")
    parser.add_argument("--validate", action="store_true", help="Validate USD layers, articulation, masses, and meshes with pxr")
    parser.add_argument("--check-only", action="store_true", help="Check existing files without writes or downloads")
    parser.add_argument("--force", action="store_true", help="Replace existing files even when their checksums differ")
    args = parser.parse_args()
    if args.check_only and args.force:
        parser.error("--check-only and --force cannot be combined")
    try:
        prepare(args)
    except (OSError, ValueError, KeyError, RuntimeError, urllib.error.URLError) as exc:
        print(f"Asset preparation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
