"""Build a portable TRON1 WF inspection scene using only OpenUSD.

The default fixture holds the base for model/joint inspection. It is not a
balancing controller. Pass --free-base to remove the fixture.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET = ROOT / "assets/robots/WF_TRON1A/WF_TRON1A.usd"
JOINT_NAMES = tuple(
    f"{joint}_{side}_Joint" for side in ("L", "R")
    for joint in ("abad", "hip", "knee", "wheel")
)


def build_scene(asset: Path, output: Path, *, free_base: bool = False) -> Path:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

    asset, output = asset.resolve(), output.resolve()
    if not asset.is_file():
        raise FileNotFoundError(f"Run scripts/prepare_assets.py first: {asset}")
    source = Usd.Stage.Open(str(asset))
    if not source or not source.GetDefaultPrim():
        raise ValueError(f"Invalid robot USD: {asset}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(120)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    physics = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(9.81)

    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr(1)
    ground.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
    ground.AddScaleOp().Set(Gf.Vec3f(6, 6, 0.1))
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.21, 0.25)])
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    robot = UsdGeom.Xform.Define(stage, "/World/TRON1")
    try:
        asset_ref = Path(os.path.relpath(asset, output.parent)).as_posix()
    except ValueError:
        # Windows cannot express a relative path across separate drive letters.
        asset_ref = asset.as_posix()
    robot.GetPrim().GetReferences().AddReference(asset_ref)
    robot.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0, 0.966))
    root_link = stage.GetPrimAtPath("/World/TRON1/base_Link")
    if not root_link or not root_link.HasAPI(UsdPhysics.ArticulationRootAPI):
        raise ValueError("Expected TRON1 articulation root at base_Link")

    for name in JOINT_NAMES:
        prim = stage.GetPrimAtPath(f"/World/TRON1/joints/{name}")
        if not prim or not prim.IsA(UsdPhysics.RevoluteJoint):
            raise ValueError(f"Missing expected robot joint: {name}")
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr("force")
        # IsaacLab gains use radians; authored USD angular drives use degrees.
        per_degree = math.pi / 180.0
        drive.CreateStiffnessAttr((0.0 if name.startswith("wheel_") else 40.0) * per_degree)
        drive.CreateDampingAttr((0.8 if name.startswith("wheel_") else 2.5) * per_degree)
        drive.CreateMaxForceAttr(80.0)
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateTargetVelocityAttr(0.0)

    if not free_base:
        fixture = UsdPhysics.FixedJoint.Define(stage, "/World/TRON1/joints/inspection_fixture")
        fixture.CreateBody1Rel().SetTargets([Sdf.Path("/World/TRON1/base_Link")])
        fixture.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0.966))
        fixture.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    stage.GetRootLayer().customLayerData = {
        "robot": "LimX TRON1 WF",
        "inspection_fixture": not free_base,
        "purpose": "Model import and articulation smoke test; no locomotion policy",
    }

    light = UsdLux.DomeLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(900)
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    view = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(2.4, 2.4, 1.7), Gf.Vec3d(0, 0, 0.55), Gf.Vec3d(0, 0, 1)
    )
    camera.AddTransformOp().Set(view.GetInverse())
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
    camera.CreateFocalLengthAttr(32)
    stage.GetRootLayer().Save()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--output", type=Path, default=ROOT / "scenes/tron1_wf.usda")
    parser.add_argument("--free-base", action="store_true")
    args = parser.parse_args()
    print(build_scene(args.asset, args.output, free_base=args.free_base))


if __name__ == "__main__":
    main()
