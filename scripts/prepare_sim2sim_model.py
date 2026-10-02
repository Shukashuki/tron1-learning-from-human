"""Derive a MuJoCo dynamics model aligned to the pinned Isaac USD parameters.

Never modifies the original model. Contact solvers remain intentionally named
as a residual difference, not claimed equivalent across PhysX and MuJoCo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim2sim_common import ROOT, load_config


def numbers(values):
    return " ".join(f"{float(x):.12g}" for x in values)


def prepare_model(output, config=None):
    config = load_config() if config is None else config
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    base_dir = ROOT / "assets/robots/WF_TRON1A/mujoco"
    tree = ET.parse(base_dir / "robot.xml")
    root = tree.getroot()
    root.find("compiler").set("meshdir", str(base_dir / "meshes"))
    option = root.find("option")
    option.set("timestep", str(config["physics_dt"]))
    option.set("gravity", "0 0 -9.81")
    option.set("integrator", "implicitfast")
    option.set("solver", "Newton")
    option.set("iterations", "100")
    option.set("tolerance", "1e-10")
    default = root.find("default")
    joint = default.find("joint")
    joint.set("damping", str(config["joint_passive_damping"]))
    joint.set("armature", str(config["joint_armature"]))
    for joint in root.findall(".//joint"):
        if "frictionloss" in joint.attrib:
            joint.set("frictionloss", str(config["joint_friction"]))
    geom = default.find("geom")
    # Robot-ground only, consistent with Isaac's disabled self-collision.
    geom.set("contype", "1")
    geom.set("conaffinity", "2")
    geom.set("friction", f'{config["ground_friction"]} 0 0')
    geom.set("margin", "0")
    geom.set("solref", "0.004 1")
    floor = root.find("worldbody/geom[@name='floor']")
    floor.set("contype", "2")
    floor.set("conaffinity", "1")
    bodies = {body.get("name"): body for body in root.iter("body")}
    inertias = json.loads((ROOT / "config/balance_wf.json").read_text())
    for entry in inertias["bodies"]:
        name = entry["link_name"]
        if name not in bodies:
            if name != "limx_imu":
                raise ValueError(f"Unexpected USD body {name}")
            bodies[name] = ET.SubElement(bodies["base_Link"], "body", name=name,
                                         pos=numbers(entry["origin_in_base"]))
            ET.SubElement(bodies[name], "inertial")
        inertial = bodies[name].find("inertial")
        inertial.set("mass", str(entry["mass"]))
        inertial.set("pos", numbers(entry["com_local"]))
        inertial.set("diaginertia", numbers(entry["diagonal_inertia"]))
        inertial.set("quat", numbers(entry["principal_axes_wxyz"]))
    # Values read from the pinned composed USD. USD Cylinder length is full
    # height; MJCF cylinder size[1] is HALF height; both use local Z as axis.
    collision_spec = {
        "base_collision": ([.135, .13, .095], [.03, 0., -.072], [1., 0., 0., 0.]),
    }
    for side, sign in (("L", 1.), ("R", -1.)):
        collision_spec.update({
            f"abad_{side}_collision": ([.05, .025], [.03, 0., 0.], [.5003980994, .4999998212, .4996018708, .4999998212]),
            f"hip_{side}_collision": ([.035, .075], [-.1, -.03 * sign, -.14], [.9650924802, 0., .2619092464, 0.]),
            f"knee_{side}_collision": ([.015, .13], [.078, 0., -.12], [.9624251723, 0., -.27154693, 0.]),
            f"wheel_{side}_collision": ([.127, .025], [0., .00707 * sign, 0.], [.7071067691, .7071067691, 0., 0.]),
        })
    for name, (size, pos, quat) in collision_spec.items():
        geom = root.find(f".//geom[@name='{name}']")
        geom.attrib.pop("euler", None)
        geom.set("size", numbers(size))
        geom.set("pos", numbers(pos))
        quat = np.asarray(quat) / np.linalg.norm(quat)
        geom.set("quat", numbers(quat))
    for motor in root.findall("actuator/motor"):
        limit = config["wheel_torque_limit_nm"] if motor.get("joint").startswith("wheel_") else config["leg_torque_limit_nm"]
        motor.set("ctrlrange", f"{-limit} {limit}")
    ET.indent(root, space="  ")
    text = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(text)
    expected_mass = sum(entry["mass"] for entry in inertias["bodies"])
    np.testing.assert_allclose(model.body_mass.sum(), expected_mass, rtol=1e-10)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n")
    metadata = {
        "kind": "USD-aligned MuJoCo dynamics diagnostic model", "total_mass_kg": float(model.body_mass.sum()),
        "aligned": ["USD masses, local COMs and principal inertias including fixed IMU",
                    "USD collision primitive dimensions and local transforms",
                    "gravity, physics timestep, actuator limits, joint passive damping/friction/armature",
                    "ground friction 0.6; self-collisions disabled; identical explicit common torque controller"],
        "residual_differences": ["PhysX vs MuJoCo contact/constraint solver and cylinder contact representation",
                                 "PhysX contactOffset .002/restOffset 0 vs MuJoCo solref .004 1/margin 0",
                                 "USD float precision and joint local-transform rounding",
                                 "PhysX finite ground box vs MuJoCo infinite plane (all trajectories inside box)"]}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/sim2sim-cmu-16_03/mujoco_model.xml")
    args = parser.parse_args()
    print(prepare_model(args.output))
