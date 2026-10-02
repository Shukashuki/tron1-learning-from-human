"""Prepare a pinned official TRON1 WF MuJoCo model for Mink IK visualization.

Only the separate mujoco output directory is written. Original official assets,
license, exact Git blob hashes, and SHA256 provenance are retained. No USD or
Isaac runtime is modified. Run with the Python environment containing MuJoCo.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request
import xml.etree.ElementTree as ET


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = PROJECT / "config/mink_model.json"


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def _verify(data: bytes, entry: dict) -> None:
    if len(data) != entry["size"] or _git_blob_sha1(data) != entry["git_blob_sha1"]:
        raise ValueError(f"Pinned official asset size/hash mismatch: {entry['source']}")


def _write_checked(path: Path, data: bytes, force: bool) -> None:
    if path.exists():
        if path.read_bytes() == data:
            return
        if not force:
            raise FileExistsError(f"Refusing to replace differing file: {path}; use --force explicitly")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        stream.write(data)
        temporary = Path(stream.name)
    temporary.replace(path)


def _prepare_file(entry: dict, lock: dict, output: Path, source_dir: Path | None, force: bool) -> dict:
    destination = output / entry["destination"]
    if destination.exists():
        content = destination.read_bytes()
        try:
            _verify(content, entry)
        except ValueError:
            if not force:
                raise FileExistsError(f"Existing asset differs from locked source: {destination}; use --force explicitly")
            content = None
    else:
        content = None
    url = f"https://raw.githubusercontent.com/limxdynamics/tron1-robot-description/{lock['revision']}/{entry['source']}"
    if content is None:
        if source_dir:
            content = (source_dir / entry["source"]).read_bytes()
        else:
            request = urllib.request.Request(url, headers={"User-Agent": "tron1-learning-from-human-asset-preparation"})
            with urllib.request.urlopen(request, timeout=60) as response:
                content = response.read(entry["size"] + 1)
        _verify(content, entry)
        _write_checked(destination, content, force)
    return {**entry, "url": url, "sha256": hashlib.sha256(content).hexdigest()}


def _build_model(lock: dict, output: Path, force: bool) -> None:
    root = ET.fromstring((output / "upstream/robot.xml").read_bytes())
    root.find("compiler").set("meshdir", "meshes")
    base = root.find("worldbody/body[@name='base_Link']")
    base.find("joint[@type='free']").set("name", "root")
    bodies = {body.get("name"): body for body in root.iter("body")}
    for name, item in lock["sites"].items():
        ET.SubElement(bodies[item["body"]], "site", name=name,
                      pos=" ".join(str(x) for x in item["pos"]),
                      size="0.008", rgba="0.15 0.85 0.35 0.8", group="4")
    ET.indent(root, space="    ")
    contents = ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"
    _write_checked(output / "robot.xml", contents, force)


def validate_model(model_path: Path, lock: dict) -> dict:
    """Validate compiled geometry and expose zero/seed FK for the IK consumer."""
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    actual_joints = [model.joint(i).name for i in range(1, model.njnt)]
    if (model.nq, model.nv, model.njnt) != (15, 14, 9) or actual_joints != lock["joint_order"]:
        raise ValueError(f"Unexpected model DOFs or order: {(model.nq, model.nv, model.njnt)}, {actual_joints}")
    np.testing.assert_allclose(model.jnt_axis[1:], lock["joint_axes"], atol=1e-12)
    np.testing.assert_allclose(model.jnt_range[1:], lock["joint_ranges_rad"], atol=1e-12)
    if model.nmesh != 9 or not np.isfinite(model.mesh_vert).all():
        raise ValueError("Expected nine finite official visual meshes")
    for side in ("L", "R"):
        geom = model.geom(f"wheel_{side}_collision")
        np.testing.assert_allclose(geom.size[:2], [lock["wheel_radius_m"], 0.005], atol=1e-12)
    for name in lock["sites"]:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) < 0:
            raise ValueError(f"Missing IK site: {name}")
    # A separate analytic FK implementation from the official URDF catches
    # joint-axis signs, order, and parent-transform mistakes at nonzero angles.
    urdf_joints = ET.parse(model_path.parent / "upstream/robot.urdf").getroot().findall("joint")

    def axis_rotation(axis, angle):
        x, y, z = axis
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)

    max_fk_position_error = 0.0
    max_fk_rotation_error = 0.0
    rng = np.random.default_rng(20261001)
    for _ in range(25):
        values = rng.uniform(-0.25, 0.25, 8)
        base_position = rng.uniform(-0.2, 0.2, 3)
        base_axis = rng.normal(size=3)
        base_axis /= np.linalg.norm(base_axis)
        base_angle = rng.uniform(-0.4, 0.4)
        base_rotation = axis_rotation(base_axis, base_angle)
        data.qpos[:3] = base_position
        data.qpos[3:7] = np.r_[np.cos(base_angle / 2), base_axis * np.sin(base_angle / 2)]
        data.qpos[7:] = values
        mujoco.mj_forward(model, data)
        poses = {"base_Link": (base_position, base_rotation)}
        joint_values = dict(zip(lock["joint_order"], values))
        pending = list(urdf_joints)
        while pending:
            available = [j for j in pending if j.find("parent").get("link") in poses]
            if not available:
                raise ValueError("Disconnected official URDF hierarchy")
            for joint in available:
                origin = joint.find("origin")
                translation = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                roll, pitch, yaw = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                origin_rotation = axis_rotation([0, 0, 1], yaw) @ axis_rotation([0, 1, 0], pitch) @ axis_rotation([1, 0, 0], roll)
                parent_position, parent_rotation = poses[joint.find("parent").get("link")]
                rotation = parent_rotation @ origin_rotation
                if joint.get("type") != "fixed":
                    axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                    rotation = rotation @ axis_rotation(axis, joint_values[joint.get("name")])
                child_name = joint.find("child").get("link")
                position = parent_position + parent_rotation @ translation
                poses[child_name] = (position, rotation)
                child_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, child_name)
                if child_id >= 0:
                    max_fk_position_error = max(max_fk_position_error, float(np.max(np.abs(position - data.xpos[child_id]))))
                    max_fk_rotation_error = max(max_fk_rotation_error, float(np.max(np.abs(rotation - data.xmat[child_id].reshape(3, 3)))))
                pending.remove(joint)
    if max(max_fk_position_error, max_fk_rotation_error) > 1e-10:
        raise ValueError(f"Official URDF/MJCF FK mismatch: {max_fk_position_error}, {max_fk_rotation_error}")
    data.qpos[:] = model.qpos0
    data.qpos[:3] = 0
    data.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    zero_sites = {name: data.site(name).xpos.tolist() for name in lock["sites"]}
    expected_wheels = {"left_wheel": [-0.02144, 0.1485, -0.77982],
                       "right_wheel": [-0.02144, -0.1485, -0.77982]}
    for name, expected in expected_wheels.items():
        np.testing.assert_allclose(zero_sites[name], expected, atol=1e-12)
    zero_body_positions = {model.body(i).name: data.xpos[i].tolist() for i in range(1, model.nbody)}
    seed_degrees = [0, 20, 40, 0, 0, -20, -40, 0]
    data.qpos[7:] = np.deg2rad(seed_degrees)
    mujoco.mj_forward(model, data)
    seed_sites = {name: data.site(name).xpos.tolist() for name in lock["sites"]}
    # For this symmetric seed, the wheel spin axes are horizontal, so this
    # radius calculation is exact. It is not a generic tilted-cylinder test.
    seed_base_height = lock["wheel_radius_m"] - min(seed_sites[n][2] for n in expected_wheels)
    data.qpos[2] = seed_base_height
    mujoco.mj_forward(model, data)
    contacts = [{"geom1": model.geom(c.geom1).name, "geom2": model.geom(c.geom2).name,
                 "distance_m": float(c.dist)} for c in data.contact[:data.ncon]]
    return {
        "mujoco_version": mujoco.__version__, "nq": model.nq, "nv": model.nv,
        "hinge_joint_order": actual_joints, "qpos_hinge_addresses": model.jnt_qposadr[1:].tolist(),
        "dof_hinge_addresses": model.jnt_dofadr[1:].tolist(),
        "joint_ranges_rad": model.jnt_range[1:].tolist(),
        "body_names": [model.body(i).name for i in range(model.nbody)],
        "sites": lock["sites"], "mass_kg": float(model.body_mass.sum()),
        "visual_meshes": model.nmesh, "mesh_vertices": model.nmeshvert,
        "independent_urdf_fk_random_poses": 25,
        "independent_urdf_fk_max_position_error_m": max_fk_position_error,
        "independent_urdf_fk_max_rotation_matrix_error": max_fk_rotation_error,
        "wheel_radius_m": lock["wheel_radius_m"], "wheel_collision_full_width_m": 0.01,
        "zero_joint_pose_body_positions_relative_to_base_m": zero_body_positions,
        "zero_joint_pose_sites_relative_to_base_m": zero_sites,
        "zero_joint_pose_contact_base_height_m": 0.77982 + lock["wheel_radius_m"],
        "seed_joint_degrees": seed_degrees,
        "seed_sites_relative_to_base_m": seed_sites,
        "seed_level_base_height_for_grounded_wheels_m": seed_base_height,
        "seed_ground_contacts": contacts,
        "note": "Geometric/FK checks only. Contacts include margin contacts and do not prove dynamically feasible balance or jumping.",
    }


def prepare(lock_path: Path = DEFAULT_LOCK, output_dir: Path | None = None,
            source_dir: Path | None = None, force: bool = False) -> Path:
    lock = json.loads(lock_path.read_text())
    output = output_dir or PROJECT / lock["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        entries = list(pool.map(lambda entry: _prepare_file(entry, lock, output, source_dir, force), lock["files"]))
    _build_model(lock, output, force)
    validation = validate_model(output / "robot.xml", lock)
    provenance = {
        "schema_version": 1, "repository": lock["repository"], "revision": lock["revision"],
        "license": lock["license"], "files": entries,
        "derived_model": "robot.xml", "derived_model_sha256": hashlib.sha256((output / "robot.xml").read_bytes()).hexdigest(),
        "changes_to_upstream_mjcf": ["compiler meshdir ../meshes/ -> meshes", "name existing free joint root", "add nine IK/display sites"],
        "notes": lock["notes"], "validation": validation,
    }
    _write_checked(output / "SOURCE.json", (json.dumps(provenance, indent=2) + "\n").encode(), force)
    print(json.dumps({"model": str((output / "robot.xml").resolve()), "validation": validation}, indent=2))
    return output / "robot.xml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-dir", type=Path, help="Optional local checkout of the pinned official description repo")
    parser.add_argument("--force", action="store_true", help="Explicitly replace differing files within the model output directory")
    args = parser.parse_args()
    prepare(args.lock, args.output_dir, args.source_dir, args.force)


if __name__ == "__main__":
    main()
