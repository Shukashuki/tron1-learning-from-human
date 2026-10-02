"""Read CMU ASF/AMC into metric, Z-up human motion (no robot retargeting).

Nodes are the root and ASF bone ENDPOINTS: rfemur is the right knee,
rtibia is the right ankle, and rfoot is the forefoot. Rotations describe the
associated segment's ASF frame, expressed in the output world frame.

Conventions follow CMU's documentation and James McCann's amc_viewer:
column vectors, fixed-axis rotations in declared order, bone C @ R @ C.T,
root Croot @ Rroot, and endpoint = parent_endpoint + Rworld @ bone_vector.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation


SOURCE_TO_WORLD = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
FOOT_NODES = ("lfoot", "ltoes", "rfoot", "rtoes")
ENDPOINT_SEMANTICS = {
    "root": "pelvis/root (not whole-body center of mass)",
    "lhipjoint": "left hip", "rhipjoint": "right hip",
    "lfemur": "left knee", "rfemur": "right knee",
    "ltibia": "left ankle", "rtibia": "right ankle",
    "lfoot": "left forefoot/toe base", "rfoot": "right forefoot/toe base",
    "ltoes": "left toe tip", "rtoes": "right toe tip",
    "lclavicle": "left shoulder", "rclavicle": "right shoulder",
    "lhumerus": "left elbow", "rhumerus": "right elbow",
    "lradius": "left forearm endpoint", "rradius": "right forearm endpoint",
}


@dataclass
class Bone:
    name: str
    direction: np.ndarray
    length_m: float
    axis_angles: np.ndarray
    axis_order: str
    dof: tuple[str, ...]


@dataclass
class Skeleton:
    bones: list[Bone]
    parent_indices: np.ndarray
    root_order: tuple[str, ...]
    root_axis_order: str
    root_position_m: np.ndarray
    root_orientation: np.ndarray
    scale_m_per_unit: float
    degrees: bool

    @property
    def names(self) -> list[str]:
        return ["root", *(bone.name for bone in self.bones)]


def _lines(path: Path):
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if line:
            yield number, line


def _fixed_rotation(axes: str, angles: np.ndarray, *, degrees: bool) -> np.ndarray:
    """Extrinsic XYZ means Rz @ Ry @ Rx for column vectors."""
    if not axes or len(set(axes.lower())) != len(axes) or any(c not in "xyz" for c in axes.lower()):
        raise ValueError(f"Invalid rotation order: {axes}")
    return Rotation.from_euler(axes.lower(), angles, degrees=degrees).as_matrix()


def _axis_rotation(angles_xyz: np.ndarray, order: str, degrees: bool) -> np.ndarray:
    return _fixed_rotation(order, angles_xyz[["XYZ".index(axis) for axis in order.upper()]], degrees=degrees)


def read_asf(path: str | Path) -> Skeleton:
    """Read the fixed-length rotational-bone subset used by CMU ASF files."""
    section, active = "", None
    units, root, definitions, edges = {}, {}, {}, {}
    for number, line in _lines(Path(path)):
        if line.startswith(":"):
            section = line[1:].split()[0].lower()
            continue
        fields = line.split()
        if section == "units":
            units[fields[0].lower()] = fields[1:]
        elif section == "root":
            root[fields[0].lower()] = fields[1:]
        elif section == "bonedata":
            if line == "begin":
                if active is not None:
                    raise ValueError(f"Nested bone at ASF line {number}")
                active = {}
            elif line == "end":
                if active is None or "name" not in active:
                    raise ValueError(f"Incomplete bone at ASF line {number}")
                name = active["name"][0]
                if name == "root" or name in definitions:
                    raise ValueError(f"Duplicate/reserved bone: {name}")
                definitions[name] = active
                active = None
            elif active is not None and fields[0] in {"id", "name", "direction", "length", "axis", "dof"}:
                active[fields[0]] = fields[1:]
        elif section == "hierarchy" and line not in {"begin", "end"}:
            if fields[0] in edges:
                raise ValueError(f"Duplicate hierarchy parent: {fields[0]}")
            edges[fields[0]] = fields[1:]
    if active is not None or not definitions or "root" not in edges:
        raise ValueError("Incomplete ASF bone definitions or hierarchy")
    length_unit = float(units.get("length", ["0"])[0])
    angle_unit = units.get("angle", ["deg"])[0].lower()
    if not np.isfinite(length_unit) or length_unit <= 0 or angle_unit not in {"deg", "rad"}:
        raise ValueError("ASF needs a positive length scale and deg/rad angles")
    # CMU stores lengths as inches multiplied by the ASF length scale.
    scale = 0.0254 / length_unit
    root_order = tuple(x.lower() for x in root.get("order", []))
    if set(root_order) != {"tx", "ty", "tz", "rx", "ry", "rz"} or len(root_order) != 6:
        raise ValueError("CMU root must declare each translation/rotation channel exactly once")

    ordered_names, parents, seen = [], [-1], {"root"}

    def visit(parent_name: str, parent_index: int) -> None:
        for name in edges.get(parent_name, []):
            if name in seen or name not in definitions:
                raise ValueError(f"Repeated, cyclic, or undefined ASF hierarchy node: {name}")
            seen.add(name)
            ordered_names.append(name)
            parents.append(parent_index)
            visit(name, len(ordered_names))

    visit("root", 0)
    if seen != {"root", *definitions} or set(edges) - seen:
        raise ValueError("ASF hierarchy contains disconnected or undefined bones")
    bones = []
    for name in ordered_names:
        value = definitions[name]
        direction = np.asarray(value["direction"], dtype=float)
        length = float(value["length"][0])
        axis = value.get("axis", ["0", "0", "0", "XYZ"])
        dof = tuple(x.lower() for x in value.get("dof", []))
        if len(direction) != 3 or not np.isfinite(direction).all() or not np.isfinite(length) or length < 0:
            raise ValueError(f"Invalid geometry for {name}")
        norm = np.linalg.norm(direction)
        if norm <= 0 and length > 0:
            raise ValueError(f"Nonzero bone length but zero direction for {name}")
        if any(channel not in {"rx", "ry", "rz"} for channel in dof) or len(set(dof)) != len(dof):
            raise ValueError(f"Unsupported nonrotational/repeated bone channels for {name}: {dof}")
        angles = np.asarray(axis[:3], dtype=float)
        if len(axis) != 4 or not np.isfinite(angles).all():
            raise ValueError(f"Invalid bone axis for {name}")
        _axis_rotation(angles, axis[3], angle_unit == "deg")
        bones.append(Bone(name, direction / norm if norm else direction, length * scale, angles, axis[3], dof))
    root_position = np.asarray(root.get("position", [0, 0, 0]), dtype=float) * scale
    root_orientation = np.asarray(root.get("orientation", [0, 0, 0]), dtype=float)
    root_axis = root.get("axis", ["XYZ"])[0]
    if root_position.shape != (3,) or root_orientation.shape != (3,) or not np.isfinite(np.r_[root_position, root_orientation]).all():
        raise ValueError("Invalid ASF root position/orientation")
    _axis_rotation(root_orientation, root_axis, angle_unit == "deg")
    return Skeleton(bones, np.asarray(parents, dtype=np.int64), root_order, root_axis,
                    root_position, root_orientation, scale, angle_unit == "deg")


def read_amc(path: str | Path, skeleton: Skeleton) -> tuple[np.ndarray, dict[str, np.ndarray], bool]:
    """Read motion channels without clipping authored joint-limit excursions."""
    expected = {"root": skeleton.root_order, **{b.name: b.dof for b in skeleton.bones if b.dof}}
    frames, samples, current = [], [], None
    fully_specified, degrees = False, skeleton.degrees
    for number, line in _lines(Path(path)):
        if line.startswith(":"):
            flag = line.upper()
            if flag == ":FULLY-SPECIFIED":
                fully_specified = True
            elif flag in {":DEGREES", ":RADIANS"}:
                degrees = flag == ":DEGREES"
            continue
        if re.fullmatch(r"\d+", line):
            frame = int(line)
            if frames and frame <= frames[-1]:
                raise ValueError("AMC frame numbers must increase strictly")
            frames.append(frame)
            current = {}
            samples.append(current)
            continue
        fields = line.split()
        if current is None or fields[0] not in expected:
            raise ValueError(f"Unexpected AMC channel at line {number}: {fields[0]}")
        name = fields[0]
        values = np.asarray(fields[1:], dtype=float)
        if name in current or len(values) != len(expected[name]) or not np.isfinite(values).all():
            raise ValueError(f"Invalid AMC values at line {number} for {name}")
        current[name] = values
    if not samples:
        raise ValueError("AMC contains no frames")
    previous = {}
    for frame, sample in zip(frames, samples):
        missing = set(expected) - set(sample)
        if missing and (fully_specified or not previous):
            raise ValueError(f"AMC frame {frame} is missing channels: {sorted(missing)}")
        for name in missing:
            sample[name] = previous[name].copy()
        previous = sample
    return np.asarray(frames, dtype=np.int64), {name: np.stack([s[name] for s in samples]) for name in expected}, degrees


def forward_kinematics(skeleton: Skeleton, channels: dict[str, np.ndarray], motion_degrees: bool = True):
    """Return endpoint positions and segment rotations in source Y-up axes."""
    count = len(channels["root"])
    positions = np.empty((count, len(skeleton.names), 3), dtype=float)
    rotations = np.empty((count, len(skeleton.names), 3, 3), dtype=float)
    root_values = channels["root"]
    translation_indices = [skeleton.root_order.index("t" + axis) for axis in "xyz"]
    rotation_indices = [i for i, channel in enumerate(skeleton.root_order) if channel.startswith("r")]
    rotation_order = "".join(skeleton.root_order[i][1] for i in rotation_indices)
    positions[:, 0] = root_values[:, translation_indices] * skeleton.scale_m_per_unit + skeleton.root_position_m
    root_basis = _axis_rotation(skeleton.root_orientation, skeleton.root_axis_order, skeleton.degrees)
    rotations[:, 0] = root_basis @ _fixed_rotation(rotation_order, root_values[:, rotation_indices], degrees=motion_degrees)
    for index, bone in enumerate(skeleton.bones, 1):
        basis = _axis_rotation(bone.axis_angles, bone.axis_order, skeleton.degrees)
        if bone.dof:
            order = "".join(channel[1] for channel in bone.dof)
            values = channels[bone.name]
            motion = _fixed_rotation(order, values, degrees=motion_degrees)
            local = basis @ motion @ basis.T
        else:
            local = np.eye(3)
        parent = skeleton.parent_indices[index]
        rotations[:, index] = rotations[:, parent] @ local
        positions[:, index] = positions[:, parent] + rotations[:, index] @ (bone.direction * bone.length_m)
    return positions, rotations


def load_motion(asf: str | Path, amc: str | Path, fps: float = 120) -> dict:
    """Load a clip with one fixed floor offset and one initial XY translation.

    Array shapes: positions (F,J,3), rotations (F,J,4) in wxyz order; names and
    parent_indices are (J,), frame_numbers and time_s are (F,). Metadata is a
    JSON-serializable dict. FPS is supplied from the dataset catalog because
    AMC does not encode it. Root motion is pelvis motion, not measured COM.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    asf, amc = Path(asf), Path(amc)
    skeleton = read_asf(asf)
    frame_numbers, channels, degrees = read_amc(amc, skeleton)
    source_positions, source_rotations = forward_kinematics(skeleton, channels, degrees)
    positions = source_positions @ SOURCE_TO_WORLD.T
    rotations = SOURCE_TO_WORLD @ source_rotations
    names = skeleton.names
    feet = [names.index(name) for name in FOOT_NODES if name in names]
    floor_nodes = [names[index] for index in feet]
    if not feet:
        feet = list(range(len(names)))
        floor_nodes = names.copy()
    window = min(len(frame_numbers), max(1, round(0.25 * fps)))
    lowest_foot = positions[:, feet, 2].min(axis=1)
    initial_floor = float(np.percentile(lowest_foot[:window], 5))
    global_floor = float(lowest_foot.min())
    # Use the clip minimum only if it agrees with the initial standing baseline
    # within 1 cm. This removes tiny display penetrations without allowing a
    # later crouch or a different support level to redefine the initial floor.
    use_global_floor = initial_floor - global_floor < 0.01
    floor_z = global_floor if use_global_floor else initial_floor
    horizontal_origin = positions[0, 0, :2].copy()
    translation = np.array([-horizontal_origin[0], -horizontal_origin[1], -floor_z])
    positions += translation
    quaternions = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_quat().reshape(len(frame_numbers), len(names), 4)
    quaternions = quaternions[..., [3, 0, 1, 2]]
    for frame in range(1, len(frame_numbers)):
        flip = np.sum(quaternions[frame] * quaternions[frame - 1], axis=-1) < 0
        quaternions[frame, flip] *= -1
    lengths = np.array([0.0, *(bone.length_m for bone in skeleton.bones)])
    measured_lengths = np.linalg.norm(positions[:, 1:] - positions[:, skeleton.parent_indices[1:]], axis=-1)
    max_length_error = float(np.max(np.abs(measured_lengths - lengths[1:])))
    feet_residual = positions[:, feet, 2].min(axis=1)

    def residual_summary(values):
        return {"min_m": float(np.min(values)), "median_m": float(np.median(values)), "max_m": float(np.max(values))}

    metadata = {
        "schema_version": 1,
        "kind": "human_skeletal_reference_not_robot_trajectory",
        "source": {
            "dataset": "CMU Graphics Lab Motion Capture Database",
            "asf_path": str(asf.resolve()), "amc_path": str(amc.resolve()),
            "asf_sha256": hashlib.sha256(asf.read_bytes()).hexdigest(),
            "amc_sha256": hashlib.sha256(amc.read_bytes()).hexdigest(),
            "terms_url": "https://mocap.cs.cmu.edu/",
            "units_documentation_url": "https://mocap.cs.cmu.edu/faqs.php",
        },
        "fps": float(fps), "fps_source": "caller-supplied CMU catalog value; not stored in AMC",
        "frames": len(frame_numbers), "nodes": len(names),
        "duration_samples_s": len(frame_numbers) / fps,
        "timestamp_span_s": float((frame_numbers[-1] - frame_numbers[0]) / fps),
        "source_up_axis": "+Y", "output_up_axis": "+Z",
        "length_scale_m_per_source_unit": skeleton.scale_m_per_unit,
        "source_to_world_rotation": SOURCE_TO_WORLD.tolist(),
        "coordinate_transform": "p_world = A @ p_source_m + constant_translation; (x,y,z)->(z,x,y); det(A)=+1; source +Z forward becomes world +X, source +X left becomes world +Y",
        "constant_translation_m": translation.tolist(),
        "orientation_transform": "R_world = A @ R_source; local ASF segment frames are unchanged; quaternion order wxyz",
        "fk_convention": "extrinsic rotation channels in declared order; root Croot@Rmotion; bones C@Rmotion@C.T; endpoints parent+Rworld@(normalized_direction*length)",
        "root_order": list(skeleton.root_order), "root_axis_order": skeleton.root_axis_order,
        "bone_direction_policy": "normalize rounded ASF direction vectors to preserve declared lengths",
        "node_definition": "root plus bone distal endpoints; rotations belong to the segment ending at the node",
        "endpoint_semantics": {name: meaning for name, meaning in ENDPOINT_SEMANTICS.items() if name in names},
        "root_is_com": False,
        "floor_alignment": {
            "method": "5th percentile of per-frame lowest selected foot endpoint in initial 0.25s; use whole-clip minimum instead only when within 0.01m below that baseline; one constant shift for all frames",
            "nodes": floor_nodes, "window_frames_each_end": window,
            "initial_window_fifth_percentile_m": initial_floor,
            "global_minimum_before_shift_m": global_floor,
            "selected_global_minimum_within_1cm": use_global_floor,
            "estimated_floor_before_shift_m": floor_z,
            "fallback_all_nodes": not any(name in names for name in FOOT_NODES),
            "first_window_lowest_foot_residual": residual_summary(feet_residual[:window]),
            "last_window_lowest_foot_residual": residual_summary(feet_residual[-window:]),
            "whole_clip_lowest_foot_z_m": float(feet_residual.min()),
            "note": "keypoint display floor, not a calibrated support surface; initial window is assumed standing; endpoints are not sole geometry or measured contact forces; different final standing residual is preserved, not corrected",
        },
        "horizontal_alignment": "subtract initial root XY once; horizontal displacement and airborne vertical motion preserved",
        "max_bone_length_error_m": max_length_error,
        "uncaptured_editing_bones": [name for name in ("lfingers", "rfingers", "lthumb", "rthumb") if name in names],
    }
    arrays = {
        "fps": np.asarray(float(fps)), "frame_numbers": frame_numbers,
        "time_s": (frame_numbers - frame_numbers[0]) / fps,
        "joint_names": np.asarray(names), "parent_indices": skeleton.parent_indices,
        "joint_positions_world_m": positions, "joint_rotations_world_wxyz": quaternions,
        "bone_lengths_m": lengths, "source_root_translation_m": source_positions[:, 0].copy(),
    }
    if not all(np.isfinite(value).all() for key, value in arrays.items() if key != "joint_names"):
        raise ValueError("Motion contains nonfinite values")
    return {**arrays, "metadata": metadata}


def save_motion(motion: dict, output_dir: str | Path) -> dict[str, Path]:
    """Write non-pickled NPZ, long-form world-keypoint CSV, and JSON metadata."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {"npz": output / "human_motion.npz", "csv": output / "human_motion.csv", "metadata": output / "human_motion_metadata.json"}
    metadata_json = json.dumps(motion["metadata"], indent=2, ensure_ascii=False)
    np.savez_compressed(paths["npz"], **{k: v for k, v in motion.items() if k != "metadata"}, metadata_json=np.asarray(metadata_json))
    with paths["csv"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_number", "time_s", "joint_name", "parent_index", "x_m", "y_m", "z_m", "qw", "qx", "qy", "qz"])
        for frame, number in enumerate(motion["frame_numbers"]):
            for joint, name in enumerate(motion["joint_names"]):
                writer.writerow([number, motion["time_s"][frame], name, motion["parent_indices"][joint],
                                 *motion["joint_positions_world_m"][frame, joint], *motion["joint_rotations_world_wxyz"][frame, joint]])
    paths["metadata"].write_text(metadata_json + "\n", encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asf", type=Path, required=True)
    parser.add_argument("--amc", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=120)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    motion = load_motion(args.asf, args.amc, args.fps)
    paths = save_motion(motion, args.output_dir)
    print(json.dumps({"frames": len(motion["frame_numbers"]), "nodes": len(motion["joint_names"]),
                      "outputs": {key: str(path.resolve()) for key, path in paths.items()}}, indent=2))


if __name__ == "__main__":
    main()
