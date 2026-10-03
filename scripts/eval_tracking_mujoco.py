"""Deploy the normalized 51-input TRON1 BeyondMimic actor in MuJoCo.

One deterministic episode starts at reference frame zero. The floating root is
initialized once (including reference COM velocities), then evolves exclusively
through physics. Policy actions run at 50 Hz; explicit joint PD runs at 200 Hz.
No policy normalization is added here: it MUST be inside the TorchScript actor.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import time
import warnings
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from eval_tracking import contact_events

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "outputs/sim2sim-cmu-16_03/mujoco_model.xml"
TASK_SOURCE = ROOT / "training/tron1_tracking.py"
ROOT_COM_LINK = np.array([0.04576, 0.00014, -0.16398])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_contract_path(contract_path=None, policy_path=None):
    """Prefer an explicitly saved or actor-adjacent contract over current code."""
    if contract_path is not None:
        return Path(contract_path)
    if policy_path is not None:
        adjacent = Path(policy_path).parent / "policy_contract.json"
        if adjacent.is_file():
            return adjacent
        warnings.warn("No saved policy contract selected/found; using CURRENT task physics. "
                      "For an older actor pass --contract with its original contract.", stacklevel=2)
    return TASK_SOURCE


def read_contract(path=TASK_SOURCE):
    """Read a saved JSON contract/manifest or literal task source without Isaac.

    Deliberately supports only literal containers and the named constants used
    by observation_action_contract. No arbitrary task code is executed.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        contract = json.loads(path.read_text())
        if isinstance(contract, dict) and "contract" in contract:
            contract = contract["contract"]
        return validate_contract(contract)
    tree = ast.parse(path.read_text())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("LEG_JOINT_NAMES", "WHEEL_JOINT_NAMES", "BODY_NAMES", "POLICY_FPS", "LEG_ACTION_SCALE"):
                constants[name] = ast.literal_eval(node.value)
    constants["ACTION_JOINT_NAMES"] = constants["LEG_JOINT_NAMES"] + constants["WHEEL_JOINT_NAMES"]
    constants["NON_WHEEL_BODY_NAMES"] = [name for name in constants["BODY_NAMES"] if not name.startswith("wheel_")]
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "observation_action_contract")
    value = next(node.value for node in function.body if isinstance(node, ast.Return))

    def literal(node):
        if isinstance(node, ast.Name):
            return constants[node.id]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [literal(item) for item in node.elts]
        if isinstance(node, ast.Dict):
            return {literal(key): literal(item) for key, item in zip(node.keys, node.values)}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -literal(node.operand)
        raise ValueError(f"Unsupported nonliteral deployment contract expression: {ast.dump(node)}")

    return validate_contract(literal(value))


def validate_contract(contract):
    """Fail closed on incompatible dimensions, unknown motors and invalid limits."""
    if not isinstance(contract, dict):
        raise ValueError("Deployment contract must be a JSON object")
    required = ("actor_dim", "action_dim", "physics_dt", "decimation", "leg_kp", "leg_kd",
                "leg_position_scale", "wheel_torque_scale_nm", "leg_torque_limit_nm", "wheel_torque_limit_nm")
    if any(name not in contract for name in required):
        raise ValueError("Incomplete deployment contract")
    if (contract["actor_dim"], contract["action_dim"], contract["physics_dt"], contract["decimation"]) != (51, 8, .005, 4):
        raise ValueError("This evaluator requires the inspected 51->8, 200 Hz / 4 deployment contract")
    actuator_model = contract.get("actuator_model", "ideal_pd")
    if actuator_model not in ("ideal_pd", "dc_motor"):
        raise ValueError(f"Unknown actuator_model: {actuator_model!r}")
    positive = ["leg_position_scale", "wheel_torque_scale_nm", "leg_torque_limit_nm", "wheel_torque_limit_nm"]
    if actuator_model == "dc_motor":
        positive += ["leg_saturation_effort_nm", "wheel_saturation_effort_nm",
                     "leg_motor_velocity_limit_rad_s", "wheel_motor_velocity_limit_rad_s",
                     "solver_joint_velocity_limit_rad_s"]
    for name in positive + ["leg_kp", "leg_kd"]:
        value = contract.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
            raise ValueError(f"Contract {name} must be finite")
        if value < 0 or (name in positive and value == 0):
            raise ValueError(f"Contract {name} is outside its valid range")
    clip = np.asarray(contract.get("raw_action_clip", []), dtype=float)
    if clip.shape != (2,) or not np.isfinite(clip).all() or clip[0] >= clip[1]:
        raise ValueError("Contract raw_action_clip must have two finite increasing bounds")
    return contract


def rotation(quat_wxyz):
    q = np.asarray(quat_wxyz, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Expected a finite, nonzero wxyz quaternion")
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


class MotionReference:
    def __init__(self, path, contract):
        self.path = Path(path).resolve()
        self.contract = contract
        with np.load(self.path, allow_pickle=False) as data:
            self.archive = {key: data[key].copy() for key in data.files}
        d = self.archive
        self.joint_names = d["joint_names"].astype(str).tolist()
        self.body_names = d["body_names"].astype(str).tolist()
        self.action_names = contract["observed_joint_velocity_order"]
        if len(set(self.joint_names)) != 8 or set(self.joint_names) != set(self.action_names):
            raise ValueError("Reference joint names must map exactly to the eight action joints")
        if len(set(self.body_names)) != len(self.body_names) or self.body_names[0] != "base_Link":
            raise ValueError("Reference body names must be unique with base_Link first")
        self.order = [self.joint_names.index(name) for name in self.action_names]
        self.wheel_bodies = [self.body_names.index(name) for name in ("wheel_L_Link", "wheel_R_Link")]
        self.fps = float(np.asarray(d["fps"]).reshape(-1)[0])
        self.frames = len(d["joint_pos"])
        if self.frames < 2 or not np.isclose(self.fps, contract["policy_fps"]):
            raise ValueError("Motion must have at least two frames at the policy's 50 Hz")
        shapes = {"joint_pos": (self.frames, 8), "joint_vel": (self.frames, 8),
                  "body_pos_w": (self.frames, len(self.body_names), 3),
                  "body_quat_w": (self.frames, len(self.body_names), 4),
                  "body_lin_vel_w": (self.frames, len(self.body_names), 3),
                  "body_ang_vel_w": (self.frames, len(self.body_names), 3)}
        for key, shape in shapes.items():
            if d[key].shape != shape or not np.isfinite(d[key]).all():
                raise ValueError(f"Nonfinite/malformed reference field {key}: {d[key].shape}, expected {shape}")
        if not np.allclose(np.linalg.norm(d["body_quat_w"], axis=-1), 1., atol=1e-5):
            raise ValueError("Reference quaternions must be normalized")
        self.q0 = d["joint_pos"][0, self.order].astype(float)
        self.duration = self.frames / self.fps

    def sample(self, frame):
        frame = int(frame)
        if not 0 <= frame < self.frames:
            raise IndexError(frame)
        d = self.archive
        return {"joint_pos": d["joint_pos"][frame, self.order],
                "joint_vel": d["joint_vel"][frame, self.order],
                "root_pos": d["body_pos_w"][frame, 0],
                "root_quat_wxyz": d["body_quat_w"][frame, 0],
                "root_com_lin_vel_w": d["body_lin_vel_w"][frame, 0],
                "root_ang_vel_w": d["body_ang_vel_w"][frame, 0],
                "wheel_pos": d["body_pos_w"][frame, self.wheel_bodies],
                "phase": 2 * np.pi * frame / (self.frames - 1)}


def build_observation(state, target, q0, previous_action):
    """Official anchor transforms; rotations use first TWO COLUMNS, row-major."""
    r = rotation(state["root_quat_wxyz"])
    relative = r.T @ rotation(target["root_quat_wxyz"])
    obs = np.r_[target["joint_pos"][:6], target["joint_vel"][:6],
                np.sin(target["phase"]), np.cos(target["phase"]),
                r.T @ (target["root_pos"] - state["root_pos"]), relative[:, :2].reshape(-1),
                r.T @ state["root_com_lin_vel_w"], r.T @ state["root_ang_vel_w"],
                state["joint_pos"][:6] - q0[:6], state["joint_vel"], previous_action]
    if obs.shape != (51,) or not np.isfinite(obs).all():
        raise FloatingPointError(f"Expected a finite 51D actor observation, got {obs.shape}")
    return obs.astype(np.float32)


def dc_motor_torque_bounds(joint_vel, saturation_effort, effort_limit, velocity_limit):
    """IsaacLab 2.3 DCMotor._clip_effort envelope; never modify actual qvel.

    The clipped local velocity only bounds the electrical braking envelope.
    Beyond no-load speed the admissible torque can be entirely braking torque.
    """
    dq, stall, effort, speed = np.broadcast_arrays(
        np.asarray(joint_vel, dtype=float), np.asarray(saturation_effort, dtype=float),
        np.asarray(effort_limit, dtype=float), np.asarray(velocity_limit, dtype=float))
    if not all(np.isfinite(value).all() for value in (dq, stall, effort, speed)):
        raise ValueError("DC motor inputs and parameters must be finite")
    if any(np.any(value <= 0) for value in (stall, effort, speed)):
        raise ValueError("DC motor stall torque, effort and no-load speed must be positive")
    velocity_at_effort_limit = speed * (1.0 + effort / stall)
    curve_velocity = np.clip(dq, -velocity_at_effort_limit, velocity_at_effort_limit)
    upper = np.minimum(stall * (1.0 - curve_velocity / speed), effort)
    lower = np.maximum(stall * (-1.0 - curve_velocity / speed), -effort)
    return lower, upper


def action_torques(action, joint_pos, joint_vel, q0, contract):
    validate_contract(contract)
    action = np.asarray(action, dtype=float).reshape(-1)
    if action.shape != (8,) or not np.isfinite(action).all():
        raise FloatingPointError("Policy must produce exactly eight finite actions")
    joint_pos, joint_vel, q0 = (np.asarray(value, dtype=float) for value in (joint_pos, joint_vel, q0))
    if any(value.shape != (8,) or not np.isfinite(value).all() for value in (joint_pos, joint_vel, q0)):
        raise FloatingPointError("Joint positions, velocities and initial positions must be finite 8-vectors")
    action = np.clip(action, *contract["raw_action_clip"])
    target = q0[:6] + contract["leg_position_scale"] * action[:6]
    torque = np.empty(8)
    torque[:6] = contract["leg_kp"] * (target - joint_pos[:6]) - contract["leg_kd"] * joint_vel[:6]
    torque[6:] = contract["wheel_torque_scale_nm"] * action[6:]
    effort = np.r_[np.full(6, contract["leg_torque_limit_nm"]), np.full(2, contract["wheel_torque_limit_nm"])]
    if contract.get("actuator_model", "ideal_pd") == "dc_motor":
        stall = np.r_[np.full(6, contract["leg_saturation_effort_nm"]), np.full(2, contract["wheel_saturation_effort_nm"])]
        speed = np.r_[np.full(6, contract["leg_motor_velocity_limit_rad_s"]), np.full(2, contract["wheel_motor_velocity_limit_rad_s"])]
        lower, upper = dc_motor_torque_bounds(joint_vel, stall, effort, speed)
        return np.clip(torque, lower, upper)
    return np.clip(torque, -effort, effort)


def training_termination_terms(state, target):
    """Checked AFTER decimation, against the held pre-update reference frame.

    Upstream anchor_ori compares projected-gravity Z, NOT an angular threshold.
    Relative wheel re-anchoring rotates yaw only, so its Z is the source Z.
    """
    terms = []
    if abs(target["root_pos"][2] - state["root_pos"][2]) > .35:
        terms.append("anchor_pos")
    if abs(rotation(target["root_quat_wxyz"])[2, 2]
           - rotation(state["root_quat_wxyz"])[2, 2]) > .6:
        terms.append("anchor_ori")
    if np.any(np.abs(target["wheel_pos"][:, 2] - state["wheel_pos"][:, 2]) > .25):
        terms.append("ee_body_pos")
    return terms


def configure_model(model, contract):
    """Only in-memory overrides: never overwrite the USD-matched MJCF."""
    model.opt.timestep = contract["physics_dt"]
    if (model.nq, model.nv, model.nu) != (15, 14, 8):
        raise ValueError("Expected floating TRON1 WF with eight motor actuators")
    # Preserve MuJoCo parent/weld/exclude collision filters. Enable nonvisual
    # robot geoms against each other, matching training's self_collision=True.
    changed = []
    for index in range(model.ngeom):
        if model.geom_bodyid[index] and (model.geom_contype[index] or model.geom_conaffinity[index]):
            model.geom_contype[index], model.geom_conaffinity[index] = 1, 3
            changed.append(model.geom(index).name)
    model.dof_damping[:] = 0.
    model.dof_armature[:] = 0.
    model.dof_frictionloss[:] = 0.
    return changed


def load_model(path, no_visual_mesh=False):
    """Optionally remove only proven non-colliding visual meshes in memory.

    Explicit inertials are required on every robot body, so visual removal
    cannot silently change inferred mass/inertia. Collision mesh assets remain.
    """
    import mujoco
    if not no_visual_mesh:
        return mujoco.MjModel.from_xml_path(str(path)), 0
    root = ET.parse(path).getroot()
    if any(body.find("inertial") is None for body in root.iter("body")):
        raise ValueError("Mesh-free evaluation requires explicit inertials on every body")
    defaults = {}

    def collect(node, inherited):
        attributes = dict(inherited)
        geom = node.find("geom")
        if geom is not None:
            attributes.update(geom.attrib)
        defaults[node.get("class", "main")] = attributes
        for nested in node.findall("default"):
            collect(nested, attributes)

    for node in root.findall("default"):
        collect(node, {})
    parent = {child: node for node in root.iter() for child in node}

    def effective(geom):
        selected = geom.get("class")
        ancestor = parent.get(geom)
        while selected is None and ancestor is not None:
            selected = ancestor.get("childclass")
            ancestor = parent.get(ancestor)
        selected = selected or "main"
        if selected not in defaults:
            raise ValueError(f"Unknown geom default class {selected}")
        return selected, {**defaults[selected], **geom.attrib}

    removed = 0
    # Only actual worldbody geometry, not default templates.
    world = root.find("worldbody")
    for geom in list(world.iter("geom")):
        selected, attrs = effective(geom)
        if selected != "visual" or not (attrs.get("type") == "mesh" or "mesh" in attrs):
            continue
        if int(attrs.get("contype", "1")) or int(attrs.get("conaffinity", "1")):
            raise ValueError(f"Refusing to remove collision-capable visual mesh: {attrs.get('name', attrs.get('mesh'))}")
        parent[geom].remove(geom)
        removed += 1
    used = {effective(geom)[1].get("mesh") for geom in world.iter("geom")}
    for asset in root.findall("asset"):
        for mesh in list(asset.findall("mesh")):
            if mesh.get("name") not in used:
                asset.remove(mesh)
    if not any(root.findall("asset/mesh")):
        compiler = root.find("compiler")
        if compiler is not None:
            compiler.attrib.pop("meshdir", None)
    xml = ET.tostring(root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml), removed


def verify_isaac_observations(trajectory_path, motion_file, atol=2e-5, rtol=2e-5, *, contract_path=None):
    """Reconstruct every applied Isaac actor input from its PRE-action state.

    eval_tracking.py row i stores the action input before its post-step state;
    thus action row i>=1 uses state row i-1, not state row i. Terminals/reset rows
    are excluded using the explicit action_was_applied/valid masks.
    """
    contract = read_contract(TASK_SOURCE if contract_path is None else contract_path)
    reference = MotionReference(motion_file, contract)
    with np.load(trajectory_path, allow_pickle=False) as archive:
        d = {key: archive[key].copy() for key in archive.files}
    order = [d["joint_names"].astype(str).tolist().index(name) for name in reference.action_names]
    max_error, count, by_term = 0., 0, np.zeros(len(contract["actor_terms"]))
    for row in range(len(d["action_observation"])):
        prior = max(0, row - 1)
        for env in range(d["valid_mask"].shape[1]):
            if not d["valid_mask"][prior, env] or (row and not d["action_was_applied"][row, env]):
                continue
            state = {"root_pos": d["root_position_m"][prior, env],
                     "root_quat_wxyz": d["root_quaternion_wxyz"][prior, env],
                     "root_com_lin_vel_w": d["root_linear_velocity_w_m_s"][prior, env],
                     "root_ang_vel_w": d["root_angular_velocity_w_rad_s"][prior, env],
                     "joint_pos": d["joint_position_rad"][prior, env, order],
                     "joint_vel": d["joint_velocity_rad_s"][prior, env, order]}
            previous = np.zeros(8) if row == 0 else d["actions_clipped"][prior, env]
            frame = int(d["reference_frame_for_action"][row, env])
            rebuilt = build_observation(state, reference.sample(frame), reference.q0, previous)
            expected = d["action_observation"][row, env]
            diff = np.abs(rebuilt - expected)
            max_error = max(max_error, float(diff.max()))
            offset = 0
            for i, (_, size) in enumerate(contract["actor_terms"]):
                by_term[i] = max(by_term[i], float(diff[offset:offset + size].max()))
                offset += size
            if not np.allclose(rebuilt, expected, atol=atol, rtol=rtol):
                dimension = int(diff.argmax())
                raise AssertionError(f"Isaac actor reconstruction mismatch row={row}, env={env}, dim={dimension}, "
                                     f"reconstructed={rebuilt[dimension]}, recorded={expected[dimension]}, max_error={diff.max()}")
            count += 1
    if not count:
        raise ValueError("No valid Isaac observations to reconstruct")
    return {"status": "passed", "observations_checked": count, "max_abs_error": max_error,
            "max_abs_error_by_term": dict(zip((name for name, _ in contract["actor_terms"]), by_term.tolist())),
            "trajectory_sha256": sha256(trajectory_path), "atol": atol, "rtol": rtol}


class MujocoState:
    def __init__(self, model, data, reference):
        import mujoco
        self.mj, self.model, self.data, self.reference = mujoco, model, data, reference
        names = reference.action_names
        self.qids = np.array([int(model.joint(name).qposadr[0]) for name in names])
        self.vids = np.array([int(model.joint(name).dofadr[0]) for name in names])
        self.aids = np.array([model.actuator(name).id for name in names])
        self.base = model.body("base_Link").id
        self.wheels = np.array([model.body(name).id for name in ("wheel_L_Link", "wheel_R_Link")])
        self.root_qid = int(model.joint("root").qposadr[0])
        self.root_vid = int(model.joint("root").dofadr[0])
        if model.joint("root").type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Root must be a free joint")
        for name, aid, vid in zip(names, self.aids, self.vids):
            if int(model.actuator_trnid[aid, 0]) != model.joint(name).id or not np.allclose(model.actuator_gear[aid], [1, 0, 0, 0, 0, 0]):
                raise ValueError(f"Expected a unit direct-torque motor for {name}")
        if not np.allclose(model.body_ipos[self.base], ROOT_COM_LINK, atol=2e-7):
            raise ValueError("Expected USD-matched unmerged base COM; refusing merged-IMU model")

    def root_jacobians(self):
        # Explicit original-URDF COM matches the training motion export; do not
        # accidentally observe the free-joint origin velocity or merged COM.
        point = self.data.xpos[self.base] + self.data.xmat[self.base].reshape(3, 3) @ ROOT_COM_LINK
        jacp, jacr = np.empty((3, self.model.nv)), np.empty((3, self.model.nv))
        self.mj.mj_jac(self.model, self.data, jacp, jacr, point, self.base)
        return jacp, jacr

    def initialize(self):
        target = self.reference.sample(0)
        d, m = self.data, self.model
        d.qpos[self.root_qid:self.root_qid + 7] = np.r_[target["root_pos"], target["root_quat_wxyz"]]
        d.qpos[self.qids] = target["joint_pos"]
        d.qvel[:] = 0.
        d.qvel[self.vids] = target["joint_vel"]
        self.mj.mj_forward(m, d)
        jacp, jacr = self.root_jacobians()
        jac = np.vstack((jacp, jacr))
        root_columns = slice(self.root_vid, self.root_vid + 6)
        desired = np.r_[target["root_com_lin_vel_w"], target["root_ang_vel_w"]]
        d.qvel[root_columns] = np.linalg.solve(jac[:, root_columns], desired - jac @ d.qvel)
        self.mj.mj_forward(m, d)
        state = self.measure()
        np.testing.assert_allclose(state["root_com_lin_vel_w"], target["root_com_lin_vel_w"], atol=1e-9)
        np.testing.assert_allclose(state["root_ang_vel_w"], target["root_ang_vel_w"], atol=1e-9)
        return state

    def measure(self):
        d = self.data
        jacp, jacr = self.root_jacobians()
        state = {"root_pos": d.xpos[self.base].copy(), "root_quat_wxyz": d.xquat[self.base].copy(),
                 "root_com_lin_vel_w": jacp @ d.qvel, "root_ang_vel_w": jacr @ d.qvel,
                 "joint_pos": d.qpos[self.qids].copy(), "joint_vel": d.qvel[self.vids].copy(),
                 "wheel_pos": d.xpos[self.wheels].copy()}
        if not all(np.isfinite(value).all() for value in state.values()):
            raise FloatingPointError("Nonfinite MuJoCo simulation state")
        return state

    def contact_forces(self):
        net, ground = np.zeros((2, 3)), np.zeros((2, 3))
        contacts = np.zeros(2, dtype=int)
        wrench = np.zeros(6)
        for index, contact in enumerate(self.data.contact):
            b1, b2 = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if b1 not in self.wheels and b2 not in self.wheels:
                continue
            self.mj.mj_contactForce(self.model, self.data, index, wrench)
            world_force = contact.frame.reshape(3, 3).T @ wrench[:3]
            for wheel, body in enumerate(self.wheels):
                if body not in (b1, b2):
                    continue
                force = world_force if b2 == body else -world_force
                net[wheel] += force
                if b1 == 0 or b2 == 0:
                    ground[wheel] += force
                    contacts[wheel] += int(np.linalg.norm(force) > 1e-6)
        return net, ground, contacts

    def nonwheel_ground_contacts(self):
        """Ground-only evidence, excluding robot self-contact and both wheels."""
        count, total_force = 0, 0.0
        wrench = np.zeros(6)
        for index, contact in enumerate(self.data.contact):
            bodies = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if 0 not in bodies:
                continue
            robot_body = bodies[1] if bodies[0] == 0 else bodies[0]
            if robot_body == 0 or robot_body in self.wheels:
                continue
            self.mj.mj_contactForce(self.model, self.data, index, wrench)
            magnitude = float(np.linalg.norm(wrench[:3]))
            total_force += magnitude
            count += int(magnitude >= 1.0)
        return count, total_force


def load_torchscript(path):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed in this Python environment. Use an existing environment with PyTorch and MuJoCo; no packages were installed.") from exc
    actor = torch.jit.load(str(path), map_location="cpu").eval()
    if hasattr(actor, "reset"):
        actor.reset()

    def policy(observation):
        with torch.inference_mode():
            result = actor(torch.from_numpy(np.asarray(observation, dtype=np.float32))[None])
        if not isinstance(result, torch.Tensor) or tuple(result.shape) != (1, 8):
            raise ValueError("Expected a feedforward normalized TorchScript actor returning shape (1, 8)")
        return result.detach().cpu().numpy()[0]

    return policy


def run_evaluation(policy, motion_file, model_path, output, *, policy_path=None,
                   contact_threshold_n=5., max_policy_steps=None, no_visual_mesh=False,
                   isaac_observation_validation=None, contract_path=None):
    """Callable policy injection is for tests; CLI always loads a TorchScript actor."""
    import mujoco
    started = time.monotonic()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserving existing evaluation results: {output}")
    contract_source = resolve_contract_path(contract_path, policy_path)
    contract = read_contract(contract_source)
    reference = MotionReference(motion_file, contract)
    model, removed_visual_geoms = load_model(model_path, no_visual_mesh)
    collision_geoms = configure_model(model, contract)
    data = mujoco.MjData(model)
    measured = MujocoState(model, data, reference)
    state = measured.initialize()
    initial_qpos, initial_qvel = data.qpos.copy(), data.qvel.copy()
    previous = np.zeros(8)
    physical, policies = [], []
    termination, terms = "motion_end", []
    count = reference.frames if max_policy_steps is None else min(reference.frames, int(max_policy_steps))
    if count < 1:
        raise ValueError("At least one policy step is required")

    def record(frame, torque):
        state = measured.measure()
        force, ground_force, contacts = measured.contact_forces()
        nonwheel_count, nonwheel_force = measured.nonwheel_ground_contacts()
        physical.append({"time_s": float(data.time), "reference_frame": frame, **state,
                         "qpos": data.qpos.copy(),
                         "joint_torque_nm": torque.copy(), "wheel_contact_force_w_n": force,
                         "wheel_ground_contact_force_w_n": ground_force, "wheel_ground_contact_count": contacts,
                         "nonwheel_ground_contact_count": nonwheel_count,
                         "nonwheel_ground_contact_force_norm_n": nonwheel_force})
        return state

    record(0, np.zeros(8))
    for frame in range(count):
        target = reference.sample(frame)
        observation = build_observation(state, target, reference.q0, previous)
        raw = np.asarray(policy(observation), dtype=float).reshape(-1)
        if raw.shape != (8,) or not np.isfinite(raw).all():
            raise FloatingPointError("Policy output is not a finite 8-vector")
        action = np.clip(raw, *contract["raw_action_clip"])
        policies.append({"policy_time_s": float(data.time), "reference_frame_for_action": frame,
                         "action_observation": observation, "actions_raw": raw.copy(),
                         "actions_clipped": action.copy(), "previous_clipped_actions": previous.copy()})
        for substep in range(contract["decimation"]):
            torque = action_torques(action, state["joint_pos"], state["joint_vel"], reference.q0, contract)
            data.ctrl[measured.aids] = torque
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)  # Refresh post-step FK, no state writes.
            state = record(frame, torque)
        previous = action.copy()
        terms = training_termination_terms(state, target)
        if terms:
            termination = "early_termination"
            break
    else:
        if count < reference.frames:
            termination = "evaluation_step_limit"

    arrays = {name: np.asarray([sample[name] for sample in physical]) for name in physical[0]}
    arrays.update({name: np.asarray([sample[name] for sample in policies]) for name in policies[0]})
    arrays.update(joint_names=np.array(reference.action_names), policy_dt_s=np.array(1 / reference.fps),
                  physics_dt_s=np.array(contract["physics_dt"]), initial_qpos=initial_qpos, initial_qvel=initial_qvel,
                  qpos_joint_names=np.array([model.joint(i).name for i in range(1, model.njnt)]))
    ref_ids = arrays["reference_frame"].astype(int)
    ref_root = reference.archive["body_pos_w"][ref_ids, 0]
    root = arrays["root_pos"]
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * np.sum(arrays["root_quat_wxyz"][:, 1:3] ** 2, axis=1), -1, 1)))
    root_gain = float(root[:, 2].max() - root[0, 2])
    events = contact_events(arrays["time_s"], arrays["wheel_ground_contact_force_w_n"], contact_threshold_n)
    events["force_source"] = "summed wheel-ground contact forces, excluding robot self-contact"
    landed = bool(events["first_landing_s"] is not None and root_gain >= .05)
    motor_model = contract.get("actuator_model", "ideal_pd")
    velocity_bounds = np.array([contract.get("leg_motor_velocity_limit_rad_s", 15.)] * 6
                              + [contract.get("wheel_motor_velocity_limit_rad_s", 100.)] * 2)
    speed_exceeded = np.abs(arrays["joint_vel"]) > velocity_bounds
    report = {
        "status": "completed" if termination == "motion_end" else "terminated",
        "engine": "MuJoCo", "engine_version": mujoco.__version__, "termination": termination,
        "termination_terms": terms, "completed_full_reference": termination == "motion_end",
        "physics_stepped": True, "root_state_writes": 1, "hidden_resets": 0,
        "start_reference_frame": 0, "final_reference_frame": int(ref_ids[-1]),
        "policy": None if policy_path is None else str(Path(policy_path).resolve()),
        "policy_sha256": None if policy_path is None else sha256(policy_path),
        "policy_kind": "injected_test_callable" if policy_path is None else "normalized_torchscript_actor",
        "observation_normalization": "included inside TorchScript actor; no second normalizer",
        "motion_file": str(reference.path), "motion_sha256": sha256(reference.path),
        "model": str(Path(model_path).resolve()), "model_sha256": sha256(model_path),
        "task_source_sha256": sha256(TASK_SOURCE), "evaluator_source_sha256": sha256(__file__),
        "contract": contract, "reference_frames": reference.frames,
        "contract_source": str(Path(contract_source).resolve()), "contract_source_sha256": sha256(contract_source),
        "actuator_model": motor_model, "actual_joint_velocity_hard_clipped": False,
        "expected_duration_s": reference.duration, "recorded_duration_s": float(data.time),
        "physics_steps": len(physical) - 1, "policy_steps": len(policies),
        "root_com_link_offset_m": ROOT_COM_LINK.tolist(), "compiled_base_com_m": model.body_ipos[measured.base].tolist(),
        "base_link_height_gain_m": root_gain, "base_link_peak_height_m": float(root[:, 2].max()),
        "reference_base_link_height_gain_m": float(reference.archive["body_pos_w"][:, 0, 2].max()
                                                        - reference.archive["body_pos_w"][0, 0, 2]),
        "reference_height_rmse_m": float(np.sqrt(np.mean((root[:, 2] - ref_root[:, 2]) ** 2))),
        "reference_root_position_rmse_m": float(np.sqrt(np.mean(np.sum((root - ref_root) ** 2, axis=1)))),
        "max_base_tilt_deg": float(tilt.max()), "final_base_tilt_deg": float(tilt[-1]),
        "max_abs_joint_speed_rad_s": np.max(np.abs(arrays["joint_vel"]), axis=0).tolist(),
        "joint_speed_reference_rad_s": velocity_bounds.tolist(),
        "joint_speed_reference_kind": "dc_motor_no_load_speed" if motor_model == "dc_motor" else "legacy_physx_solver_limit",
        "joint_speed_exceedance_fraction_by_joint": speed_exceeded.mean(axis=0).tolist(),
        "max_abs_joint_torque_nm": np.max(np.abs(arrays["joint_torque_nm"]), axis=0).tolist(),
        "contact": events, "jump_and_landing_detected": landed,
        "nonwheel_ground_contact_recorded": True,
        "max_nonwheel_ground_contact_force_norm_n": float(arrays["nonwheel_ground_contact_force_norm_n"].max()),
        "nonwheel_ground_contact_sample_count": int(np.count_nonzero(arrays["nonwheel_ground_contact_count"])),
        "full_clip_with_detected_jump_and_landing": bool(termination == "motion_end" and landed),
        "self_collision_enabled_in_memory": True, "self_collision_geoms": collision_geoms,
        "no_visual_mesh": bool(no_visual_mesh), "removed_visual_geom_count": removed_visual_geoms,
        "isaac_observation_validation": isaac_observation_validation,
        "hardware_ready": False, "wall_time_s": time.monotonic() - started,
        "limitations": [
            "One deterministic sim2sim episode, not a robustness or real-hardware validation.",
            "Contact solver/cylinder approximations and self-collision parent filtering remain engine-specific.",
            ("DC no-load speed defines a torque-speed envelope, not a hard qvel limit; external forces can exceed it."
             if motor_model == "dc_motor" else
             "Legacy IdealPD has no matching PhysX solver velocity brake in MuJoCo; qvel is never clamped."),
            "DC motor parameters are simulation assumptions, not validated manufacturer ratings. The DC model changes training physics.",
            "MuJoCo contact forces are instantaneous constraint forces; Isaac contact sensors may aggregate differently.",
            "Reference is held for each 20 ms action; training termination predicates are checked before phase advances.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "rollout.npz", **arrays)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True, help="TorchScript actor INCLUDING learned observation normalizer")
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path,
                        help="Saved JSON contract/manifest; default is actor-adjacent policy_contract.json, else current task physics")
    parser.add_argument("--contact-threshold-n", type=float, default=5.)
    parser.add_argument("--no-visual-mesh", action="store_true", help="In memory, omit only noncolliding visual mesh geometry/assets")
    parser.add_argument("--verify-isaac-trajectory", type=Path,
                        help="Reconstruct and verify all recorded Isaac actor observations before policy deployment")
    args = parser.parse_args()
    if args.contact_threshold_n <= 0:
        parser.error("contact-threshold-n must be positive")
    for name in ("policy", "motion_file", "model"):
        if not getattr(args, name).is_file():
            parser.error(f"Missing {name}: {getattr(args, name)}")
    if args.contract is not None and not args.contract.is_file():
        parser.error(f"Missing contract: {args.contract}")
    try:
        contract_path = resolve_contract_path(args.contract, args.policy)
        validation = (verify_isaac_observations(args.verify_isaac_trajectory, args.motion_file, contract_path=contract_path)
                      if args.verify_isaac_trajectory else None)
        policy = load_torchscript(args.policy)
        result = run_evaluation(policy, args.motion_file, args.model, args.output_dir,
                                policy_path=args.policy, contact_threshold_n=args.contact_threshold_n,
                                no_visual_mesh=args.no_visual_mesh, isaac_observation_validation=validation,
                                contract_path=contract_path)
    except (RuntimeError, ValueError, FileExistsError) as exc:
        parser.exit(1, f"MuJoCo evaluation error: {exc}\n")
    print(json.dumps({key: result[key] for key in ("status", "termination", "recorded_duration_s",
                      "base_link_height_gain_m", "max_base_tilt_deg", "jump_and_landing_detected")}, indent=2))


if __name__ == "__main__":
    main()
