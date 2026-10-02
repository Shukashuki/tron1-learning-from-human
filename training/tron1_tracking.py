"""TRON1 WF single-clip tracking with the official BeyondMimic MDP.

Import this module only *after* Isaac Sim's AppLauncher has started.  The
reference is kinematic supervision, not a prescribed root trajectory: only
resets write root state; policy steps apply joint motor efforts.

The upstream package has eager G1 task registration.  We intentionally import
only its MDP and generic environment configuration, so G1 robot downloads and
WandB access are not needed.  Set TRON1_BEYONDMIMIC_ROOT to the pinned upstream
checkout, or place it at third_party/whole_body_tracking.
"""

from __future__ import annotations

import importlib
import math
import os
from pathlib import Path
import sys
import types

import numpy as np
import torch


def _load_upstream():
    root = Path(os.environ.get(
        "TRON1_BEYONDMIMIC_ROOT",
        str(Path(__file__).resolve().parents[1] / "third_party" / "whole_body_tracking"),
    )).resolve()
    package = root / "source" / "whole_body_tracking" / "whole_body_tracking"
    if not (package / "tasks" / "tracking" / "tracking_env_cfg.py").is_file():
        raise FileNotFoundError(f"BeyondMimic tracking source is missing under {root}")
    for suffix in ("", ".tasks", ".tasks.tracking"):
        name = "whole_body_tracking" + suffix
        path = package.joinpath(*suffix.strip(".").split(".")) if suffix else package
        if name in sys.modules:
            previous = getattr(sys.modules[name], "__path__", ())
            if str(path) not in previous:
                raise RuntimeError(f"{name} was already imported from another checkout: {previous}")
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        module.__file__ = str(path / "__init__.py")
        sys.modules[name] = module
        if "." in name:
            parent, leaf = name.rsplit(".", 1)
            setattr(sys.modules[parent], leaf, module)
    mdp = importlib.import_module("whole_body_tracking.tasks.tracking.mdp")
    cfg_module = importlib.import_module("whole_body_tracking.tasks.tracking.tracking_env_cfg")
    return mdp, cfg_module.TrackingEnvCfg


mdp, TrackingEnvCfg = _load_upstream()

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_error_magnitude, quat_inv, quat_mul, yaw_quat


LEG_JOINT_NAMES = [
    "abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint", "knee_L_Joint", "knee_R_Joint",
]
WHEEL_JOINT_NAMES = ["wheel_L_Joint", "wheel_R_Joint"]
ACTION_JOINT_NAMES = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES
BODY_NAMES = [
    "base_Link", "abad_L_Link", "abad_R_Link", "hip_L_Link", "hip_R_Link",
    "knee_L_Link", "knee_R_Link", "wheel_L_Link", "wheel_R_Link",
]
NON_WHEEL_BODY_NAMES = [name for name in BODY_NAMES if not name.startswith("wheel_")]
WHEEL_BODY_NAMES = ["wheel_L_Link", "wheel_R_Link"]
POLICY_FPS = 50.0
LEG_ACTION_SCALE = 1.0


def reference_name_permutation(reference_names, native_names, label):
    """Map native indices to reference indices, requiring exact unique names."""
    reference = np.asarray(reference_names)
    native = np.asarray(native_names)
    for names, origin in ((reference, "reference"), (native, "runtime")):
        if names.ndim != 1 or names.dtype.kind not in "US":
            raise ValueError(f"{label}: {origin} names must be a one-dimensional string array")
        values = names.astype(str).tolist()
        if len(values) != len(set(values)):
            raise ValueError(f"{label}: duplicate {origin} names are unsafe: {values}")
        if any(not name for name in values):
            raise ValueError(f"{label}: empty {origin} name")
    reference, native = reference.astype(str).tolist(), native.astype(str).tolist()
    missing = sorted(set(native) - set(reference))
    extra = sorted(set(reference) - set(native))
    if missing or extra:
        raise ValueError(f"{label}: reference/runtime names differ; missing={missing}, extra={extra}")
    return [reference.index(name) for name in native]


def reorder_motion_reference(motion, reference_joint_names, reference_body_names,
                             native_joint_names, native_body_names):
    """Align every full reference tensor before upstream body-index selection.

    Works on either Torch or NumPy arrays; the NPZ file is never rewritten.
    Isaac 4.5 and 5.1 may enumerate identical link names in different orders.
    """
    joint_indices = reference_name_permutation(reference_joint_names, native_joint_names, "joint_names")
    body_indices = reference_name_permutation(reference_body_names, native_body_names, "body_names")
    features = (("joint_pos", joint_indices, None), ("joint_vel", joint_indices, None),
                ("_body_pos_w", body_indices, 3), ("_body_quat_w", body_indices, 4),
                ("_body_lin_vel_w", body_indices, 3), ("_body_ang_vel_w", body_indices, 3))
    frames = motion.joint_pos.shape[0]
    for name, indices, width in features:
        values = getattr(motion, name)
        expected = (frames, len(indices)) if width is None else (frames, len(indices), width)
        if tuple(values.shape) != expected:
            raise ValueError(f"Reference {name} shape {tuple(values.shape)} != named shape {expected}")
    for name, indices, _ in features:
        setattr(motion, name, getattr(motion, name)[:, indices])
    return {
        "reference_joint_names": np.asarray(reference_joint_names).astype(str).tolist(),
        "runtime_joint_names": list(native_joint_names),
        "runtime_joint_to_reference_indices": joint_indices,
        "reference_body_names": np.asarray(reference_body_names).astype(str).tolist(),
        "runtime_body_names": list(native_body_names),
        "runtime_body_to_reference_indices": body_indices,
        "full_tensors_reordered_before_body_subset": True,
        "source_npz_modified": False,
    }


class TronMotionCommand(mdp.MotionCommand):
    """Adaptive RSI, wheel-invariant command, and explicit clip-end episodes."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        with np.load(cfg.motion_file, allow_pickle=False) as archive:
            for key in ("joint_names", "body_names"):
                if key not in archive:
                    raise ValueError(f"Reference must contain {key}; positional mapping is unsafe")
            fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
            if not np.isclose(fps, 1.0 / env.step_dt, rtol=1e-6):
                raise ValueError(f"Reference fps={fps}, but policy fps={1.0 / env.step_dt}")
            self.reference_name_mapping = reorder_motion_reference(
                self.motion, archive["joint_names"], archive["body_names"],
                self.robot.joint_names, self.robot.body_names,
            )
        if cfg.body_names[0] != "base_Link" or self.robot.body_names[0] != "base_Link":
            raise ValueError("The root body must be base_Link and first in the reference and tracking body arrays")
        if self.motion.time_step_total < 2:
            raise ValueError("Reference motion must have at least two frames")
        self.leg_joint_indices = torch.tensor(
            [self.robot.joint_names.index(name) for name in LEG_JOINT_NAMES], device=self.device,
        )
        self.observed_joint_indices = torch.tensor(
            [self.robot.joint_names.index(name) for name in ACTION_JOINT_NAMES], device=self.device,
        )
        self.nonwheel_body_indices = torch.tensor(
            [cfg.body_names.index(name) for name in NON_WHEEL_BODY_NAMES], device=self.device,
        )
        # Upstream uses approximately one-second bins; this one short jump needs
        # finer separation between compression, take-off, flight, and landing.
        self.bin_count = max(2, math.ceil(self.motion.time_step_total * env.step_dt / cfg.sampling_bin_seconds))
        self.bin_failed_count = torch.zeros(self.bin_count, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, device=self.device)

    @property
    def command(self):
        phase = self.time_steps.to(torch.float32) / (self.motion.time_step_total - 1)
        phase = (2.0 * math.pi * phase).unsqueeze(-1)
        return torch.cat((
            self.joint_pos[:, self.leg_joint_indices],
            self.joint_vel[:, self.leg_joint_indices],
            torch.sin(phase), torch.cos(phase),
        ), dim=-1)

    def _adaptive_sampling(self, env_ids):
        if self.cfg.deterministic_start:
            self.time_steps[env_ids] = 0
            self.metrics["sampling_entropy"][:] = 0.0
            self.metrics["sampling_top1_prob"][:] = 1.0
            self.metrics["sampling_top1_bin"][:] = 0.0
        else:
            super()._adaptive_sampling(env_ids)
            # A full-clip component ensures take-off is trained from standing,
            # rather than only learning recovery from privileged flying RSI.
            indices = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            start_mask = torch.rand(len(indices), device=self.device) < self.cfg.start_fraction
            self.time_steps[indices[start_mask]] = 0

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        # Explicit env.reset() does not call command.compute().  Initialize
        # targets now, otherwise its first reward/termination sees zeros or
        # the previous episode's targets (especially harmful for airborne RSI).
        if len(env_ids) > 0:
            self._refresh_relative_targets()

    def _refresh_relative_targets(self):
        """Refresh targets at the current phase, without advancing time."""
        anchor_pos = self.anchor_pos_w[:, None, :].expand(-1, len(self.cfg.body_names), -1)
        anchor_quat = self.anchor_quat_w[:, None, :].expand(-1, len(self.cfg.body_names), -1)
        robot_pos = self.robot_anchor_pos_w[:, None, :].expand(-1, len(self.cfg.body_names), -1)
        robot_quat = self.robot_anchor_quat_w[:, None, :].expand(-1, len(self.cfg.body_names), -1)
        delta_pos = robot_pos.clone()
        # Preserve global reference height, including flight.  Only XY and yaw
        # are re-anchored for relative body tracking, exactly as in upstream.
        delta_pos[..., 2] = anchor_pos[..., 2]
        delta_quat = yaw_quat(quat_mul(robot_quat, quat_inv(anchor_quat)))
        self.body_quat_relative_w = quat_mul(delta_quat, self.body_quat_w)
        self.body_pos_relative_w = delta_pos + quat_apply(delta_quat, self.body_pos_w - anchor_pos)

    def _update_command(self):
        # Never call _resample_command here.  Doing so, as upstream does at
        # clip-end, teleports the robot outside the RL episode reset boundary.
        # IsaacLab calls command.compute AFTER resetting done environments.
        # A reset env has episode_length_buf=0 and has not simulated a step
        # from its sampled pose yet: preserve frame 0 (or its sampled RSI
        # frame) rather than silently skipping it.  Non-reset envs advance.
        advance = self._env.episode_length_buf > 0
        self.time_steps = torch.clamp(self.time_steps + advance, max=self.motion.time_step_total - 1)
        self._refresh_relative_targets()
        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed
            + (1.0 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _update_metrics(self):
        super()._update_metrics()
        # Upstream joint metrics include the meaningless accumulated wheel
        # angle.  Replace them with meaningful leg-only errors.
        self.metrics["error_joint_pos"] = torch.linalg.vector_norm(
            (self.joint_pos - self.robot_joint_pos)[:, self.leg_joint_indices], dim=-1,
        )
        self.metrics["error_joint_vel"] = torch.linalg.vector_norm(
            (self.joint_vel - self.robot_joint_vel)[:, self.leg_joint_indices], dim=-1,
        )
        ids = self.nonwheel_body_indices
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w[:, ids], self.robot_body_quat_w[:, ids],
        ).mean(-1)
        self.metrics["error_body_ang_vel"] = torch.linalg.vector_norm(
            (self.body_ang_vel_w - self.robot_body_ang_vel_w)[:, ids], dim=-1,
        ).mean(-1)


@configclass
class TronMotionCommandCfg(mdp.MotionCommandCfg):
    class_type: type = TronMotionCommand
    deterministic_start: bool = False
    sampling_bin_seconds: float = 0.2
    start_fraction: float = 0.2


def leg_joint_positions(env, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    ids = command.leg_joint_indices
    return (command.robot.data.joint_pos - command.robot.data.default_joint_pos)[:, ids]


def ordered_joint_velocities(env, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    return command.robot.data.joint_vel[:, command.observed_joint_indices]


def nonwheel_body_orientation(env, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    quat = command.robot_body_quat_w[:, command.nonwheel_body_indices]
    root_inv = quat_inv(command.robot_anchor_quat_w)[:, None, :].expand(-1, quat.shape[1], -1)
    rotation = matrix_from_quat(quat_mul(root_inv, quat))
    return rotation[..., :2].reshape(env.num_envs, -1)


def motion_ended(env, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    return command.time_steps >= command.motion.time_step_total - 1


@configclass
class TronActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=LEG_JOINT_NAMES, preserve_order=True,
        scale=LEG_ACTION_SCALE, use_default_offset=True,
    )
    wheel_effort = mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=WHEEL_JOINT_NAMES, preserve_order=True, scale=12.0,
    )


@configclass
class TronObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        command = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObservationTermCfg(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObservationTermCfg(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
        joint_pos = ObservationTermCfg(func=leg_joint_positions)
        joint_vel = ObservationTermCfg(func=ordered_joint_velocities)
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        body_pos = ObservationTermCfg(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObservationTermCfg(func=nonwheel_body_orientation)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class TronEventsCfg:
    # A fixed material assignment, not domain randomization.  Startup joint
    # offset/CoM/push events in upstream are intentionally absent in the pilot.
    physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material, mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 0.6), "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0), "num_buckets": 1,
        },
    )


@configclass
class TronTrackingEnvCfg(TrackingEnvCfg):
    actions: TronActionsCfg = TronActionsCfg()
    observations: TronObservationsCfg = TronObservationsCfg()
    events: TronEventsCfg = TronEventsCfg()


def make_env_cfg(motion_file, asset_path, num_envs, device="cuda:0", eval_mode=False):
    """Return a registry-free ManagerBasedRLEnv config for the local motion.

    The entrypoint must clip all raw policy actions to [-1, 1], identically in
    training, evaluation and MuJoCo deployment.  Explicit actuators additionally
    clip final torques, so Gaussian exploration cannot bypass motor limits.
    """
    motion_file = Path(motion_file).resolve()
    asset_path = Path(asset_path).resolve()
    if not motion_file.is_file() or not asset_path.is_file():
        raise FileNotFoundError(f"Missing motion or robot asset: {motion_file}, {asset_path}")
    with np.load(motion_file, allow_pickle=False) as archive:
        fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
        if not np.isclose(fps, POLICY_FPS):
            raise ValueError(f"Expected a {POLICY_FPS:g} Hz motion, got {fps}")
        joint_names = archive["joint_names"].astype(str).tolist()
        if set(joint_names) != set(ACTION_JOINT_NAMES):
            raise ValueError(f"Reference is not an 8-DOF TRON1 WF motion: {joint_names}")
        initial_positions = dict(zip(joint_names, archive["joint_pos"][0].tolist()))
        initial_root = tuple(archive["body_pos_w"][0, 0].tolist())
        frames = len(archive["joint_pos"])

    cfg = TronTrackingEnvCfg()
    cfg.scene.num_envs = int(num_envs)
    cfg.scene.env_spacing = 3.0
    cfg.sim.device = device
    cfg.sim.dt = 0.005
    cfg.decimation = 4
    cfg.sim.render_interval = cfg.decimation
    cfg.episode_length_s = frames / POLICY_FPS + 0.04
    cfg.scene.terrain.visual_material = None  # No remote Nucleus materials.
    cfg.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="average", restitution_combine_mode="average",
        static_friction=0.6, dynamic_friction=0.6, restitution=0.0,
    )
    cfg.sim.physics_material = cfg.scene.terrain.physics_material
    cfg.scene.contact_forces.debug_vis = False
    cfg.scene.contact_forces.update_period = cfg.sim.dt

    cfg.scene.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(asset_path), activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, linear_damping=0.0, angular_damping=0.0,
                max_linear_velocity=100.0, max_angular_velocity=100.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8, solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=initial_root, joint_pos=initial_positions, joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.95,
        actuators={
            "legs": IdealPDActuatorCfg(
                joint_names_expr=LEG_JOINT_NAMES, stiffness=500.0, damping=10.0,
                effort_limit=80.0, effort_limit_sim=80.0,
                velocity_limit=15.0, velocity_limit_sim=15.0,
                armature=0.0, friction=0.0,
            ),
            "wheels": IdealPDActuatorCfg(
                joint_names_expr=WHEEL_JOINT_NAMES, stiffness=0.0, damping=0.0,
                effort_limit=12.0, effort_limit_sim=12.0,
                velocity_limit=100.0, velocity_limit_sim=100.0,
                armature=0.0, friction=0.0,
            ),
        },
    )
    cfg.commands.motion = TronMotionCommandCfg(
        asset_name="robot", motion_file=str(motion_file), anchor_body_name="base_Link",
        body_names=BODY_NAMES, resampling_time_range=(1.0e9, 1.0e9), debug_vis=False,
        pose_range={}, velocity_range={}, joint_position_range=(0.0, 0.0),
        deterministic_start=bool(eval_mode), sampling_bin_seconds=0.2,
        start_fraction=0.2,
    )
    # Keep the official exponential tracking objectives, but omit wheel axial
    # rotation.  The arbitrary zero wheel angle in the IK reference is not a
    # physical rolling target.
    cfg.rewards.motion_body_ori.params["body_names"] = NON_WHEEL_BODY_NAMES
    cfg.rewards.motion_body_ang_vel.params["body_names"] = NON_WHEEL_BODY_NAMES
    cfg.rewards.joint_limit.params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True,
    )
    cfg.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=NON_WHEEL_BODY_NAMES,
    )
    cfg.terminations.anchor_pos.params["threshold"] = 0.35
    cfg.terminations.anchor_ori.params["threshold"] = 0.6
    cfg.terminations.ee_body_pos.params["body_names"] = WHEEL_BODY_NAMES
    cfg.terminations.ee_body_pos.params["threshold"] = 0.25
    cfg.terminations.motion_end = TerminationTermCfg(
        func=motion_ended, params={"command_name": "motion"}, time_out=True,
    )
    return cfg


def observation_action_contract():
    """Serializable deployment contract; include this beside every checkpoint."""
    return {
        "policy_fps": POLICY_FPS, "physics_dt": 0.005, "decimation": 4,
        "actor_dim": 51, "critic_dim": 120, "action_dim": 8,
        "leg_joint_names": LEG_JOINT_NAMES, "wheel_joint_names": WHEEL_JOINT_NAMES,
        "observed_joint_velocity_order": ACTION_JOINT_NAMES,
        "body_names": BODY_NAMES, "orientation_body_names": NON_WHEEL_BODY_NAMES,
        "actor_terms": [
            ["ref_leg_q_ref_leg_dq_phase_sin_cos", 14], ["motion_anchor_pos_b", 3],
            ["motion_anchor_ori_b_6d_first_two_columns_row_major", 6],
            ["base_lin_vel_b", 3], ["base_ang_vel_b", 3], ["leg_joint_pos_rel", 6],
            ["joint_velocities", 8], ["previous_clipped_actions", 8],
        ],
        "raw_action_clip": [-1.0, 1.0], "leg_position_scale": LEG_ACTION_SCALE,
        "leg_position_offset": "reference_frame_0_joint_pos",
        "wheel_torque_scale_nm": 12.0, "leg_kp": 500.0, "leg_kd": 10.0,
        "leg_torque_limit_nm": 80.0, "wheel_torque_limit_nm": 12.0,
        "quaternion_order": "wxyz", "root_prescribed_during_steps": False,
        "wheel_orientation_tracking": False, "wheel_angle_observation": False,
        "reference_state_initialization_training_only": True,
        "training_full_clip_start_fraction": 0.2,
        "deterministic_evaluation_start_frame": 0, "motion_end_hidden_teleport": False,
        "domain_randomization": "disabled for initial pilot",
        "actuator_limits_provenance": "project sim2sim baseline settings; NOT verified manufacturer hardware ratings",
        "hardware_ready": False,
    }
