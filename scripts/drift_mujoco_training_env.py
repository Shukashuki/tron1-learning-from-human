"""Stateful 50/200 Hz MuJoCo drift environment with evaluator-identical dynamics.

This is a training scaffold, not a separate deployment evaluator. The actor
observation, action clipping, DC-motor envelope, PD, reference timing, and
termination use the frozen evaluator helpers. Native substep contact/torque
arrays are retained for an exact zero-residual replay control.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from eval_tracking_mujoco import (
    MotionReference, MujocoState, action_torques, build_observation,
    configure_model, load_model, read_contract, training_termination_terms,
)
from training.tron1_drift import configure_mujoco_materials


class MujocoDriftEnv:
    def __init__(self, motion_file: Path, model_file: Path, contract_file: Path):
        self.contract = read_contract(contract_file)
        self.reference = MotionReference(motion_file, self.contract)
        self.model, _ = load_model(model_file, no_visual_mesh=True)
        configure_model(self.model, self.contract)
        if self.contract.get("task_profile") is not None:
            configure_mujoco_materials(self.model, self.contract["task_profile"])
        self.data = mujoco.MjData(self.model)
        self.measured = MujocoState(self.model, self.data, self.reference)
        self.frame = 0
        self.previous_action = np.zeros(8)
        self.state = None
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.frame = 0
        self.previous_action[:] = 0.
        self.state = self.measured.initialize()
        return self.observation()

    def observation(self):
        if self.frame >= self.reference.frames:
            return None
        return build_observation(self.state, self.reference.sample(self.frame),
                                 self.reference.q0, self.previous_action)

    def step(self, raw_action):
        if self.frame >= self.reference.frames:
            raise RuntimeError("Cannot step after full reference completion")
        raw = np.asarray(raw_action, dtype=float)
        if raw.shape != (8,) or not np.isfinite(raw).all():
            raise ValueError("Expected finite raw eight-action actor output")
        action = np.clip(raw, *self.contract["raw_action_clip"])
        target = self.reference.sample(self.frame)
        substeps = []
        for _ in range(self.contract["decimation"]):
            torque = action_torques(action, self.state["joint_pos"],
                                    self.state["joint_vel"], self.reference.q0, self.contract)
            self.data.ctrl[self.measured.aids] = torque
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.state = self.measured.measure()
            contact = self.measured.ground_contact_slip()
            substeps.append({"time_s": float(self.data.time),
                             "qpos": self.data.qpos.copy(),
                             "joint_torque_nm": torque.copy(),
                             "root_pos": self.state["root_pos"].copy(),
                             "root_com_lin_vel_w": self.state["root_com_lin_vel_w"].copy(),
                             "root_quat_wxyz": self.state["root_quat_wxyz"].copy(),
                             "wheel_ground_contact_count": contact["wheel_ground_contact_count"].copy(),
                             "wheel_ground_contact_force_w_n": contact["wheel_ground_contact_force_w_n"].copy(),
                             "wheel_lateral_contact_slip_m_s": contact["wheel_lateral_contact_slip_m_s"].copy()})
        self.previous_action = action.copy()
        termination = training_termination_terms(self.state, target)
        self.frame += 1
        done = bool(termination) or self.frame == self.reference.frames
        return (None if done else self.observation(), done,
                {"frame": self.frame - 1, "termination_terms": termination,
                 "completed_full_reference": self.frame == self.reference.frames and not termination,
                 "state": self.state, "substeps": substeps})
