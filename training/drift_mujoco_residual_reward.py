"""Dense residual-policy reward; never substitutes for the frozen drift gate.

The same 200 Hz physical samples used by assessment enter the core reward.
Tracking outside the turn prevents the policy from sacrificing the approach.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _yaw(quat_wxyz):
    q = np.asarray(quat_wxyz, dtype=float)
    r = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    return float(np.arctan2(r[1, 0], r[0, 0]))


def _smooth_above(value, center, width):
    return float(1. / (1. + np.exp(-np.clip((value - center) / width, -30., 30.))))


def reward_components(target, state, substeps, residual, *, core_start_s=1.44, core_end_s=4.):
    """Return explicit shaping terms from an evaluator-identical action window."""
    if len(substeps) != 4:
        raise ValueError("The pinned WF drift contract has four native steps per action")
    pos_error = float(np.linalg.norm(state["root_pos"][:2] - target["root_pos"][:2]))
    vel_error = float(np.linalg.norm(state["root_com_lin_vel_w"][:2]
                                     - target["root_com_lin_vel_w"][:2]))
    yaw_error = float(np.arctan2(np.sin(_yaw(state["root_quat_wxyz"])
                                       - _yaw(target["root_quat_wxyz"])),
                                      np.cos(_yaw(state["root_quat_wxyz"])
                                             - _yaw(target["root_quat_wxyz"]))))
    path = float(np.exp(-(pos_error / .18)**2))
    velocity = float(np.exp(-(vel_error / .5)**2))
    heading = float(np.exp(-(yaw_error / .4)**2))
    native = []
    for sample in substeps:
        forces = np.asarray(sample["wheel_ground_contact_force_w_n"], dtype=float)
        counts = np.asarray(sample["wheel_ground_contact_count"])
        slip = np.asarray(sample["wheel_lateral_contact_slip_m_s"], dtype=float)
        vel = np.asarray(sample["root_com_lin_vel_w"], dtype=float)[:2]
        speed = float(np.linalg.norm(vel))
        beta = abs((np.arctan2(vel[1], vel[0]) - _yaw(sample["root_quat_wxyz"])
                    + np.pi) % (2*np.pi) - np.pi) * 180. / np.pi
        normal = np.linalg.norm(forces, axis=1)
        supported = float(np.all((counts > 0) & (normal >= 5.)))
        load = float(np.mean(np.clip(normal / 20., 0., 1.) * (counts > 0)))
        sliding = _smooth_above(float(np.mean(np.abs(slip))), .04, .015)
        moving = _smooth_above(speed, .8, .1)
        sideways = _smooth_above(beta, 8., 3.)
        native.append((supported, load, sliding, moving, sideways,
                       float(core_start_s <= sample["time_s"] <= core_end_s)))
    values = np.asarray(native)
    is_core = float(np.any(values[:, 5] > 0))
    support = float(values[:, 0].mean())
    load = float(values[:, 1].mean())
    qualifying = float(np.mean(np.prod(values[:, [0, 2, 3, 4]], axis=1)))
    slip = float(values[:, 2].mean())
    speed = float(values[:, 3].mean())
    sideslip = float(values[:, 4].mean())
    residual_cost = float(np.sum(np.asarray(residual, dtype=float)**2))
    # A small residual prior protects the previously Isaac-PASS actor. Contact
    # reward alone would admit slow, non-sliding high-load behavior.
    reward = (.2 * (2.*path + velocity + .5*heading)
              + is_core * (.25*load + .4*qualifying + .15*slip + .15*speed + .15*sideslip)
              - .05*residual_cost)
    result = {"reward": float(reward), "path": path, "velocity": velocity,
              "heading": heading, "core": is_core, "support_fraction": support,
              "load": load, "joint_qualifying_soft_fraction": qualifying,
              "slip": slip, "speed": speed, "sideslip": sideslip,
              "residual_cost": residual_cost}
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError("Nonfinite residual reward component")
    return result
