"""Retrospective saved-state wheel geometry, never a new rollout or drift gate.

Only frozen trials 013 and 016 are supported. MuJoCo 3.9.0 mj_fwdPosition
reconstructs FK and the CURRENT collision manifold, without force solving,
actuation, integration, an actor, or a viewer. Original archive forces remain
separate. No reconstructed force is read. PhysX manifolds were not archived
and no Isaac geometry reconstruction is attempted by this tool.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PHYSICS_DT = .005
POLICY_DT = .02
SAMPLES = 1153
QUATERNION_STORAGE_ATOL = 32 * np.finfo(np.float64).eps
WINDOWS = ((.78, 1.44), (1.82, 2.32))
JOINT_NAMES = tuple(f"{j}_{s}_Joint" for j in ("abad", "hip", "knee") for s in ("L", "R")) + (
    "wheel_L_Joint", "wheel_R_Joint")
NATIVE_JOINT_NAMES = tuple(f"{j}_{s}_Joint" for s in ("L", "R") for j in ("abad", "hip", "knee", "wheel"))
WHEEL_NAMES = ("wheel_L_Link", "wheel_R_Link")
ROOT_COM_LINK = np.array([.04576, .00014, -.16398])
MODEL_PATH = ROOT / "outputs/sim2sim-cmu-16_03/mujoco_model.xml"
PINS = {
    "model": "bd92264b72b4d09084732888dbd81fd94a64a1678978e58a2ab0dd191887f513",
    "mujoco_evaluator": "da6d1da378b5065589a49f2befaebfb0576665dbd2ad5d34a157d11b8984a3aa",
    "drift_module": "6d5f7974602adec9313a5e4feceb8e3caab392ca99351ac4a5f717e57fe1cf8a",
    "contact_slip": "52ef38de9cece19a8ee02c120cb85d472ffd831e4587e5f0d1f340974ab699f4",
    "assessor": "a2214c47d0f8b262fb8f5b70a709908ef1d7487fd2b1393becb8ad8dfc672fa5",
}
TRIALS = {
    "trial013": ("config/car_drift_ground_load_task.json",
                 "174a0543dfbf3c65bdd54867ab8e9b591047c8a572b0212eaaddbcd0573f2064",
                 "dc71ee407f56147d1c6e7ff91b797213d5a6786fa70304c8254794adc123b182",
                 "8342bc119ca81a2c3f182fa383024fdc24c82c210504e2d031087f25009be47f"),
    "trial016": ("config/car_drift_dense_ground_load_task.json",
                 "989215d55b548e245d6add928411d9e6fa16852a21f6863a777179014e3230b5",
                 "da44cf91d156d6097050795a7c3b4051974c74bcf516ef29ad7ae750b161cd7e",
                 "9832b7c1ba7d90bb06094cb435aa630be37aed43f31946c0c3caf187dc61199a"),
}
MJ_KEYS = frozenset("time_s reference_frame root_pos root_quat_wxyz root_com_lin_vel_w root_ang_vel_w "
    "joint_pos joint_vel wheel_pos qpos joint_torque_nm wheel_contact_force_w_n "
    "wheel_ground_total_contact_force_w_n wheel_ground_contact_count wheel_ground_contact_force_w_n "
    "wheel_lateral_contact_slip_m_s nonwheel_ground_contact_count nonwheel_ground_contact_force_norm_n "
    "policy_time_s reference_frame_for_action action_observation actions_raw actions_clipped "
    "previous_clipped_actions joint_names policy_dt_s physics_dt_s initial_qpos initial_qvel qpos_joint_names".split())
LIMITS = [
    "Retrospective descriptive saved-state audit, not a new episode, causal test, hardware or robustness proof.",
    "Original acceptance gates, clocks, thresholds and source windows are unchanged; no gaps are bridged or tolerated.",
    "Archived bearing counts use Fn > 5 N; an empty bearing count does not identify an empty collision manifold.",
    "Archived normal vectors sum all supplied ground normal forces; MuJoCo ground total additionally includes friction.",
    "MuJoCo archived forces follow mj_step then mj_forward and are refreshed-current-state forces, not past applied impulses.",
    "Reconstructed MuJoCo manifolds/distances are current saved-state geometry, not originally saved raw contacts or forces.",
    "Actual PhysX manifold, separation, cooked cylinder geometry and raw contact point buffers are unavailable.",
    "No Isaac geometry/manifold reconstruction is attempted; no PhysX/MuJoCo contact-equivalence claim.",
    "Instantaneous support derivatives are undefined at zero-camber or axial cusps; interval finite differences are separate.",
    "No regenerated forces, phase fitting, new teacher, contact-law identification, reserve-cost reconstruction or gate changes.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def typed_equal(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(typed_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON field {key}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"Nonfinite JSON number {value}")
    result = json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=constant)
    require(isinstance(result, dict), "JSON root must be an object")
    return result


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), "Duplicate NPZ field")
        return {k: archive[k].copy() for k in archive.files}


def numeric(array, shape, label, *, integer=False, boolean=False):
    value = np.asarray(array)
    kind = "b" if boolean else "iu" if integer else "iuf"
    require(value.shape == shape and value.dtype.kind in kind and np.isfinite(value).all(),
            f"Malformed/nonfinite {label}: expected {shape}")
    return value


def names(array, expected, label):
    array = np.asarray(array)
    require(array.shape == (len(expected),) and array.dtype.kind in "US"
            and tuple(array.tolist()) == tuple(expected), f"Unrecognized {label}")


def normalized_quaternions(quaternions):
    value = np.asarray(quaternions, float)
    require(value.ndim == 2 and value.shape[1] == 4 and np.isfinite(value).all(), "Malformed wxyz quaternion")
    length = np.linalg.norm(value, axis=1)
    require(np.all(np.abs(length - 1) <= 1e-3), "Saved quaternion is not near unit")
    return value / length[:, None]


def normalize_archive(arrays, *, samples=SAMPLES):
    """Strict existing MuJoCo physical schema; no actor/action inference."""
    require(set(arrays) == MJ_KEYS, "Unrecognized physical archive fields")
    n = samples
    t = numeric(arrays["time_s"], (n,), "time_s")
    require(abs(float(t[0])) <= 1e-9 and np.allclose(np.diff(t), PHYSICS_DT, atol=1e-9, rtol=0),
            "Physical time must begin at zero with uniform 5 ms increments")
    for key, dt in (("physics_dt_s", PHYSICS_DT), ("policy_dt_s", POLICY_DT)):
        value = numeric(arrays[key], (), key)
        require(float(value) == dt, f"Unrecognized {key}")
    names(arrays["joint_names"], JOINT_NAMES, "joint order")
    names(arrays["qpos_joint_names"], NATIVE_JOINT_NAMES, "saved qpos joint order")
    for key, value in arrays.items():
        require(value.dtype.kind in "biufUS", f"Unsupported dtype in {key}")
        if value.dtype.kind in "biuf":
            require(np.isfinite(value).all(), f"Nonfinite {key}")
    shapes = {"root_pos": (3,), "root_quat_wxyz": (4,), "root_com_lin_vel_w": (3,),
              "root_ang_vel_w": (3,), "joint_pos": (8,), "joint_vel": (8,), "joint_torque_nm": (8,),
              "wheel_pos": (2, 3), "qpos": (15,), "wheel_ground_contact_count": (2,),
              "wheel_ground_contact_force_w_n": (2, 3), "wheel_ground_total_contact_force_w_n": (2, 3),
              "wheel_contact_force_w_n": (2, 3),
              "wheel_lateral_contact_slip_m_s": (2,), "reference_frame": (),
              "nonwheel_ground_contact_count": (), "nonwheel_ground_contact_force_norm_n": ()}
    result = {"time_s": t.copy()}
    for target, tail in shapes.items():
        value = numeric(arrays[target], (n, *tail), target,
                        integer=target.endswith("count") or target == "reference_frame")
        result[target] = value.copy()
    for key in ("wheel_ground_contact_count", "nonwheel_ground_contact_count", "reference_frame",
                "wheel_lateral_contact_slip_m_s", "nonwheel_ground_contact_force_norm_n"):
        require(np.all(result[key] >= 0), f"Negative {key}")
    require(np.all(result["reference_frame"] < 288), "Unknown reference phase")
    p = (n - 1) // 4
    require((n - 1) % 4 == 0, "Unrecognized physical/policy sample ratio")
    numeric(arrays["initial_qpos"], (15,), "initial qpos")
    numeric(arrays["initial_qvel"], (14,), "initial qvel")
    require(np.array_equal(arrays["initial_qpos"], arrays["qpos"][0]), "Initial qpos mismatch")
    for key, tail in (("action_observation", (51,)), ("actions_raw", (8,)), ("actions_clipped", (8,)),
                      ("previous_clipped_actions", (8,)), ("reference_frame_for_action", ())):
        numeric(arrays[key], (p, *tail), key, integer=not tail)
    pt = numeric(arrays["policy_time_s"], (p,), "policy time")
    require(np.allclose(pt, np.arange(p) * POLICY_DT, atol=1e-9, rtol=0), "Invalid policy action clock")
    require(np.all(np.abs(arrays["actions_clipped"]) <= 1), "Clipped action outside [-1,1]")
    result["policy_actions_clipped"] = arrays["actions_clipped"].copy()
    result["policy_time_s"] = pt.copy()
    result["initial_qvel"] = arrays["initial_qvel"].copy()
    # qpos is a check, not an assumed common native ordering.
    # d.xquat FK output and native qpos can differ by double-precision
    # normalization roundoff. This fixed coefficient STORAGE bound is not
    # a clearance/contact/gate tolerance; native qpos is never rewritten.
    native_to_observed = [JOINT_NAMES.index(name) for name in NATIVE_JOINT_NAMES]
    require(np.array_equal(result["qpos"][:, :3], result["root_pos"])
            and np.all(np.abs(result["qpos"][:, 3:7] - result["root_quat_wxyz"]) <= QUATERNION_STORAGE_ATOL)
            and np.array_equal(result["qpos"][:, 7:], result["joint_pos"][:, native_to_observed]),
            "Saved qpos does not match named root/joint states")
    normalized_quaternions(result["root_quat_wxyz"])  # validate only; preserve archived qpos bytes
    return result


def cylinder_plane_geometry(center, axis, radius, halfwidth, plane_point, plane_normal,
                            center_velocity, angular_velocity):
    """Exact finite-cylinder support and derivative where differentiable.

    A zero-camber cylinder has a |n.axis| cusp. No zero/sign(0) instantaneous
    derivative is invented: return a valid mask and zero-valued storage sentinel.
    Finite differences are a different interval measurement.
    """
    center, axis, plane_point, normal, velocity, omega = [np.asarray(v, float) for v in
        (center, axis, plane_point, plane_normal, center_velocity, angular_velocity)]
    require(all(v.shape == (3,) and np.isfinite(v).all() for v in
                (center, axis, plane_point, normal, velocity, omega)), "Geometry vectors must be finite 3-vectors")
    sizes = np.asarray([radius, halfwidth])
    require(sizes.shape == (2,) and sizes.dtype.kind in "iuf" and np.isfinite(sizes).all()
            and np.all(sizes > 0) and not any(isinstance(v, (bool, np.bool_)) for v in (radius, halfwidth)),
            "Cylinder sizes must be finite, nonboolean and positive")
    require(abs(np.linalg.norm(axis) - 1) <= 1e-9 and abs(np.linalg.norm(normal) - 1) <= 1e-9,
            "Cylinder axis and plane normal must be unit vectors")
    u = float(np.clip(axis @ normal, -1, 1))
    radial = float(np.sqrt(max(0., 1 - u * u)))
    support = halfwidth * abs(u) + radius * radial
    u_rate = float(normal @ np.cross(omega, axis))
    defined = u != 0. and radial != 0.
    support_rate = (halfwidth * np.sign(u) - radius * u / radial) * u_rate if defined else 0.
    return {"clearance_m": float(normal @ (center - plane_point) - support),
            "camber_rad": float(np.arcsin(u)), "axis_normal_dot": u, "axis_normal_dot_rate_s": u_rate,
            "support_extent_m": float(support), "center_plane_velocity_m_s": float(normal @ velocity),
            "clearance_rate_m_s": float(normal @ velocity - support_rate) if defined else 0.,
            "clearance_rate_defined": bool(defined)}


def interval_derivatives(times, values):
    times, values = np.asarray(times, float), np.asarray(values, float)
    require(times.ndim == 1 and len(times) >= 2 and values.shape[0] == len(times)
            and np.isfinite(times).all() and np.isfinite(values).all() and np.all(np.diff(times) > 0),
            "Finite differences need finite increasing times and aligned values")
    dt = np.diff(times).reshape((-1,) + (1,) * (values.ndim - 1))
    return {"start_time_s": times[:-1].copy(), "end_time_s": times[1:].copy(),
            "rate": np.diff(values, axis=0) / dt}


def recover_generalized_velocity(jacobian, named_joint_velocity, dof_ids, root_columns, desired_com_twist):
    jac, dq, desired = np.asarray(jacobian, float), np.asarray(named_joint_velocity, float), np.asarray(desired_com_twist, float)
    ids, root = np.asarray(dof_ids), np.asarray(root_columns)
    require(jac.shape == (6, 14) and dq.shape == (8,) and desired.shape == (6,)
            and all(np.isfinite(v).all() for v in (jac, dq, desired)), "Invalid COM Jacobian/velocity shape")
    require(ids.shape == (8,) and root.shape == (6,) and ids.dtype.kind in "iu" and root.dtype.kind in "iu"
            and set(ids.tolist()).isdisjoint(root.tolist()) and set(np.r_[ids, root].tolist()) == set(range(14)),
            "Named velocity columns must partition root and eight motor DOFs")
    qvel = np.zeros(14)
    qvel[ids] = dq
    qvel[root] = np.linalg.solve(jac[:, root], desired - jac @ qvel)
    require(np.isfinite(qvel).all() and np.allclose(jac @ qvel, desired, atol=1e-9, rtol=0),
            "Root COM/world angular velocity reconstruction failed")
    return qvel


def position_only(mj, model, data):
    """Position-stage collision only: qpos/qvel/time must not be changed."""
    before = (data.qpos.copy(), data.qvel.copy(), float(data.time))
    mj.mj_fwdPosition(model, data)
    require(np.array_equal(data.qpos, before[0]) and np.array_equal(data.qvel, before[1])
            and float(data.time) == before[2], "Position-only FK changed saved state/time")


def event_ledgers(recorded, geometry):
    """All count-zero events, whole episode and fixed source windows; no selection/bridging."""
    times = recorded["time_s"]
    result = []
    for row, wheel in zip(*np.where(recorded["wheel_ground_contact_count"] == 0)):
        context = []
        for j in range(max(0, int(row) - 2), min(len(times), int(row) + 3)):
            count = None if "manifold_count" not in geometry else int(geometry["manifold_count"][j, wheel])
            minimum = (None if count in (None, 0) else float(geometry["manifold_min_distance_m"][j, wheel]))
            defined = bool(geometry["clearance_rate_defined"][j, wheel])
            context.append({"sample": j, "time_s": float(times[j]), "mod4": j % 4,
                "archived_bearing_count_gt5N": int(recorded["wheel_ground_contact_count"][j, wheel]),
                "archived_ground_normal_vector_n": recorded["wheel_ground_contact_force_w_n"][j, wheel].tolist(),
                "archived_ground_total_vector_n": (recorded["wheel_ground_total_contact_force_w_n"][j, wheel].tolist()
                                                   if "wheel_ground_total_contact_force_w_n" in recorded else None),
                "reconstructed_manifold_count": count, "reconstructed_min_contact_distance_m": minimum,
                "analytic_clearance_m": float(geometry["clearance_m"][j, wheel]),
                "analytic_clearance_rate_m_s": float(geometry["clearance_rate_m_s"][j, wheel]) if defined else None,
                "analytic_clearance_rate_defined": defined,
                "wheel_link_plane_velocity_m_s": float(geometry["wheel_link_plane_velocity_m_s"][j, wheel]),
                "wheel_joint_velocity_rad_s": float(recorded["joint_vel"][j, 6 + wheel]),
                "joint_torques_nm": recorded["joint_torque_nm"][j].tolist(),
                "root_position_m": recorded["root_pos"][j].tolist(),
                "root_com_velocity_w_m_s": recorded["root_com_lin_vel_w"][j].tolist(),
                "root_angular_velocity_w_rad_s": recorded["root_ang_vel_w"][j].tolist()})
        raw_windows = [k + 1 for k, (a, b) in enumerate(WINDOWS) if a <= times[row] < b]
        nominal_windows = [k + 1 for k, (a, b) in enumerate(((156, 288), (364, 464))) if a <= row < b]
        result.append({"sample": int(row), "time_s": float(times[row]), "mod4": int(row) % 4,
                       "wheel": WHEEL_NAMES[wheel],
                       "raw_timestamp_halfopen_window_ids": raw_windows,
                       "nominal5msgrid_halfopen_window_ids": nominal_windows,
                       "window_assignment_disagrees": raw_windows != nominal_windows,
                       "context_minus2_plus2": context})
    return result


def reconstruct(mj, model, recorded):
    """Read the actual native qpos; solve named joint/root-COM velocities."""
    require(mj.__version__ == "3.9.0", "Exact MuJoCo 3.9.0 is required")
    require((model.nq, model.nv, model.nu) == (15, 14, 8), "Unexpected model dimensions")
    root = model.joint("root")
    require(int(root.type[0]) == int(mj.mjtJoint.mjJNT_FREE) and int(root.qposadr[0]) == 0
            and int(root.dofadr[0]) == 0, "Expected original floating root layout")
    qids = np.array([int(model.joint(k).qposadr[0]) for k in JOINT_NAMES])
    vids = np.array([int(model.joint(k).dofadr[0]) for k in JOINT_NAMES])
    base = model.body("base_Link").id
    require(np.allclose(model.body_ipos[base], ROOT_COM_LINK, atol=2e-7, rtol=0), "Unmerged original root COM required")
    wheel_ids = [model.body(name).id for name in WHEEL_NAMES]
    gids = [model.geom(name.replace("_Link", "_collision")).id for name in WHEEL_NAMES]
    floor = model.geom("floor").id
    require(int(model.geom_type[floor]) == int(mj.mjtGeom.mjGEOM_PLANE) and model.geom_bodyid[floor] == 0,
            "Expected original fixed plane")
    for side, (gid, body) in enumerate(zip(gids, wheel_ids)):
        require(int(model.geom_type[gid]) == int(mj.mjtGeom.mjGEOM_CYLINDER) and model.geom_bodyid[gid] == body
                and np.allclose(model.geom_size[gid, :2], [.127, .025], atol=1e-14, rtol=0)
                and np.allclose(model.geom_pos[gid], [0, .00707 * (1 if side == 0 else -1), 0], atol=1e-14, rtol=0),
                "Unexpected wheel cylinder geometry/offset")
    data = mj.MjData(model)
    n = len(recorded["time_s"])
    scalar_keys = ("clearance_m", "camber_rad", "axis_normal_dot", "axis_normal_dot_rate_s", "support_extent_m",
                   "center_plane_velocity_m_s", "clearance_rate_m_s", "wheel_link_plane_velocity_m_s")
    geometry = {k: np.zeros((n, 2)) for k in scalar_keys}
    geometry.update(clearance_rate_defined=np.zeros((n, 2), bool), qvel_reconstructed=np.zeros((n, 14)),
                    geom_center_w_m=np.zeros((n, 2, 3)), geom_axis_w=np.zeros((n, 2, 3)),
                    wheel_angular_velocity_w_rad_s=np.zeros((n, 2, 3)), plane_origin_w_m=np.zeros((n, 3)),
                    plane_normal_w=np.zeros((n, 3)))
    contacts = {k: [] for k in ("sample", "wheel", "distance_m", "position_w_m", "normal_geom1_to_geom2_w",
                              "geom1", "geom2", "dimension", "excluded", "efc_address", "includemargin_m")}
    geometry.update(manifold_count=np.zeros((n, 2), int), manifold_min_distance_m=np.zeros((n, 2)))
    max_errors = dict(root_position_m=0., root_com_velocity_m_s=0., root_angular_velocity_rad_s=0., wheel_link_position_m=0.)
    for row in range(n):
        data.qpos[:] = recorded["qpos"][row]
        require(np.array_equal(data.qpos[qids], recorded["joint_pos"][row]), "Native joint qpos mismatch")
        data.qvel[:] = 0.
        data.time = float(recorded["time_s"][row])
        position_only(mj, model, data)
        point = data.xpos[base] + data.xmat[base].reshape(3, 3) @ ROOT_COM_LINK
        jp, jr = np.zeros((3, 14)), np.zeros((3, 14))
        mj.mj_jac(model, data, jp, jr, point, base)
        jac = np.vstack((jp, jr))
        twist = np.r_[recorded["root_com_lin_vel_w"][row], recorded["root_ang_vel_w"][row]]
        data.qvel[:] = recover_generalized_velocity(jac, recorded["joint_vel"][row], vids, np.arange(6), twist)
        if row == 0:
            require(np.allclose(data.qvel, recorded["initial_qvel"], atol=1e-9, rtol=0), "Initial qvel reconstruction mismatch")
        geometry["qvel_reconstructed"][row] = data.qvel
        for key, actual, expected in (("root_position_m", data.xpos[base], recorded["root_pos"][row]),
                                     ("root_com_velocity_m_s", jp @ data.qvel, twist[:3]),
                                     ("root_angular_velocity_rad_s", jr @ data.qvel, twist[3:]),
                                     ("wheel_link_position_m", data.xpos[wheel_ids], recorded["wheel_pos"][row])):
            max_errors[key] = max(max_errors[key], float(np.max(np.abs(actual - expected))))
        plane_normal = data.geom_xmat[floor].reshape(3, 3)[:, 2]
        geometry["plane_origin_w_m"][row] = data.geom_xpos[floor]
        geometry["plane_normal_w"][row] = plane_normal
        for wheel, (gid, body) in enumerate(zip(gids, wheel_ids)):
            center, axis = data.geom_xpos[gid].copy(), data.geom_xmat[gid].reshape(3, 3)[:, 2].copy()
            mj.mj_jac(model, data, jp, jr, center, body)
            omega = jr @ data.qvel
            values = cylinder_plane_geometry(center, axis, *model.geom_size[gid, :2], data.geom_xpos[floor],
                                             plane_normal, jp @ data.qvel, omega)
            for key, value in values.items():
                geometry[key][row, wheel] = value
            mj.mj_jac(model, data, jp, jr, data.xpos[body], body)
            geometry["wheel_link_plane_velocity_m_s"][row, wheel] = float(plane_normal @ (jp @ data.qvel))
            geometry["geom_center_w_m"][row, wheel] = center
            geometry["geom_axis_w"][row, wheel] = axis
            geometry["wheel_angular_velocity_w_rad_s"][row, wheel] = omega
        for contact in data.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            if floor not in pair:
                continue
            for wheel, gid in enumerate(gids):
                if gid not in pair:
                    continue
                fields = (row, wheel, float(contact.dist), contact.pos.copy(), contact.frame[:3].copy(),
                          int(contact.geom1), int(contact.geom2), int(contact.dim), int(contact.exclude),
                          int(contact.efc_address), float(contact.includemargin))
                for key, value in zip(contacts, fields):
                    contacts[key].append(value)
                count = geometry["manifold_count"][row, wheel]
                if count == 0 or contact.dist < geometry["manifold_min_distance_m"][row, wheel]:
                    geometry["manifold_min_distance_m"][row, wheel] = contact.dist
                geometry["manifold_count"][row, wheel] += 1
    # Tolerances check FK storage/model agreement, never redefine a physical gap.
    tolerance = 1e-9
    require(all(v <= tolerance for v in max_errors.values()), f"Saved-state FK verification failed: {max_errors}")
    fd = interval_derivatives(recorded["time_s"], geometry["clearance_m"])
    geometry.update(clearance_fd_interval_start_s=fd["start_time_s"], clearance_fd_interval_end_s=fd["end_time_s"],
                    clearance_fd_interval_rate_m_s=fd["rate"])
    fd_link = interval_derivatives(recorded["time_s"], recorded["wheel_pos"])
    geometry["wheel_link_fd_interval_velocity_w_m_s"] = fd_link["rate"]
    for key, values in contacts.items():
        integer = key in ("sample", "wheel", "geom1", "geom2", "dimension", "excluded", "efc_address")
        tail = (3,) if key in ("position_w_m", "normal_geom1_to_geom2_w") else ()
        geometry[f"reconstructed_contact_{key}"] = np.asarray(values, dtype=int if integer else float).reshape((-1, *tail))
    summary = {"fk_verification_max_abs_error": max_errors, "fk_tolerance_not_gap_tolerance": tolerance,
               "samples": n, "geometry_scope": "current MuJoCo saved-state manifold",
               "manifold_count_semantics": "all wheel/plane geometric candidates, including excluded points; NOT bearing/load count",
               "compiled_wheel_geometry": [{"name": model.geom(gid).name,
                   "radius_m": float(model.geom_size[gid, 0]), "halfwidth_m": float(model.geom_size[gid, 1]),
                   "local_center_m": model.geom_pos[gid].tolist(), "local_quaternion_wxyz": model.geom_quat[gid].tolist(),
                   "axis": "geom local Z"} for gid in gids],
               "clearance_range_m_by_wheel": [[float(geometry["clearance_m"][:, w].min()), float(geometry["clearance_m"][:, w].max())] for w in range(2)],
               "instantaneous_derivative_undefined_samples_by_wheel": (~geometry["clearance_rate_defined"]).sum(axis=0).tolist(),
               "zero_bearing_count_events": event_ledgers(recorded, geometry)}
    return geometry, summary


def check_report(mujoco_report, contract, profile):
    require("schema_version" not in mujoco_report and mujoco_report.get("engine") == "MuJoCo"
            and mujoco_report.get("engine_version") == "3.9.0",
            "Original recorded MuJoCo version must be 3.9.0")
    require(typed_equal(mujoco_report.get("contract"), contract) and typed_equal(contract.get("task_profile"), profile),
            "Report/contract/profile mismatch")
    for key, expected in (("actor_dim", 51), ("critic_dim", 120), ("action_dim", 8), ("decimation", 4)):
        require(type(contract.get(key)) is int and contract[key] == expected, f"Unrecognized contract {key}")
    for key, value in (("physics_dt", PHYSICS_DT), ("policy_fps", 50), ("leg_kp", 500), ("leg_kd", 10)):
        require(type(contract.get(key)) in (int, float) and contract[key] == value, f"Unrecognized contract {key}")
    require(contract.get("terrain") is None and not any(k in contract for k in
            ("physics_clock_refinement", "controller_gain_diagnostic", "domain_parameters")), "Unsupported contract override")
    require(mujoco_report.get("terrain") is None and mujoco_report.get("physics_stepped") is True
            and mujoco_report.get("completed_full_reference") is True,
            "Require original complete single-episode MuJoCo archive")
    for key, expected in (("hidden_resets", 0), ("root_state_writes", 1), ("physics_steps", 1152),
                          ("policy_steps", 288), ("removed_visual_geom_count", 9)):
        require(type(mujoco_report.get(key)) is int and mujoco_report[key] == expected, f"Unrecognized report {key}")
    require(mujoco_report.get("root_com_link_offset_m") == ROOT_COM_LINK.tolist()
            and mujoco_report.get("no_visual_mesh") is True and mujoco_report.get("removed_visual_geom_count") == 9,
            "Unrecognized original model/COM/visual-removal convention")
    require(mujoco_report.get("model_sha256") == PINS["model"]
            and mujoco_report.get("evaluator_source_sha256") == PINS["mujoco_evaluator"], "Unrecognized original evaluator/model SHA")
    require(mujoco_report.get("contact_slip_helper_sha256") == PINS["contact_slip"], "Unknown normal/contact aggregate semantics")
    require(mujoco_report.get("contact", {}).get("force_threshold_n") == 5., "Original bearing threshold must be 5 N")


def input_files(trial_dir):
    directory = Path(trial_dir).resolve()
    require(directory.name in TRIALS and directory == ROOT / "outputs/drift-sim2sim-20261012" / directory.name,
            "Only original local trial013 and trial016 directories are recognized")
    profile_path = ROOT / TRIALS[directory.name][0]
    files = {"model": MODEL_PATH, "profile": profile_path,
             "contract": directory / "isaac/policy_contract.json",
             "mujoco_report": directory / "mujoco/report.json", "mujoco_rollout": directory / "mujoco/rollout.npz",
             "mujoco_evaluator": ROOT / "scripts/eval_tracking_mujoco.py", "drift_module": ROOT / "training/tron1_drift.py",
             "contact_slip": ROOT / "training/tron1_contact_slip.py", "assessor": ROOT / "scripts/assess_drift.py",
             "diagnostic": Path(__file__).resolve()}
    # Imported pure helper source dependencies are also immutable during the audit.
    for relative in ("scripts/compare_tracking.py", "scripts/eval_tracking.py", "training/tron1_physics_clock.py",
                     "training/tron1_domain_randomization.py", "training/tron1_ground_load_reserve.py"):
        files[relative] = ROOT / relative
    require(all(path.is_file() and not path.is_symlink() for path in files.values()), "Missing/symlinked input")
    return files


def verify_hashes(files, expected):
    actual = {name: sha256(path) for name, path in files.items()}
    require(typed_equal(actual, expected), "Input bytes changed during diagnostic")
    return actual


def audit_trial(trial_dir):
    files = input_files(trial_dir)
    before = {k: sha256(p) for k, p in files.items()}
    for key, expected in PINS.items():
        require(before[key] == expected, f"Frozen source/model pin mismatch: {key}")
    trial_id = Path(trial_dir).name
    require(before["profile"] == TRIALS[trial_id][1], "Frozen trial profile mismatch")
    require(before["mujoco_rollout"] == TRIALS[trial_id][2] and before["contract"] == TRIALS[trial_id][3],
            "Frozen original archive/contract identity mismatch (public evidence chain)")
    profile, contract = read_json(files["profile"]), read_json(files["contract"])
    mreport = read_json(files["mujoco_report"])
    require(mreport.get("contract_source_sha256") == before["contract"], "Original report contract source SHA mismatch")
    check_report(mreport, contract, profile)
    # Importing these pinned tools is inert; none of their CLI/run functions is called.
    for path in (ROOT, ROOT / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    evaluator = importlib.import_module("eval_tracking_mujoco")
    drift = importlib.import_module("training.tron1_drift")
    mj = importlib.import_module("mujoco")
    require(mj.__version__ == "3.9.0", "Exact MuJoCo 3.9.0 required; no fallback")
    recorded = normalize_archive(read_npz(files["mujoco_rollout"]))
    model, removed = evaluator.load_model(MODEL_PATH, no_visual_mesh=True)
    require(removed == 9, "Original noncolliding visual removal changed")
    collisions = evaluator.configure_model(model, contract)
    material = drift.configure_mujoco_materials(model, profile)
    require(typed_equal(material, mreport["contact_material_audit"]), "Compiled original material readback mismatch")
    require(np.all(model.geom_margin == 0) and np.all(model.geom_gap == 0), "Original margin/gap changed")
    require(float(model.opt.timestep) == PHYSICS_DT, "Original physics dt mismatch")
    geometry, summary = reconstruct(mj, model, recorded)
    payload = {f"archived_{k}": v for k, v in recorded.items()}
    payload.update({f"geometry_{k}": v for k, v in geometry.items()})
    verify_hashes(files, before)
    return {"trial_id": trial_id, "input_sha256": before,
            "input_files_relative_to_repository": {k: str(p.relative_to(ROOT)) for k, p in files.items()},
            "compiled_material": material, "self_collision_geoms": collisions,
            "mujoco_version": mj.__version__, "engine": "MuJoCo", "diagnostic": summary}, payload, files


def write_exclusive(output_dir, report, payload):
    output = Path(output_dir).resolve()
    require(output.is_relative_to(ROOT / "outputs") and not output.exists(), "Output must be a NEW private outputs directory")
    encoded = json.dumps(report, indent=2, allow_nan=False).encode() + b"\n"
    require(all(np.asarray(v).dtype.kind in "biufUS" and
                (np.asarray(v).dtype.kind in "US" or np.isfinite(v).all()) for v in payload.values()), "Nonfinite output payload")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "wheel_clearance.npz").open("xb") as stream:
        np.savez_compressed(stream, **payload)
    with (output / "wheel_clearance_audit.json").open("xb") as stream:
        stream.write(encoded)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="NEW private directory under repository outputs")
    args = parser.parse_args(argv)
    try:
        require(not args.output_dir.resolve().exists() and args.output_dir.resolve().is_relative_to(ROOT / "outputs"),
                "Preserving output: require NEW private directory")
        require(len(args.trial_dir) <= 2 and len({p.resolve() for p in args.trial_dir}) == len(args.trial_dir),
                "Select each original trial at most once")
        report = {"schema_version": 1, "status": "recorded_state_geometry_audited", "retrospective": True,
                  "new_physics_steps": 0, "new_actor_calls": 0, "new_episodes": 0, "solver_forces_recomputed": False,
                  "position_stage": "MuJoCo 3.9.0 mj_fwdPosition only",
                  "position_stage_primary_source": "https://github.com/google-deepmind/mujoco/blob/3.9.0/src/engine/engine_forward.c#L141",
                  "regenerated_forces_used": False, "actual_physx_manifold_available": False,
                  "gap_tolerance_or_bridging": False, "acceptance_gates_changed": False,
                  "quaternion_storage_coefficient_bound": {"absolute": float(QUATERNION_STORAGE_ATOL),
                      "definition": "32 * float64 epsilon; FK storage roundoff only, native qpos unchanged; NOT a gap or behavior tolerance"},
                  "source_windows_s": [list(v) for v in WINDOWS], "event_context_samples_each_side": 2,
                  "source_window_endpoint_convention": "half-open [start,end); event context may extend outside",
                  "window_clock_labels": "raw timestamp versus nominal integer grid [156,288),[364,464); both disclosed without tolerance or alignment",
                  "phase_label": "absolute physical sample index modulo 4; zero is initial/policy boundary, not a fitted phase",
                  "derivative_storage": "undefined instantaneous rate has mask false and NPZ zero sentinel; JSON null, NOT physical zero",
                  "contact_distance_storage": "empty manifold has count zero, NPZ minimum-distance zero sentinel and JSON null, NOT physical zero",
                  "policy_action_clock": "separate pre-action policy_time_s; actions are not shifted or assigned to instantaneous geometry",
                  "finite_difference_clock": "interval [sample_i,sample_i+1], not an instantaneous velocity or shifted contact phase",
                  "limitations": LIMITS, "trials": {}}
        payload, manifests = {}, []
        for path in args.trial_dir:
            result, arrays, files = audit_trial(path)
            report["trials"][result["trial_id"]] = result
            payload.update({f"{result['trial_id']}_{k}": v for k, v in arrays.items()})
            manifests.append((files, result["input_sha256"]))
        for files, expected in manifests:
            verify_hashes(files, expected)
        report["all_input_hashes_unchanged_before_output"] = True
        output = write_exclusive(args.output_dir, report, payload)
        print(json.dumps({"status": report["status"], "trials": list(report["trials"]), "output": str(output)}))
    except (ValueError, RuntimeError, FileExistsError, KeyError) as exc:
        parser.exit(1, f"Recorded-state wheel geometry audit refused: {exc}\n")


if __name__ == "__main__":
    main()
