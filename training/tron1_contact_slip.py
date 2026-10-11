"""Ground-only, actual-contact wheel slip measurements for both simulators.

The geometry helper consumes recorded contact points rather than a guessed
wheel bottom. Linear velocities belong to the body COM, whose world position
must be supplied with them. The measured lateral direction is the wheel's
local +Y axle projected into each contact tangent plane.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


WHEEL_BODY_NAMES = ("wheel_L_Link", "wheel_R_Link")
WHEEL_GROUND_SENSOR_NAMES = ("wheel_L_ground", "wheel_R_ground")


def _vector(value, size, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite {size}-vector")
    return result


def _contacts(value, label):
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError(f"{label} must have finite shape (contacts, 3)")
    return result


def _unit_normals(normals):
    lengths = np.linalg.norm(normals, axis=-1)
    if np.any(lengths <= 1e-12):
        raise ValueError("Contact normals must be nonzero")
    return normals / lengths[:, None]


def _world_axle(link_quaternion_wxyz):
    quaternion = _vector(link_quaternion_wxyz, 4, "link_quaternion_wxyz")
    length = np.linalg.norm(quaternion)
    if length <= 1e-12:
        raise ValueError("Link quaternion must be nonzero")
    w, x, y, z = quaternion / length
    # Second column of the local-to-world quaternion rotation matrix.
    return np.array([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)])


def contact_point_lateral_slip_m_s(
    points_w, normals_w, body_com_position_w, body_com_linear_velocity_w,
    body_angular_velocity_w, body_link_quaternion_wxyz, ground_point_velocity_w=None,
):
    """Return signed relative lateral speed at every real contact point.

    ``ground_point_velocity_w`` may be a (3,) vector or (contacts, 3) array;
    it defaults to zero for the experiment's stationary ground. The returned
    values describe sliding at the contact, including link and wheel spin.
    """
    points = _contacts(points_w, "points_w")
    normals = _contacts(normals_w, "normals_w")
    if points.shape != normals.shape:
        raise ValueError("Point and normal counts differ")
    center = _vector(body_com_position_w, 3, "body_com_position_w")
    velocity = _vector(body_com_linear_velocity_w, 3, "body_com_linear_velocity_w")
    angular = _vector(body_angular_velocity_w, 3, "body_angular_velocity_w")
    axle = _world_axle(body_link_quaternion_wxyz)
    normals = _unit_normals(normals)
    tangent = axle - (normals @ axle)[:, None] * normals
    lengths = np.linalg.norm(tangent, axis=-1)
    if np.any(lengths <= 1e-8):
        raise ValueError("Wheel axle is normal to the contact plane; lateral direction is undefined")
    tangent /= lengths[:, None]
    if ground_point_velocity_w is None:
        ground = np.zeros_like(points)
    else:
        ground = np.asarray(ground_point_velocity_w, dtype=float)
        if ground.shape == (3,):
            ground = np.broadcast_to(ground, points.shape)
        if ground.shape != points.shape or not np.isfinite(ground).all():
            raise ValueError("Ground point velocity must be a finite 3-vector or match contact points")
    point_velocity = velocity + np.cross(angular, points - center) - ground
    return np.einsum("ij,ij->i", point_velocity, tangent)


def aggregate_ground_contacts(
    points_w, normals_w, normal_force_n, body_com_position_w,
    body_com_linear_velocity_w, body_angular_velocity_w,
    body_link_quaternion_wxyz, force_threshold_n=5.0, ground_point_velocity_w=None,
):
    """Return active count, summed normal force, and force-weighted slip RMS.

    A contact is active only when its normal force exceeds the threshold.
    RMS preserves opposite sliding directions within a wheel's contact patch.
    No active contact returns zero slip, which must be interpreted together
    with the count: zero slip with zero contact is not evidence of rolling.
    The force vector sums all supplied ground normal forces, not friction.
    """
    threshold = float(force_threshold_n)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("force_threshold_n must be finite and nonnegative")
    points = _contacts(points_w, "points_w")
    normals = _contacts(normals_w, "normals_w")
    force = np.asarray(normal_force_n, dtype=float)
    if points.shape != normals.shape or force.shape != (len(points),):
        raise ValueError("Contact points, normals and normal force shapes differ")
    if not np.isfinite(force).all() or np.any(force < 0):
        raise ValueError("Normal forces must be finite and nonnegative")
    normal_vectors = _unit_normals(normals)
    ground_force_w = np.sum(normal_vectors * force[:, None], axis=0)
    active = force > threshold
    count = int(np.count_nonzero(active))
    if not count:
        return count, ground_force_w, 0.0
    ground = ground_point_velocity_w
    if ground is not None and np.asarray(ground).ndim == 2:
        ground = np.asarray(ground)[active]
    slip = contact_point_lateral_slip_m_s(
        points[active], normals[active], body_com_position_w,
        body_com_linear_velocity_w, body_angular_velocity_w,
        body_link_quaternion_wxyz, ground,
    )
    rms = np.sqrt(np.sum(force[active] * slip * slip) / np.sum(force[active]))
    return count, ground_force_w, float(rms)


def configure_wheel_ground_sensors(scene_cfg, ground_prim_paths: Sequence[str], max_contacts_per_wheel=32):
    """Attach one filtered Isaac sensor per wheel for actual ground contacts.

    Call before constructing the scene. Caller supplies the actual collider
    prim paths after inspecting the ground asset; a parent prim is not assumed
    to be a valid rigid-contact filter. This uses the installed Isaac Lab 2.3
    API and does not depend on newer ``track_friction_forces`` support.
    """
    from isaaclab.sensors import ContactSensorCfg

    paths = list(ground_prim_paths)
    if not paths or any(not isinstance(path, str) or not path.startswith("/") for path in paths):
        raise ValueError("At least one absolute ground collider prim path is required")
    if not isinstance(max_contacts_per_wheel, int) or max_contacts_per_wheel < 1:
        raise ValueError("max_contacts_per_wheel must be a positive integer")
    for sensor_name, body_name in zip(WHEEL_GROUND_SENSOR_NAMES, WHEEL_BODY_NAMES):
        if getattr(scene_cfg, sensor_name, None) is not None:
            raise ValueError(f"Refusing to overwrite scene sensor {sensor_name}")
        setattr(scene_cfg, sensor_name, ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + body_name,
            update_period=0.0, history_length=0,
            filter_prim_paths_expr=paths,
            track_contact_points=True,
            max_contact_data_count_per_prim=max_contacts_per_wheel,
            debug_vis=False,
        ))
    return WHEEL_GROUND_SENSOR_NAMES


def configure_nonwheel_ground_sensors(
    scene_cfg, ground_prim_paths: Sequence[str], body_names: Sequence[str], max_contacts_per_body=32,
):
    """Create individual filtered sensors for chassis/leg ground-contact checks.

    Separate sensors are required because the installed PhysX contact filtering
    supports one sensor body per environment. These are evaluation sensors;
    they need not be added to the PPO observation or reward configuration.
    """
    from isaaclab.sensors import ContactSensorCfg

    paths, bodies = list(ground_prim_paths), list(body_names)
    if not paths or any(not isinstance(path, str) or not path.startswith("/") for path in paths):
        raise ValueError("At least one absolute ground collider prim path is required")
    if not bodies or len(set(bodies)) != len(bodies):
        raise ValueError("Nonwheel body names must be nonempty and unique")
    if any(not isinstance(name, str) or not name or "/" in name or name in WHEEL_BODY_NAMES for name in bodies):
        raise ValueError("Nonwheel sensors require exact nonwheel body leaf names")
    if not isinstance(max_contacts_per_body, int) or max_contacts_per_body < 1:
        raise ValueError("max_contacts_per_body must be a positive integer")
    sensor_names = tuple("nonwheel_ground_" + body for body in bodies)
    for sensor_name in sensor_names:
        if getattr(scene_cfg, sensor_name, None) is not None:
            raise ValueError(f"Refusing to overwrite scene sensor {sensor_name}")
    for sensor_name, body_name in zip(sensor_names, bodies):
        setattr(scene_cfg, sensor_name, ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + body_name,
            update_period=0.0, history_length=0,
            filter_prim_paths_expr=paths,
            track_contact_points=True,
            max_contact_data_count_per_prim=max_contacts_per_body,
            debug_vis=False,
        ))
    return sensor_names


def _numpy(tensor):
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy().copy()
    return np.asarray(tensor).copy()


def sample_raw_isaac_ground_contacts(
    env, wheel_sensor_names=WHEEL_GROUND_SENSOR_NAMES, force_threshold_n=5.0,
):
    """Sample real PhysX contact buffers and COM velocities into finite arrays.

    Returns three record fields shaped (environments, 2[, 3]). The lateral
    speed field is a nonnegative force-weighted RMS over active ground contact
    points. All coordinates remain in the world frame during the calculation.
    Sensors must each contain exactly one wheel body and at least one filter.
    The helper is intended for bounded evaluation, not every PPO physics step.
    """
    names = tuple(wheel_sensor_names)
    if len(names) != 2:
        raise ValueError("Exactly two wheel ground sensors are required")
    robot = env.scene["robot"]
    wheel_ids = [robot.body_names.index(name) for name in WHEEL_BODY_NAMES]
    center = _numpy(robot.data.body_com_pos_w[:, wheel_ids])
    velocity = _numpy(robot.data.body_com_lin_vel_w[:, wheel_ids])
    angular = _numpy(robot.data.body_com_ang_vel_w[:, wheel_ids])
    quaternion = _numpy(robot.data.body_link_quat_w[:, wheel_ids])
    count = np.zeros((env.num_envs, 2), dtype=np.int32)
    force = np.zeros((env.num_envs, 2, 3), dtype=np.float64)
    slip = np.zeros((env.num_envs, 2), dtype=np.float64)
    for wheel, sensor_name in enumerate(names):
        sensor = env.scene.sensors[sensor_name]
        if sensor.num_bodies != 1 or sensor.body_names != [WHEEL_BODY_NAMES[wheel]]:
            raise ValueError(f"{sensor_name} must select only {WHEEL_BODY_NAMES[wheel]}")
        if not sensor.cfg.filter_prim_paths_expr or sensor.contact_physx_view.filter_count < 1:
            raise ValueError(f"{sensor_name} has no ground filter")
        buffers = sensor.contact_physx_view.get_contact_data(dt=env.cfg.sim.dt)
        if len(buffers) != 6:
            raise ValueError("Unexpected PhysX get_contact_data schema")
        raw_force, points, normals, _, counts, starts = map(_numpy, buffers)
        raw_force = raw_force.reshape(-1)
        points, normals = points.reshape(-1, 3), normals.reshape(-1, 3)
        counts = counts.reshape(env.num_envs, -1)
        starts = starts.reshape(env.num_envs, -1)
        if counts.shape != starts.shape or counts.shape[1] != sensor.contact_physx_view.filter_count:
            raise ValueError("Unexpected PhysX sensor/filter row layout")
        for env_index in range(env.num_envs):
            indexes = []
            for n, start in zip(counts[env_index], starts[env_index]):
                n, start = int(n), int(start)
                if n < 0 or (n and (start < 0 or start + n > len(raw_force))):
                    raise ValueError("PhysX contact buffer overrun or invalid count")
                indexes.extend(range(start, start + n))
            indexes = np.asarray(indexes, dtype=int)
            count[env_index, wheel], force[env_index, wheel], slip[env_index, wheel] = aggregate_ground_contacts(
                points[indexes], normals[indexes], raw_force[indexes], center[env_index, wheel],
                velocity[env_index, wheel], angular[env_index, wheel], quaternion[env_index, wheel],
                force_threshold_n=force_threshold_n,
            )
    return {
        "wheel_ground_contact_count": count,
        "wheel_ground_contact_force_w_n": force,
        "wheel_lateral_contact_slip_m_s": slip,
    }


def sample_nonwheel_ground_force_norm(env, sensor_names, force_threshold_n=1.0):
    """Count actual force-bearing nonwheel ground points and sum normal forces.

    Only the configured ground partners contribute. Robot self-contact cannot
    contribute to these fields. The count uses a strict per-point threshold,
    while the force scalar includes all actual normal forces. Thus many weak
    contacts cannot conceal sustained chassis/leg support on the ground.
    """
    threshold = float(force_threshold_n)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("force_threshold_n must be finite and nonnegative")
    names = tuple(sensor_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Nonwheel ground sensor names must be nonempty and unique")
    total_count = np.zeros(env.num_envs, dtype=np.int32)
    total_normal_force = np.zeros(env.num_envs, dtype=np.float64)
    for sensor_name in names:
        sensor = env.scene.sensors[sensor_name]
        if sensor.num_bodies != 1 or sensor.body_names[0] in WHEEL_BODY_NAMES:
            raise ValueError(f"{sensor_name} must select exactly one nonwheel body")
        if not sensor.cfg.filter_prim_paths_expr or sensor.contact_physx_view.filter_count < 1:
            raise ValueError(f"{sensor_name} has no ground filter")
        buffers = sensor.contact_physx_view.get_contact_data(dt=env.cfg.sim.dt)
        if len(buffers) != 6:
            raise ValueError("Unexpected PhysX get_contact_data schema")
        raw_force = _numpy(buffers[0]).reshape(-1)
        counts = _numpy(buffers[4]).reshape(env.num_envs, -1)
        starts = _numpy(buffers[5]).reshape(env.num_envs, -1)
        if counts.shape != starts.shape or counts.shape[1] != sensor.contact_physx_view.filter_count:
            raise ValueError("Unexpected PhysX sensor/filter row layout")
        for env_index in range(env.num_envs):
            for n, start in zip(counts[env_index], starts[env_index]):
                n, start = int(n), int(start)
                if n < 0 or (n and (start < 0 or start + n > len(raw_force))):
                    raise ValueError("PhysX contact buffer overrun or invalid count")
                forces = raw_force[start:start + n] if n else np.empty(0)
                if not np.isfinite(forces).all() or np.any(forces < 0):
                    raise ValueError("Normal forces must be finite and nonnegative")
                total_count[env_index] += np.count_nonzero(forces > threshold)
                total_normal_force[env_index] += forces.sum()
    return {"nonwheel_ground_contact_count": total_count,
            "nonwheel_ground_contact_force_norm_n": total_normal_force}
