"""Optional car-drift task profile; nominal tracking remains unchanged."""
import copy
import json
import math
from pathlib import Path


def validate_task_profile(profile):
    if not isinstance(profile, dict) or type(profile.get("schema_version")) is not int or profile.get("schema_version") != 1 or profile.get("task") != "car_drift":
        raise ValueError("Expected schema 1 car_drift task profile")
    out = copy.deepcopy(profile)
    for key, low, high in (("contact_friction", .01, 1.), ("env_spacing_m", 3., 20.)):
        value = out.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Invalid task {key}")
    weights = out.get("reward_weights", {})
    if not isinstance(weights, dict) or set(weights) != {"sideslip", "yaw_rate", "speed", "grounded", "tilt"}:
        raise ValueError("Drift profile requires all five reward weights")
    for key, value in weights.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 10:
            raise ValueError(f"Invalid drift reward {key}")
        if (key == "tilt" and value > 0) or (key != "tilt" and value < 0):
            raise ValueError(f"Invalid drift reward sign: {key}")
    for key, low, high in (("speed_std_m_s", .03, 1.), ("action_rate_weight", -2., 0.), ("leg_kd", 5., 50.),
                           ("leg_target_rate_weight", -5., 0.), ("wheel_vertical_velocity_weight", -5., 0.),
                           ("wheel_vertical_velocity_std_m_s", .005, .1)):
        if key in out:
            value = out[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"Invalid drift task {key}")
    if "wheel_vertical_velocity_weight" in out and "wheel_vertical_velocity_std_m_s" not in out:
        raise ValueError("Wheel vertical-velocity weight requires an explicit wheel_vertical_velocity_std_m_s")
    if "mujoco_contact_solref" in out:
        value = out["mujoco_contact_solref"]
        if (not isinstance(value, list) or len(value) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)
                or not .01 <= value[0] <= .05 or not 1. <= value[1] <= 2.):
            raise ValueError("Invalid explicit MuJoCo contact time constant/damping ratio")
    json.dumps(out, allow_nan=False)
    return out


def load_task_profile(path):
    return validate_task_profile(json.loads(Path(path).read_text()))


def apply_task_profile_to_contract(contract, profile):
    out = copy.deepcopy(contract)
    out["task_profile"] = validate_task_profile(profile)
    if "leg_kd" in out["task_profile"]:
        out["leg_kd"] = out["task_profile"]["leg_kd"]
    return out


def drift_motion_terms(env):
    import torch
    from isaaclab.utils.math import quat_apply, quat_inv
    command = env.command_manager.get_term("motion")
    robot = env.scene["robot"]
    actual_vel = quat_apply(quat_inv(robot.data.root_quat_w), robot.data.root_lin_vel_w)
    ref_vel = quat_apply(quat_inv(command.anchor_quat_w), command.body_lin_vel_w[:, 0])
    beta = torch.atan2(actual_vel[:, 1], actual_vel[:, 0])
    ref_beta = torch.atan2(ref_vel[:, 1], ref_vel[:, 0])
    error = torch.atan2(torch.sin(beta - ref_beta), torch.cos(beta - ref_beta))
    actual_speed = torch.linalg.vector_norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
    ref_speed = torch.linalg.vector_norm(command.body_lin_vel_w[:, 0, :2], dim=-1)
    return command, actual_speed, ref_speed, error


def sideslip_tracking(env):
    import torch
    _, speed, reference_speed, error = drift_motion_terms(env)
    # Low-speed angular readings are not rewarded as drift.
    return torch.exp(-(error / .12) ** 2) * (reference_speed > .3) * (speed > .25)


def speed_tracking(env, std=.35):
    import torch
    _, speed, target, _ = drift_motion_terms(env)
    return torch.exp(-((speed - target) / std) ** 2)


def yaw_rate_tracking(env):
    import torch
    command = env.command_manager.get_term("motion")
    actual = env.scene["robot"].data.root_ang_vel_w[:, 2]
    target = command.body_ang_vel_w[:, 0, 2]
    return torch.exp(-((actual - target) / .5) ** 2)


def grounded_reward(env):
    import torch
    sensor = env.scene.sensors["contact_forces"]
    names = ("wheel_L_Link", "wheel_R_Link")
    ids = [sensor.body_names.index(name) for name in names]
    forces = sensor.data.net_forces_w[:, ids]
    return torch.all(torch.linalg.vector_norm(forces, dim=-1) > 5., dim=-1).float()


def tilt_penalty(env):
    return (env.scene["robot"].data.projected_gravity_b[:, :2] ** 2).sum(-1)


def leg_target_rate_penalty(env):
    """Positive leg-only target-rate cost in rad^2/s^2; cfg supplies its sign.

    The action manager receives clipped [-1,1] actions in the tracking wrapper.
    Its first six actions map to leg targets with the existing 1 rad scale.
    Reuse its previous-action/reset state; no reward history is maintained.
    """
    delta = env.action_manager.action[:, :6] - env.action_manager.prev_action[:, :6]
    target_rate_rad_s = delta * 1.0 / env.step_dt
    return (target_rate_rad_s ** 2).sum(-1)


def wheel_vertical_velocity_penalty(env, std):
    """Positive dimensionless flat-drift cost sum((wheel-link vz/std)^2).

    Read world-frame LINK-ORIGIN velocity, not the legacy body/COM alias, on
    the tensor's current device. Reward evaluation at the 50 Hz policy clock
    sees the last physics substep only: this cannot certify continuous contact
    or constrain all intervening substeps. Target vz=0 is specific to flat drift.
    """
    robot = env.scene["robot"]
    ids = [robot.body_names.index(name) for name in ("wheel_L_Link", "wheel_R_Link")]
    vertical_speed = robot.data.body_link_lin_vel_w[:, ids, 2]
    return ((vertical_speed / std) ** 2).sum(-1)


def configure_drift(cfg, profile, eval_mode=False):
    profile = validate_task_profile(profile)
    from isaaclab.managers import RewardTermCfg
    mu = profile["contact_friction"]
    cfg.scene.env_spacing = profile["env_spacing_m"]
    if "leg_kd" in profile:
        cfg.scene.robot.actuators["legs"].damping = profile["leg_kd"]
    cfg.scene.terrain.physics_material.static_friction = mu
    cfg.scene.terrain.physics_material.dynamic_friction = mu
    cfg.sim.physics_material = cfg.scene.terrain.physics_material
    params = cfg.events.physics_material.params
    params["static_friction_range"] = (mu, mu)
    params["dynamic_friction_range"] = (mu, mu)
    # Existing global pose objectives remain active in addition to slip/velocity.
    cfg.rewards.motion_global_anchor_pos.weight = 2.0
    cfg.rewards.motion_global_anchor_ori.weight = 2.0
    cfg.rewards.motion_body_lin_vel.params["std"] = .5
    cfg.rewards.motion_body_ang_vel.params["std"] = .8
    funcs = {"sideslip": sideslip_tracking, "speed": speed_tracking, "yaw_rate": yaw_rate_tracking,
             "grounded": grounded_reward, "tilt": tilt_penalty}
    for key, weight in profile["reward_weights"].items():
        params = {"std": profile.get("speed_std_m_s", .35)} if key == "speed" else {}
        setattr(cfg.rewards, "drift_" + key, RewardTermCfg(func=funcs[key], weight=weight, params=params))
    if "action_rate_weight" in profile:
        cfg.rewards.action_rate_l2.weight = profile["action_rate_weight"]
    if "leg_target_rate_weight" in profile:
        cfg.rewards.drift_leg_target_rate = RewardTermCfg(
            func=leg_target_rate_penalty, weight=profile["leg_target_rate_weight"], params={})
    if "wheel_vertical_velocity_weight" in profile:
        cfg.rewards.drift_wheel_vertical_velocity = RewardTermCfg(
            func=wheel_vertical_velocity_penalty, weight=profile["wheel_vertical_velocity_weight"],
            params={"std": profile["wheel_vertical_velocity_std_m_s"]})
    if eval_mode:
        from training.tron1_contact_slip import configure_wheel_ground_sensors, configure_nonwheel_ground_sensors
        from training.tron1_tracking import NON_WHEEL_BODY_NAMES
        ground_paths = ["/World/ground/terrain/GroundPlane/CollisionPlane"]
        configure_wheel_ground_sensors(cfg.scene, ground_paths)
        configure_nonwheel_ground_sensors(cfg.scene, ground_paths, NON_WHEEL_BODY_NAMES)
    return profile


def audit_isaac_materials(env, profile):
    import numpy as np
    from pxr import UsdPhysics, UsdShade
    import omni.usd
    mu = validate_task_profile(profile)["contact_friction"]
    materials = env.scene["robot"].root_physx_view.get_material_properties().cpu().numpy()
    if not np.allclose(materials[..., :2], mu, atol=1e-6):
        raise RuntimeError("Runtime robot friction differs from the declared drift contract")
    legs = env.scene["robot"].actuators["legs"]
    kp, kd = legs.stiffness.detach().cpu().numpy(), legs.damping.detach().cpu().numpy()
    if not np.allclose(kp, 500.) or not np.allclose(kd, profile.get("leg_kd", 10.)):
        raise RuntimeError("Runtime leg PD gains differ from the declared drift contract")
    stage = omni.usd.get_context().get_stage()
    ground = []
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith("/World/ground") and prim.HasAPI(UsdPhysics.CollisionAPI):
            mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            if not mat:
                raise RuntimeError(f"Ground collider has no bound physics material: {prim.GetPath()}")
            api = UsdPhysics.MaterialAPI(mat.GetPrim())
            static, dynamic = api.GetStaticFrictionAttr().Get(), api.GetDynamicFrictionAttr().Get()
            if static is None or dynamic is None or not np.allclose([static, dynamic], mu, atol=1e-6):
                raise RuntimeError("Authored ground friction differs from the declared drift contract")
            ground.append({"collider": str(prim.GetPath()), "material": str(mat.GetPath()),
                           "static_friction": float(static), "dynamic_friction": float(dynamic)})
    if not ground:
        raise RuntimeError("No ground collider found during material audit")
    ground_paths = {item["collider"] for item in ground}
    for name, sensor in env.scene.sensors.items():
        if (name in ("wheel_L_ground", "wheel_R_ground") or name.startswith("nonwheel_ground_")) and not set(sensor.cfg.filter_prim_paths_expr) <= ground_paths:
            raise RuntimeError(f"Contact filter does not resolve to an audited ground collider: {name}")
    return {"expected_friction": mu, "robot_runtime_friction_range": [float(materials[..., :2].min()), float(materials[..., :2].max())],
            "leg_pd_gain_readback": {"kp_range": [float(kp.min()), float(kp.max())], "kd_range": [float(kd.min()), float(kd.max())]},
            "ground_colliders": ground, "ground_audit": "USD-bound material; robot materials additionally read back from live PhysX"}


def configure_mujoco_materials(model, profile):
    import numpy as np
    profile = validate_task_profile(profile)
    ids = np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    model.geom_friction[ids, 0] = profile["contact_friction"]
    if "mujoco_contact_solref" in profile:
        model.geom_solref[ids] = profile["mujoco_contact_solref"]
    if not np.allclose(model.geom_friction[ids, 0], profile["contact_friction"]):
        raise RuntimeError("MuJoCo contact-material assignment failed")
    return {"expected_friction": profile["contact_friction"], "colliding_geom_count": len(ids),
            "explicit_contact_solref": profile.get("mujoco_contact_solref"),
            "compiled_solref_ranges": [[float(model.geom_solref[ids, i].min()), float(model.geom_solref[ids, i].max())] for i in range(2)],
            "refsafe_minimum_timeconst_s": 2. * float(model.opt.timestep),
            "contact_margin_and_gap_changed": False,
            "compiled_friction_range": [float(model.geom_friction[ids, 0].min()), float(model.geom_friction[ids, 0].max())],
            "combination_note": "Both contacting materials have the same value; max/average combination agree."}
