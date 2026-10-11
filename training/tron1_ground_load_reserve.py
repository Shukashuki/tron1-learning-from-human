"""Explicitly enabled, last-substep wheel/ground normal-load reserve cost.

The dimensionless cost is sum(exp(-max(F_ground,z, 0)/(r*m*g))).  ``r`` is
an exponential SCALE, not a minimum-force threshold or a required load split.
The symmetric cost has a soft balanced-load preference and can trade against
tracking.  It does not prove continuous contact, stability, or hardware safety.
Only the last physical substep is read at the unchanged 50 Hz reward clock;
there is no history, accumulator, reset hook, or extra physics/actor call.

Torch and simulator imports are lazy.  Live masses are CPU tensors in the
installed native API: double-precision startup reduction is intentionally a
one-time host operation.  The cached ``mg_n`` tensor stays on the GPU.  Runtime
finite checks and Isaac's lazy sensor refresh can synchronize the host; this
module does NOT claim a synchronization-free reward.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import importlib
import math
from numbers import Real
from pathlib import Path


WHEEL_BODY_NAMES = ("wheel_L_Link", "wheel_R_Link")
SENSOR_NAMES = ("wheel_L_ground", "wheel_R_ground")
GROUND_COLLIDER = "/World/ground/terrain/GroundPlane/CollisionPlane"
CACHE_ATTRIBUTE = "_tron1_ground_load_reserve"
PHYSICS_DT_S = .005
POLICY_DT_S = .02


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite non-boolean number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite non-boolean number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite non-boolean number")
    return result


def validate_load_reserve_settings(settings):
    """Validate and copy exactly the two opt-in parameters; never mutate input."""
    if not isinstance(settings, dict) or set(settings) != {"weight", "scale_fraction"}:
        raise ValueError("Ground-load reserve requires exactly weight and scale_fraction")
    weight = _number(settings["weight"], "weight")
    scale = _number(settings["scale_fraction"], "scale_fraction")
    if not -2. <= weight <= 0.:
        raise ValueError("Ground-load reserve weight must be in [-2, 0]")
    if not .05 <= scale <= .30:
        raise ValueError("Ground-load reserve scale_fraction must be in [.05, .30]")
    return {"weight": weight, "scale_fraction": scale}


def configure_training_ground_force_sensors(scene_cfg):
    """Add only two single-wheel/one-ground sensors before scene construction.

    Preflight both names before modifying the scene.  Evaluation has its own
    point-recording sensors and must not call this training-only configurator.
    """
    for name in SENSOR_NAMES:
        if getattr(scene_cfg, name, None) is not None:
            raise ValueError(f"Refusing to overwrite existing ground sensor: {name}")
    from isaaclab.sensors import ContactSensorCfg
    sensors = [ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + body,
        update_period=0., history_length=0,
        filter_prim_paths_expr=[GROUND_COLLIDER],
        track_contact_points=False, track_air_time=False, track_pose=False,
        max_contact_data_count_per_prim=32, debug_vis=False,
    ) for body in WHEEL_BODY_NAMES]
    for name, sensor in zip(SENSOR_NAMES, sensors):
        setattr(scene_cfg, name, sensor)


def _validate_clock(env):
    values = (env.cfg.sim.dt, env.physics_dt, env.sim.get_physics_dt())
    if any(_number(value, "physics dt") != PHYSICS_DT_S for value in values):
        raise RuntimeError("Ground-load reserve requires the declared 200 Hz physics clock")
    if type(env.cfg.decimation) is not int or env.cfg.decimation != 4:
        raise RuntimeError("Ground-load reserve requires four physical substeps per reward")
    if _number(env.step_dt, "policy dt") != POLICY_DT_S:
        raise RuntimeError("Ground-load reserve requires the unchanged 50 Hz reward clock")


def _validate_events(env):
    """Reject anything beyond the three inspected, identity-checked event terms."""
    tracking = importlib.import_module("training.tron1_tracking")
    motor = importlib.import_module("training.tron1_domain_randomization")
    expected = {
        "physics_material": ("startup", tracking.mdp.randomize_rigid_body_material),
        "legacy_joint_friction": ("startup", tracking.clear_legacy_joint_friction),
        "domain_randomization": ("reset", motor.reset_domain_randomization),
    }
    active = env.event_manager.active_terms
    if not isinstance(active, dict):
        raise RuntimeError("Cannot audit active event terms for cached mass/gravity")
    seen, audit = set(), []
    for mode, names in active.items():
        if not isinstance(names, (list, tuple)):
            raise RuntimeError("Malformed active event term list")
        for name in names:
            if name in seen or name not in expected:
                raise RuntimeError(f"Unsafe or duplicate event for cached mass/gravity: {name}")
            seen.add(name)
            expected_mode, function = expected[name]
            term = env.event_manager.get_term_cfg(name)
            # Isaac's ManagerBase initializes ManagerTermBase classes in place.
            # Accept ONLY that exact class, not subclasses or name lookalikes.
            identity_matches = (term.func is function or
                                (isinstance(function, type) and type(term.func) is function))
            if mode != expected_mode or term.mode != expected_mode or not identity_matches:
                raise RuntimeError(f"Unexpected event identity or mode: {name}")
            if name == "domain_randomization":
                if not isinstance(term.params, dict) or set(term.params) != {"profile"}:
                    raise RuntimeError("Motor reset event must contain only its inspected profile")
                motor.validate_profile(term.params["profile"])
            audit.append({"name": name, "mode": mode,
                          "function": function.__module__ + "." + function.__name__,
                          "initialized_exact_class_instance": type(term.func) is function})
    if not {"physics_material", "legacy_joint_friction"} <= seen:
        raise RuntimeError("Missing inspected material or legacy joint-friction startup event")
    return audit


def _gravity_tuple(value, label):
    try:
        if len(value) != 3:
            raise ValueError
        gravity = tuple(_number(value[i], label) for i in range(3))
    except (TypeError, IndexError, ValueError) as exc:
        raise RuntimeError(f"{label} must be a finite three-vector") from exc
    if gravity[0] != 0. or gravity[1] != 0. or gravity[2] >= 0.:
        raise RuntimeError(f"{label} must be strictly downward vertical gravity")
    return gravity


def _ground_prim_audit():
    import omni.usd
    from pxr import UsdPhysics
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(GROUND_COLLIDER)
    if not prim or not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError("Ground filter must resolve to the actual plane collision prim")
    prim_type = prim.GetTypeName()
    if prim_type not in ("Mesh", "Plane"):
        raise RuntimeError("Ground collider must retain the declared plane or plane-mesh representation")
    return {"path": GROUND_COLLIDER, "usd_type": prim_type,
            "collision_api_verified": True,
            "native_cooked_geometry_readback_performed": False}


def _sensor_matrix(env, sensor, expected_body, torch):
    cfg = sensor.cfg
    if (cfg.update_period != 0. or cfg.history_length != 0
            or cfg.track_air_time or tuple(cfg.filter_prim_paths_expr) != (GROUND_COLLIDER,)):
        raise RuntimeError("Wheel ground sensor must use one actual-ground filter with no history")
    if tuple(sensor.body_names) != (expected_body,) or sensor.num_bodies != 1:
        raise RuntimeError("Each filtered ground sensor must observe exactly its single wheel")
    expected_paths = {"{ENV_REGEX_NS}/Robot/" + expected_body,
                      env.scene.env_regex_ns + "/Robot/" + expected_body}
    if cfg.prim_path not in expected_paths:
        raise RuntimeError("Wheel sensor prim expression differs from the exact wheel body")
    view = sensor.contact_physx_view
    if view.filter_count != 1 or view.sensor_count != env.num_envs:
        raise RuntimeError("Native wheel-ground view must have N sensors and one filter")
    if (cfg.max_contact_data_count_per_prim < 32
            or view.max_contact_data_count < 32 * env.num_envs):
        raise RuntimeError("Ground contact point capacity must preserve at least 32 entries per wheel")
    if _number(sensor._sim_physics_dt, "sensor impulse dt") != PHYSICS_DT_S:
        raise RuntimeError("Native ground force must convert impulses with physical dt, not policy dt")
    matrix = sensor.data.force_matrix_w
    if matrix is None or tuple(matrix.shape) != (env.num_envs, 1, 1, 3):
        raise RuntimeError("Ground normal-force matrix must have shape (N,1,1,3)")
    if matrix.dtype != torch.float32 or torch.device(matrix.device) != torch.device(env.device):
        raise RuntimeError("Ground force matrix must be float32 on the environment CUDA device")
    if not bool(torch.isfinite(matrix).all()):
        raise RuntimeError("Nonfinite ground normal force")
    return matrix


@dataclass(frozen=True)
class _ReserveCache:
    """Private explicit cache; ``mg_n`` is available for independent smoke audit."""
    settings: dict
    mg_n: object
    device: object
    num_envs: int
    gravity_cfg: tuple
    robot: object
    sensors: tuple
    audit: dict


def initialize_ground_load_reserve(env, settings):
    """Validate startup physics/sensors/events, cache GPU mg once, return JSON audit.

    Zero-mass welded links are allowed, but every environment must have positive
    finite total mass.  Initial zero forces are legitimate; loaded contact-vector
    orientation/aggregate agreement still requires the prospective runtime smoke.
    A second initializer call is rejected rather than silently replacing a cache.
    """
    settings = validate_load_reserve_settings(settings)
    if hasattr(env, CACHE_ATTRIBUTE):
        raise RuntimeError("Ground-load reserve has already been explicitly initialized")
    torch = importlib.import_module("torch")
    _validate_clock(env)
    event_audit = _validate_events(env)
    device = torch.device(env.device)
    if device.type != "cuda" or device.index is None:
        raise RuntimeError("Ground-load reserve requires an explicit CUDA device ordinal")
    if type(env.num_envs) is not int or env.num_envs < 1:
        raise RuntimeError("Ground-load reserve requires a positive integer environment count")
    ground = _ground_prim_audit()
    robot = env.scene["robot"]
    if any(tuple(robot.body_names).count(body) != 1 for body in WHEEL_BODY_NAMES):
        raise RuntimeError("Robot must contain exactly one of each expected wheel body")
    sensors = tuple(env.scene.sensors.get(name) for name in SENSOR_NAMES)
    if any(sensor is None for sensor in sensors):
        raise RuntimeError("Explicit wheel-ground sensors must exist before reserve initialization")
    matrices = [_sensor_matrix(env, sensor, body, torch)
                for sensor, body in zip(sensors, WHEEL_BODY_NAMES)]
    if any(bool((matrix[..., 2] < -1.e-4).any()) for matrix in matrices):
        raise RuntimeError("Startup ground normal force points downward on a wheel")

    masses = robot.root_physx_view.get_masses()
    if (torch.device(masses.device).type != "cpu"
            or tuple(masses.shape) != (env.num_envs, len(robot.body_names))
            or not torch.is_floating_point(masses)):
        raise RuntimeError("Live native masses must be floating CPU tensors of shape (N, bodies)")
    masses64 = masses.detach().to(dtype=torch.float64).clone()
    if not bool(torch.isfinite(masses64).all()) or bool((masses64 < 0.).any()):
        raise RuntimeError("Live link masses must be finite and nonnegative")
    total_mass = masses64.sum(dim=1)
    if not bool((total_mass > 0.).all()):
        raise RuntimeError("Every environment must have strictly positive total robot mass")
    gravity_cfg = _gravity_tuple(env.cfg.sim.gravity, "configured gravity")
    manager = importlib.import_module("isaacsim.core.simulation_manager").SimulationManager
    gravity_native = _gravity_tuple(manager.get_physics_sim_view().get_gravity(), "native gravity")
    if any(not math.isclose(a, b, rel_tol=1.e-6, abs_tol=1.e-7)
           for a, b in zip(gravity_cfg, gravity_native)):
        raise RuntimeError("Native gravity differs from configured gravity")
    mg_cpu64 = total_mass * abs(gravity_native[2])
    mg_n = mg_cpu64.to(device=device, dtype=torch.float32)
    if not bool(torch.isfinite(mg_n).all()) or not bool((mg_n > 0.).all()):
        raise RuntimeError("Cached CUDA mg must be finite and positive")
    audit = {
        "schema_version": 1, "settings": copy.deepcopy(settings),
        "formula": "sum(exp(-clamp(ground_normal_force_world_z_N,min=0)/(scale_fraction*mg_N)))",
        "sampling": "last_physics_substep", "physics_dt_s": PHYSICS_DT_S,
        "policy_dt_s": POLICY_DT_S, "decimation": 4,
        "force_evidence": "filtered ContactSensor.data.force_matrix_w; normal forces only",
        "sensor_names": list(SENSOR_NAMES), "wheel_body_names": list(WHEEL_BODY_NAMES),
        "sensor_matrix_shape": [env.num_envs, 1, 1, 3], "ground_collider": ground,
        "training_contact_point_tracking_required": False,
        "sensor_track_contact_points": [bool(sensor.cfg.track_contact_points) for sensor in sensors],
        "native_mass_source": "ArticulationView.get_masses() CPU (N,bodies)",
        "mass_reduction_dtype": "float64", "cached_mg_dtype": "float32",
        "mass_kg_range": [float(total_mass.min()), float(total_mass.max())],
        "first_environment_link_mass_kg": masses64[0].tolist(),
        "mg_N_range": [float(mg_cpu64.min()), float(mg_cpu64.max())],
        "configured_gravity_m_s2": list(gravity_cfg),
        "native_gravity_m_s2": list(gravity_native), "cuda_device": str(device),
        "num_envs": env.num_envs, "active_event_identity_audit": event_audit,
        "startup_force_finite_and_nonnegative_z": True,
        "startup_loaded_contact_required": False,
        "loaded_normal_vector_and_raw_aggregate_runtime_smoke_required": True,
        "raw_normal_forces_thresholded": False, "scale_is_minimum_force": False,
        "cost_bounds": [0., 2.], "continuous_contact_certified": False,
        "host_synchronization_free": False, "reset_or_history_state_added": False,
        "soft_balanced_load_preference": True,
        "tracking_tradeoff_and_no_hardware_safety_claim": True,
        "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    setattr(env, CACHE_ATTRIBUTE, _ReserveCache(settings, mg_n, device, env.num_envs,
                                              gravity_cfg, robot, sensors, copy.deepcopy(audit)))
    return copy.deepcopy(audit)


def wheel_ground_load_reserve_penalty(env, scale_fraction):
    """Return a positive per-env cost; the configured reward weight supplies sign."""
    cache = getattr(env, CACHE_ATTRIBUTE, None)
    if not isinstance(cache, _ReserveCache):
        raise RuntimeError("Ground-load reserve reward requires explicit startup initialization")
    scale = _number(scale_fraction, "scale_fraction")
    if scale != cache.settings["scale_fraction"]:
        raise RuntimeError("Reward scale differs from the explicitly initialized reserve settings")
    torch = importlib.import_module("torch")
    if (env.num_envs != cache.num_envs or torch.device(env.device) != cache.device
            or tuple(cache.mg_n.shape) != (cache.num_envs,)
            or cache.mg_n.dtype != torch.float32
            or torch.device(cache.mg_n.device) != cache.device):
        raise RuntimeError("Ground-load reserve cache shape/device changed")
    _validate_clock(env)
    _validate_events(env)
    if _gravity_tuple(env.cfg.sim.gravity, "configured gravity") != cache.gravity_cfg:
        raise RuntimeError("Configured gravity changed after mg initialization")
    if env.scene["robot"] is not cache.robot:
        raise RuntimeError("Robot changed after live mass initialization")
    if not bool(torch.isfinite(cache.mg_n).all()) or not bool((cache.mg_n > 0.).all()):
        raise RuntimeError("Cached mg is invalid")
    matrices = []
    for name, body, sensor in zip(SENSOR_NAMES, WHEEL_BODY_NAMES, cache.sensors):
        if env.scene.sensors.get(name) is not sensor:
            raise RuntimeError("Wheel ground sensor changed after initialization")
        matrices.append(_sensor_matrix(env, sensor, body, torch))
    ground_fn_z = torch.stack([matrix[:, 0, 0, 2] for matrix in matrices], dim=1)
    # Out-of-place clamp: raw sensor vectors (including unloaded zeros) survive.
    return torch.exp(-torch.clamp(ground_fn_z, min=0.) / (scale * cache.mg_n[:, None])).sum(dim=1)
