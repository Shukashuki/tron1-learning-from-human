"""Pure, strict clocks for evaluation-only numerical timestep refinement.

The policy interface remains 50 Hz with 51 observations and eight actions.
This module does not change training, task profiles, rewards, or motor models.
"""
from __future__ import annotations

import copy
import math


PHYSICS_CLOCKS = {0.005: 4, 0.0025: 8, 0.00125: 16}
POLICY_FPS = 50.0
POLICY_DT_S = 1.0 / POLICY_FPS


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite non-boolean number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite non-boolean number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite non-boolean number")
    return number


def _clock_pair(dt, decimation):
    dt = _number(dt, "physics_dt")
    if dt not in PHYSICS_CLOCKS:
        raise ValueError("Supported physics timesteps are 0.005, 0.0025, and 0.00125 seconds")
    if type(decimation) is not int or decimation != PHYSICS_CLOCKS[dt]:
        raise ValueError("Physics timestep and integer decimation must preserve the 0.02s policy interval")
    return dt, decimation


def validate_physics_clock(contract):
    """Return the inspected (physics_dt, decimation), failing closed on drift.

    Legacy minimal contracts may omit policy_fps, which means the unchanged
    50 Hz interface. Refined metadata, when present, must describe this clock.
    """
    if not isinstance(contract, dict):
        raise ValueError("Physics clock requires a deployment contract object")
    for key, expected in (("actor_dim", 51), ("action_dim", 8)):
        if type(contract.get(key)) is not int or contract[key] != expected:
            raise ValueError("Physics refinement requires the inspected 51-observation / 8-action interface")
    if _number(contract.get("policy_fps", POLICY_FPS), "policy_fps") != POLICY_FPS:
        raise ValueError("Physics refinement must preserve the 50 Hz policy")
    dt, decimation = _clock_pair(contract.get("physics_dt"), contract.get("decimation"))
    if "physics_clock_refinement" in contract:
        metadata = contract["physics_clock_refinement"]
        if (not isinstance(metadata, dict) or type(metadata.get("schema_version")) is not int
                or metadata["schema_version"] != 1
                or metadata.get("scope") != "evaluation_only_numerical_refinement"):
            raise ValueError("Invalid evaluation physics-clock refinement metadata")
        actual = _clock_pair(metadata.get("actual_physics_dt_s"), metadata.get("actual_decimation"))
        _clock_pair(metadata.get("trained_physics_dt_s"), metadata.get("trained_decimation"))
        if actual != (dt, decimation):
            raise ValueError("Refinement metadata differs from the actual deployment clock")
        if (_number(metadata.get("policy_dt_s"), "refinement policy_dt_s") != POLICY_DT_S
                or _number(metadata.get("policy_fps"), "refinement policy_fps") != POLICY_FPS):
            raise ValueError("Refinement metadata must preserve the 50 Hz policy interval")
    return dt, decimation


def configure_physics_clock(cfg, dt):
    """Apply an explicit evaluation override before constructing the environment.

    Filtered contact-point sensors already update every step (period zero).
    The aggregate net-force sensor must follow the refined physical clock.
    """
    validate_physics_clock({"actor_dim": 51, "action_dim": 8, "policy_fps": POLICY_FPS,
                            "physics_dt": cfg.sim.dt, "decimation": cfg.decimation})
    requested = _number(dt, "requested physics timestep")
    pair = _clock_pair(requested, PHYSICS_CLOCKS.get(requested))
    cfg.sim.dt, cfg.decimation = pair
    cfg.sim.render_interval = cfg.decimation
    cfg.scene.contact_forces.update_period = cfg.sim.dt
    return pair


def apply_physics_clock_to_contract(contract, dt, trained_contract=None):
    """Deep-copy actual and nominal trained clocks; never change task semantics.

    None means no explicit override, retaining the historical contract exactly.
    The caller supplies the checkpoint manifest's contract when available.
    """
    validate_physics_clock(contract)
    out = copy.deepcopy(contract)
    if dt is None:
        return out
    trained = contract if trained_contract is None else trained_contract
    trained_dt, trained_decimation = validate_physics_clock(trained)
    if trained.get("task_profile") != contract.get("task_profile"):
        raise ValueError("Numerical refinement must not change the trained task profile")
    requested = _number(dt, "requested physics timestep")
    actual_dt, actual_decimation = _clock_pair(requested, PHYSICS_CLOCKS.get(requested))
    out.update(physics_dt=actual_dt, decimation=actual_decimation)
    out["physics_clock_refinement"] = {
        "schema_version": 1, "scope": "evaluation_only_numerical_refinement",
        "trained_physics_dt_s": trained_dt, "trained_decimation": trained_decimation,
        "actual_physics_dt_s": actual_dt, "actual_decimation": actual_decimation,
        "policy_dt_s": POLICY_DT_S, "policy_fps": POLICY_FPS,
    }
    validate_physics_clock(out)
    return out
