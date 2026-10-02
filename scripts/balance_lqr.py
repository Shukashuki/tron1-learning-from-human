"""Planar wheeled-inverted-pendulum dynamics and a sampled torque LQR.

Coordinates are axle travel x (forward positive) and body lean theta (forward
from upright positive). State order is [x, xdot, theta, thetadot], in SI units.
The input is the SUM of both wheels' forward drive torques, in N m. Equal
straight-line commands therefore give half this input to each wheel, after
accounting for each joint's axis sign.

This rigid-body approximation assumes fixed leg geometry, a horizontal floor,
continuous contact, rolling without slip, and negligible yaw/roll. Body inertia
is about its COM, not the axle. Both wheel masses and axle inertias are totals.

Virtual work is tau * (delta(x)/r - delta(theta)): the wheel drive also exerts
-tau on the body. Wheel rotational energy adds J_wheels/r**2 to the effective
translational mass. A cart-pole with external force tau/r omits this reaction.

References for the modeling conventions and LQR:
  https://mediatum.ub.tum.de/doc/1296097/document.pdf (Section 8.2, Eq. 8.4)
  https://wolfgangmerkt.com/publications/2020/iros20vlwip.pdf (Eq. 5)
  https://underactuated.mit.edu/lqr.html
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class WIPParameters:
    """Physical parameters of a rigid body riding on two coaxial wheels."""

    body_mass: float
    wheel_mass_total: float
    wheel_inertia_total: float
    radius: float
    length: float
    body_inertia: float
    gravity: float = 9.81

    def __post_init__(self) -> None:
        for name in ("body_mass", "radius", "length", "gravity"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("wheel_mass_total", "wheel_inertia_total", "body_inertia"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.determinant_upright) or self.determinant_upright <= 0:
            raise ValueError("Parameters must give a positive definite mass matrix")

    @property
    def effective_mass(self) -> float:
        return self.body_mass + self.wheel_mass_total + self.wheel_inertia_total / self.radius**2

    @property
    def inertia_about_axle(self) -> float:
        return self.body_inertia + self.body_mass * self.length**2

    @property
    def coupling(self) -> float:
        return self.body_mass * self.length

    @property
    def determinant_upright(self) -> float:
        # Expanded form avoids subtracting the common (m*l)**2 term.
        return (
            self.effective_mass * self.body_inertia
            + (self.wheel_mass_total + self.wheel_inertia_total / self.radius**2)
            * self.body_mass * self.length**2
        )


def linear_model(parameters: WIPParameters) -> tuple[np.ndarray, np.ndarray]:
    """Return continuous A (4x4), B (4x1) at upright rest with zero torque.

    theta > 0 with zero torque is unstable. At upright, positive torque gives
    positive axle acceleration and negative body angular acceleration.
    """
    p = parameters
    mass, inertia, coupling = p.effective_mass, p.inertia_about_axle, p.coupling
    determinant = p.determinant_upright
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -coupling**2 * p.gravity / determinant, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, mass * coupling * p.gravity / determinant, 0.0],
    ])
    B = np.array([
        [0.0],
        [(inertia / p.radius + coupling) / determinant],
        [0.0],
        [-(mass + coupling / p.radius) / determinant],
    ])
    return A, B


def nonlinear_rhs(
    state: Sequence[float], torque: float, parameters: WIPParameters
) -> np.ndarray:
    """Return state derivative for the nonlinear, frictionless planar model.

    The two acceleration equations are:
      H*xdd + m*l*cos(theta)*thetadd = tau/r + m*l*sin(theta)*thetadot**2
      m*l*cos(theta)*xdd + P*thetadd = m*g*l*sin(theta) - tau
    where H includes wheel spin inertia and P is body inertia about the axle.
    No saturation is applied here; the caller owns actuator/contact limits.
    """
    state = np.asarray(state, dtype=float)
    if state.shape != (4,) or not np.isfinite(state).all():
        raise ValueError("state must be a finite vector with shape (4,)")
    if np.ndim(torque) != 0 or not np.isfinite(torque):
        raise ValueError("torque must be a finite scalar")
    p = parameters
    _, velocity, theta, theta_velocity = state
    sine, cosine = math.sin(theta), math.cos(theta)
    mass, inertia, coupling = p.effective_mass, p.inertia_about_axle, p.coupling
    cross = coupling * cosine
    determinant = p.determinant_upright + coupling**2 * sine**2
    force_rhs = float(torque) / p.radius + coupling * sine * theta_velocity**2
    moment_rhs = coupling * p.gravity * sine - float(torque)
    acceleration = (inertia * force_rhs - cross * moment_rhs) / determinant
    theta_acceleration = (mass * moment_rhs - cross * force_rhs) / determinant
    return np.array([velocity, acceleration, theta_velocity, theta_acceleration])


@dataclass(frozen=True)
class LQRDesign:
    """Continuous plant, exact sampled plant, and discrete stabilizing gain."""

    A: np.ndarray
    B: np.ndarray
    Ad: np.ndarray
    Bd: np.ndarray
    K: np.ndarray
    P: np.ndarray
    closed_loop_poles: np.ndarray


def design_lqr(
    parameters: WIPParameters, dt: float, Qdiag: Sequence[float], R: float
) -> LQRDesign:
    """Design u[k] = -K @ (state[k] - reference[k]) with held wheel torque.

    Ad/Bd use the exact zero-order-hold matrix exponential. Qdiag and R specify
    the DISCRETE per-step cost sum(e.T @ diag(Qdiag) @ e + R*u**2); they are not
    an exact conversion of a continuous-time integral cost. Use the actual
    controller sample period for dt, including any physics-step decimation.

    SciPy is imported only for controller synthesis. Continuous dynamics and
    nonlinear evaluations otherwise require NumPy alone.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    Qdiag = np.asarray(Qdiag, dtype=float)
    if Qdiag.shape != (4,) or not np.isfinite(Qdiag).all() or np.any(Qdiag < 0):
        raise ValueError("Qdiag must contain four finite nonnegative weights")
    if not math.isfinite(R) or R <= 0:
        raise ValueError("R must be finite and positive")

    from scipy.linalg import expm, solve_discrete_are

    A, B = linear_model(parameters)
    augmented = np.zeros((5, 5))
    augmented[:4, :4] = A
    augmented[:4, 4:] = B
    transition = expm(dt * augmented)
    Ad, Bd = transition[:4, :4], transition[:4, 4:]
    input_cost = np.array([[R]])
    P = solve_discrete_are(Ad, Bd, np.diag(Qdiag), input_cost)
    K = np.linalg.solve(input_cost + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    poles = np.linalg.eigvals(Ad - Bd @ K)
    if not np.isfinite(K).all() or not np.isfinite(poles).all() or np.any(np.abs(poles) >= 1):
        raise ValueError("Weights/sample period did not produce an asymptotically stable LQR")
    return LQRDesign(A=A, B=B, Ad=Ad, Bd=Bd, K=K, P=P, closed_loop_poles=poles)
