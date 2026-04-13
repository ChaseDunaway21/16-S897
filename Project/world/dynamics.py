"""
Author: Chase Dunaway

Integrates the orbital dynamics of the ARGUS Satellite using RK4 and spherical acceleration.

OUTPUT:
    Time
    State
"""

from __future__ import annotations

import numpy as np

from world.math import skew_symmetric
import world.models.gravity as gravity

def f(
    state: np.ndarray,
    state_index: dict,
    current_time: float,
    dt: float,
    inertia_tensor: np.ndarray,
) -> np.ndarray:
    """Compute full state derivative from orbital and attitude dynamics."""
    _ = current_time
    _ = dt

    state_dot = np.zeros_like(state)

    orbital_dynamics(state, state_dot, state_index)
    attitude_dynamics(state, state_dot, state_index, inertia_tensor)

    return state_dot
    
def attitude_dynamics(
    state: np.ndarray,
    state_dot: np.ndarray,
    state_index: dict,
    inertia_tensor: np.ndarray,
) -> np.ndarray:
    """Compute quaternion and angular-velocity dynamics."""

    attitude_slice = state_index["ATTITUDE"]
    attitude_rate_slice = state_index["ATTITUDE_RATE"]

    q = state[attitude_slice]
    w = state[attitude_rate_slice]

    # Scalar-first quaternion kinematics
    q_scalar = q[0]
    q_vector = q[1:4]

    # This is the same as the in-class lecture notes, but made to run efficiently with numpy
    qdot_scalar = -0.5 * float(np.dot(q_vector, w))
    qdot_vector = 0.5 * (q_scalar * w + np.cross(q_vector, w))
    qdot = np.hstack((qdot_scalar, qdot_vector))
    wdot = np.linalg.solve(inertia_tensor, -skew_symmetric(w) @ (inertia_tensor @ w))

    state_dot[attitude_slice] = qdot
    state_dot[attitude_rate_slice] = wdot
    return state_dot

def orbital_dynamics(state: np.ndarray, state_dot: np.ndarray, state_index: dict) -> np.ndarray:
    """Compute translational orbital dynamics in the full state vector."""

    pos_slice = state_index["POS_ECI"]
    vel_slice = state_index["VEL_ECI"]

    state_dot[pos_slice] = state[vel_slice]
    state_dot[vel_slice] = gravity.acceleration(state[pos_slice])

    return state_dot

def rk4_step(state: np.ndarray, current_time: float, dt: float, derivative_fn) -> np.ndarray:
    """Generic RK4 step for any state dimension and derivative function."""
    k1 = derivative_fn(state, current_time, dt)
    k2 = derivative_fn(state + 0.5 * dt * k1, current_time + 0.5 * dt, dt)
    k3 = derivative_fn(state + 0.5 * dt * k2, current_time + 0.5 * dt, dt)
    k4 = derivative_fn(state + dt * k3, current_time + dt, dt)

    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def integrate_dynamics(
    spacecraft,
    current_time: float,
    dt: float,
    method: str = "rk4",
    derivative_fn=None,
) -> np.ndarray:
    """Integrate spacecraft dynamics while using state/index/inertia from the spacecraft object."""

    state = spacecraft.get_state().astype(float, copy=True)
    state_index = spacecraft.Idx["X"]
    inertia_tensor = spacecraft.compute_inertia_tensor()

    if derivative_fn is None:
        derivative_fn = lambda x, t, h: f(x, state_index, t, h, inertia_tensor)

    if method == "rk4":
        attitude_slice = state_index["ATTITUDE"]

        x_new = rk4_step(state, current_time, dt, derivative_fn)
        quat_norm = np.linalg.norm(x_new[attitude_slice])
        if quat_norm > 0.0:
            x_new[attitude_slice] /= quat_norm

        spacecraft.set_state(x_new)

    else:
        raise ValueError(f"only RK4")
    
    return x_new