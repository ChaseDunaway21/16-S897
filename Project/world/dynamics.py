"""
Author: Chase Dunaway

Integrates the orbital dynamics of the ARGUS Satellite using RK4 and spherical acceleration.

OUTPUT:
    Time
    State
"""

from __future__ import annotations

import numpy as np
import world.models.gravity as gravity


def f(state: np.ndarray, current_time: float, dt: float) -> np.ndarray:
    """Compute the derivative of the state, different dynamic models can be added here later"""
    return orbital_dynamics(state, current_time, dt)

def orbital_dynamics(
    state: np.ndarray,
    current_time: float,
    dt: float,
    state_index: dict | None = None,
) -> np.ndarray:
    """Compute orbital dynamics."""
    _ = current_time
    _ = dt

    xdot = np.zeros_like(state)
    if state_index is None:
        xdot[:3] = state[3:6]
        xdot[3:6] = gravity.acceleration(state[:3])
        return xdot

    pos_slice = state_index["POS_ECEF"]
    vel_slice = state_index["VEL_ECEF"]

    xdot[pos_slice] = state[vel_slice]
    xdot[vel_slice] = gravity.acceleration(state[pos_slice])
    return xdot

def rk4_step(state: np.ndarray, current_time: float, dt: float, derivative_fn) -> np.ndarray:
    """Generic RK4 step for any state dimension and derivative function."""
    k1 = derivative_fn(state, current_time, dt)
    k2 = derivative_fn(state + 0.5 * dt * k1, current_time + 0.5 * dt, dt)
    k3 = derivative_fn(state + 0.5 * dt * k2, current_time + 0.5 * dt, dt)
    k4 = derivative_fn(state + dt * k3, current_time + dt, dt)

    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def integrate_dynamics(
    state: np.ndarray,
    current_time: float,
    dt: float,
    method: str = "rk4",
    derivative_fn=None,
    state_index: dict | None = None,
) -> np.ndarray:
    """Integrate the state using the defined method"""

    if derivative_fn is None:
        if state_index is not None:
            derivative_fn = lambda x, t, h: orbital_dynamics(x, t, h, state_index)
        else:
            derivative_fn = f

    if method == "rk4":
        x_new = rk4_step(state, current_time, dt, derivative_fn)

    else:
        raise ValueError(f"only RK4")
    
    return x_new