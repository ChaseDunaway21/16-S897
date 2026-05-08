"""
Integrates the orbital dynamics of the ARGUS Satellite using RK4 and spherical acceleration.

OUTPUT:
    Time
    State
"""

from __future__ import annotations

import numpy as np

from world.math_utils import skew_symmetric
from world.rotations_and_transformations import attitude_jacobian as G
import world.models.gravity as gravity
import world.models.drag as drag
import world.models.solar_radiation_pressure as srp


def f(
    state: np.ndarray,
    state_index: dict,
    current_time: float,
    dt: float,
    inertia_tensor: np.ndarray,
    environment_model: dict | None = None,
    actuator_model: dict | None = None,
) -> np.ndarray:
    """Compute full state derivative from orbital and attitude dynamics."""
    _ = dt

    return (
        orbital_dynamics(state, state_index)
        + attitude_dynamics(
            state,
            state_index,
            inertia_tensor,
            current_time,
            environment_model,
            actuator_model,
        )
        + environmental_dynamics(
            state,
            state_index,
            current_time,
            environment_model,
        )
    )


def attitude_dynamics(
    state: np.ndarray,
    state_index: dict,
    inertia_tensor: np.ndarray,
    current_time: float,
    environment_model: dict | None = None,
    actuator_model: dict | None = None,
) -> np.ndarray:
    """Compute quaternion and angular-velocity dynamics."""
    state_dot = np.zeros_like(state)

    attitude_slice = state_index["ATTITUDE"]
    attitude_rate_slice = state_index["ATTITUDE_RATE"]

    q = state[attitude_slice]
    w = state[attitude_rate_slice]
    rho = state[state_index["RHO"]]

    qdot = 0.5 * G(q) @ w  # Attitude Jacobian from Notes

    # Using the numpy solver to speed up the simulation, but this is the same as the notes
    # (without external torques):
    # First, convert J into the principal components frame
    # Then solve each component of euler separately:
    # wdot_1 = -(J33 - J22)* w2 * w3 / J11
    # wdot_2 = -(J11 - J33)* w1 * w3 / J22
    # wdot_3 = -(J22 - J11)* w1 * w2 / J33

    # This time add the gyrostat momentum and body-frame torque terms.
    environmental_torque = environmental_torque_body(
        state, state_index, current_time, environment_model
    )
    actuator_torque = actuator_torque_body(
        state, state_index, current_time, actuator_model
    )
    wdot = np.linalg.solve(
        inertia_tensor,
        environmental_torque
        + actuator_torque
        - skew_symmetric(w) @ (inertia_tensor @ w + rho),
    )  # J wdot + w x (Jw + rho) = tau_env + tau_act

    state_dot[attitude_slice] = qdot
    state_dot[attitude_rate_slice] = wdot
    state_dot[state_index["RHO"]] = actuator_rho_dot_body(
        state, state_index, current_time, actuator_model
    )
    return state_dot


def orbital_dynamics(
    state: np.ndarray,
    state_index: dict,
) -> np.ndarray:
    """Return the position-kinematics and J2 perturbation to the full xdot."""
    state_dot = np.zeros_like(state)

    state_dot[state_index["POS_ECI"]] = state[state_index["VEL_ECI"]]
    state_dot[state_index["VEL_ECI"]] = gravity.j2_acceleration(
        state[state_index["POS_ECI"]]
    )

    return state_dot


def environmental_dynamics(
    state: np.ndarray,
    state_index: dict,
    current_time: float,
    environment_model: dict | None = None,
) -> np.ndarray:
    """Return non-gravitational translational acceleration contributions."""
    state_dot = np.zeros_like(state)
    acceleration_eci = environmental_acceleration(
        state, state_index, current_time, environment_model
    )

    state_dot[state_index["VEL_ECI"]] += acceleration_eci

    return state_dot


# This is wired into the accelerometer, but that sensor is often entirely unused
# I may just move this into environmental dynamics later
def environmental_acceleration(
    state: np.ndarray,
    state_index: dict,
    current_time: float,
    environment_model: dict | None = None,
) -> np.ndarray:
    """Return non-gravitational translational acceleration in ECI [m/s^2]."""
    if not environment_model:
        return np.zeros(3, dtype=float)

    position = state[state_index["POS_ECI"]]
    velocity = state[state_index["VEL_ECI"]]
    q = state[state_index["ATTITUDE"]]

    acceleration_eci = np.zeros(3, dtype=float)
    if environment_model.get("use_drag", False):
        acceleration_eci += drag.drag_acceleration(
            position,
            velocity,
            q,
            current_time,
            environment_model["drag_coefficient"],
            environment_model["reference_area_m2"],
            environment_model["mass_kg"],
            sun_model=environment_model.get("sun_model"),
            surface_areas_m2=environment_model.get("surface_areas_m2"),
            surface_normals_body=environment_model.get("surface_normals_body"),
        )

    if environment_model.get("use_srp", False):
        acceleration_eci += srp.srp_acceleration(
            q,
            position,
            current_time,
            environment_model["coefficient_reflectivity"],
            environment_model["reference_area_m2"],
            environment_model["mass_kg"],
            sun_model=environment_model.get("sun_model"),
            surface_areas_m2=environment_model.get("surface_areas_m2"),
            surface_normals_body=environment_model.get("surface_normals_body"),
        )

    return acceleration_eci


def environmental_torque_body(
    state: np.ndarray,
    state_index: dict,
    current_time: float,
    environment_model: dict | None = None,
) -> np.ndarray:
    """Return non-gravitational external torque about the COM in body coordinates [N m]."""
    if not environment_model:
        return np.zeros(3, dtype=float)

    surface_areas_m2 = environment_model.get("surface_areas_m2")
    surface_normals_body = environment_model.get("surface_normals_body")
    surface_centers_body = environment_model.get("surface_centers_body")
    if (
        surface_areas_m2 is None
        or surface_normals_body is None
        or surface_centers_body is None
    ):
        return np.zeros(3, dtype=float)

    position = state[state_index["POS_ECI"]]
    velocity = state[state_index["VEL_ECI"]]
    q = state[state_index["ATTITUDE"]]

    torque_body = np.zeros(3, dtype=float)
    if environment_model.get("use_drag", False):
        torque_body += drag.drag_torque_body(
            position,
            velocity,
            q,
            current_time,
            environment_model["drag_coefficient"],
            surface_areas_m2,
            surface_normals_body,
            surface_centers_body,
            sun_model=environment_model.get("sun_model"),
        )

    if environment_model.get("use_srp", False):
        torque_body += srp.srp_torque_body(
            q,
            position,
            current_time,
            environment_model["coefficient_reflectivity"],
            surface_areas_m2,
            surface_normals_body,
            surface_centers_body,
            sun_model=environment_model.get("sun_model"),
        )

    return torque_body


def actuator_torque_body(
    state: np.ndarray,
    state_index: dict,
    current_time: float,
    actuator_model: dict | None = None,
) -> np.ndarray:
    """Return actuator torque applied to the spacecraft body [N m]."""
    if not actuator_model:
        return np.zeros(3, dtype=float)

    torque_body = np.zeros(3, dtype=float)
    reaction_wheel = actuator_model.get("reaction_wheel")
    if reaction_wheel is not None:
        torque_body += reaction_wheel.get_torque(
            actuator_model.get("reaction_wheel_speeds", np.zeros(3))
        )

    magnetorquer = actuator_model.get("magnetorquer")
    if magnetorquer is not None:
        magnetic_field_model = actuator_model.get("magnetic_field_model")
        if magnetic_field_model is not None:
            magnetic_field_eci = magnetic_field_model.field_eci(
                state[state_index["POS_ECI"]], current_time
            )
            torque_body += magnetorquer.get_torque(
                actuator_model.get(
                    "magnetorquer_voltages", np.zeros(magnetorquer.N_MTBs)
                ),
                state[state_index["ATTITUDE"]],
                magnetic_field_eci,
            )

    return torque_body


def actuator_rho_dot_body(
    state: np.ndarray,
    state_index: dict,
    current_time: float,
    actuator_model: dict | None = None,
) -> np.ndarray:
    """Return gyrostat-momentum derivative from internal actuators [N m]."""
    return -actuator_torque_body(state, state_index, current_time, actuator_model)


def rk4_step(
    state: np.ndarray,
    current_time: float,
    dt: float,
    derivative_fn,
    state_index: dict,
    inertia_tensor: np.ndarray,
    environment_model: dict | None = None,
    actuator_model: dict | None = None,
) -> np.ndarray:
    """Generic RK4 step for any state dimension and derivative function."""
    k1 = derivative_fn(
        state,
        state_index,
        current_time,
        dt,
        inertia_tensor,
        environment_model,
        actuator_model,
    )
    k2 = derivative_fn(
        state + 0.5 * dt * k1,
        state_index,
        current_time + 0.5 * dt,
        dt,
        inertia_tensor,
        environment_model,
        actuator_model,
    )
    k3 = derivative_fn(
        state + 0.5 * dt * k2,
        state_index,
        current_time + 0.5 * dt,
        dt,
        inertia_tensor,
        environment_model,
        actuator_model,
    )
    k4 = derivative_fn(
        state + dt * k3,
        state_index,
        current_time + dt,
        dt,
        inertia_tensor,
        environment_model,
        actuator_model,
    )

    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_dynamics(
    spacecraft,
    current_time: float,
    dt: float,
    method: str = "rk4",
    derivative_fn=None,
    environment_model: dict | None = None,
    actuator_model: dict | None = None,
) -> np.ndarray:
    """Integrate spacecraft dynamics while using state/index/inertia from the spacecraft object."""

    state = spacecraft.get_state().astype(float, copy=True)
    state_index = spacecraft.Idx["X"]
    inertia_tensor = spacecraft.inertia_tensor

    if derivative_fn is None:
        derivative_fn = f

    if method == "rk4":
        attitude_slice = state_index["ATTITUDE"]

        x_new = rk4_step(
            state,
            current_time,
            dt,
            derivative_fn,
            state_index,
            inertia_tensor,
            environment_model,
            actuator_model,
        )
        quat_norm = np.linalg.norm(x_new[attitude_slice])
        if quat_norm > 0.0:
            x_new[attitude_slice] /= quat_norm

        spacecraft.set_state(x_new)

    else:
        raise ValueError("only RK4")

    return x_new
