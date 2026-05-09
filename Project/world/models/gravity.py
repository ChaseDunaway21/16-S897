"""
Gravity Model for the Satellite Simulation.
The model is simply based on a spherical acceleration model.
"Position" is assumed to be in ECI

Inspiration comes from GNC-Simulation/ARGUS-2's simulation framework

References:
[1] F. L. Markley and J. L. Crassidis, Fundamentals of Spacecraft Attitude Determination and Control, ser. Space Technology Library. New
    York, NY: Springer, 2014, vol. 33.
"""

from __future__ import annotations

import numpy as np
from world.models.constants import MU_EARTH, J2, RADIUS_EARTH
from world.rotations_and_transformations import R_body_to_inertial, inertial_to_body


def j2_acceleration(position: np.ndarray) -> np.ndarray:  # Eq 10.85 [1]
    """Compute gravitational acceleration."""

    acceleration = spherical_acceleration(position) + j2_perturbation(position)

    return acceleration


def spherical_acceleration(position: np.ndarray) -> np.ndarray:  # Eq 10.85 [1]
    """Compute spherical gravity acceleration."""

    position = np.asarray(position, dtype=float).reshape(-1)
    if position.size != 3:
        raise ValueError("Position invalid size")

    r = np.linalg.norm(position)
    if r == 0.0:
        raise ValueError("Position norm is 0")

    accel = -MU_EARTH * position / r**3

    return accel


def j2_perturbation(position: np.ndarray) -> np.ndarray:  # Eq 10.103a [1]
    """J2 Perturbation model. Returns a 3x1 column vector."""

    perturbation = np.zeros(3, dtype=float)

    J2_term = (3 / 2) * J2 * MU_EARTH * RADIUS_EARTH**2 / np.linalg.norm(position) ** 5

    perturbation[0] = (
        J2_term
        * position[0]
        * (5 * (position[2] ** 2) / np.linalg.norm(position) ** 2 - 1)
    )
    perturbation[1] = (
        J2_term
        * position[1]
        * (5 * (position[2] ** 2) / np.linalg.norm(position) ** 2 - 1)
    )
    perturbation[2] = (
        J2_term
        * position[2]
        * (5 * (position[2] ** 2) / np.linalg.norm(position) ** 2 - 3)
    )

    return perturbation


def gravity_gradient_torque_body(
    position_eci: np.ndarray, q: np.ndarray, inertia_tensor: np.ndarray
) -> np.ndarray:
    """Lecture 13 gravity-gradient torque [N m]."""
    r_eci = np.asarray(position_eci, dtype=float).reshape(3)
    r = np.linalg.norm(r_eci)
    R_body_to_eci = R_body_to_inertial(q)
    J_eci = R_body_to_eci @ np.asarray(inertia_tensor, dtype=float) @ R_body_to_eci.T
    torque_eci = 3.0 * MU_EARTH / r**5 * np.cross(r_eci, J_eci @ r_eci)
    return inertial_to_body(q, torque_eci)
