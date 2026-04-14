"""
Author: Chase Dunaway

Gravity Model for the Satellite Simulation.
The model is simply based on a spherical acceleration model.
"Position" is assumed to be in ECI

Inspiration comes from ARGUS-2's simulation framework

References:
[1] F. L. Markley and J. L. Crassidis, Fundamentals of Spacecraft Attitude Determination and Control, ser. Space Technology Library. New
    York, NY: Springer, 2014, vol. 33.
"""

from __future__ import annotations

import numpy as np
from world.models.constants import MU_EARTH, J2, RADIUS_EARTH
    
def acceleration(position: np.ndarray) -> np.ndarray:  # Eq 10.85 [1]
    """Compute gravitational acceleration."""
    
    accel = spherical_acceleration(position) + j2_perturbation(position)
    
    return accel


def spherical_acceleration(position: np.ndarray) -> np.ndarray: # Eq 10.85 [1]
    """Compute spherical gravity acceleration."""

    position = np.asarray(position, dtype=float).reshape(-1)
    if position.size != 3:
        raise ValueError("Position invalid size")

    r = np.linalg.norm(position)
    if r == 0.0:
        raise ValueError("Position norm is 0")

    accel = -MU_EARTH * position / r**3

    return accel


def j2_perturbation(position: np.ndarray) -> np.ndarray: # Eq 10.103a [1]
    """J2 Perturbation model. Returns a 3x1 column vector."""

    accel = np.zeros(3, dtype=float)

    J2_term = (3/2) * J2 * MU_EARTH * RADIUS_EARTH**2 / np.linalg.norm(position)**5

    accel[0] = J2_term * position[0] * (5 * (position[2]**2) / np.linalg.norm(position)**2 - 1)
    accel[1] = J2_term * position[1] * (5 * (position[2]**2) / np.linalg.norm(position)**2 - 1)
    accel[2] = J2_term * position[2] * (5 * (position[2]**2) / np.linalg.norm(position)**2 - 3)

    return accel