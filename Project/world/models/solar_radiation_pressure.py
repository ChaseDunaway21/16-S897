"""Solar-radiation-pressure model.

This model is heavily inspired by GNC-Simulation:
https://github.com/cmu-argus-2/GNC-Simulation.

Though, the GNC-Simulation used a strange frontal-area-factor that was not based on the geometry of the satellite.
This replaces it with a summation of the projected surface areas of the satellite in a naive way as
the deployables do not 'cast' a shadow persay on the rest of the satellite.
"""

from __future__ import annotations

import numpy as np

from Project.world.math_utils import unit_vector
from world.models.constants import (
    ASTRONOMICAL_UNIT,
    SOLAR_CONSTANT_1_AU,
    SPEED_OF_LIGHT,
)
from world.models.sun import SunModel, partial_illumination
from world.rotations_and_transformations import R_body_to_inertial


def projected_area(
    q_body_to_eci: np.ndarray,
    vector_eci: np.ndarray,
    surface_areas_m2: np.ndarray,
    surface_normals_body: np.ndarray,
) -> float:
    """Return summed one-sided (i.e. facing the sun) projected area for configured body surfaces."""
    return float(
        np.sum(
            projected_surface_areas(
                q_body_to_eci,
                vector_eci,
                surface_areas_m2,
                surface_normals_body,
            )
        )
    )


def projected_surface_areas(
    q_body_to_eci: np.ndarray,
    vector_eci: np.ndarray,
    surface_areas_m2: np.ndarray,
    surface_normals_body: np.ndarray,
) -> np.ndarray:
    """Return per-surface one-sided projected areas along the given ECI vector."""
    direction = np.asarray(vector_eci, dtype=float).reshape(3)
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0.0:
        raise ValueError("projection vector norm is zero")
    direction = direction / direction_norm

    areas = np.asarray(surface_areas_m2, dtype=float).reshape(-1)
    normals_body = np.asarray(surface_normals_body, dtype=float)
    if normals_body.shape != (areas.size, 3):
        raise ValueError("surface normals must have shape (N, 3)")

    normal_norms = np.linalg.norm(normals_body, axis=1)
    if np.any(normal_norms == 0.0):
        raise ValueError("surface normals must be nonzero")

    normals_body = normals_body / normal_norms[:, np.newaxis]
    normals_eci = normals_body @ R_body_to_inertial(q_body_to_eci).T
    projection = np.maximum(0.0, normals_eci @ direction)
    return areas * projection


def srp_acceleration(
    q_body_to_eci: np.ndarray,
    position_eci_m: np.ndarray,
    time_s: float,
    coefficient_reflectivity: float,
    area_m2: float,
    mass_kg: float,
    sun_model: SunModel | None = None,
    surface_areas_m2: np.ndarray | None = None,
    surface_normals_body: np.ndarray | None = None,
) -> np.ndarray:
    """Return solar-radiation-pressure acceleration in ECI [m/s^2]."""
    if surface_areas_m2 is not None and surface_normals_body is not None:
        forces_eci = srp_forces_eci(
            q_body_to_eci,
            position_eci_m,
            time_s,
            coefficient_reflectivity,
            surface_areas_m2,
            surface_normals_body,
            sun_model=sun_model,
        )
        return np.sum(forces_eci, axis=0) / float(mass_kg)

    position = np.asarray(position_eci_m, dtype=float).reshape(3)
    sun = sun_model or SunModel()
    sun_position = sun.position_eci(time_s)
    sun_to_sc = position - sun_position
    sun_area_m2 = float(area_m2)
    shadow_factor = partial_illumination(position, sun_position)
    pressure = (SOLAR_CONSTANT_1_AU / SPEED_OF_LIGHT) * (
        ASTRONOMICAL_UNIT / np.linalg.norm(sun_to_sc)
    ) ** 2

    return (
        shadow_factor
        * float(coefficient_reflectivity)
        * pressure
        * sun_area_m2
        / float(mass_kg)
        * unit_vector(sun_to_sc)
    )


def srp_forces_eci(
    q_body_to_eci: np.ndarray,
    position_eci_m: np.ndarray,
    time_s: float,
    coefficient_reflectivity: float,
    surface_areas_m2: np.ndarray,
    surface_normals_body: np.ndarray,
    sun_model: SunModel | None = None,
) -> np.ndarray:
    """Return per-surface solar-radiation-pressure forces in ECI [N]."""
    position = np.asarray(position_eci_m, dtype=float).reshape(3)
    sun = sun_model or SunModel()
    sun_position = sun.position_eci(time_s)
    sun_to_sc = position - sun_position
    sc_to_sun = -sun_to_sc

    projected_areas = projected_surface_areas(
        q_body_to_eci,
        sc_to_sun,
        surface_areas_m2,
        surface_normals_body,
    )
    shadow_factor = partial_illumination(position, sun_position)
    pressure = (SOLAR_CONSTANT_1_AU / SPEED_OF_LIGHT) * (
        ASTRONOMICAL_UNIT / np.linalg.norm(sun_to_sc)
    ) ** 2

    return (
        shadow_factor
        * float(coefficient_reflectivity)
        * pressure
        * projected_areas[:, np.newaxis]
        * unit_vector(sun_to_sc)
    )


def srp_torque_body(
    q_body_to_eci: np.ndarray,
    position_eci_m: np.ndarray,
    time_s: float,
    coefficient_reflectivity: float,
    surface_areas_m2: np.ndarray,
    surface_normals_body: np.ndarray,
    surface_centers_body: np.ndarray,
    sun_model: SunModel | None = None,
) -> np.ndarray:
    """Return solar-radiation-pressure torque about the COM in body coordinates [N m]."""
    centers_body = np.asarray(surface_centers_body, dtype=float)
    areas = np.asarray(surface_areas_m2, dtype=float).reshape(-1)
    if centers_body.shape != (areas.size, 3):
        raise ValueError("surface centers must have shape (N, 3)")

    forces_eci = srp_forces_eci(
        q_body_to_eci,
        position_eci_m,
        time_s,
        coefficient_reflectivity,
        areas,
        surface_normals_body,
        sun_model=sun_model,
    )
    forces_body = (R_body_to_inertial(q_body_to_eci).T @ forces_eci.T).T
    return np.sum(np.cross(centers_body, forces_body), axis=0)
