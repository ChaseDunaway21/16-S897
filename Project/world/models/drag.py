"""Atmospheric drag model.

This model is heavily inspired by GNC-Simulation:
https://github.com/cmu-argus-2/GNC-Simulation.

Harris, Isadore, and Wolfgang Priester. “Time-Dependent Structure of the Upper Atmosphere.”
Journal of the Atmospheric Sciences, vol. 19, no. 4, 1962, pp. 286–301.
https://doi.org/10.1175/1520-0469(1962)019<0286:TDSOTU>2.0.CO;2
"""

from __future__ import annotations

import numpy as np

from world.models.constants import (
    EARTH_ROTATION_RATE,
    GMST_J2000,
    HP_ALTITUDES_KM,
    HP_RHO_MAX,
    HP_RHO_MIN,
    HP_PARAMETER,
    RA_LAG_RAD,
)
from world.models.solar_radiation_pressure import projected_surface_areas
from world.models.sun import SunModel
from world.rotations_and_transformations import (
    R_body_to_inertial,
    rotate_around_z,
    geodetic_from_ecef,
)


def density(
    position_eci_m: np.ndarray, time_s: float, sun_model: SunModel | None = None
) -> float:
    """Return Harris-Priester atmospheric density [kg/m^3]."""
    position = np.asarray(position_eci_m, dtype=float).reshape(3)
    gmst = GMST_J2000 + EARTH_ROTATION_RATE * float(time_s)
    position_ecef = rotate_around_z(-gmst) @ position
    _, _, altitude_km = geodetic_from_ecef(position_ecef)

    if altitude_km < HP_ALTITUDES_KM[0] or altitude_km > HP_ALTITUDES_KM[-1]:
        return 0.0

    ih = np.searchsorted(HP_ALTITUDES_KM, altitude_km, side="right") - 1
    ih = int(np.clip(ih, 0, HP_ALTITUDES_KM.size - 2))

    h_min = (HP_ALTITUDES_KM[ih] - HP_ALTITUDES_KM[ih + 1]) / np.log(
        HP_RHO_MIN[ih + 1] / HP_RHO_MIN[ih]
    )
    h_max = (HP_ALTITUDES_KM[ih] - HP_ALTITUDES_KM[ih + 1]) / np.log(
        HP_RHO_MAX[ih + 1] / HP_RHO_MAX[ih]
    )
    d_min = HP_RHO_MIN[ih] * np.exp((HP_ALTITUDES_KM[ih] - altitude_km) / h_min)
    d_max = HP_RHO_MAX[ih] * np.exp((HP_ALTITUDES_KM[ih] - altitude_km) / h_max)

    sun = sun_model or SunModel()
    sun_position = sun.position_eci(time_s)
    right_ascension = np.arctan2(sun_position[1], sun_position[0])
    if right_ascension < 0.0:
        right_ascension += 2.0 * np.pi
    declination = np.arcsin(sun_position[2] / np.linalg.norm(sun_position))

    cos_dec = np.cos(declination)
    bulge_unit = np.array(
        [
            cos_dec * np.cos(right_ascension + RA_LAG_RAD),
            cos_dec * np.sin(right_ascension + RA_LAG_RAD),
            np.sin(declination),
        ],
        dtype=float,
    )
    c_psi2 = 0.5 + 0.5 * np.dot(position, bulge_unit) / np.linalg.norm(position)

    return float((d_min + (d_max - d_min) * c_psi2**HP_PARAMETER) * 1.0e-12)


def drag_acceleration(
    position_eci_m: np.ndarray,
    velocity_eci_mps: np.ndarray,
    q_body_to_eci: np.ndarray,
    time_s: float,
    drag_coefficient: float,
    area_m2: float,
    mass_kg: float,
    sun_model: SunModel | None = None,
    surface_areas_m2: np.ndarray | None = None,
    surface_normals_body: np.ndarray | None = None,
) -> np.ndarray:
    """Return atmospheric-drag acceleration in ECI [m/s^2]."""
    if surface_areas_m2 is not None and surface_normals_body is not None:
        forces_eci = drag_forces_eci(
            position_eci_m,
            velocity_eci_mps,
            q_body_to_eci,
            time_s,
            drag_coefficient,
            surface_areas_m2,
            surface_normals_body,
            sun_model=sun_model,
        )
        return np.sum(forces_eci, axis=0) / float(mass_kg)

    velocity = np.asarray(velocity_eci_mps, dtype=float).reshape(3)
    rho = density(position_eci_m, time_s, sun_model=sun_model)
    area_frontal = float(area_m2)

    return (
        -0.5
        * float(drag_coefficient)
        * rho
        * area_frontal
        * np.linalg.norm(velocity)
        / float(mass_kg)
        * velocity
    )


def drag_forces_eci(
    position_eci_m: np.ndarray,
    velocity_eci_mps: np.ndarray,
    q_body_to_eci: np.ndarray,
    time_s: float,
    drag_coefficient: float,
    surface_areas_m2: np.ndarray,
    surface_normals_body: np.ndarray,
    sun_model: SunModel | None = None,
) -> np.ndarray:
    """Return per-surface atmospheric-drag forces in ECI [N]."""
    velocity = np.asarray(velocity_eci_mps, dtype=float).reshape(3)
    velocity_norm = np.linalg.norm(velocity)
    areas = np.asarray(surface_areas_m2, dtype=float).reshape(-1)
    if velocity_norm == 0.0:
        return np.zeros((areas.size, 3), dtype=float)

    rho = density(position_eci_m, time_s, sun_model=sun_model)
    projected_areas = projected_surface_areas(
        q_body_to_eci,
        velocity,
        areas,
        surface_normals_body,
    )

    return (
        -0.5
        * float(drag_coefficient)
        * rho
        * projected_areas[:, np.newaxis]
        * velocity_norm
        * velocity
    )


def drag_torque_body(
    position_eci_m: np.ndarray,
    velocity_eci_mps: np.ndarray,
    q_body_to_eci: np.ndarray,
    time_s: float,
    drag_coefficient: float,
    surface_areas_m2: np.ndarray,
    surface_normals_body: np.ndarray,
    surface_centers_body: np.ndarray,
    sun_model: SunModel | None = None,
) -> np.ndarray:
    """Return atmospheric-drag torque about the COM in body coordinates [N m]."""
    centers_body = np.asarray(surface_centers_body, dtype=float)
    areas = np.asarray(surface_areas_m2, dtype=float).reshape(-1)
    if centers_body.shape != (areas.size, 3):
        raise ValueError("surface centers must have shape (N, 3)")

    forces_eci = drag_forces_eci(
        position_eci_m,
        velocity_eci_mps,
        q_body_to_eci,
        time_s,
        drag_coefficient,
        areas,
        surface_normals_body,
        sun_model=sun_model,
    )
    forces_body = (R_body_to_inertial(q_body_to_eci).T @ forces_eci.T).T
    return np.sum(np.cross(centers_body, forces_body), axis=0)
