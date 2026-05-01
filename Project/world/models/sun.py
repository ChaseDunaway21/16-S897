"""Sun position, eclipse, and solar radiation pressure models.

This model is heavily inspired by GNC-Simulation:
https://github.com/cmu-argus-2/GNC-Simulation.

References (Also including references from GNC-Simulation):
[1] O. Montenbruck and E. Gill, Satellite Orbits: Models, Methods, and
    Applications, Springer, 2000, Ch. 3 force models.
[2] cmu-argus-2/GNC-Simulation, argusim/world/physics/models/SRP.cpp and
    argusim/world/math/utils_and_transforms.cpp.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import spiceypy as spice
except ImportError:
    spice = None

from world.models.constants import (
    ASTRONOMICAL_UNIT,
    RADIUS_EARTH,
    RADIUS_SUN,
    SOLAR_CONSTANT_1_AU,
    SPEED_OF_LIGHT,
)
from world.math import unit_vector
from world.rotations_and_transformations import R_body_to_inertial


def sun_position_approx_eci(time_j2000_s: float) -> np.ndarray:
    """Return an analytical Sun position approximation in ECI [m]."""
    # Analytical fallback ported from GNC-Simulation sun_position_mod. [2]
    centuries = float(time_j2000_s) / (36525.0 * 86400.0)

    mean_anomaly = np.deg2rad(357.5291092 + 35999.05034 * centuries)
    mean_anomaly = np.mod(mean_anomaly, 2.0 * np.pi)
    sin_m = np.sin(mean_anomaly)
    cos_m = np.cos(mean_anomaly)
    sin_2m = 2.0 * sin_m * cos_m
    cos_2m = cos_m**2 - sin_m**2

    mean_longitude = 280.460 + 36000.771 * centuries
    ecliptic_longitude = mean_longitude + 1.914666471 * sin_m + 0.019994643 * sin_2m
    obliquity = 23.439291 - 0.0130042 * centuries

    lam = np.mod(np.deg2rad(ecliptic_longitude), 2.0 * np.pi)
    eps = np.mod(np.deg2rad(obliquity), 2.0 * np.pi)
    distance = (
        1.000140612 - 0.016708617 * cos_m - 0.000139589 * cos_2m
    ) * ASTRONOMICAL_UNIT

    return distance * np.array(
        [
            np.cos(lam),
            np.cos(eps) * np.sin(lam),
            np.sin(eps) * np.sin(lam),
        ]
    )


def partial_illumination_rel(
    r_earth_to_sc_m: np.ndarray, r_sun_to_sc_m: np.ndarray
) -> float:
    """Return sunlight fraction using the Montenbruck-Gill conical shadow model."""
    # Conical shadow geometry from Montenbruck-Gill, implemented in GNC-Simulation. [1], [2]
    r_earth = np.asarray(r_earth_to_sc_m, dtype=float)
    r_sun = np.asarray(r_sun_to_sc_m, dtype=float)
    r_mag = np.linalg.norm(r_earth)
    d_mag = np.linalg.norm(r_sun)

    if r_mag <= RADIUS_EARTH:
        return 0.0

    a = np.arcsin(np.clip(RADIUS_SUN / d_mag, -1.0, 1.0))
    b = np.arcsin(np.clip(RADIUS_EARTH / r_mag, -1.0, 1.0))
    c = np.arccos(np.clip(np.dot(r_earth, r_sun) / (r_mag * d_mag), -1.0, 1.0))

    if (a + b) <= c:
        return 1.0
    if c < (b - a):
        return 0.0

    x = (c**2 + a**2 - b**2) / (2.0 * c)
    y = np.sqrt(max(a**2 - x**2, 0.0))
    overlap = a**2 * np.arccos(np.clip(x / a, -1.0, 1.0))
    overlap += b**2 * np.arccos(np.clip((c - x) / b, -1.0, 1.0))
    overlap -= c * y
    return float(1.0 - overlap / (np.pi * a**2))


def partial_illumination(
    position_eci_m: np.ndarray, sun_position_eci_m: np.ndarray
) -> float:
    """Return sunlight fraction for an Earth-centered spacecraft position."""
    position = np.asarray(position_eci_m, dtype=float)
    sun_position = np.asarray(sun_position_eci_m, dtype=float)
    return partial_illumination_rel(position, position - sun_position)  # [2]


def frontal_area_factor(
    q_body_to_eci: np.ndarray,
    sun_direction_from_sc_eci: np.ndarray,
    surface_normal_body: np.ndarray | None = None,
) -> float:
    """Return projected-area scale factor for an illuminated body-frame normal."""
    # Project the chosen body-frame surface normal onto the Sun ray. [2]
    if surface_normal_body is None:
        return 1.0
    normal_eci = R_body_to_inertial(q_body_to_eci) @ unit_vector(surface_normal_body)
    return float(max(0.0, np.dot(normal_eci, unit_vector(sun_direction_from_sc_eci))))


class SunModel:
    """SPICE-backed Sun model with analytical fallback."""

    def __init__(
        self,
        direction_eci: np.ndarray | None = None,
        kernel_paths: list[str | Path] | None = None,
        use_spice: bool = True,
        require_spice: bool = False,
    ) -> None:
        self.fixed_direction = (
            None if direction_eci is None else unit_vector(direction_eci)
        )
        self.kernel_paths = [Path(path) for path in (kernel_paths or [])]
        self.use_spice = bool(use_spice)
        self.require_spice = bool(require_spice)
        self._kernels_loaded = False

    def _load_kernels(self) -> None:
        if self._kernels_loaded:
            return
        if spice is None:
            if self.require_spice:
                raise ImportError("spiceypy is required for this SunModel")
            return
        for kernel_path in self.kernel_paths:
            spice.furnsh(str(kernel_path))
        self._kernels_loaded = True

    def position_eci(self, time_j2000_s: float = 0.0) -> np.ndarray:
        """Return geocentric Sun position in J2000 ECI [m]."""
        if self.fixed_direction is not None:
            return ASTRONOMICAL_UNIT * self.fixed_direction

        if self.use_spice:
            self._load_kernels()
            if spice is not None and self._kernels_loaded:
                try:
                    # SPICE returns km
                    state_km, _ = spice.spkpos(
                        "SUN", float(time_j2000_s), "J2000", "NONE", "EARTH"
                    )
                    return 1000.0 * np.asarray(state_km, dtype=float)
                except Exception:
                    if self.require_spice:
                        raise

        return sun_position_approx_eci(time_j2000_s)

    def direction_eci(
        self, position_eci_m: np.ndarray | None = None, time_s: float = 0.0
    ) -> np.ndarray:
        """Return the unit Sun ray in ECI, from spacecraft to Sun when position is provided."""
        sun_position = self.position_eci(time_s)
        if position_eci_m is None:
            return unit_vector(sun_position)
        return unit_vector(sun_position - np.asarray(position_eci_m, dtype=float))

    def eclipse_factor(self, position_eci_m: np.ndarray, time_s: float = 0.0) -> float:
        return partial_illumination(position_eci_m, self.position_eci(time_s))

    def srp_acceleration(
        self,
        position_eci_m: np.ndarray,
        q_body_to_eci: np.ndarray,
        time_s: float,
        coefficient_reflectivity: float,
        area_m2: float,
        mass_kg: float,
        surface_normal_body: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return solar radiation pressure acceleration in ECI [m/s^2]."""
        position = np.asarray(position_eci_m, dtype=float)
        sun_position = self.position_eci(time_s)
        sun_to_sc = position - sun_position
        sc_to_sun = -sun_to_sc

        illumination = partial_illumination(position, sun_position)
        area_factor = frontal_area_factor(q_body_to_eci, sc_to_sun, surface_normal_body)
        pressure = (
            SOLAR_CONSTANT_1_AU
            / SPEED_OF_LIGHT
            * (ASTRONOMICAL_UNIT / np.linalg.norm(sun_to_sc)) ** 2
        )  # [1], [2]

        return (
            illumination
            * float(coefficient_reflectivity)
            * pressure
            * float(area_m2)
            * area_factor
            / float(mass_kg)
            * unit_vector(sun_to_sc)
        )
