"""Minimal magnetic-field models for sensor simulation.

This model is heavily inspired by GNC-Simulation:
https://github.com/cmu-argus-2/GNC-Simulation.

However, GNC-Simulation uses IGRF-13

References:
[1] International Association of Geomagnetism and Aeronomy.
    IGRF-14. Zenodo, 22 Nov. 2024, https://doi.org/10.5281/zenodo.14012303.
[2] cmu-argus-2/GNC-Simulation, argusim/world/physics/models/MagneticField.cpp.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import ppigrf

from world.models.constants import (
    EARTH_ROTATION_RATE,
    GMST_J2000,
    J2000_UTC,
)
from world.math_utils import scalar_value
from world.rotations_and_transformations import (
    rotate_around_z,
    enu_to_ecef,
    geodetic_from_ecef,
)


class MagneticFieldModel:
    """Earth magnetic field in ECI using IGRF-14."""

    def __init__(self) -> None:
        self._cached_time_s: float | None = None
        self._cached_position_eci_m: np.ndarray | None = None
        self._cached_field_eci: np.ndarray | None = None

    def field_eci(self, position_eci_m: np.ndarray, time_s: float = 0.0) -> np.ndarray:
        """Return magnetic flux density [uT] at an ECI position."""
        r = np.asarray(position_eci_m, dtype=float).reshape(3)
        t = float(time_s)

        if (
            self._cached_time_s == t
            and self._cached_position_eci_m is not None
            and self._cached_field_eci is not None
            and np.array_equal(r, self._cached_position_eci_m)
        ):
            return self._cached_field_eci.copy()

        field_eci = 1e-3 * self._igrf14_field_eci(r, t)
        self._cached_time_s = t
        self._cached_position_eci_m = r.copy()
        self._cached_field_eci = field_eci.copy()
        return field_eci

    def clear_cache(self) -> None:
        """Forget the last magnetic-field sample."""
        self._cached_time_s = None
        self._cached_position_eci_m = None
        self._cached_field_eci = None

    def _igrf14_field_eci(
        self, position_eci_m: np.ndarray, time_s: float
    ) -> np.ndarray:
        gmst = GMST_J2000 + EARTH_ROTATION_RATE * float(time_s)
        position_ecef = rotate_around_z(-gmst) @ position_eci_m
        lon_deg, lat_deg, alt_km = geodetic_from_ecef(position_ecef)
        Be, Bn, Bu = ppigrf.igrf(
            lon_deg, lat_deg, alt_km, J2000_UTC + timedelta(seconds=float(time_s))
        )  # [1], [2]
        field_enu_nt = np.array([scalar_value(Be), scalar_value(Bn), scalar_value(Bu)])
        field_ecef = enu_to_ecef(field_enu_nt, np.deg2rad(lon_deg), np.deg2rad(lat_deg))
        return rotate_around_z(gmst) @ field_ecef  # [2]
