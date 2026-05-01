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
from world.math import scalar_value
from world.rotations_and_transformations import R_z, enu_to_ecef, geodetic_from_ecef


class MagneticFieldModel:
    """Earth magnetic field in ECI using IGRF-14."""

    def field_eci(self, position_eci_m: np.ndarray, time_s: float = 0.0) -> np.ndarray:
        """Return magnetic flux density [uT] at an ECI position."""
        r = np.asarray(position_eci_m, dtype=float)
        return 1e-3 * self._igrf14_field_eci(r, time_s)

    def _igrf14_field_eci(
        self, position_eci_m: np.ndarray, time_s: float
    ) -> np.ndarray:
        gmst = GMST_J2000 + EARTH_ROTATION_RATE * float(time_s)
        position_ecef = R_z(-gmst) @ position_eci_m
        lon_deg, lat_deg, alt_km = geodetic_from_ecef(position_ecef)
        Be, Bn, Bu = ppigrf.igrf(
            lon_deg, lat_deg, alt_km, J2000_UTC + timedelta(seconds=float(time_s))
        )  # [1], [2]
        field_enu_nt = np.array([scalar_value(Be), scalar_value(Bn), scalar_value(Bu)])
        field_ecef = enu_to_ecef(field_enu_nt, np.deg2rad(lon_deg), np.deg2rad(lat_deg))
        return R_z(gmst) @ field_ecef  # [2]
