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
import spiceypy as spice

from world.models.constants import (
    ASTRONOMICAL_UNIT,
    RADIUS_EARTH,
    RADIUS_SUN,
    SOLAR_CONSTANT_1_AU,
    SPEED_OF_LIGHT,
)
from world.math import unit_vector
from world.rotations_and_transformations import R_body_to_inertial


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


class SunModel:
    """SPICE-backed Sun model."""

    def __init__(
        self,
        kernel_paths: list[str | Path] | None = None,
    ) -> None:
        self.kernel_paths = self._resolve_kernel_paths(kernel_paths)
        self._kernels_loaded = False

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def _default_kernel_paths(cls) -> list[Path]:
        kernel_dir = cls._project_root() / "kernels"
        kernel_paths = []
        leapseconds_kernel = kernel_dir / "naif0012.tls"
        if leapseconds_kernel.exists():
            kernel_paths.append(leapseconds_kernel)

        planetary_ephemeris_kernel = kernel_dir / "de442s.bsp"
        if planetary_ephemeris_kernel.exists():
            kernel_paths.append(planetary_ephemeris_kernel)
        return kernel_paths

    @classmethod
    def _resolve_kernel_paths(cls, kernel_paths: list[str | Path] | None) -> list[Path]:
        if not kernel_paths:
            return cls._default_kernel_paths()

        resolved_paths: list[Path] = []
        repo_root = cls._project_root().parent
        search_roots = [Path.cwd(), repo_root, cls._project_root()]
        for kernel_path in kernel_paths:
            path = Path(kernel_path)
            if path.is_absolute():
                resolved_paths.append(path)
                continue
            if path.exists():
                resolved_paths.append(path.resolve())
                continue

            resolved_path = None
            for root in search_roots:
                candidate = root / path
                if candidate.exists():
                    resolved_path = candidate.resolve()
                    break
            resolved_paths.append(resolved_path if resolved_path is not None else path)

        return resolved_paths

    def _load_kernels(self) -> None:
        if self._kernels_loaded:
            return
        if not self.kernel_paths:
            raise ValueError(
                "No SPICE kernels configured. Run setup_simulation.sh or pass "
                "Project/kernels/de442s.bsp to SunModel."
            )
        for kernel_path in self.kernel_paths:
            if not kernel_path.exists():
                raise FileNotFoundError(
                    f"SPICE kernel not found: {kernel_path}. Run setup_simulation.sh "
                    "or update kernel_paths in Project/config.yaml."
                )
            spice.furnsh(str(kernel_path))
        self._kernels_loaded = True

    def position_eci(self, time_j2000_s: float = 0.0) -> np.ndarray:
        """Return geocentric Sun position in J2000 ECI [m]."""
        self._load_kernels()
        # SPICE returns km.
        state_km, _ = spice.spkpos("SUN", float(time_j2000_s), "J2000", "NONE", "EARTH")
        return 1000.0 * np.asarray(state_km, dtype=float)

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
        projected_area_m2 = float(area_m2)
        if surface_normal_body is not None:
            normal_eci = R_body_to_inertial(q_body_to_eci) @ unit_vector(
                surface_normal_body
            )
            projected_area_m2 *= max(0.0, np.dot(normal_eci, unit_vector(sc_to_sun)))
        pressure = (
            SOLAR_CONSTANT_1_AU
            / SPEED_OF_LIGHT
            * (ASTRONOMICAL_UNIT / np.linalg.norm(sun_to_sc)) ** 2
        )  # [1], [2]

        return (
            illumination
            * float(coefficient_reflectivity)
            * pressure
            * projected_area_m2
            / float(mass_kg)
            * unit_vector(sun_to_sc)
        )
