"""
A simple camera model for the ARGUS Satellite.

This is heavily simplified, but ideally the camera returns unit bearing vectors
from the spacecraft to landmarks on the Earth's surface in the F.o.V. of the camera.

References:
[1] Arducam, https://www.arducam.com/arducam-12mp-imx708-autofocus-camera-module-with-hdr-pdaf-for-raspberry-pi.html,
    12MP IMX708 Autofocus Camera Module 3.
"""

from __future__ import annotations

import numpy as np

from world.math import add_noise, covariance_matrix, unit_vector
from world.rotations_and_transformations import inertial_to_body


class VisualCamera:
    """Return a unit bearing vector to an ECI target in the body frame."""

    def __init__(
        self,
        boresight_body: np.ndarray = np.array([1.0, 0.0, 0.0]),
        field_of_view_rad: float = np.deg2rad(75.0),
        covariance: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.boresight_body = unit_vector(boresight_body)
        self.field_of_view_rad = float(field_of_view_rad)
        self.covariance = covariance_matrix(covariance)
        self.rng = rng or np.random.default_rng()

    def clean_measurement(
        self,
        state: np.ndarray,
        state_index: dict,
        target_position_eci: np.ndarray = np.zeros(3),
    ) -> np.ndarray | None:
        position = state[state_index["POS_ECI"]]
        q = state[state_index["ATTITUDE"]]
        bearing_eci = unit_vector(np.asarray(target_position_eci, dtype=float) - position)
        bearing_body = unit_vector(inertial_to_body(q, bearing_eci))

        if bearing_body @ self.boresight_body < np.cos(0.5 * self.field_of_view_rad):
            return None
        return bearing_body

    def get_measurement(
        self,
        state: np.ndarray,
        state_index: dict,
        time_s: float = 0.0,
        target_position_eci: np.ndarray = np.zeros(3),
    ) -> np.ndarray | None:
        _ = time_s
        clean = self.clean_measurement(state, state_index, target_position_eci)
        if clean is None:
            return None
        return unit_vector(add_noise(clean, self.covariance, self.rng))
