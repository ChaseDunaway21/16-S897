"""
A simple camera model for the ARGUS Satellite.

This is heavily simplified, but ideally the camera returns unit bearing vectors
from the spacecraft to landmarks on the Earth's surface.

References:
[1] Arducam, https://www.arducam.com/arducam-12mp-imx708-autofocus-camera-module-with-hdr-pdaf-for-raspberry-pi.html,
    12MP IMX708 Autofocus Camera Module 3.
"""

from __future__ import annotations

import numpy as np

from world.math_utils import add_bearing_noise, unit_vector
from world.rotations_and_transformations import inertial_to_body


VISUAL_CAMERA_MODEL = np.eye(3)


class VisualCamera:
    """
    Return a unit bearing vector to an ECI target in the body frame.
    ARGUS has cameras on multiple sides, so this model does not gate by FOV.
    For now the "model" is just identity.
    """

    def __init__(
        self,
        sigma_angle_deg: float = 0.0,
        bias: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.sigma_angle_rad = np.deg2rad(float(sigma_angle_deg))
        self.bias = (
            np.zeros(3, dtype=float)
            if bias is None
            else np.asarray(bias, dtype=float).reshape(3)
        )
        self.rng = rng or np.random.default_rng()

    def clean_measurement(
        self,
        state: np.ndarray,
        state_index: dict,
        target_position_eci: np.ndarray = np.zeros(3),
    ) -> np.ndarray:
        position = state[state_index["POS_ECI"]]
        q = state[state_index["ATTITUDE"]]
        bearing_eci = unit_vector(
            np.asarray(target_position_eci, dtype=float) - position
        )
        bearing_body = unit_vector(inertial_to_body(q, bearing_eci))

        return VISUAL_CAMERA_MODEL @ bearing_body + self.bias

    def get_measurement(
        self,
        state: np.ndarray,
        state_index: dict,
        time_s: float = 0.0,
        target_position_eci: np.ndarray = np.zeros(3),
    ) -> np.ndarray:
        _ = time_s
        clean = self.clean_measurement(state, state_index, target_position_eci)
        return add_bearing_noise(clean, self.sigma_angle_rad, self.rng)
