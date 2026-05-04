"""BMX160-style magnetometer model.

This sensor model is heavily inspired by GNC-Simulation:
https://github.com/cmu-argus-2/GNC-Simulation.

References:
[1] Bosch Sensortec, https://www.mouser.com/pdfdocs/BST-BMX160-DS000-11.pdf, BMX160 datasheet.
[2] cmu-argus-2/GNC-Simulation, argusim/sensors/Magnetometer.py.
"""

from __future__ import annotations

import numpy as np

from world.math import add_noise, covariance_matrix
from world.models.magnetic_field import MagneticFieldModel
from world.rotations_and_transformations import inertial_to_body


MAGNETOMETER_MODEL = np.eye(3)


class Magnetometer:
    """
    Return the local magnetic-field vector in the body frame.
    For now the "model" is just identity.
    """

    def __init__(
        self,
        magnetic_field_model: MagneticFieldModel | None = None,
        covariance: np.ndarray | None = None,
        bias: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.magnetic_field_model = magnetic_field_model or MagneticFieldModel()
        self.covariance = covariance_matrix(covariance)
        self.bias = (
            np.zeros(3, dtype=float)
            if bias is None
            else np.asarray(bias, dtype=float).reshape(3)
        )
        self.rng = rng or np.random.default_rng()

    def clean_measurement(
        self, state: np.ndarray, state_index: dict, time_s: float = 0.0
    ) -> np.ndarray:
        position = state[state_index["POS_ECI"]]
        q = state[state_index["ATTITUDE"]]
        b_eci = self.magnetic_field_model.field_eci(position, time_s)  # [2]
        ideal_measurement = inertial_to_body(q, b_eci)
        return MAGNETOMETER_MODEL @ ideal_measurement + self.bias

    def get_measurement(
        self, state: np.ndarray, state_index: dict, time_s: float = 0.0
    ) -> np.ndarray:
        return add_noise(
            self.clean_measurement(state, state_index, time_s),
            self.covariance,
            self.rng,
        )
