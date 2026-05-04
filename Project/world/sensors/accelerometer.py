"""BMX160-style accelerometer model.
This is entirely unused by ARGUS

This sensor model is heavily inspired by GNC-Simulation:
https://github.com/cmu-argus-2/GNC-Simulation.

References:
[1] Bosch Sensortec, https://www.mouser.com/pdfdocs/BST-BMX160-DS000-11.pdf, BMX160 datasheet.
[2] cmu-argus-2/GNC-Simulation, argusim/sensors/Sensor.py.
"""

from __future__ import annotations

import numpy as np

import world.models.gravity as gravity
from world.math import add_noise, covariance_matrix
from world.rotations_and_transformations import inertial_to_body


ACCELEROMETER_MODEL = np.eye(3)


class Accelerometer:
    """
    Return specific force in the body frame.
    For now the "model" is just identity.
    """

    def __init__(
        self,
        covariance: np.ndarray | None = None,
        bias: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.covariance = covariance_matrix(covariance)
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
        acceleration_eci: np.ndarray | None = None,
    ) -> np.ndarray:
        position = state[state_index["POS_ECI"]]
        q = state[state_index["ATTITUDE"]]
        if acceleration_eci is None:
            specific_force_eci = np.zeros(3, dtype=float)
        else:
            specific_force_eci = np.asarray(
                acceleration_eci, dtype=float
            ) - gravity.acceleration(position)  # [1]
        ideal_measurement = inertial_to_body(q, specific_force_eci)
        return ACCELEROMETER_MODEL @ ideal_measurement + self.bias

    def get_measurement(
        self,
        state: np.ndarray,
        state_index: dict,
        time_s: float = 0.0,
        acceleration_eci: np.ndarray | None = None,
    ) -> np.ndarray:
        _ = time_s
        clean = self.clean_measurement(state, state_index, acceleration_eci)
        return add_noise(clean, self.covariance, self.rng)
