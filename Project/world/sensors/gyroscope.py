"""BMX160-style gyroscope model.

This sensor model is heavily inspired by GNC-Simulation:
https://github.com/cmu-argus-2/GNC-Simulation.

References:
[1] Bosch Sensortec, https://www.mouser.com/pdfdocs/BST-BMX160-DS000-11.pdf, BMX160 datasheet.
[2] cmu-argus-2/GNC-Simulation, argusim/sensors/Sensor.py.
"""

from __future__ import annotations

import numpy as np

from world.math import add_noise, covariance_matrix


class Gyroscope:
    """Return angular velocity in body coordinates."""

    def __init__(
        self,
        covariance: np.ndarray | None = None,
        bias: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.covariance = covariance_matrix(covariance)
        self.bias = (
            np.zeros(3, dtype=float) if bias is None else np.asarray(bias, dtype=float)
        )
        self.rng = rng or np.random.default_rng()

    def clean_measurement(self, state: np.ndarray, state_index: dict) -> np.ndarray:
        return (
            np.asarray(state[state_index["ATTITUDE_RATE"]], dtype=float) + self.bias
        )  # [1], [2]

    def get_measurement(
        self, state: np.ndarray, state_index: dict, time_s: float = 0.0
    ) -> np.ndarray:
        _ = time_s
        return add_noise(
            self.clean_measurement(state, state_index), self.covariance, self.rng
        )
