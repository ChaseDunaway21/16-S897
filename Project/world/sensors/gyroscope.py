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


GYROSCOPE_MODEL = np.eye(3)


class Gyroscope:
    """
    Return angular velocity in body coordinates.
    For now the "model" is just identity.
    """

    def __init__(
        self,
        covariance: np.ndarray | None = None,
        bias: np.ndarray | None = None,
        bias_random_walk_sigma: np.ndarray | float | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.covariance = covariance_matrix(covariance)
        self.bias = (
            np.zeros(3, dtype=float)
            if bias is None
            else np.asarray(bias, dtype=float).reshape(3)
        )
        self.bias_random_walk_sigma = (
            np.zeros(3, dtype=float)
            if bias_random_walk_sigma is None
            else np.asarray(bias_random_walk_sigma, dtype=float)
        )
        self.rng = rng or np.random.default_rng()
        self.last_bias_update_time = None

    def _advance_bias(self, time_s: float) -> None:
        time_s = float(time_s)
        if self.last_bias_update_time is None:
            self.last_bias_update_time = time_s
            return

        dt = time_s - self.last_bias_update_time
        self.last_bias_update_time = time_s
        if dt <= 0.0 or not np.any(self.bias_random_walk_sigma):
            return

        self.bias = self.bias + (
            self.bias_random_walk_sigma * np.sqrt(dt) * self.rng.standard_normal(3)
        )

    def clean_measurement(self, state: np.ndarray, state_index: dict) -> np.ndarray:
        omega_true = np.asarray(state[state_index["ATTITUDE_RATE"]], dtype=float)
        return GYROSCOPE_MODEL @ omega_true + self.bias  # [1], [2]

    def get_measurement(
        self, state: np.ndarray, state_index: dict, time_s: float = 0.0
    ) -> np.ndarray:
        self._advance_bias(time_s)
        return add_noise(
            self.clean_measurement(state, state_index), self.covariance, self.rng
        )
