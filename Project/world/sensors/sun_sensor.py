"""OPT4003-style Sun bearing sensor model.

This sensor model is heavily inspired by GNC-Simulation:
https://github.com/cmu-argus-2/GNC-Simulation.

References:
[1] Texas Instruments, https://www.ti.com/product/OPT4003-Q1, OPT4003-Q1 High-Speed High-Precision
    Digital Ambient Light Sensor datasheet.
[2] cmu-argus-2/GNC-Simulation, argusim/sensors/SunSensor.py.
"""

from __future__ import annotations

import numpy as np

from world.math import add_noise, covariance_matrix, unit
from world.models.sun import SunModel
from world.rotations_and_transformations import inertial_to_body


class SunSensor:
    """Return the Sun unit vector in the body frame."""

    def __init__(
        self,
        sun_model: SunModel | None = None,
        covariance: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        return_none_if_eclipsed: bool = True,
    ) -> None:
        self.sun_model = sun_model or SunModel()
        self.covariance = covariance_matrix(covariance)
        self.rng = rng or np.random.default_rng()
        self.return_none_if_eclipsed = return_none_if_eclipsed

    def clean_measurement(
        self, state: np.ndarray, state_index: dict, time_s: float = 0.0
    ) -> np.ndarray | None:
        position = state[state_index["POS_ECI"]]
        if (
            self.return_none_if_eclipsed
            and self.sun_model.eclipse_factor(position) == 0.0
        ):
            return None

        q = state[state_index["ATTITUDE"]]
        sun_eci = self.sun_model.direction_eci(position, time_s)  # [2]
        return unit(inertial_to_body(q, sun_eci))

    def get_measurement(
        self, state: np.ndarray, state_index: dict, time_s: float = 0.0
    ) -> np.ndarray | None:
        clean = self.clean_measurement(state, state_index, time_s)
        if clean is None:
            return None
        return unit(add_noise(clean, self.covariance, self.rng))
