"""ARGUS style magnetorquer model.

This model is heavily inspired by GNC-Simulation:
https://github.com/cmu-argus-2/GNC-Simulation.

Store the torquer constants and body-frame orientation matrix, then compute
the body-frame torque from commanded voltages and the local magnetic field.
"""

from __future__ import annotations

import numpy as np

from world.rotations_and_transformations import inertial_to_body


DEFAULT_MTB_ORIENTATION = np.array(  # This also matches the ARGUS model
    [
        [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
    ],
    dtype=float,
)


class Magnetorquer:
    """
    Configurable Magnetorquer model.
    ARGUS defaults to Six body-frame magnetorquers.
    """

    def __init__(
        self,
        N_MTBs: int = 6,
        resistance: float | np.ndarray = 3.25e-7,
        A_cross: float = 5.432e-3,
        N_turns: int = 64,
        max_voltage: float = 5.0,
        max_current_rating: float = 1.0,
        max_power: float = 1.0,
        G_MTB_b: np.ndarray = DEFAULT_MTB_ORIENTATION,
    ) -> None:
        self.N_MTBs = int(N_MTBs)
        self.resistance = np.asarray(resistance, dtype=float)
        if self.resistance.ndim == 0:
            self.resistance = np.full(self.N_MTBs, float(self.resistance))
        else:
            self.resistance = self.resistance.reshape(self.N_MTBs)

        self.A_cross = float(A_cross)
        self.N_turns = int(N_turns)
        self.max_voltage = float(max_voltage)
        self.max_current_rating = float(max_current_rating)
        self.max_power = float(max_power)

        self.G_MTB_b = np.asarray(G_MTB_b, dtype=float)

    def get_torque(
        self,
        voltages: np.ndarray,
        q_body_to_eci: np.ndarray,
        magnetic_field_eci: np.ndarray,
    ) -> np.ndarray:
        """Return total magnetorquer torque in the body frame [N m]."""
        magnetic_field_body = inertial_to_body(q_body_to_eci, magnetic_field_eci)
        return self.get_torque_body(voltages, magnetic_field_body)

    def get_torque_body(
        self, voltages: np.ndarray, magnetic_field_body: np.ndarray
    ) -> np.ndarray:
        """Return total torque from body-frame magnetic field [N m]."""
        currents = self._currents_from_voltages(voltages)
        dipole_moments = self.N_turns * self.A_cross * self.G_MTB_b * currents
        torques = np.cross(
            dipole_moments.T, np.asarray(magnetic_field_body, dtype=float)
        )
        return np.sum(torques, axis=0)

    def _currents_from_voltages(self, voltages: np.ndarray) -> np.ndarray:
        """Get the current from the commanded voltage."""

        voltages = np.asarray(voltages, dtype=float).reshape(self.N_MTBs)
        currents = voltages / self.resistance
        power = voltages * currents

        if np.any(np.abs(currents) > self.max_current_rating):
            raise ValueError("Current exceeds maximum current rating.")
        if np.any(np.abs(voltages) > self.max_voltage):
            raise ValueError("Voltage exceeds maximum voltage rating.")
        if np.any(np.abs(power) > self.max_power):
            raise ValueError("Power exceeds maximum power rating.")

        return currents
