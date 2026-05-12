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
        resistance: float | np.ndarray = 25,
        A_cross: float = 5.432e-3,
        N_turns: int = 64,
        max_voltage: float = 8.4,
        max_current_rating: float = 9999.0,
        max_power: float = 9999.0,
        G_MTB_b: np.ndarray = DEFAULT_MTB_ORIENTATION,
    ) -> None:
        self.N_MTBs = int(N_MTBs)
        if self.N_MTBs <= 0:
            raise ValueError("N_MTBs must be positive")
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
        if self.G_MTB_b.shape != (3, self.N_MTBs):
            raise ValueError(f"G_MTB_b must have shape (3, {self.N_MTBs})")

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
        dipole_moments = (
            self.N_turns * self.A_cross * self.G_MTB_b * currents[np.newaxis, :]
        )
        magnetic_field_tesla = 1e-6 * np.asarray(magnetic_field_body, dtype=float)
        torques = np.cross(
            dipole_moments.T,
            magnetic_field_tesla,
        )
        return np.sum(torques, axis=0)

    def _currents_from_voltages(self, voltages: np.ndarray) -> np.ndarray:
        """Get the current from the commanded voltage."""

        voltages = np.asarray(voltages, dtype=float).reshape(self.N_MTBs)
        voltage_limit = np.full(self.N_MTBs, self.max_voltage, dtype=float)
        if self.max_current_rating > 0.0:
            voltage_limit = np.minimum(
                voltage_limit,
                self.max_current_rating * self.resistance,
            )
        if self.max_power > 0.0:
            voltage_limit = np.minimum(
                voltage_limit,
                np.sqrt(self.max_power * self.resistance),
            )

        voltages = np.clip(voltages, -voltage_limit, voltage_limit)
        currents = voltages / self.resistance
        return currents
