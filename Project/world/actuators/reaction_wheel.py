"""
Very basic reaction wheel model.

The defaults are based on the:
Astrofein RW1 Type A Reaction Wheel for Small Satellites:
https://www.astrofein.com/en/reaction-wheels/reaction-wheel-rw1/
"""

from __future__ import annotations

import numpy as np


DEFAULT_RW_ORIENTATION = np.eye(3, dtype=float)
DEFAULT_RW_POSITIONS_BODY = np.zeros((3, 3), dtype=float)


class ReactionWheel:
    """
    Configurable Reaction Wheel model.
    Defaults to three orthogonal reaction wheels.
    """

    def __init__(
        self,
        max_torque: float = 23e-6,
        max_angular_momentum: float = 5.8e-4,
        G_RW_b: np.ndarray = DEFAULT_RW_ORIENTATION,
        wheel_positions_body: np.ndarray | None = None,
    ) -> None:
        self.max_torque = float(max_torque)
        self.max_angular_momentum = float(max_angular_momentum)
        if self.max_angular_momentum <= 0.0:
            raise ValueError("max_angular_momentum must be positive")

        self.G_RW_b = np.asarray(G_RW_b, dtype=float)
        if self.G_RW_b.shape != (3, 3):
            raise ValueError("G_RW_b must have shape (3, 3)")

        self.wheel_positions_body = self._positions_body(
            DEFAULT_RW_POSITIONS_BODY
            if wheel_positions_body is None
            else wheel_positions_body
        )

    def _positions_body(self, positions_body: np.ndarray) -> np.ndarray:
        """Return one body-frame position vector per wheel [m]."""
        positions = np.asarray(positions_body, dtype=float)
        n_wheels = self.G_RW_b.shape[1]
        if positions.ndim == 1:
            return np.tile(positions.reshape(1, 3), (n_wheels, 1))
        if positions.shape == (n_wheels, 3):
            return positions
        raise ValueError(
            f"wheel_positions_body must have shape (3,) or ({n_wheels}, 3)"
        )

    def get_torque(self, wheel_speeds: np.ndarray) -> np.ndarray:
        """Return total reaction wheel torque in the body frame [N m]."""
        wheel_speeds = np.asarray(wheel_speeds, dtype=float).reshape(3)
        wheel_torques = np.clip(
            wheel_speeds, -self.max_angular_momentum, self.max_angular_momentum
        ) * (self.max_torque / self.max_angular_momentum)
        return self.G_RW_b @ wheel_torques
