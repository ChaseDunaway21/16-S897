"""
Very basic reaction wheel model.

The defaults are based on the:
Astrofein RW1 Type A Reaction Wheel for Small Satellites:
https://www.astrofein.com/en/reaction-wheels/reaction-wheel-rw1/
"""

from __future__ import annotations

import numpy as np


DEFAULT_RW_ORIENTATION = np.eye(3, dtype=float)


class ReactionWheel:
    """
    Configurable Reaction Wheel model.
    Defaults to three orthogonal reaction wheels.
    """

    def __init__(
        self,
        N_RWs: int = 3,
        max_torque: float = 23e-6,
        max_angular_momentum: float = 5.8e-4,
        G_RW_b: np.ndarray = DEFAULT_RW_ORIENTATION,
    ) -> None:
        self.N_RWs = int(N_RWs)
        self.max_torque = float(max_torque)
        self.max_angular_momentum = float(max_angular_momentum)
        if self.max_angular_momentum <= 0.0:
            raise ValueError("max_angular_momentum must be positive")

        self.G_RW_b = np.asarray(G_RW_b, dtype=float)
        if self.G_RW_b.shape != (3, self.N_RWs):
            raise ValueError("G_RW_b must have shape (3, N_RWs)")

    def get_torque(self, wheel_speeds: np.ndarray) -> np.ndarray:
        """Return total reaction wheel torque in the body frame [N m]."""
        wheel_speeds = np.asarray(wheel_speeds, dtype=float).reshape(self.N_RWs)
        wheel_torques = np.clip(
            wheel_speeds, -self.max_angular_momentum, self.max_angular_momentum
        ) * (self.max_torque / self.max_angular_momentum)
        return self.G_RW_b @ wheel_torques
