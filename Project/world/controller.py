"""
Controller classes for attitude control.

ARGUS is designed for magnetorquer-only control, with a separate
reaction-wheel controller path available for simulated spacecraft.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are

from actuators import ReactionWheel


class ReactionWheelTVLQRController:
    """
    Reaction-wheel TVLQR Controller.
    This is as described in lecture 17 using an eigen-slew for a reference trajectory
    and TVLQR for feedback.
    """

    def __init__(
        self,
        n_reaction_wheels: int = 3,
        wheel_speeds_command: np.ndarray | None = None,
        update_period_s: float = 0.0,
        inertia_tensor=np.eye(3) * 1e-3,
        Q=np.eye(6) * 1 / (ReactionWheel.max_torque**2),  # Bryson's rule-ish
        R=np.eye(3) * 1 / (0.08**2),  # Bryson's rule through empirical testing
    ) -> None:
        self.update_period_s = float(update_period_s)
        self.n_reaction_wheels = int(n_reaction_wheels)
        if wheel_speeds_command is None:
            wheel_speeds_command = np.zeros(self.n_reaction_wheels, dtype=float)
        self.wheel_speeds_command = np.asarray(
            wheel_speeds_command, dtype=float
        ).reshape(self.n_reaction_wheels)
        self.Q = np.asarray(Q, dtype=float).reshape(6, 6)
        self.R = np.asarray(R, dtype=float).reshape(3, 3)
        self.S = Q.copy()  # Initial guess for the cost-to-go matrix
        self.K = self._compute_K(
            self, np.eye(6), np.eye(3)
        )  # Initial guess for the gain matrix

    def eigen_slew(self, target_attitude: np.ndarray) -> None:
        """Set wheel speeds to slew about the eigenaxis of the target attitude."""

    def _compute_K(self, A: np.ndarray, B: np.ndarray) -> None:
        """Compute the TVLQR gain matrix K offline."""
        # Solve DARE
        self.S = solve_discrete_are(A, B, self.Q, self.R)

        # Compute K
        self.K = np.linalg.inv(B.T @ self.S @ B + self.R) @ (B.T @ self.S @ A)

    def compute_command(
        self,
        state: np.ndarray,
        state_index: dict,
        time_s: float,
        *,
        spacecraft=None,
        environment_model: dict | None = None,
        actuator_model: dict | None = None,
        estimator_state: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return reaction-wheel commands."""

        _ = (
            state,
            state_index,
            time_s,
            spacecraft,
            environment_model,
            actuator_model,
            estimator_state,
        )

        return self.wheel_speeds_command.copy()


class MagnetorquerOnlyController:
    """Magnetorquer-only controller for the ARGUS."""

    def __init__(
        self,
        n_magnetorquers: int = 6,
        voltages_command: np.ndarray | None = None,
        update_period_s: float = 0.0,
    ) -> None:
        self.update_period_s = float(update_period_s)
        self.n_magnetorquers = int(n_magnetorquers)
        if voltages_command is None:
            voltages_command = np.zeros(self.n_magnetorquers, dtype=float)
        self.voltages_command = np.asarray(voltages_command, dtype=float).reshape(
            self.n_magnetorquers
        )

    # TODO
    def compute_command(
        self,
        state: np.ndarray,
        state_index: dict,
        time_s: float,
        *,
        spacecraft=None,
        environment_model: dict | None = None,
        actuator_model: dict | None = None,
        estimator_state: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return magnetorquer voltage commands."""
        _ = (
            state,
            state_index,
            time_s,
            spacecraft,
            environment_model,
            actuator_model,
            estimator_state,
        )
        return self.voltages_command.copy()
