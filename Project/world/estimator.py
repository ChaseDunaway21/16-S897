"""Multiplicative extended Kalman filter for attitude estimation.

The implementation is adapted from the ARGUS GNC-Simulation MEKF:
https://github.com/cmu-argus-2/GNC-Simulation

The state is [q_w, q_x, q_y, q_z, gyro_bias_x, gyro_bias_y, gyro_bias_z].
The covariance is over the 6D error state [attitude_error, gyro_bias_error].
"""

from __future__ import annotations

import numpy as np

from world.math import skew_symmetric, unit_rows, unit_vector
from world.rotations_and_transformations import attitude_jacobian as G
from world.rotations_and_transformations import (
    H,
    L,
    R,
    T,
    normalize_quaternion,
    quaternion_to_rotation_matrix,
    quaternion_from_rotation_matrix,
    quaternion_from_rotation_vector,
)


def _direction_covariance(sigma: float, direction_eci: np.ndarray) -> np.ndarray:
    """Return direction-vector covariance from an angular noise sigma."""
    direction_cross = skew_symmetric(unit_vector(direction_eci))
    return float(sigma) ** 2 * direction_cross @ direction_cross.T


def delta_q(phi: np.ndarray) -> np.ndarray:
    """Return the MEKF delta-q correction used in the lecture."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    scalar = np.sqrt(max(0.0, 1.0 - float(phi @ phi)))
    return normalize_quaternion(np.hstack([scalar, phi]))


def wahba(body_vectors: np.ndarray, reference_vectors: np.ndarray) -> np.ndarray:
    """
    Return estimated attitude from >= 2 bearing vectors using Wahba's SVD.
    This copies the Wahba SVD from Project/wahbas/wahbas_main.py.
    For now this is used to intialize the MEKF.
    """

    body_unit = unit_rows(body_vectors)
    reference_unit = unit_rows(reference_vectors)

    B = body_unit.T @ reference_unit
    U, _, Vt = np.linalg.svd(B)
    M = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])
    return (U @ M @ Vt).T  # Body to ECI


class MEKF:
    """
    The relevant states are the quaternion attitude and the gyro bias
    [q1; q2; q3; q4; b1; b2; b3]

    The gyro measurement is an input to the prediction step, not its own state.
    """

    def __init__(
        self,
        sigma_initial_attitude: float = 0.0,
        sigma_initial_gyro_bias: float = 0.0,
        sigma_gyro_white: float = 0.0,
        sigma_gyro_bias_deriv: float = 0.0,
        sigma_sunsensor_direction: float = 0.0,
        sigma_magnetometer_direction: float = 0.0,
    ) -> None:
        """Initialize the nominal state, error covariance, and noise models."""
        self.state_vector = np.zeros(7, dtype=float)
        self.state_vector[0] = 1.0

        # For P, I am just squaring the initial variance to initialize P.
        self.P = np.zeros((6, 6), dtype=float)
        self.P[0:3, 0:3] = float(sigma_initial_attitude) ** 2 * np.eye(3)
        self.P[3:6, 3:6] = float(sigma_initial_gyro_bias) ** 2 * np.eye(3)

        self.Q = np.zeros((6, 6), dtype=float)
        self.Q[0:3, 0:3] = float(sigma_gyro_white) ** 2 * np.eye(3)
        self.Q[3:6, 3:6] = float(sigma_gyro_bias_deriv) ** 2 * np.eye(3)

        self.sigma_sunsensor_direction = float(sigma_sunsensor_direction)
        self.sigma_magnetometer_direction = float(sigma_magnetometer_direction)
        self.last_prediction_time = None

    #################################################################################################
    # PREDICTION AND UPDATE
    #################################################################################################

    def predict(self, gyro_measurement: np.ndarray, t: float) -> None:
        """Propagate attitude and covariance using the measured gyro body rate."""
        gyro = np.asarray(gyro_measurement, dtype=float).reshape(3)

        dt = (
            0.0
            if self.last_prediction_time is None
            else float(t) - self.last_prediction_time
        )
        self.last_prediction_time = float(t)
        if dt <= 0.0:
            return

        q = self.get_attitude()
        omega = gyro - self.get_gyro_bias()
        # quaternion_from_rotation_vector is expm but doesn't need scipy
        dq_body = quaternion_from_rotation_vector(omega * dt)
        q_next = normalize_quaternion(L(q) @ dq_body)
        self.set_attitude(q_next)

        # A = [ dphi_k+1/dphi_k dphi_k+1/db_k ]
        #     [ 0            I                ]
        A_attitude = G(q_next).T @ R(dq_body) @ G(q)
        A = np.eye(6)
        A[0:3, 0:3] = A_attitude
        A[0:3, 3:6] = -0.5 * dt * (G(q).T @ G(q))
        self.P = A @ self.P @ A.T + self.Q * dt

    def update(
        self, C: np.ndarray, innovation: np.ndarray, R_noise: np.ndarray
    ) -> None:
        """Apply the measurement update to the error state."""
        C = np.asarray(C, dtype=float)
        innovation = np.asarray(innovation, dtype=float).reshape(C.shape[0])
        R_noise = np.asarray(R_noise, dtype=float)

        S = C @ self.P @ C.T + R_noise
        K = self.P @ C.T @ np.linalg.pinv(S, rcond=1e-8)
        dx = K @ innovation

        dq_body = delta_q(dx[0:3])
        self.set_attitude(L(self.get_attitude()) @ dq_body)
        self.set_gyro_bias(self.get_gyro_bias() + dx[3:6])

        I6 = np.eye(6)  # Can't just name this I because of linting...
        self.P = (I6 - K @ C) @ self.P @ (
            I6 - K @ C
        ).T + K @ R_noise @ K.T  # Joseph form

    #################################################################################################
    # MEASUREMENT MODELS
    #################################################################################################

    def vector_update(
        self,
        measured_vector_body: np.ndarray,
        true_vector_eci: np.ndarray,
        sigma_direction: float | None = None,
        R_noise: np.ndarray | None = None,
    ) -> None:
        """Update attitude using one measured body vector and its ECI reference."""
        measured_body = unit_vector(measured_vector_body)
        true_eci = unit_vector(true_vector_eci)
        predicted_body = self.get_ECI_R_b().T @ true_eci
        innovation = measured_body - predicted_body

        q = self.get_attitude()
        C = np.zeros((3, 6), dtype=float)
        C[0:3, 0:3] = (
            H.T @ (L(q).T @ L(H @ true_eci) + R(q) @ R(H @ true_eci) @ T) @ G(q)
        )
        if R_noise is None:
            sigma = 0.0 if sigma_direction is None else sigma_direction
            R_noise = _direction_covariance(sigma, predicted_body)

        self.update(C, innovation, R_noise)

    def sun_sensor_update(
        self, measured_sun_ray_body: np.ndarray, true_sun_ray_eci: np.ndarray
    ) -> None:
        """Update attitude with a Sun-sensor body vector measurement."""
        self.vector_update(
            measured_sun_ray_body,
            true_sun_ray_eci,
            self.sigma_sunsensor_direction,
        )

    def Bfield_update(
        self, measured_Bfield_body: np.ndarray, true_Bfield_eci: np.ndarray
    ) -> None:
        """Update attitude with a magnetometer body vector measurement."""
        self.vector_update(
            measured_Bfield_body,
            true_Bfield_eci,
            self.sigma_magnetometer_direction,
        )

    def initialize_from_vectors(
        self, body_vectors: np.ndarray, reference_vectors_eci: np.ndarray
    ) -> None:
        """Initialize attitude from at least two body/ECI vector pairs."""
        self.set_ECI_R_b(wahba(body_vectors, reference_vectors_eci))

    #################################################################################################
    # SETTERS, GETTERS
    #################################################################################################

    def get_state(self) -> np.ndarray:
        """Return a copy of the 7D nominal MEKF state."""
        return self.state_vector.copy()

    def set_state(self, state: np.ndarray) -> None:
        """Set the nominal quaternion and gyro-bias state."""
        state = np.asarray(state, dtype=float).reshape(7)
        self.state_vector[0:4] = normalize_quaternion(state[0:4])
        self.state_vector[4:7] = state[4:7]

    def get_attitude(self) -> np.ndarray:
        """Return the estimated body-to-ECI attitude quaternion."""
        return self.state_vector[0:4].copy()

    def set_attitude(self, q_body_to_eci: np.ndarray) -> None:
        """Set the estimated body-to-ECI attitude quaternion."""
        self.state_vector[0:4] = normalize_quaternion(q_body_to_eci)

    def get_gyro_bias(self) -> np.ndarray:
        """Return the estimated gyroscope bias in body coordinates."""
        return self.state_vector[4:7].copy()

    def set_gyro_bias(self, gyro_bias: np.ndarray) -> None:
        """Set the estimated gyroscope bias in body coordinates."""
        self.state_vector[4:7] = np.asarray(gyro_bias, dtype=float).reshape(3)

    def get_ECI_R_b(self) -> np.ndarray:
        """Return the rotation matrix that maps body vectors to ECI."""
        q = self.state_vector[0:4]
        return quaternion_to_rotation_matrix(q)

    def set_ECI_R_b(self, ECI_R_b: np.ndarray) -> None:
        """Set attitude from a body-to-ECI rotation matrix."""
        self.set_attitude(quaternion_from_rotation_matrix(ECI_R_b))

    def get_uncertainty_sigma(self) -> np.ndarray:
        """Return one-sigma uncertainty values for the 6D error state."""
        return np.sqrt(np.clip(np.diag(self.P), 0.0, None))
