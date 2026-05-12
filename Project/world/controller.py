"""
Attitude controllers and TVLQR helper math.

References:
[1] Fisch, Paulo Rotband Marchtein.
    Advancing Spacecraft Autonomy: Optimal GNC, Vision-Based Estimation, and Systems Integration for Small Spacecraft.
    2026. Carnegie Mellon University, PhD dissertation. CMU Robotics Institute,
    Technical Report CMU-RI-TR-26-12.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from world.math_utils import matrix_from_config, skew_symmetric, unit_vector
from world.rotations_and_transformations import (
    L,
    short_quaternion,
    normalize_quaternion,
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_from_rotation_vector,
    rotation_vector_from_quaternion,
    quaternion_to_rotation_matrix,
    inertial_to_body,
    angle_to_unit_vector_distance,
)


def gyrostat_linearization(
    omega: np.ndarray,
    inertia_tensor: np.ndarray,
    wheel_momentum: np.ndarray,
    wheel_axes_body: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearize [phi, omega, wheel momentum] using wheel momentum rate input.
    These equations are derived in lecture 17.
    """
    omega = np.asarray(omega, dtype=float).reshape(3)
    inertia_tensor = np.asarray(inertia_tensor, dtype=float).reshape(3, 3)
    wheel_axes_body = np.asarray(wheel_axes_body, dtype=float)
    wheel_momentum = np.asarray(wheel_momentum, dtype=float).reshape(
        wheel_axes_body.shape[1]
    )
    inertia_inverse = np.linalg.inv(inertia_tensor)

    # H = J omega + B_w r
    angular_momentum = inertia_tensor @ omega + wheel_axes_body @ wheel_momentum

    n_wheels = wheel_axes_body.shape[1]
    A = np.zeros((6 + n_wheels, 6 + n_wheels), dtype=float)

    # phi_dot = -omega^ phi + delta_omega
    A[0:3, 0:3] = -skew_symmetric(omega)
    A[0:3, 3:6] = np.eye(3)

    # J omega_dot = tau - omega^ (J omega + B_w r)
    A[3:6, 3:6] = inertia_inverse @ (
        skew_symmetric(omega) @ inertia_tensor - skew_symmetric(angular_momentum)
    )

    # Wheel momentum
    A[3:6, 6:] = -inertia_inverse @ skew_symmetric(omega) @ wheel_axes_body

    B = np.zeros((6 + n_wheels, n_wheels), dtype=float)

    # Wheel control torque
    B[3:6, :] = -inertia_inverse @ wheel_axes_body

    # r_dot = u
    B[6:, :] = np.eye(n_wheels)
    return A, B


def zoh_discretize(
    A_tilde: np.ndarray, B_tilde: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Discretize with expm([[A_tilde, B_tilde], [0, 0]] * dt).
    These equations are derived in lecture 17.
    """
    n_x, n_u = B_tilde.shape

    # exp([[A, B], [0, 0]] dt) = [[A_d, B_d], [0, I]]
    block = np.zeros((n_x + n_u, n_x + n_u), dtype=float)
    block[:n_x, :n_x] = A_tilde
    block[:n_x, n_x:] = B_tilde
    discrete = expm(block * float(dt))
    return discrete[:n_x, :n_x], discrete[:n_x, n_x:]


def riccati_gain(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, P: np.ndarray
) -> np.ndarray:
    """DARE K"""
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def dare_gain_history(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Used for plotting K."""
    steps = max(1, int(steps))
    P = Q.copy()
    gains = np.zeros((steps, B.shape[1], A.shape[0]), dtype=float)
    for k in range(steps):
        K = riccati_gain(A, B, Q, R, P)
        gains[k] = K

        # P_next = Q + A.T P A - A.T P B K.
        P = Q + A.T @ P @ A - A.T @ P @ B @ K

    return riccati_gain(A, B, Q, R, P), P, gains


def versine_profile(
    elapsed_s: float, duration_s: float, final_angle: float
) -> tuple[float, float, float]:
    """Lecture 16 rest-to-rest angle profile."""
    if elapsed_s < 0.0:
        return 0.0, 0.0, 0.0
    if elapsed_s > duration_s:
        return final_angle, 0.0, 0.0

    alpha = np.pi / max(duration_s, 1e-9)
    alpha_t = alpha * float(elapsed_s)
    theta = 0.5 * final_angle * (1.0 - np.cos(alpha_t))
    theta_dot = 0.5 * final_angle * alpha * np.sin(alpha_t)
    theta_ddot = 0.5 * final_angle * alpha**2 * np.cos(alpha_t)
    return theta, theta_dot, theta_ddot


def rk4_step(t: float, y: np.ndarray, dt: float, derivative) -> np.ndarray:
    """
    RK4 step to propogate rho. Using the RK4 from dynamics.py
    does not work since that requires full simulation state input.
    """
    k1 = derivative(t, y)
    k2 = derivative(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = derivative(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = derivative(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def interpolate_rows(times: np.ndarray, values: np.ndarray, t: float) -> np.ndarray:
    """Interpolate a vector-valued time history."""
    if times.size == 1 or t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    return np.array([np.interp(t, times, values[:, i]) for i in range(values.shape[1])])


class ReactionWheelTVLQRController:
    """Reaction-wheel TVLQR using reduced quaternion error dynamics."""

    def __init__(
        self,
        target_attitude: np.ndarray,
        target_rate_body: np.ndarray | None = None,
        update_period_s: float = 0.0,
        use_eigen_slew: bool = True,
        slew_duration_s: float = 60.0,
        lqr_dt_s: float = 0.1,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        inertia_tensor: np.ndarray | None = None,
        wheel_axes_body: np.ndarray | None = None,
        wheel_max_torque: float = 23e-6,
        wheel_max_angular_momentum: float = 5.8e-4,
        nominal_gain_steps: int = 54_000,
    ) -> None:
        self.target_attitude = normalize_quaternion(target_attitude)
        self.target_rate_body = (
            np.zeros(3)
            if target_rate_body is None
            else np.asarray(target_rate_body, dtype=float).reshape(3)
        )
        self.update_period_s = float(update_period_s)
        self.use_eigen_slew = bool(use_eigen_slew)
        self.slew_duration_s = max(float(slew_duration_s), 1e-9)
        self.lqr_dt_s = max(float(lqr_dt_s), 1e-6)
        self.inertia_tensor = (
            np.eye(3) * 1e-3
            if inertia_tensor is None
            else np.asarray(inertia_tensor, dtype=float)
        )
        self.wheel_axes_body = (
            np.eye(3)
            if wheel_axes_body is None
            else np.asarray(wheel_axes_body, dtype=float)
        )
        if self.wheel_axes_body.shape != (3, 3):
            raise ValueError("wheel_axes_body must have shape (3, 3)")
        self.wheel_max_torque = float(wheel_max_torque)
        self.wheel_max_angular_momentum = float(wheel_max_angular_momentum)

        # LQR State[phi(3), delta_omega(3), delta_wheel_momentum(3)]
        self.state_size = 9

        # Bryson-ish rule
        wheel_momentum_weight = 1.0 / self.wheel_max_angular_momentum**2
        self.Q = (
            np.diag([400.0] * 3 + [10_000.0] * 3 + [wheel_momentum_weight] * 3)
            if Q is None
            else matrix_from_config(Q, (self.state_size, self.state_size))
        )
        self.R = (
            np.eye(3) / self.wheel_max_torque**2
            if R is None
            else matrix_from_config(R, (3, 3))
        )
        (
            self.K_nominal,
            self.P_nominal,
            self.K_nominal_history,
        ) = self._nominal_gain_history(nominal_gain_steps)
        self.reference_start_s = 0.0
        self.reference_initialized = False
        self.q0 = self.target_attitude
        self.slew_axis_body = np.zeros(3)
        self.slew_angle = 0.0
        self.reference_times_s = np.zeros(1)
        self.reference_wheel_momentum = np.zeros((1, 3))
        self.reference_wheel_rate = np.zeros((1, 3))
        self.reference_gains = self.K_nominal.reshape(1, 3, self.state_size)

    def _nominal_gain_history(
        self, steps: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Nominal steady gain is computed about the target rate with zero wheel momentum
        A_c, B_c = gyrostat_linearization(
            self.target_rate_body,
            self.inertia_tensor,
            np.zeros(3),
            self.wheel_axes_body,
        )

        # Discretize dynamics
        A_d, B_d = zoh_discretize(A_c, B_c, self.lqr_dt_s)
        return dare_gain_history(A_d, B_d, self.Q, self.R, max(1, int(steps)))

    def initialize_reference(
        self, state: np.ndarray, state_index: dict, time_s: float
    ) -> None:
        self.reference_start_s = float(time_s)
        self.q0 = normalize_quaternion(state[state_index["ATTITUDE"]])
        initial_wheel_momentum = self._wheel_momentum_coordinates(
            state[state_index["RHO"]]
        )

        if not self.use_eigen_slew:
            self.reference_times_s = np.zeros(1)
            self.reference_wheel_momentum = initial_wheel_momentum.reshape(1, 3)
            self.reference_wheel_rate = np.zeros((1, 3))
            self.reference_gains = self.K_nominal.reshape(1, 3, self.state_size)
            self.reference_initialized = True
            return

        self.slew_axis_body, self.slew_angle = self._eigen_axis_and_angle(
            self.q0, self.target_attitude
        )
        self._make_eigen_slew_reference(initial_wheel_momentum)
        self._compute_gain_schedule()
        self.reference_initialized = True

    def _eigen_axis_and_angle(
        self, q: np.ndarray, q_desired: np.ndarray
    ) -> tuple[np.ndarray, float]:
        Q = quaternion_to_rotation_matrix(q)
        Q_desired = quaternion_to_rotation_matrix(q_desired)
        relative_rotation = Q_desired.T @ Q
        eigenvalues, eigenvectors = np.linalg.eig(relative_rotation)
        axis = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            return np.zeros(3), 0.0
        axis /= axis_norm

        q_delta = short_quaternion(
            quaternion_multiply(quaternion_conjugate(q), q_desired)
        )
        rotation_vector = rotation_vector_from_quaternion(q_delta)
        angle = float(np.linalg.norm(rotation_vector))
        if angle < 1e-12:
            return np.zeros(3), 0.0
        if np.dot(axis, rotation_vector) < 0.0:
            axis = -axis
        return axis, angle

    def _reference_kinematics(
        self, elapsed_s: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if (not self.use_eigen_slew) or self.slew_angle < 1e-12:
            return self.target_attitude, self.target_rate_body, np.zeros(3)
        if elapsed_s > self.slew_duration_s:
            return self.target_attitude, self.target_rate_body, np.zeros(3)

        theta, theta_dot, theta_ddot = versine_profile(
            elapsed_s, self.slew_duration_s, self.slew_angle
        )
        q_ref = quaternion_multiply(
            self.q0, quaternion_from_rotation_vector(self.slew_axis_body * theta)
        )
        return q_ref, self.slew_axis_body * theta_dot, self.slew_axis_body * theta_ddot

    def _reference_rho_dot(self, elapsed_s: float, rho_body: np.ndarray) -> np.ndarray:
        _, omega_ref, omega_dot_ref = self._reference_kinematics(elapsed_s)
        return -self.inertia_tensor @ omega_dot_ref - skew_symmetric(omega_ref) @ (
            self.inertia_tensor @ omega_ref + rho_body
        )

    def _reference_wheel_rate(
        self, elapsed_s: float, rho_body: np.ndarray
    ) -> np.ndarray:
        return self._wheel_momentum_coordinates(
            self._reference_rho_dot(elapsed_s, rho_body)
        )

    def _make_eigen_slew_reference(self, initial_wheel_momentum: np.ndarray) -> None:
        n_steps = max(1, int(np.ceil(self.slew_duration_s / self.lqr_dt_s)))
        self.reference_times_s = np.linspace(0.0, self.slew_duration_s, n_steps + 1)
        rho_history = np.zeros((n_steps + 1, 3))
        rho_history[0] = self.wheel_axes_body @ initial_wheel_momentum
        self.reference_wheel_momentum = np.zeros((n_steps + 1, 3))
        self.reference_wheel_rate = np.zeros((n_steps + 1, 3))

        for i in range(n_steps):
            t0 = self.reference_times_s[i]
            dt = self.reference_times_s[i + 1] - t0
            self.reference_wheel_momentum[i] = self._wheel_momentum_coordinates(
                rho_history[i]
            )
            self.reference_wheel_rate[i] = self._reference_wheel_rate(
                t0, rho_history[i]
            )
            rho_history[i + 1] = rk4_step(
                t0, rho_history[i], dt, self._reference_rho_dot
            )

        self.reference_wheel_momentum[-1] = self._wheel_momentum_coordinates(
            rho_history[-1]
        )
        self.reference_wheel_rate[-1] = np.zeros(3)

    def _compute_gain_schedule(self) -> None:
        """Compute Ks along the reference trajectory."""
        n_steps = self.reference_times_s.size - 1
        self.reference_gains = np.zeros((n_steps + 1, 3, self.state_size), dtype=float)
        P = self.Q.copy()  # S_N = Q_N

        for i in range(n_steps):
            # Backward pass: start at the terminal state and go backwards
            A_d, B_d = self._reference_discrete_matrices(n_steps - 1 - i)
            K = riccati_gain(A_d, B_d, self.Q, self.R, P)
            self.reference_gains[n_steps - 1 - i] = K
            P = self.Q + A_d.T @ P @ A_d - A_d.T @ P @ B_d @ K

        # After the slew ends, use Kss
        self.reference_gains[-1] = self.K_nominal

    def _reference_discrete_matrices(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        t0 = self.reference_times_s[index]
        _, omega0, _ = self._reference_kinematics(t0)
        A_c, B_c = gyrostat_linearization(
            omega0,
            self.inertia_tensor,
            self.reference_wheel_momentum[index],
            self.wheel_axes_body,
        )
        t1 = self.reference_times_s[index + 1]

        return zoh_discretize(A_c, B_c, t1 - t0)

    def _gain(self, elapsed_s: float) -> np.ndarray:
        """Use the gain along the reference trajectory"""
        if self.reference_gains.shape[0] == 1:
            return self.reference_gains[0]
        index = np.searchsorted(self.reference_times_s, elapsed_s, side="right") - 1
        return self.reference_gains[
            int(np.clip(index, 0, self.reference_gains.shape[0] - 1))
        ]

    def _wheel_reference(self, elapsed_s: float) -> tuple[np.ndarray, np.ndarray]:
        wheel_momentum = interpolate_rows(
            self.reference_times_s, self.reference_wheel_momentum, elapsed_s
        )
        wheel_rate = interpolate_rows(
            self.reference_times_s, self.reference_wheel_rate, elapsed_s
        )
        return wheel_momentum, wheel_rate

    def _body_torque_to_wheel_command(
        self, torque_body: np.ndarray, actuator_model: dict | None
    ) -> np.ndarray:
        """Maps wheel command to torque"""
        axes = self.wheel_axes_body
        max_torque = self.wheel_max_torque
        max_momentum = self.wheel_max_angular_momentum
        if actuator_model and actuator_model.get("reaction_wheel") is not None:
            wheel = actuator_model["reaction_wheel"]
            axes = wheel.G_RW_b
            max_torque = wheel.max_torque
            max_momentum = wheel.max_angular_momentum

        command = np.linalg.solve(axes, np.asarray(torque_body).reshape(3))
        command *= max_momentum / max_torque
        return np.clip(command, -max_momentum, max_momentum)

    def _wheel_momentum_coordinates(self, rho_body: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self.wheel_axes_body, np.asarray(rho_body).reshape(3))

    def _state_error(
        self,
        state: np.ndarray,
        state_index: dict,
        q_ref: np.ndarray,
        omega_ref: np.ndarray,
        wheel_momentum_ref: np.ndarray,
    ) -> np.ndarray:
        error = np.empty(self.state_size)
        q_error = short_quaternion(L(q_ref).T @ state[state_index["ATTITUDE"]])
        error[:3] = q_error[1:4]
        error[3:6] = state[state_index["ATTITUDE_RATE"]] - omega_ref
        error[6:] = (
            self._wheel_momentum_coordinates(state[state_index["RHO"]])
            - wheel_momentum_ref
        )
        return error

    def compute_command(
        self,
        state: np.ndarray,
        state_index: dict,
        time_s: float,
        actuator_model: dict | None = None,
        estimator_state: np.ndarray | None = None,
    ) -> np.ndarray:
        feedback_state = state if estimator_state is None else estimator_state
        if not self.reference_initialized:
            self.initialize_reference(feedback_state, state_index, time_s)

        elapsed = float(time_s) - self.reference_start_s
        q_ref, omega_ref, _ = self._reference_kinematics(elapsed)
        wheel_momentum_ref, wheel_rate_ref = self._wheel_reference(elapsed)
        delta_x = self._state_error(
            feedback_state, state_index, q_ref, omega_ref, wheel_momentum_ref
        )

        # TVLQR update: u is wheel momentum rate.
        u = wheel_rate_ref - self._gain(elapsed) @ delta_x
        body_torque = -self.wheel_axes_body @ u
        return self._body_torque_to_wheel_command(body_torque, actuator_model)


class MagnetorquerOnlyController:
    """Magnetorquer-only angular-momentum controller."""

    def __init__(
        self,
        target_rate_body: np.ndarray | None = None,
        target_spin_stable_axis_body: np.ndarray | None = None,
        target_pointing_axis_inertial: np.ndarray | None = None,
        spin_stable_tolerance_rad: float = 0.1,
        pointing_tolerance_rad: float = 0.1,
        update_period_s: float = 0.0,
        inertia_tensor: np.ndarray | None = None,
        max_voltage: float = 8.4,
    ) -> None:
        self.target_spin_stable_axis_body = self._unit_vector(
            [0.0, 0.0, 1.0]
            if target_spin_stable_axis_body is None
            else target_spin_stable_axis_body,
            "target_spin_stable_axis_body",
        )
        self.target_pointing_axis_inertial = self._unit_vector(
            [1.0, 0.0, 0.0]
            if target_pointing_axis_inertial is None
            else target_pointing_axis_inertial,
            "target_pointing_axis_inertial",
        )

        if target_rate_body is None:
            self.target_rate_body = (
                10.0 * 2.0 * np.pi / 60.0
            ) * self.target_spin_stable_axis_body
        else:
            target_rate = np.asarray(target_rate_body, dtype=float).reshape(-1)
            if target_rate.size == 1:
                self.target_rate_body = (
                    float(target_rate[0]) * self.target_spin_stable_axis_body
                )
            elif target_rate.size == 3:
                self.target_rate_body = target_rate
            else:
                raise ValueError("target_rate_body must be a scalar or length-3 vector")
        self.target_rate = self.target_rate_body

        self.inertia_tensor = (
            np.eye(3) * 1e-3
            if inertia_tensor is None
            else np.asarray(inertia_tensor, dtype=float)
        )
        self.update_period_s = float(update_period_s)
        self.spin_stable_tolerance = angle_to_unit_vector_distance(
            spin_stable_tolerance_rad
        )
        self.pointing_tolerance = angle_to_unit_vector_distance(pointing_tolerance_rad)
        self.max_voltage = float(max_voltage)
        self.momentum_target = self.inertia_tensor @ self.target_rate_body

    @staticmethod
    def _unit_vector(vector: np.ndarray, field_name: str) -> np.ndarray:
        vector = np.asarray(vector, dtype=float).reshape(3)
        norm = np.linalg.norm(vector)
        if norm <= 1e-12:
            raise ValueError(f"{field_name} must be a nonzero vector")
        return vector / norm

    def _alpha_gain(
        self,
        command_vector: np.ndarray,
        k: int = 1,
    ) -> float:
        """
        As defined in [1], alpha is a smoothing scalar gain that takes in the
        command vector and outputs a value between 0 and 1 of that vector.
        The command is for spin-stabilization first or sun-pointing second, depending
        on the state of the system.
        """
        if np.linalg.norm(command_vector) < 1e-12:
            return 0.0
        return np.tanh(k * np.linalg.norm(command_vector))

    def _coil_voltages_from_command(
        self, voltages: np.ndarray, actuator_model: dict | None
    ) -> np.ndarray:
        """Map a 3D body-axis command to the configured physical coils."""
        voltage_command = np.asarray(voltages, dtype=float).reshape(-1)
        magnetorquer = actuator_model.get("magnetorquer")
        return np.linalg.pinv(magnetorquer.G_MTB_b) @ voltage_command

    def compute_command(
        self,
        state: np.ndarray,
        state_index: dict,
        time_s: float,
        actuator_model: dict | None = None,
        estimator_state: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        From [1], the controller aligns the angular momentum with the target spin-stable axis first,
        then aligns the angular momentum with the time-varying target pointing axis.
        """

        feedback_state = state if estimator_state is None else estimator_state
        magnetic_field_model = (
            None
            if actuator_model is None
            else actuator_model.get("magnetic_field_model")
        )
        if magnetic_field_model is None:
            return np.zeros(3, dtype=float)

        magnetic_field_eci = magnetic_field_model.field_eci(
            feedback_state[state_index["POS_ECI"]], time_s
        )
        b = inertial_to_body(
            feedback_state[state_index["ATTITUDE"]], magnetic_field_eci
        )
        b_hat = skew_symmetric(b)

        q = feedback_state[state_index["ATTITUDE"]]
        omega = feedback_state[state_index["ATTITUDE_RATE"]]

        h = self.inertia_tensor @ omega
        h_tgt = self.momentum_target
        h_tgt_norm = np.linalg.norm(h_tgt)
        h_heuristic = h / h_tgt_norm if h_tgt_norm > 1e-12 else np.zeros(3)

        a = self.target_spin_stable_axis_body
        s = inertial_to_body(q, self.target_pointing_axis_inertial)
        s = s / np.linalg.norm(s)

        # From [1]
        if np.linalg.norm(a - h_heuristic) > self.spin_stable_tolerance:
            command_prime_b = b_hat @ (h_tgt - h)
            voltage_command = (
                self.max_voltage
                * self._alpha_gain(command_prime_b)
                * unit_vector(command_prime_b)
            )

        elif np.linalg.norm(s - h_heuristic) > self.pointing_tolerance:
            command_prime_i = b_hat @ (s * h_tgt_norm - h)
            voltage_command = (
                self.max_voltage
                * self._alpha_gain(command_prime_i)
                * unit_vector(command_prime_i)
            )

        return self._coil_voltages_from_command(voltage_command, actuator_model)
