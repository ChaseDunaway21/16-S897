"""Attitude controllers and TVLQR helper math."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from world.math_utils import matrix_from_config, skew_symmetric
from world.rotations_and_transformations import (
    short_quaternion,
    normalize_quaternion,
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_from_rotation_vector,
    rotation_vector_from_quaternion,
    quaternion_to_rotation_matrix,
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


# A lot of this storage class for the reference trajectory came from help with ChatGPT,
# It is bulky, I should skim this down to just computing the reference trajectory, but
# this makes plotting easier
@dataclass
class EigenSlewReference:
    q0: np.ndarray
    qf: np.ndarray
    target_rate_body: np.ndarray
    duration_s: float
    inertia_tensor: np.ndarray
    wheel_axes_body: np.ndarray
    initial_wheel_momentum: np.ndarray
    sample_dt_s: float
    start_time_s: float = 0.0
    sample_times_s: np.ndarray | None = None
    wheel_momentum_history: np.ndarray | None = None
    wheel_momentum_rate_history: np.ndarray | None = None

    @classmethod
    def build(
        cls,
        initial_attitude: np.ndarray,
        target_attitude: np.ndarray,
        target_rate_body: np.ndarray,
        duration_s: float,
        inertia_tensor: np.ndarray,
        wheel_axes_body: np.ndarray,
        initial_wheel_momentum: np.ndarray,
        sample_dt_s: float,
        start_time_s: float,
    ) -> "EigenSlewReference":
        # Renormalizing these helped with numerical issues
        q0 = normalize_quaternion(initial_attitude)
        qf = normalize_quaternion(target_attitude)

        reference = cls(
            q0,
            qf,
            np.asarray(target_rate_body, dtype=float).reshape(3),
            duration_s,
            np.asarray(inertia_tensor, dtype=float).reshape(3, 3),
            np.asarray(wheel_axes_body, dtype=float),
            np.asarray(initial_wheel_momentum, dtype=float).reshape(
                np.asarray(wheel_axes_body, dtype=float).shape[1]
            ),
            max(float(sample_dt_s), 1e-6),
            start_time_s,
        )
        reference._precompute_inverse_dynamics()
        return reference

    @property
    def initial_rho_body(self) -> np.ndarray:
        return self.wheel_axes_body @ self.initial_wheel_momentum

    def _rotation_axis(self) -> tuple[np.ndarray, float]:
        Q = quaternion_to_rotation_matrix(self.q0)
        Q_desired = quaternion_to_rotation_matrix(self.qf)

        # Lecture 16 eigenaxis: use Q_desired.T Q and take the eigenvector
        # corresponding to the eigenvalue 1
        relative_rotation = Q_desired.T @ Q
        eigenvalues, eigenvectors = np.linalg.eig(relative_rotation)
        axis = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            return np.zeros(3), 0.0
        axis = axis / axis_norm

        # The eigenvector sign is arbitrary. Pick the sign that matches the
        # right-multiplied body-frame slew qf = q0 * q_delta
        q_delta_command = short_quaternion(
            quaternion_multiply(quaternion_conjugate(self.q0), self.qf)
        )
        rotation_vector = rotation_vector_from_quaternion(q_delta_command)
        if np.dot(axis, rotation_vector) < 0.0:
            axis = -axis

        angle = float(np.linalg.norm(rotation_vector))
        if angle < 1e-12:
            return np.zeros(3), 0.0
        return axis, angle

    def _versine(self, elapsed_s: float, angle: float) -> tuple[float, float, float]:
        if elapsed_s < 0.0:
            return 0.0, 0.0, 0.0
        if elapsed_s > self.duration_s:
            return angle, 0.0, 0.0

        # Lecture 16 eigen-slew: theta(t) = theta_f/2 * (1 - cos(alpha t)),
        # with alpha = pi / T so theta_dot is zero at both endpoints
        alpha = np.pi / max(self.duration_s, 1e-9)
        alpha_t = alpha * float(elapsed_s)
        theta = 0.5 * angle * (1.0 - np.cos(alpha_t))
        theta_dot = 0.5 * angle * alpha * np.sin(alpha_t)
        theta_ddot = 0.5 * angle * alpha**2 * np.cos(alpha_t)
        return theta, theta_dot, theta_ddot

    def _kinematics_at(
        self, elapsed_s: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        axis_body, angle = self._rotation_axis()
        if angle < 1e-12:
            return self.qf.copy(), self.target_rate_body.copy(), np.zeros(3)

        theta, theta_dot, theta_ddot = self._versine(elapsed_s, angle)

        # Right-multiply by the body-frame eigenaxis rotation so q_ref traces
        # q0 -> qf while keeping the commanded rate in body coordinates
        q_ref = quaternion_multiply(
            self.q0, quaternion_from_rotation_vector(axis_body * theta)
        )

        # Reference rate and accleration
        omega_ref = axis_body * theta_dot
        omega_dot_ref = axis_body * theta_ddot

        if elapsed_s > self.duration_s:
            # After the slew, track the requested final body rate
            omega_ref = self.target_rate_body.copy()
            omega_dot_ref = np.zeros(3)

        return q_ref, omega_ref, omega_dot_ref

    def _rho_dot_body(self, elapsed_s: float, rho_body: np.ndarray) -> np.ndarray:
        _, omega_ref, omega_dot_ref = self._kinematics_at(elapsed_s)
        # Lecture 16 inverse dynamics for wheels
        # J omega_dot + rho_dot + omega x (J omega + rho) = 0
        return -self.inertia_tensor @ omega_dot_ref - skew_symmetric(omega_ref) @ (
            self.inertia_tensor @ omega_ref + rho_body
        )

    def _rk4_rho_step(
        self, elapsed_s: float, rho_body: np.ndarray, dt: float
    ) -> np.ndarray:  # TODO: Should I make a shared one with dynamics.py?
        k1 = self._rho_dot_body(elapsed_s, rho_body)
        k2 = self._rho_dot_body(elapsed_s + 0.5 * dt, rho_body + 0.5 * dt * k1)
        k3 = self._rho_dot_body(elapsed_s + 0.5 * dt, rho_body + 0.5 * dt * k2)
        k4 = self._rho_dot_body(elapsed_s + dt, rho_body + dt * k3)
        return rho_body + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _precompute_inverse_dynamics(self) -> None:
        """
        Inverse dynamics precomputation from lecture 16.
        """
        # The total number of steps is based on the slew duration and the LQR dt
        n_steps = max(1, int(np.ceil(self.duration_s / self.sample_dt_s)))
        self.sample_times_s = np.linspace(0.0, self.duration_s, n_steps + 1)
        rho_history = np.zeros((n_steps + 1, 3), dtype=float)
        rho_dot_history = np.zeros((n_steps + 1, 3), dtype=float)
        rho_history[0] = self.initial_rho_body

        # From the notes:
        # Start with rho(0), solve p_dot(t), integrate to get rho(t)
        for i in range(n_steps):
            t0 = self.sample_times_s[i]
            t1 = self.sample_times_s[i + 1]
            rho_dot_history[i] = self._rho_dot_body(t0, rho_history[i])
            rho_history[i + 1] = self._rk4_rho_step(t0, rho_history[i], t1 - t0)

        rho_dot_history[-1] = np.zeros(3)

        #
        self.wheel_momentum_history = (
            np.linalg.pinv(self.wheel_axes_body) @ rho_history.T
        ).T
        self.wheel_momentum_rate_history = (
            np.linalg.pinv(self.wheel_axes_body) @ rho_dot_history.T
        ).T

    def at(self, elapsed_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._kinematics_at(elapsed_s)

    def wheel_momentum_at(self, elapsed_s: float) -> tuple[np.ndarray, np.ndarray]:
        if (
            self.sample_times_s is None
            or self.wheel_momentum_history is None
            or self.wheel_momentum_rate_history is None
        ):
            self._precompute_inverse_dynamics()

        if elapsed_s < 0.0:
            return self.wheel_momentum_history[0].copy(), np.zeros_like(
                self.initial_wheel_momentum
            )
        if elapsed_s == 0.0:
            return (
                self.wheel_momentum_history[0].copy(),
                self.wheel_momentum_rate_history[0].copy(),
            )
        if elapsed_s >= self.duration_s:
            return self.wheel_momentum_history[-1].copy(), np.zeros_like(
                self.initial_wheel_momentum
            )

        wheel_momentum = np.array(
            [
                np.interp(elapsed_s, self.sample_times_s, component)
                for component in self.wheel_momentum_history.T
            ],
            dtype=float,
        )
        wheel_momentum_rate = np.array(
            [
                np.interp(elapsed_s, self.sample_times_s, component)
                for component in self.wheel_momentum_rate_history.T
            ],
            dtype=float,
        )
        return wheel_momentum, wheel_momentum_rate


@dataclass
class FixedAttitudeReference:
    q_ref: np.ndarray
    target_rate_body: np.ndarray
    wheel_momentum: np.ndarray
    start_time_s: float = 0.0

    @classmethod
    def build(
        cls,
        target_attitude: np.ndarray,
        target_rate_body: np.ndarray,
        wheel_momentum: np.ndarray,
        start_time_s: float,
    ) -> "FixedAttitudeReference":
        # Fixed-target mode skips trajectory tracking and regulates directly to
        # the requested endpoint attitude/rate
        return cls(
            normalize_quaternion(target_attitude),
            np.asarray(target_rate_body, dtype=float).reshape(3),
            np.asarray(wheel_momentum, dtype=float).reshape(-1),
            start_time_s,
        )

    def at(self, elapsed_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _ = elapsed_s
        return self.q_ref.copy(), self.target_rate_body.copy(), np.zeros(3)

    def wheel_momentum_at(self, elapsed_s: float) -> tuple[np.ndarray, np.ndarray]:
        _ = elapsed_s
        return self.wheel_momentum.copy(), np.zeros_like(self.wheel_momentum)


class ReactionWheelTVLQRController:
    """Reaction-wheel TVLQR using reduced quaternion error dynamics."""

    def __init__(
        self,
        target_attitude: np.ndarray,
        n_reaction_wheels: int = 3,
        update_period_s: float = 0.0,
        inertia_tensor: np.ndarray | None = None,
        target_rate_body: np.ndarray | None = None,
        slew_duration_s: float = 60.0,
        lqr_dt_s: float = 0.1,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        wheel_axes_body: np.ndarray | None = None,
        wheel_max_torque: float = 23e-6,
        wheel_max_angular_momentum: float = 5.8e-4,
        nominal_gain_steps: int = 54_000,
        use_eigen_slew: bool = True,
    ) -> None:
        self.update_period_s = float(update_period_s)
        self.inertia_tensor = (
            np.eye(3) * 1e-3
            if inertia_tensor is None
            else np.asarray(inertia_tensor, dtype=float)
        )
        self.target_attitude = normalize_quaternion(target_attitude)
        self.target_rate_body = (
            np.zeros(3)
            if target_rate_body is None
            else np.asarray(target_rate_body, dtype=float).reshape(3)
        )
        self.slew_duration_s = max(float(slew_duration_s), 1e-9)
        self.lqr_dt_s = max(float(lqr_dt_s), 1e-6)
        self.use_eigen_slew = bool(use_eigen_slew)
        self.wheel_axes_body = (
            np.eye(3, int(n_reaction_wheels))
            if wheel_axes_body is None
            else np.asarray(wheel_axes_body, dtype=float)
        )
        if self.wheel_axes_body.shape != (3, int(n_reaction_wheels)):
            raise ValueError("wheel_axes_body must have shape (3, n_reaction_wheels)")
        self.wheel_max_torque = float(wheel_max_torque)
        self.wheel_max_angular_momentum = float(wheel_max_angular_momentum)
        self.n_reaction_wheels = int(n_reaction_wheels)

        # LQR State[phi(3), delta_omega(3), delta_wheel_momentum(n)]
        self.state_size = 6 + self.n_reaction_wheels

        # Bryson-ish rule
        wheel_momentum_weight = 1.0 / self.wheel_max_angular_momentum**2
        self.Q = (
            np.diag(
                [400.0] * 3
                + [10_000.0] * 3
                + [wheel_momentum_weight] * self.n_reaction_wheels
            )
            if Q is None
            else matrix_from_config(Q, (self.state_size, self.state_size))
        )
        self.R = (
            np.eye(self.n_reaction_wheels) / self.wheel_max_torque**2
            if R is None
            else matrix_from_config(R, (self.n_reaction_wheels, self.n_reaction_wheels))
        )
        self.reference: EigenSlewReference | FixedAttitudeReference | None = None
        self.reference_times_s = np.zeros(1)
        self.reference_gains = np.zeros((1, self.n_reaction_wheels, self.state_size))
        self.reference_wheel_momentum = np.zeros(self.n_reaction_wheels)
        (
            self.K_nominal,
            self.P_nominal,
            self.K_nominal_history,
        ) = self._nominal_gain_history(nominal_gain_steps)

    def _nominal_gain_history(
        self, steps: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Nominal steady gain is computed about the target rate with zero wheel momentum
        A_c, B_c = gyrostat_linearization(
            self.target_rate_body,
            self.inertia_tensor,
            np.zeros(self.n_reaction_wheels),
            self.wheel_axes_body,
        )

        # Discretize dynamics
        A_d, B_d = zoh_discretize(A_c, B_c, self.lqr_dt_s)
        return dare_gain_history(A_d, B_d, self.Q, self.R, max(1, int(steps)))

    def initialize_reference(
        self, state: np.ndarray, state_index: dict, time_s: float
    ) -> None:
        rho_body = np.asarray(state[state_index["RHO"]], dtype=float).reshape(3)

        # Convert body gyrostat momentum into wheel-coordinate momentum so the
        # LQR error state matches the linearization
        self.reference_wheel_momentum = np.linalg.pinv(self.wheel_axes_body) @ rho_body

        if self.use_eigen_slew:
            # Freeze the current state as the start of the eigenaxis reference
            self.reference = EigenSlewReference.build(
                state[state_index["ATTITUDE"]],
                self.target_attitude,
                self.target_rate_body,
                self.slew_duration_s,
                self.inertia_tensor,
                self.wheel_axes_body,
                self.reference_wheel_momentum,
                self.lqr_dt_s,
                float(time_s),
            )
            self._compute_gain_schedule()
            return

        self.reference = FixedAttitudeReference.build(
            self.target_attitude,
            self.target_rate_body,
            self.reference_wheel_momentum,
            float(time_s),
        )
        self._use_Kss()

    def _use_Kss(self) -> None:
        """Use nominal Kss for all time."""
        self.reference_times_s = np.zeros(1)
        self.reference_gains = self.K_nominal.reshape(
            1, self.n_reaction_wheels, self.state_size
        )

    def _compute_gain_schedule(self) -> None:
        """Compute Ks along the reference trajectory."""

        # Build a backward finite-horizon TVLQR schedule along the eigen-slew
        n_steps = max(1, int(np.ceil(self.slew_duration_s / self.lqr_dt_s)))
        self.reference_times_s = np.linspace(0.0, self.slew_duration_s, n_steps + 1)
        self.reference_gains = np.zeros(
            (n_steps + 1, self.n_reaction_wheels, self.state_size), dtype=float
        )
        P = self.P_nominal.copy()

        for i in range(n_steps):
            # Backward pass: start at the terminal steady-state cost and walk
            # toward the start of the reference trajectory
            A_d, B_d = self._reference_discrete_matrices(n_steps - 1 - i)
            K = riccati_gain(A_d, B_d, self.Q, self.R, P)
            self.reference_gains[n_steps - 1 - i] = K
            P = self.Q + A_d.T @ P @ A_d - A_d.T @ P @ B_d @ K

        # At and beyond the terminal time, use the nominal infinite-horizon gain
        self.reference_gains[-1] = self.K_nominal

    def _reference_discrete_matrices(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        assert self.reference is not None
        t0 = self.reference_times_s[index]

        # Linearize around the reference angular velocity
        _, omega0, _ = self.reference.at(t0)
        wheel_momentum0, _ = self.reference.wheel_momentum_at(t0)
        A_c, B_c = gyrostat_linearization(
            omega0,
            self.inertia_tensor,
            wheel_momentum0,
            self.wheel_axes_body,
        )
        t1 = self.reference_times_s[index + 1]

        return zoh_discretize(A_c, B_c, t1 - t0)

    def _gain(self, elapsed_s: float) -> np.ndarray:
        """Use the gain along the reference trajectory"""
        index = np.searchsorted(self.reference_times_s, elapsed_s, side="right") - 1
        return self.reference_gains[
            int(np.clip(index, 0, self.reference_gains.shape[0] - 1))
        ]

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

        command = np.linalg.pinv(axes * (max_torque / max_momentum)) @ np.asarray(
            torque_body
        ).reshape(3)
        return np.clip(command, -max_momentum, max_momentum)

    def _wheel_momentum_coordinates(self, rho_body: np.ndarray) -> np.ndarray:
        # Least-squares map from body gyrostat momentum to wheel-axis momenta
        return np.linalg.pinv(self.wheel_axes_body) @ np.asarray(
            rho_body, dtype=float
        ).reshape(3)

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
        _ = spacecraft, environment_model, estimator_state
        if self.reference is None:
            self.initialize_reference(state, state_index, time_s)

        elapsed = float(time_s) - self.reference.start_time_s
        q_ref, omega_ref, omega_dot_ref = self.reference.at(elapsed)
        wheel_momentum_ref, _ = self.reference.wheel_momentum_at(elapsed)
        wheel_momentum = self._wheel_momentum_coordinates(state[state_index["RHO"]])

        # Reduced attitude error: q_err = q_ref^-1 * q, then phi ~= 0.5 log(q_err)
        # to match the small-angle coordinates used by the LQR linearization
        error = np.hstack(
            [
                0.5
                * rotation_vector_from_quaternion(
                    quaternion_multiply(
                        quaternion_conjugate(q_ref),
                        state[state_index["ATTITUDE"]],
                    )
                ),
                state[state_index["ATTITUDE_RATE"]] - omega_ref,
                wheel_momentum - wheel_momentum_ref,
            ]
        )

        # tau_feedforward = J omega_dot_ref + omega_ref^ (J omega_ref + rho)
        feedforward = self.inertia_tensor @ omega_dot_ref + skew_symmetric(
            omega_ref
        ) @ (
            self.inertia_tensor @ omega_ref + self.wheel_axes_body @ wheel_momentum_ref
        )

        # tau_body = -B_w r_dot
        wheel_momentum_rate = -np.linalg.pinv(self.wheel_axes_body) @ feedforward

        # Apply u = - K @ delta x
        wheel_momentum_rate -= self._gain(elapsed) @ error

        # Map the commanded internal momentum rate back to body torque.
        body_torque = -self.wheel_axes_body @ wheel_momentum_rate
        return self._body_torque_to_wheel_command(body_torque, actuator_model)

    def save_gain_convergence_plot(self, path: str | Path) -> Path:
        import matplotlib.pyplot as plt

        path = Path(path)
        time_s = np.arange(self.K_nominal_history.shape[0]) * self.lqr_dt_s

        # Plot distance from each stored Riccati gain to the nominal fixed-point
        # gain computed by the extra post-history Riccati update.
        error = np.linalg.norm(self.K_nominal_history - self.K_nominal, axis=(1, 2))

        # Avoid drawing exact zeros as artificial 1e-308 cliffs on a log axis.
        error[error <= 0.0] = np.nan
        finite_error = error[np.isfinite(error)]
        fig, (ax, k_ax) = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.4]},
        )
        ax.plot(time_s, error, color="#2563eb", linewidth=1.4)
        if finite_error.size:
            ax.set_yscale("log")
        ax.set_title("Nominal TVLQR Gain Convergence")
        ax.set_ylabel("||K_k - K_ss||_F")
        ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.7)

        state_labels = ["phi_x", "phi_y", "phi_z", "omega_x", "omega_y", "omega_z"] + [
            f"r_{i + 1}" for i in range(self.n_reaction_wheels)
        ]
        max_abs_gain = np.max(np.abs(self.K_nominal_history), axis=0)
        active_entries = np.argwhere(max_abs_gain > 1e-14)
        if active_entries.size == 0:
            active_entries = np.argwhere(np.ones_like(max_abs_gain, dtype=bool))
        for control_index, state_index in active_entries:
            label = f"K[{control_index + 1},{state_labels[state_index]}]"
            k_ax.plot(
                time_s,
                self.K_nominal_history[:, control_index, state_index],
                linewidth=1.0,
                label=label,
            )

        k_ax.set_title("Nominal TVLQR Gain Entries")
        k_ax.set_xlabel("nominal DARE time [s]")
        k_ax.set_ylabel("K_k entries")
        k_ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.7)
        if active_entries.shape[0] <= 18:
            k_ax.legend(loc="best", fontsize=8, ncol=3)
        else:
            k_ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.22),
                fontsize=7,
                ncol=4,
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path


class MagnetorquerOnlyController:
    """Fixed-command magnetorquer controller."""

    def __init__(
        self,
        n_magnetorquers: int = 6,
        voltages_command: np.ndarray | None = None,
        update_period_s: float = 0.0,
    ) -> None:
        self.update_period_s = float(update_period_s)
        self.n_magnetorquers = int(n_magnetorquers)
        command = (
            np.zeros(self.n_magnetorquers)
            if voltages_command is None
            else voltages_command
        )
        self.voltages_command = np.asarray(command, dtype=float).reshape(
            self.n_magnetorquers
        )

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
