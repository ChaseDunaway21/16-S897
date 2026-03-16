"""Simulation driver for propagating spacecraft state over a configured duration."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import yaml
import matplotlib.pyplot as plt

from world.models.constants import MU_EARTH
from world.dynamics import integrate_dynamics
from world.spacecraft import Spacecraft

class Simulator:
    """Run Simulation of Satellite"""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)

        with self.config_path.open("r", encoding="utf-8") as file:
            self.cfg = yaml.safe_load(file) or {}

        self.spacecraft = Spacecraft(self.config_path)
        self.dt = float(self._property_value(self.cfg.get("simulation_properties", []), "time_step", 1.0))
        self.max_simulation_time = float(
            self._property_value(self.cfg.get("simulation_properties", []), "max_simulation_time", 7200.0)
        )
        self.integration_method = str(
            self._property_value(self.cfg.get("integration_properties", []), "integration_method", "rk4")
        ).lower()

        if self.integration_method != "rk4":
            raise ValueError("Only rk4 integration_method is currently supported")

        self.idx = self.spacecraft.Idx["X"]
        self.log_interval_steps = int(
            self._property_value(self.cfg.get("simulation_properties", []), "log_interval_steps", 1000)
        )
        self.log_file = self._setup_logger()

    @staticmethod
    def _property_value(items: Iterable[Dict], target_name: str, default: float | int | str) -> float | int | str:
        for item in items:
            if str(item.get("name", "")).strip() == target_name:
                return item.get("value", default)
        return default

    @staticmethod
    def _orbit_period_seconds(position_m: np.ndarray) -> float:
        r = np.linalg.norm(position_m)
        if r <= 0.0:
            raise ValueError("Initial position norm must be positive")
        return 2.0 * np.pi * np.sqrt(r**3 / MU_EARTH) # TODO Change for non-circular orbits later

    def _setup_logger(self) -> Path:
        """Create a file logger under Project/results for simulation run diagnostics."""
        results_dir = self.config_path.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = results_dir / f"simulation_{timestamp}.log"

        logger_name = f"simulator.{id(self)}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        self.logger.handlers.clear()
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self.logger.addHandler(handler)

        self.logger.info("Logger initialized")
        self.logger.info("Config file: %s", self.config_path)
        return log_file

    @staticmethod
    def _vector_to_string(vec: np.ndarray, precision: int = 6) -> str:
        """Format a vector for readable component logging."""
        return np.array2string(np.asarray(vec, dtype=float), precision=precision, separator=", ")

    def _log_state_components(
        self,
        prefix: str,
        state: np.ndarray,
        step: int | None = None,
        total_steps: int | None = None,
        time_s: float | None = None,
    ) -> None:
        """Log each state component vector in a consistent format."""
        message = prefix
        if step is not None and total_steps is not None:
            message += f" | step={step}/{total_steps}"
        if time_s is not None:
            message += f" | t={time_s:.2f} s"

        self.logger.info(message)
        self.logger.info("  position [m]: %s", self._vector_to_string(state[self.idx["POS_ECEF"]]))
        self.logger.info("  velocity [m/s]: %s", self._vector_to_string(state[self.idx["VEL_ECEF"]]))
        self.logger.info("  attitude [-]: %s", self._vector_to_string(state[self.idx["ATTITUDE"]]))
        self.logger.info("  omega [rad/s]: %s", self._vector_to_string(state[self.idx["ATTITUDE_RATE"]]))

    def run(self) -> dict[str, np.ndarray | float | int]:
        state = self.spacecraft.get_state().astype(float, copy=True)

        orbit_period = self._orbit_period_seconds(state[self.idx["POS_ECEF"]])
        sim_duration = self.max_simulation_time
        num_steps = int(np.ceil(sim_duration / self.dt))
        self.logger.info(
            "Simulation start | dt=%.3f s | duration=%.3f s | steps=%d | method=%s",
            self.dt,
            sim_duration,
            num_steps,
            self.integration_method,
        )
        self._log_state_components("Initial state", state, step=0, total_steps=num_steps, time_s=0.0)

        times = np.zeros(num_steps + 1, dtype=float)
        history = np.zeros((num_steps + 1, state.size), dtype=float)
        history[0] = state

        t = 0.0
        for k in range(1, num_steps + 1):
            state = integrate_dynamics(
                self.spacecraft,
                t,
                self.dt,
                method=self.integration_method,
            )
            t += self.dt
            times[k] = t
            history[k] = state

            if self.log_interval_steps > 0 and (k % self.log_interval_steps == 0 or k == num_steps):
                self._log_state_components("Progress", state, step=k, total_steps=num_steps, time_s=t)

        final_state = history[-1]

        final_pos_m = final_state[self.idx["POS_ECEF"]]
        final_vel_ms = final_state[self.idx["VEL_ECEF"]]
        updated_state = self.spacecraft.get_state()
        updated_state[self.idx["POS_ECEF"]] = final_pos_m
        updated_state[self.idx["VEL_ECEF"]] = final_vel_ms
        self.spacecraft.set_state(updated_state)

        self.logger.info("Simulation complete")
        self._log_state_components("Final state", updated_state, step=num_steps, total_steps=num_steps, time_s=t)
        self.logger.info("Log file saved: %s", self.log_file)

        return {
            "times_s": times,
            "state_history_si": history,
            "orbit_period_s": orbit_period,
            "sim_duration_s": sim_duration,
            "num_steps": num_steps,
            "log_file": str(self.log_file),
        }

    def plot_simulation(self, result: dict[str, np.ndarray | float | int], show: bool = True, save_path: str | Path | None = None) -> None:
        """Plot trajectory and state histories from a simulation result."""

        times = np.asarray(result["times_s"], dtype=float)
        history = np.asarray(result["state_history_si"], dtype=float)

        pos_km = history[:, self.idx["POS_ECEF"]] / 1_000.0
        vel_kms = history[:, self.idx["VEL_ECEF"]] / 1_000.0
        att = history[:, self.idx["ATTITUDE"]]
        att_rate = history[:, self.idx["ATTITUDE_RATE"]]

        fig = plt.figure(figsize=(15, 10), facecolor="#f7f8fa")
        fig.suptitle("ARGUS Simulation Overview", fontsize=16, fontweight="bold", y=0.98)

        ax_orbit = fig.add_subplot(2, 2, 1, projection="3d")
        ax_orbit.set_facecolor("#f2f4f8")
        ax_orbit.plot(pos_km[:, 0], pos_km[:, 1], pos_km[:, 2], linewidth=2.2, color="#0f766e", alpha=0.95)
        ax_orbit.scatter(
            pos_km[0, 0],
            pos_km[0, 1],
            pos_km[0, 2],
            color="#19a04b",
            edgecolors="white",
            linewidths=0.8,
            s=45,
            label="start",
            zorder=3,
        )
        ax_orbit.scatter(
            pos_km[-1, 0],
            pos_km[-1, 1],
            pos_km[-1, 2],
            color="#e12121",
            edgecolors="white",
            linewidths=0.8,
            s=45,
            label="end",
            zorder=3,
        )
        ax_orbit.set_title("Trajectory (ECEF)")
        ax_orbit.set_xlabel("x [km]")
        ax_orbit.set_ylabel("y [km]")
        ax_orbit.set_zlabel("z [km]")
        ax_orbit.legend(loc="best")

        x_min, x_max = float(np.min(pos_km[:, 0])), float(np.max(pos_km[:, 0]))
        y_min, y_max = float(np.min(pos_km[:, 1])), float(np.max(pos_km[:, 1]))
        z_min, z_max = float(np.min(pos_km[:, 2])), float(np.max(pos_km[:, 2]))
        x_mid = 0.5 * (x_min + x_max)
        y_mid = 0.5 * (y_min + y_max)
        z_mid = 0.5 * (z_min + z_max)
        half_range = 0.5 * max(x_max - x_min, y_max - y_min, z_max - z_min)
        if half_range == 0.0:
            half_range = 1.0
        ax_orbit.set_xlim(x_mid - half_range, x_mid + half_range)
        ax_orbit.set_ylim(y_mid - half_range, y_mid + half_range)
        ax_orbit.set_zlim(z_mid - half_range, z_mid + half_range)
        ax_orbit.set_box_aspect((1.0, 1.0, 1.0))

        ax_vel = fig.add_subplot(2, 2, 2)
        ax_vel.set_facecolor("#f2f4f8")
        ax_vel.plot(times, vel_kms[:, 0], label="vx", linewidth=2.0, color="#2563eb")
        ax_vel.plot(times, vel_kms[:, 1], label="vy", linewidth=2.0, color="#f59e0b")
        ax_vel.plot(times, vel_kms[:, 2], label="vz", linewidth=2.0, color="#14b8a6")
        ax_vel.set_title("Velocity Components")
        ax_vel.set_xlabel("time [s]")
        ax_vel.set_ylabel("velocity [km/s]")
        ax_vel.legend(loc="best")
        ax_vel.grid(True, alpha=0.35, linestyle="--", linewidth=0.7)
        ax_vel.set_box_aspect(1)

        ax_att = fig.add_subplot(2, 2, 3)
        ax_att.set_facecolor("#f2f4f8")
        ax_att.plot(times, att[:, 0], label="q0", linewidth=2.0, color="#6d28d9")
        ax_att.plot(times, att[:, 1], label="q1", linewidth=2.0, color="#db2777")
        ax_att.plot(times, att[:, 2], label="q2", linewidth=2.0, color="#0ea5e9")
        if att.shape[1] > 3:
            ax_att.plot(times, att[:, 3], label="q3", linewidth=2.0, color="#16a34a")
        ax_att.set_title("Attitude Components")
        ax_att.set_xlabel("time [s]")
        ax_att.set_ylabel("attitude [-]")
        ax_att.legend(loc="best")
        ax_att.grid(True, alpha=0.35, linestyle="--", linewidth=0.7)
        ax_att.set_box_aspect(1)

        ax_att_rate = fig.add_subplot(2, 2, 4)
        ax_att_rate.set_facecolor("#f2f4f8")
        ax_att_rate.plot(times, att_rate[:, 0], label="wx", linewidth=2.0, color="#2563eb")
        ax_att_rate.plot(times, att_rate[:, 1], label="wy", linewidth=2.0, color="#f97316")
        ax_att_rate.plot(times, att_rate[:, 2], label="wz", linewidth=2.0, color="#059669")
        ax_att_rate.set_title("Angular Velocity Components")
        ax_att_rate.set_xlabel("time [s]")
        ax_att_rate.set_ylabel("omega [rad/s]")
        ax_att_rate.legend(loc="best")
        ax_att_rate.grid(True, alpha=0.35, linestyle="--", linewidth=0.7)
        ax_att_rate.set_box_aspect(1)

        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))

        if save_path is None:
            save_path = self.config_path.parent / "results" / "simulation_plot.png"

        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        self.logger.info("Plot saved: %s", output_path)

        if show:
            plt.show()

        return fig

    def plot_momentum_sphere(self, result: dict[str, np.ndarray | float | int], show: bool = True, save_path: str | Path | None = None) -> None:
        """Plot the spacecraft's angular momentum sphere"""

        history = np.asarray(result["state_history_si"], dtype=float)
        w = history[:, self.idx["ATTITUDE_RATE"]]
        n = w.shape[0]
        J = self.spacecraft.compute_inertia_tensor()

        h_body = (J @ w.T).T
        h_norm = np.linalg.norm(h_body, axis=1, keepdims=True)
        h_norm[h_norm == 0.0] = 1.0
        momentum_history = (h_body / h_norm).T
        
        u = np.linspace(-np.pi, np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        U, V = np.meshgrid(u, v)

        x = np.cos(U) * np.sin(V)
        y = np.sin(U) * np.sin(V)
        z = np.cos(V)

        fig = plt.figure(figsize=(9, 8), facecolor="#f7f8fa")
        ax = plt.subplot(1, 1, 1, projection="3d")
        ax.set_facecolor("#f2f4f8")
        ax.plot_surface(x, y, z, cmap="Blues", alpha=0.22, edgecolor="none", antialiased=True)
        ax.plot(
            momentum_history[0, :],
            momentum_history[1, :],
            momentum_history[2, :],
            linewidth=2.0,
            color="#0f172a",
            alpha=0.9,
        )
        color_progress = np.linspace(0.0, 1.0, n)
        ax.scatter(
            momentum_history[0, :],
            momentum_history[1, :],
            momentum_history[2, :],
            c=color_progress,
            cmap="plasma",
            s=6,
            alpha=0.85,
        )
        ax.set_title("Angular Momentum Sphere")
        ax.set_xlabel("Lx [kg*m^2/s]")
        ax.set_ylabel("Ly [kg*m^2/s]")
        ax.set_zlabel("Lz [kg*m^2/s]")
        ax.set_box_aspect((1.0, 1.0, 1.0))
        fig.tight_layout()

        if save_path is None:
            save_path = self.config_path.parent / "results" / "momentum_sphere.png"


        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        self.logger.info("Momentum sphere plot saved: %s", output_path)

        if show:
            fig.show()

        return fig


    def render_3D(self):
        """Use Meshcat to produce a 3D visualization of the spacecraft trajectory and attitude over time."""
        # TODO

        return
    

        