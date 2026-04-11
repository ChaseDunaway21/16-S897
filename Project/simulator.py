"""Simulation driver for propagating spacecraft state over a configured duration."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from world.models.constants import MU_EARTH
from world.dynamics import integrate_dynamics
from world.math import quaternion_to_euler
from world.spacecraft import Spacecraft


def _run_single_monte_carlo_trial(
    config_path: str,
    output_dir: str,
    save_plots: bool,
) -> dict[str, object]:
    """Run one Monte Carlo trial in an isolated output directory."""
    run_dir = Path(output_dir)
    sim = Simulator(config_path=Path(config_path), output_dir=run_dir)
    result = sim.run()
    state_file = run_dir / "state_history.npz"
    np.savez_compressed(
        state_file,
        times_s=np.asarray(result["times_s"], dtype=float),
        state_history_si=np.asarray(result["state_history_si"], dtype=float),
    )
    if save_plots:
        sim.plot_simulation(result, show=False)
        sim.plot_momentum_sphere(result, show=False)

    final_state = np.asarray(result["state_history_si"])[-1]
    idx = sim.idx
    return {
        "status": "ok",
        "output_dir": str(run_dir),
        "log_file": str(result["log_file"]),
        "state_file": str(state_file),
        "num_steps": int(result["num_steps"]),
        "final_position_m": final_state[idx["POS_ECEF"]].tolist(),
        "final_velocity_ms": final_state[idx["VEL_ECEF"]].tolist(),
        "final_attitude": final_state[idx["ATTITUDE"]].tolist(),
        "final_omega_rads": final_state[idx["ATTITUDE_RATE"]].tolist(),
    }

class Simulator:
    """Run Simulation of Satellite"""

    def __init__(self, config_path: str | Path, output_dir: str | Path | None = None) -> None:
        self.config_path = Path(config_path)
        self._output_dir_explicit = output_dir is not None
        self.single_run_seed: int | None = None

        if output_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = self.config_path.parent / "results" / f"simulation_{stamp}"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        with self.config_path.open("r", encoding="utf-8") as file:
            self.cfg = yaml.safe_load(file) or {}

        sim_props = self.cfg.get("simulation_properties", []) or []
        mc_item = self._property_item(sim_props, "monte_carlo") or {}
        ideal_item = self._property_item(sim_props, "ideal") or {}
        monte_carlo_enabled = bool(mc_item.get("value", False))
        ideal_enabled = bool(ideal_item.get("value", True))

        effective_config_path = self.config_path
        if (not monte_carlo_enabled) and (not ideal_enabled):
            base_seed = int(mc_item.get("seed", 42))
            self.single_run_seed = base_seed
            trial_cfg = self._build_trial_config(
                trial_index=0,
                seed=base_seed,
                use_nominal_attitude_rate=False,
            )
            effective_config_path = self.output_dir / "config_effective.yaml"
            with effective_config_path.open("w", encoding="utf-8") as file:
                yaml.safe_dump(trial_cfg, file, sort_keys=False)
            self.cfg = trial_cfg

        self.spacecraft = Spacecraft(effective_config_path)
        self.dt = float(self._property_value(self.cfg.get("simulation_properties", []), "time_step", 1.0))
        self.max_simulation_time = float(
            self._property_value(self.cfg.get("simulation_properties", []), "max_simulation_time", 7200.0)
        )
        self.integration_method = str(
            self._property_value(self.cfg.get("integration_properties", []), "integration_method", "rk4")
        ).lower()
        self.plot_layout = self._validated_option(
            self._property_value(self.cfg.get("plotting_properties", []), "layout", "together"),
            {"together", "separate"},
            "plotting_properties.layout",
        )
        self.attitude_plot_layout = self._validated_option(
            self._property_value(
                self.cfg.get("plotting_properties", []),
                "attitude_plot_layout",
                "overlay",
            ),
            {"overlay", "stacked"},
            "plotting_properties.attitude_plot_layout",
        )
        self.attitude_plot_mode = self._validated_option(
            self._property_value(
                self.cfg.get("plotting_properties", []),
                "attitude_representation",
                "quaternion",
            ),
            {"quaternion", "euler"},
            "plotting_properties.attitude_representation",
        )

        if self.integration_method != "rk4":
            raise ValueError("Only rk4 integration_method is currently supported")

        self.idx = self.spacecraft.Idx["X"]
        self.log_interval_steps = int(
            self._property_value(self.cfg.get("simulation_properties", []), "log_interval_steps", 1000)
        )
        self.log_file = self._setup_logger()
        if self.single_run_seed is not None:
            self.logger.info(
                "Single non-ideal run uses deterministic sampled parameters with seed=%d",
                self.single_run_seed,
            )

    @staticmethod
    def _property_value(items: Iterable[Dict], target_name: str, default: float | int | str) -> float | int | str:
        for item in items:
            if str(item.get("name", "")).strip() == target_name:
                return item.get("value", default)
        return default

    @staticmethod
    def _property_item(items: Iterable[Dict], target_name: str) -> Dict | None:
        for item in items:
            if str(item.get("name", "")).strip() == target_name:
                return item
        return None

    @staticmethod
    def _validated_option(value: object, valid_options: set[str], field_name: str) -> str:
        option = str(value).strip().lower()
        if option not in valid_options:
            valid_list = ", ".join(sorted(valid_options))
            raise ValueError(f"{field_name} must be one of: {valid_list}")
        return option

    @staticmethod
    def _sample_with_uncertainty(
        rng: np.random.Generator,
        base_value: object,
        deviation: object | None,
        perturbation: object | None,
    ) -> object:
        base = np.asarray(base_value, dtype=float)
        sampled = base.copy()

        if deviation is not None:
            dev = np.asarray(deviation, dtype=float)
            sampled = sampled + dev * rng.standard_normal(size=sampled.shape)

        if perturbation is not None:
            pert = np.asarray(perturbation, dtype=float)
            sampled = sampled + rng.uniform(-pert, pert, size=sampled.shape)

        if sampled.ndim == 0:
            return float(sampled)
        return sampled.tolist()

    def _build_trial_config(
        self,
        trial_index: int,
        seed: int,
        use_nominal_attitude_rate: bool = True,
    ) -> dict:
        """Create one trial config with sampled initial-condition and mass uncertainties."""
        trial_cfg = deepcopy(self.cfg)
        rng = np.random.default_rng(seed + trial_index)

        initial_conditions = trial_cfg.get("initial_conditions", []) or []
        for item in initial_conditions:
            item_name = str(item.get("name", "")).strip()
            if item_name == "attitude_rate" and not use_nominal_attitude_rate:
                base = item.get("value")
            else:
                base = item.get("nominal_value", item.get("value"))
            deviation = item.get("deviation")
            perturbation = item.get("perturbation")

            if deviation is None and perturbation is None and "nominal_value" not in item:
                continue

            sampled_value = self._sample_with_uncertainty(rng, base, deviation, perturbation)

            if item_name == "attitude":
                q = np.asarray(sampled_value, dtype=float)
                if q.size == 4:
                    norm_q = float(np.linalg.norm(q))
                    if norm_q > 0.0:
                        sampled_value = (q / norm_q).tolist()

            item["value"] = sampled_value

        physical_properties = trial_cfg.get("physical_properties", []) or []
        for component in physical_properties:
            base_mass = component.get("nominal_mass", component.get("mass"))
            if base_mass is None:
                continue

            mass_deviation = component.get("mass_deviation")
            mass_perturbation = component.get("mass_perturbation")

            if mass_deviation is None and mass_perturbation is None and "nominal_mass" not in component:
                continue

            sampled_mass = self._sample_with_uncertainty(
                rng,
                base_mass,
                mass_deviation,
                mass_perturbation,
            )
            sampled_mass_value = max(0.0, float(sampled_mass))
            component["mass"] = sampled_mass_value

        return trial_cfg

    @staticmethod
    def _orbit_period_seconds(position_m: np.ndarray) -> float:
        r = np.linalg.norm(position_m)
        if r <= 0.0:
            raise ValueError("Initial position norm must be positive")
        return 2.0 * np.pi * np.sqrt(r**3 / MU_EARTH) # TODO Change for non-circular orbits later

    def _setup_logger(self) -> Path:
        """Create a file logger under Project/results for simulation run diagnostics."""
        results_dir = self.output_dir
        log_file = results_dir / "simulation.log"

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

    def run_monte_carlo(
        self,
        trials: int | None = None,
        max_workers: int | None = None,
        seed: int | None = None,
        save_plots: bool = False,
    ) -> dict[str, object]:
        """Run Monte Carlo trials in parallel and store each trial in its own folder."""
        sim_props = self.cfg.get("simulation_properties", []) or []
        mc_item = self._property_item(sim_props, "monte_carlo") or {}

        mc_enabled = bool(mc_item.get("value", False))
        if not mc_enabled:
            raise ValueError("monte_carlo is disabled in simulation_properties")

        total_trials = int(trials if trials is not None else mc_item.get("trials", 1))
        if total_trials <= 0:
            raise ValueError("Monte Carlo trials must be positive")

        base_seed = int(seed if seed is not None else mc_item.get("seed", 42))
        default_root = self.output_dir if self._output_dir_explicit else (self.config_path.parent / "results")
        mc_root = default_root / "monte_carlo"
        root_dir = mc_root / datetime.now().strftime('%Y%m%d_%H%M%S')
        root_dir.mkdir(parents=True, exist_ok=True)

        trial_jobs: list[tuple[str, str, bool]] = []
        for i in range(total_trials):
            run_dir = root_dir / f"run_{i + 1:04d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            run_cfg = self._build_trial_config(i, base_seed)
            run_cfg_path = run_dir / "config.yaml"
            with run_cfg_path.open("w", encoding="utf-8") as file:
                yaml.safe_dump(run_cfg, file, sort_keys=False)
            trial_jobs.append((str(run_cfg_path), str(run_dir), save_plots))

        if max_workers is None:
            cpu_count = os.cpu_count() or 1
            max_workers = min(total_trials, max(1, cpu_count - 1))
        max_workers = max(1, int(max_workers))

        summaries: list[dict[str, object]] = []
        if max_workers == 1:
            for job in trial_jobs:
                summaries.append(_run_single_monte_carlo_trial(*job))
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_run_single_monte_carlo_trial, *job) for job in trial_jobs]
                for future in as_completed(futures):
                    summaries.append(future.result())

        summaries.sort(key=lambda item: str(item.get("output_dir", "")))

        summary_payload = {
            "root_dir": str(root_dir),
            "trials": total_trials,
            "seed": base_seed,
            "max_workers": max_workers,
            "completed": sum(1 for item in summaries if item.get("status") == "ok"),
            "runs": summaries,
        }

        summary_path = root_dir / "summary.yaml"
        with summary_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(summary_payload, file, sort_keys=False)

        return summary_payload

    def _default_plot_dir(self) -> Path:
        return self.output_dir if self.output_dir is not None else (self.config_path.parent / "results")

    @staticmethod
    def _style_time_axis(ax: plt.Axes) -> None:
        ax.set_facecolor("#f2f4f8")
        ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.7)

    @staticmethod
    def _set_equal_orbit_axes(ax: plt.Axes, pos_km: np.ndarray) -> None:
        x_min, x_max = float(np.min(pos_km[:, 0])), float(np.max(pos_km[:, 0]))
        y_min, y_max = float(np.min(pos_km[:, 1])), float(np.max(pos_km[:, 1]))
        z_min, z_max = float(np.min(pos_km[:, 2])), float(np.max(pos_km[:, 2]))
        x_mid = 0.5 * (x_min + x_max)
        y_mid = 0.5 * (y_min + y_max)
        z_mid = 0.5 * (z_min + z_max)
        half_range = 0.5 * max(x_max - x_min, y_max - y_min, z_max - z_min)
        if half_range == 0.0:
            half_range = 1.0
        ax.set_xlim(x_mid - half_range, x_mid + half_range)
        ax.set_ylim(y_mid - half_range, y_mid + half_range)
        ax.set_zlim(z_mid - half_range, z_mid + half_range)
        ax.set_box_aspect((1.0, 1.0, 1.0))

    def _save_figure(
        self,
        fig: plt.Figure,
        save_path: str | Path,
        log_message: str,
        dpi: int = 150,
    ) -> Path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        self.logger.info("%s: %s", log_message, output_path)
        return output_path

    def _attitude_plot_values(self, attitudes: np.ndarray) -> np.ndarray:
        attitude_history = np.asarray(attitudes, dtype=float)
        if self.attitude_plot_mode == "quaternion":
            return attitude_history

        norms = np.linalg.norm(attitude_history, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized_attitude = attitude_history / norms
        return np.rad2deg(
            np.asarray([quaternion_to_euler(q) for q in normalized_attitude], dtype=float)
        )

    def _attitude_plot_spec(self, attitudes: np.ndarray) -> dict[str, object]:
        if self.attitude_plot_mode == "quaternion":
            return {
                "values": self._attitude_plot_values(attitudes),
                "labels": ["q0 [-]", "q1 [-]", "q2 [-]", "q3 [-]"],
                "colors": ["#6d28d9", "#db2777", "#0ea5e9", "#16a34a"],
                "title": "Quaternion Components",
                "filename": "simulation_attitude_quaternion.png",
            }

        return {
            "values": self._attitude_plot_values(attitudes),
            "labels": ["roll [deg]", "pitch [deg]", "yaw [deg]"],
            "colors": ["#6d28d9", "#db2777", "#0ea5e9"],
            "title": "Euler Angle Components",
            "filename": "simulation_attitude_euler.png",
        }

    def _simulation_plot_paths(
        self,
        save_path: str | Path | None,
        attitude_filename: str,
    ) -> dict[str, Path]:
        if save_path is None:
            root_dir = self._default_plot_dir()
            return {
                "overview": root_dir / "simulation_plot.png",
                "trajectory": root_dir / "simulation_trajectory.png",
                "velocity": root_dir / "simulation_velocity.png",
                "attitude": root_dir / attitude_filename,
                "angular_velocity": root_dir / "simulation_angular_velocity.png",
            }

        base_path = Path(save_path)
        if self.plot_layout == "together":
            return {"overview": base_path}

        if base_path.suffix:
            stem_path = base_path.with_suffix("")
            prefix = stem_path.name
            root_dir = stem_path.parent
            return {
                "trajectory": root_dir / f"{prefix}_trajectory.png",
                "velocity": root_dir / f"{prefix}_velocity.png",
                "attitude": root_dir / f"{prefix}_{Path(attitude_filename).stem}.png",
                "angular_velocity": root_dir / f"{prefix}_angular_velocity.png",
            }

        return {
            "trajectory": base_path / "simulation_trajectory.png",
            "velocity": base_path / "simulation_velocity.png",
            "attitude": base_path / attitude_filename,
            "angular_velocity": base_path / "simulation_angular_velocity.png",
        }

    def _plot_orbit_figure(self, pos_km: np.ndarray) -> plt.Figure:
        fig = plt.figure(figsize=(10, 9), facecolor="#f7f8fa")
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.set_facecolor("#f2f4f8")
        ax.plot(pos_km[:, 0], pos_km[:, 1], pos_km[:, 2], linewidth=2.2, color="#0f766e", alpha=0.95)
        ax.scatter(
            pos_km[0, 0],
            pos_km[0, 1],
            pos_km[0, 2],
            color="#19a04b",
            edgecolors="white",
            linewidths=0.8,
            s=55,
            label="start",
            zorder=3,
        )
        ax.scatter(
            pos_km[-1, 0],
            pos_km[-1, 1],
            pos_km[-1, 2],
            color="#e12121",
            edgecolors="white",
            linewidths=0.8,
            s=55,
            label="end",
            zorder=3,
        )
        ax.set_title("Trajectory (ECEF)")
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
        ax.set_zlabel("z [km]")
        ax.legend(loc="best")
        self._set_equal_orbit_axes(ax, pos_km)
        fig.tight_layout()
        return fig

    def _plot_velocity_figure(self, times: np.ndarray, vel_kms: np.ndarray) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 5), facecolor="#f7f8fa")
        self._style_time_axis(ax)
        ax.plot(times, vel_kms[:, 0], label="vx", linewidth=2.0, color="#2563eb")
        ax.plot(times, vel_kms[:, 1], label="vy", linewidth=2.0, color="#f59e0b")
        ax.plot(times, vel_kms[:, 2], label="vz", linewidth=2.0, color="#14b8a6")
        ax.set_title("Velocity Components")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("velocity [km/s]")
        ax.legend(loc="best")
        fig.tight_layout()
        return fig

    def _plot_component_stack(
        self,
        times: np.ndarray,
        values: np.ndarray,
        labels: list[str],
        colors: list[str],
        title: str,
    ) -> plt.Figure:
        component_values = np.asarray(values, dtype=float)
        if component_values.ndim == 1:
            component_values = component_values[:, np.newaxis]

        n_components = component_values.shape[1]
        fig, axes = plt.subplots(
            n_components,
            1,
            figsize=(12, max(4.5, 2.35 * n_components)),
            facecolor="#f7f8fa",
            sharex=True,
        )
        axes_array = np.atleast_1d(axes)
        fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)

        for i, ax in enumerate(axes_array):
            self._style_time_axis(ax)
            ax.plot(times, component_values[:, i], linewidth=1.6, color=colors[i % len(colors)])
            ax.set_ylabel(labels[i])

        axes_array[-1].set_xlabel("time [s]")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
        return fig

    def _plot_component_overlay(
        self,
        times: np.ndarray,
        values: np.ndarray,
        labels: list[str],
        colors: list[str],
        title: str,
        ylabel: str,
    ) -> plt.Figure:
        component_values = np.asarray(values, dtype=float)
        if component_values.ndim == 1:
            component_values = component_values[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(12, 5), facecolor="#f7f8fa")
        self._style_time_axis(ax)
        for i in range(component_values.shape[1]):
            ax.plot(
                times,
                component_values[:, i],
                linewidth=1.6,
                color=colors[i % len(colors)],
                label=labels[i],
            )
        ax.set_title(title)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        fig.tight_layout()
        return fig

    def _plot_monte_carlo_component_stack(
        self,
        trial_series: list[tuple[np.ndarray, np.ndarray]],
        labels: list[str],
        colors: list[str],
        title: str,
        line_alpha: float,
    ) -> plt.Figure:
        n_components = len(labels)
        fig, axes = plt.subplots(
            n_components,
            1,
            figsize=(12, max(4.5, 2.35 * n_components)),
            facecolor="#f7f8fa",
            sharex=True,
        )
        axes_array = np.atleast_1d(axes)
        fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)

        for i, ax in enumerate(axes_array):
            self._style_time_axis(ax)
            for times, values in trial_series:
                if i < values.shape[1]:
                    ax.plot(times, values[:, i], color=colors[i % len(colors)], alpha=line_alpha, linewidth=1.0)
            ax.set_ylabel(labels[i])

        axes_array[-1].set_xlabel("time [s]")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
        return fig

    def _plot_monte_carlo_component_overlay(
        self,
        trial_series: list[tuple[np.ndarray, np.ndarray]],
        labels: list[str],
        colors: list[str],
        title: str,
        ylabel: str,
        line_alpha: float,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 5), facecolor="#f7f8fa")
        self._style_time_axis(ax)

        for trial_index, (times, values) in enumerate(trial_series):
            for component_index in range(values.shape[1]):
                ax.plot(
                    times,
                    values[:, component_index],
                    color=colors[component_index % len(colors)],
                    alpha=line_alpha,
                    linewidth=1.0,
                    label=labels[component_index] if trial_index == 0 else None,
                )

        ax.set_title(title)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        fig.tight_layout()
        return fig

    def _plot_monte_carlo_overview(
        self,
        plot_groups: list[dict[str, object]],
        line_alpha: float,
    ) -> plt.Figure:
        total_components = sum(
            1 if bool(group.get("overlay", False)) else len(group["labels"])
            for group in plot_groups
        )
        fig, axes = plt.subplots(
            total_components,
            1,
            figsize=(14, max(10.0, 1.95 * total_components)),
            facecolor="#f7f8fa",
            sharex=True,
        )
        axes_array = np.atleast_1d(axes)
        fig.suptitle("Monte Carlo Trials: State Components", fontsize=16, fontweight="bold", y=0.995)

        axis_index = 0
        for group in plot_groups:
            labels = list(group["labels"])
            colors = list(group["colors"])
            trials = list(group["trial_series"])
            title = str(group["title"])
            overlay = bool(group.get("overlay", False))
            ylabel = str(group.get("ylabel", "value"))

            if overlay:
                ax = axes_array[axis_index]
                self._style_time_axis(ax)
                for trial_index, (times, values) in enumerate(trials):
                    for component_index, label in enumerate(labels):
                        if component_index >= values.shape[1]:
                            continue
                        ax.plot(
                            times,
                            values[:, component_index],
                            color=colors[component_index % len(colors)],
                            alpha=line_alpha,
                            linewidth=1.0,
                            label=label if trial_index == 0 else None,
                        )
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.legend(loc="best")
                axis_index += 1
                continue

            for component_index, label in enumerate(labels):
                ax = axes_array[axis_index]
                self._style_time_axis(ax)
                for times, values in trials:
                    if component_index < values.shape[1]:
                        ax.plot(
                            times,
                            values[:, component_index],
                            color=colors[component_index % len(colors)],
                            alpha=line_alpha,
                            linewidth=1.0,
                        )
                ax.set_ylabel(label)
                if component_index == 0:
                    ax.set_title(title)
                axis_index += 1

        axes_array[-1].set_xlabel("time [s]")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
        return fig

    def _monte_carlo_plot_paths(
        self,
        root_dir: str | Path,
        save_path: str | Path | None,
        attitude_plot_suffix: str,
    ) -> dict[str, Path]:
        if save_path is None:
            root_dir = Path(root_dir)
            return {
                "overview": root_dir / "monte_carlo_components.png",
                "position": root_dir / "monte_carlo_position.png",
                "velocity": root_dir / "monte_carlo_velocity.png",
                "attitude": root_dir / f"monte_carlo_attitude_{attitude_plot_suffix}.png",
                "angular_velocity": root_dir / "monte_carlo_angular_velocity.png",
            }

        base_path = Path(save_path)
        if self.plot_layout == "together":
            return {"overview": base_path}

        if base_path.suffix:
            stem_path = base_path.with_suffix("")
            prefix = stem_path.name
            root_dir = stem_path.parent
            return {
                "position": root_dir / f"{prefix}_position.png",
                "velocity": root_dir / f"{prefix}_velocity.png",
                "attitude": root_dir / f"{prefix}_attitude_{attitude_plot_suffix}.png",
                "angular_velocity": root_dir / f"{prefix}_angular_velocity.png",
            }

        return {
            "position": base_path / "monte_carlo_position.png",
            "velocity": base_path / "monte_carlo_velocity.png",
            "attitude": base_path / f"monte_carlo_attitude_{attitude_plot_suffix}.png",
            "angular_velocity": base_path / "monte_carlo_angular_velocity.png",
        }

    def _plot_simulation_overview(
        self,
        times: np.ndarray,
        pos_km: np.ndarray,
        vel_kms: np.ndarray,
        attitude_spec: dict[str, object],
        att_rate: np.ndarray,
    ) -> plt.Figure:
        att_values = np.asarray(attitude_spec["values"], dtype=float)
        att_labels = list(attitude_spec["labels"])
        att_colors = list(attitude_spec["colors"])
        angle_ylabel = "quaternion [-]" if self.attitude_plot_mode == "quaternion" else "angle [deg]"
        omega_colors = ["#2563eb", "#f97316", "#059669"]
        omega_labels = ["wx [rad/s]", "wy [rad/s]", "wz [rad/s]"]
        overlay_attitude_plots = self.attitude_plot_layout == "overlay"

        if overlay_attitude_plots:
            fig_height = 14.5
        else:
            fig_height = 11.5 + 0.65 * (att_values.shape[1] + att_rate.shape[1])
        fig = plt.figure(figsize=(17, fig_height), facecolor="#f7f8fa")
        fig.suptitle("ARGUS Simulation Overview", fontsize=16, fontweight="bold", y=0.98)

        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 2.2])

        ax_orbit = fig.add_subplot(gs[0, 0], projection="3d")
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
        self._set_equal_orbit_axes(ax_orbit, pos_km)

        ax_vel = fig.add_subplot(gs[0, 1])
        self._style_time_axis(ax_vel)
        ax_vel.plot(times, vel_kms[:, 0], label="vx", linewidth=2.0, color="#2563eb")
        ax_vel.plot(times, vel_kms[:, 1], label="vy", linewidth=2.0, color="#f59e0b")
        ax_vel.plot(times, vel_kms[:, 2], label="vz", linewidth=2.0, color="#14b8a6")
        ax_vel.set_title("Velocity Components")
        ax_vel.set_xlabel("time [s]")
        ax_vel.set_ylabel("velocity [km/s]")
        ax_vel.legend(loc="best")
        ax_vel.set_box_aspect(1)

        if overlay_attitude_plots:
            component_gs = gs[1, :].subgridspec(2, 1, hspace=0.25)

            ax_att = fig.add_subplot(component_gs[0, 0])
            self._style_time_axis(ax_att)
            for i in range(att_values.shape[1]):
                ax_att.plot(times, att_values[:, i], linewidth=1.5, color=att_colors[i], label=att_labels[i])
            ax_att.set_title(str(attitude_spec["title"]))
            ax_att.set_ylabel(angle_ylabel)
            ax_att.legend(loc="best")
            ax_att.tick_params(labelbottom=False)

            ax_omega = fig.add_subplot(component_gs[1, 0])
            self._style_time_axis(ax_omega)
            for i in range(att_rate.shape[1]):
                ax_omega.plot(
                    times,
                    att_rate[:, i],
                    linewidth=1.5,
                    color=omega_colors[i % len(omega_colors)],
                    label=omega_labels[i],
                )
            ax_omega.set_title("Angular Velocity Components")
            ax_omega.set_xlabel("time [s]")
            ax_omega.set_ylabel("angular velocity [rad/s]")
            ax_omega.legend(loc="best")
        else:
            total_rows = att_values.shape[1] + att_rate.shape[1]
            component_gs = gs[1, :].subgridspec(total_rows, 1, hspace=0.22)

            for i in range(att_values.shape[1]):
                ax = fig.add_subplot(component_gs[i, 0])
                self._style_time_axis(ax)
                ax.plot(times, att_values[:, i], linewidth=1.4, color=att_colors[i], label=att_labels[i])
                ax.set_ylabel(att_labels[i])
                if i == 0:
                    ax.set_title(str(attitude_spec["title"]))
                ax.tick_params(labelbottom=False)

            for i in range(att_rate.shape[1]):
                ax = fig.add_subplot(component_gs[att_values.shape[1] + i, 0])
                self._style_time_axis(ax)
                ax.plot(
                    times,
                    att_rate[:, i],
                    linewidth=1.4,
                    color=omega_colors[i % len(omega_colors)],
                    label=omega_labels[i],
                )
                ax.set_ylabel(omega_labels[i])
                if i == 0:
                    ax.set_title("Angular Velocity Components")
                if i < (att_rate.shape[1] - 1):
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel("time [s]")

        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
        return fig

    def plot_simulation(
        self,
        result: dict[str, np.ndarray | float | int],
        show: bool = True,
        save_path: str | Path | None = None,
    ) -> plt.Figure | dict[str, plt.Figure]:
        """Plot trajectory and state histories from a simulation result."""

        times = np.asarray(result["times_s"], dtype=float)
        history = np.asarray(result["state_history_si"], dtype=float)

        pos_km = history[:, self.idx["POS_ECEF"]] / 1_000.0
        vel_kms = history[:, self.idx["VEL_ECEF"]] / 1_000.0
        att = history[:, self.idx["ATTITUDE"]]
        att_rate = history[:, self.idx["ATTITUDE_RATE"]]
        attitude_spec = self._attitude_plot_spec(att)
        plot_paths = self._simulation_plot_paths(save_path, str(attitude_spec["filename"]))

        if self.plot_layout == "together":
            fig = self._plot_simulation_overview(times, pos_km, vel_kms, attitude_spec, att_rate)
            self._save_figure(fig, plot_paths["overview"], "Simulation plot saved")

            if show:
                plt.show()

            return fig

        figures = {
            "trajectory": self._plot_orbit_figure(pos_km),
            "velocity": self._plot_velocity_figure(times, vel_kms),
            "attitude": (
                self._plot_component_overlay(
                    times,
                    np.asarray(attitude_spec["values"], dtype=float),
                    list(attitude_spec["labels"]),
                    list(attitude_spec["colors"]),
                    str(attitude_spec["title"]),
                    "quaternion [-]" if self.attitude_plot_mode == "quaternion" else "angle [deg]",
                )
                if self.attitude_plot_layout == "overlay"
                else self._plot_component_stack(
                    times,
                    np.asarray(attitude_spec["values"], dtype=float),
                    list(attitude_spec["labels"]),
                    list(attitude_spec["colors"]),
                    str(attitude_spec["title"]),
                )
            ),
            "angular_velocity": (
                self._plot_component_overlay(
                    times,
                    att_rate,
                    ["wx [rad/s]", "wy [rad/s]", "wz [rad/s]"],
                    ["#2563eb", "#f97316", "#059669"],
                    "Angular Velocity Components",
                    "angular velocity [rad/s]",
                )
                if self.attitude_plot_layout == "overlay"
                else self._plot_component_stack(
                    times,
                    att_rate,
                    ["wx [rad/s]", "wy [rad/s]", "wz [rad/s]"],
                    ["#2563eb", "#f97316", "#059669"],
                    "Angular Velocity Components",
                )
            ),
        }
        self._save_figure(figures["trajectory"], plot_paths["trajectory"], "Trajectory plot saved")
        self._save_figure(figures["velocity"], plot_paths["velocity"], "Velocity plot saved")
        self._save_figure(figures["attitude"], plot_paths["attitude"], "Attitude plot saved")
        self._save_figure(
            figures["angular_velocity"],
            plot_paths["angular_velocity"],
            "Angular velocity plot saved",
        )

        if show:
            plt.show()

        return figures

    def plot_monte_carlo_trials(
        self,
        summary: dict[str, object],
        show: bool = True,
        save_path: str | Path | None = None,
        line_alpha: float = 0.15,
    ) -> plt.Figure | dict[str, plt.Figure]:
        """Overlay all Monte Carlo trials by component with transparent lines."""
        runs = summary.get("runs", []) if isinstance(summary, dict) else []
        valid_runs = [run for run in runs if isinstance(run, dict) and run.get("state_file")]
        if not valid_runs:
            raise ValueError("No Monte Carlo state files found in summary")

        position_trials: list[tuple[np.ndarray, np.ndarray]] = []
        velocity_trials: list[tuple[np.ndarray, np.ndarray]] = []
        attitude_trials: list[tuple[np.ndarray, np.ndarray]] = []
        omega_trials: list[tuple[np.ndarray, np.ndarray]] = []
        for run in valid_runs:
            state_path = Path(str(run["state_file"]))
            data = np.load(state_path)
            times = np.asarray(data["times_s"], dtype=float)
            history = np.asarray(data["state_history_si"], dtype=float)
            position_trials.append((times, history[:, self.idx["POS_ECEF"]]))
            velocity_trials.append((times, history[:, self.idx["VEL_ECEF"]]))
            attitude_trials.append((times, self._attitude_plot_values(history[:, self.idx["ATTITUDE"]])))
            omega_trials.append((times, history[:, self.idx["ATTITUDE_RATE"]]))

        attitude_suffix = "quaternion" if self.attitude_plot_mode == "quaternion" else "euler"
        summary_root_dir = Path(str(summary.get("root_dir", self.config_path.parent / "results")))
        plot_paths = self._monte_carlo_plot_paths(summary_root_dir, save_path, attitude_suffix)
        plot_groups = [
            {
                "title": "Position Components",
                "labels": ["x [m]", "y [m]", "z [m]"],
                "colors": ["#2563eb", "#1d4ed8", "#1e40af"],
                "trial_series": position_trials,
            },
            {
                "title": "Velocity Components",
                "labels": ["vx [m/s]", "vy [m/s]", "vz [m/s]"],
                "colors": ["#f59e0b", "#d97706", "#b45309"],
                "trial_series": velocity_trials,
            },
            {
                "title": "Quaternion Components" if self.attitude_plot_mode == "quaternion" else "Euler Angle Components",
                "labels": (
                    ["q0 [-]", "q1 [-]", "q2 [-]", "q3 [-]"]
                    if self.attitude_plot_mode == "quaternion"
                    else ["roll [deg]", "pitch [deg]", "yaw [deg]"]
                ),
                "colors": ["#7c3aed", "#db2777", "#0ea5e9", "#16a34a"],
                "trial_series": attitude_trials,
                "overlay": self.attitude_plot_layout == "overlay",
                "ylabel": "quaternion [-]" if self.attitude_plot_mode == "quaternion" else "angle [deg]",
            },
            {
                "title": "Angular Velocity Components",
                "labels": ["wx [rad/s]", "wy [rad/s]", "wz [rad/s]"],
                "colors": ["#2563eb", "#f97316", "#059669"],
                "trial_series": omega_trials,
                "overlay": self.attitude_plot_layout == "overlay",
                "ylabel": "angular velocity [rad/s]",
            },
        ]

        if self.plot_layout == "together":
            fig = self._plot_monte_carlo_overview(plot_groups, line_alpha)
            self._save_figure(fig, plot_paths["overview"], "Monte Carlo component plot saved", dpi=180)

            if show:
                plt.show()

            return fig

        figures = {
            "position": self._plot_monte_carlo_component_stack(
                position_trials,
                ["x [m]", "y [m]", "z [m]"],
                ["#2563eb", "#1d4ed8", "#1e40af"],
                "Position Components",
                line_alpha,
            ),
            "velocity": self._plot_monte_carlo_component_stack(
                velocity_trials,
                ["vx [m/s]", "vy [m/s]", "vz [m/s]"],
                ["#f59e0b", "#d97706", "#b45309"],
                "Velocity Components",
                line_alpha,
            ),
            "attitude": (
                self._plot_monte_carlo_component_overlay(
                    attitude_trials,
                    (
                        ["q0 [-]", "q1 [-]", "q2 [-]", "q3 [-]"]
                        if self.attitude_plot_mode == "quaternion"
                        else ["roll [deg]", "pitch [deg]", "yaw [deg]"]
                    ),
                    ["#7c3aed", "#db2777", "#0ea5e9", "#16a34a"],
                    "Quaternion Components" if self.attitude_plot_mode == "quaternion" else "Euler Angle Components",
                    "quaternion [-]" if self.attitude_plot_mode == "quaternion" else "angle [deg]",
                    line_alpha,
                )
                if self.attitude_plot_layout == "overlay"
                else self._plot_monte_carlo_component_stack(
                    attitude_trials,
                    (
                        ["q0 [-]", "q1 [-]", "q2 [-]", "q3 [-]"]
                        if self.attitude_plot_mode == "quaternion"
                        else ["roll [deg]", "pitch [deg]", "yaw [deg]"]
                    ),
                    ["#7c3aed", "#db2777", "#0ea5e9", "#16a34a"],
                    "Quaternion Components" if self.attitude_plot_mode == "quaternion" else "Euler Angle Components",
                    line_alpha,
                )
            ),
            "angular_velocity": (
                self._plot_monte_carlo_component_overlay(
                    omega_trials,
                    ["wx [rad/s]", "wy [rad/s]", "wz [rad/s]"],
                    ["#2563eb", "#f97316", "#059669"],
                    "Angular Velocity Components",
                    "angular velocity [rad/s]",
                    line_alpha,
                )
                if self.attitude_plot_layout == "overlay"
                else self._plot_monte_carlo_component_stack(
                    omega_trials,
                    ["wx [rad/s]", "wy [rad/s]", "wz [rad/s]"],
                    ["#2563eb", "#f97316", "#059669"],
                    "Angular Velocity Components",
                    line_alpha,
                )
            ),
        }
        self._save_figure(figures["position"], plot_paths["position"], "Monte Carlo position plot saved", dpi=180)
        self._save_figure(figures["velocity"], plot_paths["velocity"], "Monte Carlo velocity plot saved", dpi=180)
        self._save_figure(figures["attitude"], plot_paths["attitude"], "Monte Carlo attitude plot saved", dpi=180)
        self._save_figure(
            figures["angular_velocity"],
            plot_paths["angular_velocity"],
            "Monte Carlo angular velocity plot saved",
            dpi=180,
        )

        if show:
            plt.show()

        return figures

    def plot_momentum_sphere(self, result: dict[str, np.ndarray | float | int], show: bool = True, save_path: str | Path | None = None) -> None:
        """Plot the spacecraft's normalized body angular momentum on the unit sphere."""

        history = np.asarray(result["state_history_si"], dtype=float)
        w = history[:, self.idx["ATTITUDE_RATE"]]
        n = w.shape[0]
        J = self.spacecraft.compute_inertia_tensor()

        h_body = (J @ w.T).T
        h_norm = np.linalg.norm(h_body, axis=1, keepdims=True)
        h_norm[h_norm == 0.0] = 1.0
        momentum_history = (h_body / h_norm).T
        principal_moments, principal_axes = np.linalg.eigh(J)
        
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
            linewidth=2.4,
            color="#0f172a",
            alpha=0.95,
        )
        axis_colors = ["#dc2626", "#16a34a", "#2563eb"]
        legend_handles = [
            Line2D([0], [0], color="#0f172a", linewidth=2.0, label="Momentum path"),
        ]
        for i, (moment, color) in enumerate(zip(principal_moments, axis_colors), start=1):
            axis_vec = principal_axes[:, i - 1]
            pos_point = axis_vec
            neg_point = -axis_vec

            ax.scatter(
                pos_point[0],
                pos_point[1],
                pos_point[2],
                color=color,
                s=80,
                marker="o",
                edgecolors="white",
                linewidths=0.9,
                zorder=5,
            )
            ax.scatter(
                neg_point[0],
                neg_point[1],
                neg_point[2],
                color=color,
                s=90,
                marker="X",
                edgecolors="white",
                linewidths=0.9,
                zorder=5,
            )

            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=0.9,
                    markersize=9,
                    label=f"+ Principal Axis {i}",
                )
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="X",
                    color="w",
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=0.9,
                    markersize=9,
                    label=f"- Principal Axis {i}",
                )
            )

        ax.set_title("Normalized Body Angular Momentum Sphere")
        ax.set_xlabel("Lx / ||L|| [-]")
        ax.set_ylabel("Ly / ||L|| [-]")
        ax.set_zlabel("Lz / ||L|| [-]")
        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=8)
        fig.tight_layout()

        if save_path is None:
            default_dir = self.output_dir if self.output_dir is not None else (self.config_path.parent / "results")
            save_path = default_dir / "momentum_sphere.png"


        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        self.logger.info("Momentum sphere plot saved: %s", output_path)

        if show:
            plt.show(block=True)

        return fig


    def render_3D(self):
        """Use Meshcat to produce a 3D visualization of the spacecraft trajectory and attitude over time."""
        # TODO

        return
