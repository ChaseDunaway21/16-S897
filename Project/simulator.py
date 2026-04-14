"""Simulation driver for propagating spacecraft state over a configured duration."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import yaml
from matplotlib.figure import Figure

from visualization import (
    plot_momentum_sphere as build_momentum_sphere_plot,
    plot_monte_carlo_trials as build_monte_carlo_plots,
    plot_simulation as build_simulation_plots,
)
from world.models.constants import MU_EARTH
from world.dynamics import integrate_dynamics
from world.spacecraft import Spacecraft


def _run_single_monte_carlo_trial(
    config_path: str,
    output_dir: str,
    save_plots: bool,
) -> dict[str, object]:
    """Run one Monte Carlo trial in an isolated output directory."""
    run_dir = Path(output_dir)
    sim = Simulator(config_path=Path(config_path), output_dir=run_dir)
    result = sim.run(show_progress=False)
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
        "final_position_m": final_state[idx["POS_ECI"]].tolist(),
        "final_velocity_ms": final_state[idx["VEL_ECI"]].tolist(),
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
        self.logger.info("  position [m]: %s", self._vector_to_string(state[self.idx["POS_ECI"]]))
        self.logger.info("  velocity [m/s]: %s", self._vector_to_string(state[self.idx["VEL_ECI"]]))
        self.logger.info("  attitude [-]: %s", self._vector_to_string(state[self.idx["ATTITUDE"]]))
        self.logger.info("  omega [rad/s]: %s", self._vector_to_string(state[self.idx["ATTITUDE_RATE"]]))

    @staticmethod
    def _progress_fraction(current: int, total: int) -> float:
        if total <= 0:
            return 1.0
        return min(max(current / total, 0.0), 1.0)

    @classmethod
    def _progress_line(cls, label: str, current: int, total: int, unit: str, width: int = 28) -> str:
        fraction = cls._progress_fraction(current, total)
        filled = int(width * fraction)
        bar = "#" * filled + "-" * (width - filled)
        return f"{label:<11} [{bar}] {current}/{total} {unit} ({100.0 * fraction:5.1f}%)"

    @classmethod
    def _print_progress(cls, label: str, current: int, total: int, unit: str) -> None:
        line = cls._progress_line(label, current, total, unit)
        sys.stdout.write(f"\r{line}")
        if current >= total:
            sys.stdout.write("\n")
        sys.stdout.flush()

    def run(self, show_progress: bool = True) -> dict[str, np.ndarray | float | int]:
        state = self.spacecraft.get_state().astype(float, copy=True)

        orbit_period = self._orbit_period_seconds(state[self.idx["POS_ECI"]])
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

        if show_progress:
            self._print_progress("Simulation", 0, num_steps, "steps")

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

            if show_progress:
                self._print_progress("Simulation", k, num_steps, "steps")

            if self.log_interval_steps > 0 and (k % self.log_interval_steps == 0 or k == num_steps):
                self._log_state_components("Progress", state, step=k, total_steps=num_steps, time_s=t)

        final_state = history[-1]

        final_pos_m = final_state[self.idx["POS_ECI"]]
        final_vel_ms = final_state[self.idx["VEL_ECI"]]
        updated_state = self.spacecraft.get_state()
        updated_state[self.idx["POS_ECI"]] = final_pos_m
        updated_state[self.idx["VEL_ECI"]] = final_vel_ms
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
        show_progress: bool = True,
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
        completed_trials = 0
        if show_progress:
            self._print_progress("Monte Carlo", completed_trials, total_trials, "trials")

        if max_workers == 1:
            for job in trial_jobs:
                summaries.append(_run_single_monte_carlo_trial(*job))
                completed_trials += 1
                if show_progress:
                    self._print_progress("Monte Carlo", completed_trials, total_trials, "trials")
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_run_single_monte_carlo_trial, *job) for job in trial_jobs]
                for future in as_completed(futures):
                    summaries.append(future.result())
                    completed_trials += 1
                    if show_progress:
                        self._print_progress("Monte Carlo", completed_trials, total_trials, "trials")

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

    def plot_simulation(
        self,
        result: dict[str, np.ndarray | float | int],
        show: bool = True,
        save_path: str | Path | None = None,
    ) -> Figure | dict[str, Figure]:
        """Plot trajectory and state histories from a simulation result."""
        return build_simulation_plots(self, result, show=show, save_path=save_path)

    def plot_monte_carlo_trials(
        self,
        summary: dict[str, object],
        show: bool = True,
        save_path: str | Path | None = None,
        line_alpha: float = 0.15,
    ) -> Figure | dict[str, Figure]:
        """Overlay all Monte Carlo trials by component with transparent lines."""
        return build_monte_carlo_plots(
            self,
            summary,
            show=show,
            save_path=save_path,
            line_alpha=line_alpha,
        )

    def plot_momentum_sphere(
        self,
        result: dict[str, np.ndarray | float | int],
        show: bool = True,
        save_path: str | Path | None = None,
    ) -> Figure:
        """Plot the spacecraft's normalized body angular momentum on the unit sphere."""
        return build_momentum_sphere_plot(self, result, show=show, save_path=save_path)

    def render_3D(self):
        """Use Meshcat to produce a 3D visualization of the spacecraft trajectory and attitude over time."""
        # TODO

        return
