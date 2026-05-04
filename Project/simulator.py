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
from world.estimator import MEKF
from world.models.constants import MU_EARTH
from world.dynamics import integrate_dynamics
import world.models.gravity as gravity
from world.models.sun import SunModel
from world.sensors import (
    Accelerometer,
    Gyroscope,
    Magnetometer,
    SunSensor,
    VisualCamera,
)
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
        if sim.show_momentum_sphere_plot:
            sim.plot_momentum_sphere(result, show=False)

    final_state = np.asarray(result["state_history_si"])[-1]
    idx = sim.idx
    return {
        "status": "ok",
        "output_dir": str(run_dir),
        "log_file": str(result["log_file"]),
        "state_file": str(state_file),
        "sensor_file": result.get("sensor_history_file"),
        "estimator_file": result.get("estimator_history_file"),
        "num_steps": int(result["num_steps"]),
        "final_position_m": final_state[idx["POS_ECI"]].tolist(),
        "final_velocity_ms": final_state[idx["VEL_ECI"]].tolist(),
        "final_attitude": final_state[idx["ATTITUDE"]].tolist(),
        "final_omega_rads": final_state[idx["ATTITUDE_RATE"]].tolist(),
        "final_rho_kgm2s": final_state[idx["RHO"]].tolist(),
    }


class Simulator:
    """Run Simulation of Satellite"""

    def __init__(
        self, config_path: str | Path, output_dir: str | Path | None = None
    ) -> None:
        self.config_path = Path(config_path)
        self._output_dir_explicit = output_dir is not None
        self.single_run_seed: int | None = None

        if output_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = (
                self.config_path.parent / "results" / f"simulation_{stamp}"
            )
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
        self.dt = float(
            self._property_value(
                self.cfg.get("simulation_properties", []), "time_step", 1.0
            )
        )
        self.max_simulation_time = float(
            self._property_value(
                self.cfg.get("simulation_properties", []), "max_simulation_time", 7200.0
            )
        )
        self.integration_method = str(
            self._property_value(
                self.cfg.get("integration_properties", []), "integration_method", "rk4"
            )
        ).lower()
        plotting_properties = self.cfg.get("plotting_properties", [])
        self.plot_layout = self._validated_option(
            self._property_value(plotting_properties, "layout", "together"),
            {"together", "separate"},
            "plotting_properties.layout",
        )
        self.attitude_plot_layout = self._validated_option(
            self._property_value(
                plotting_properties,
                "attitude_plot_layout",
                "overlay",
            ),
            {"overlay", "stacked"},
            "plotting_properties.attitude_plot_layout",
        )
        self.sensor_plot_layout = self._validated_option(
            self._property_value(
                plotting_properties,
                "sensor_plot_layout",
                "overlay",
            ),
            {"overlay", "stacked"},
            "plotting_properties.sensor_plot_layout",
        )
        self.attitude_plot_mode = self._validated_option(
            self._property_value(
                plotting_properties,
                "attitude_representation",
                "quaternion",
            ),
            {"quaternion", "euler"},
            "plotting_properties.attitude_representation",
        )
        self.show_simulation_overview = self._property_bool(
            plotting_properties,
            "show_simulation_overview",
            True,
        )
        self.show_trajectory_plot = self._property_bool(
            plotting_properties, "show_trajectory_plot", True
        )
        self.show_velocity_plot = self._property_bool(
            plotting_properties, "show_velocity_plot", True
        )
        self.show_attitude_plot = self._property_bool(
            plotting_properties, "show_attitude_plot", True
        )
        self.show_angular_velocity_plot = self._property_bool(
            plotting_properties,
            "show_angular_velocity_plot",
            True,
        )
        self.show_gyrostat_components = self._property_bool(
            plotting_properties,
            "show_gyrostat_components",
            True,
        )
        self.show_sun_safe_mode_axis_plot = self._property_bool(
            plotting_properties,
            "show_sun_safe_mode_axis_plot",
            True,
        )
        self.show_sensor_plot = self._property_bool(
            plotting_properties,
            "show_sensor_plot",
            True,
        )
        self.show_camera_measurement_plot = self._property_bool(
            plotting_properties,
            "show_camera_measurement_plot",
            True,
        )
        self.show_estimator_plot = self._property_bool(
            plotting_properties,
            "show_estimator_plot",
            True,
        )
        self.show_momentum_sphere_plot = self._property_bool(
            plotting_properties,
            "show_momentum_sphere_plot",
            True,
        )

        if self.integration_method != "rk4":
            raise ValueError("Only rk4 integration_method is currently supported")

        self.idx = self.spacecraft.Idx["X"]
        self.log_interval_steps = int(
            self._property_value(
                self.cfg.get("simulation_properties", []), "log_interval_steps", 1000
            )
        )
        self.log_file = self._setup_logger()
        if self.single_run_seed is not None:
            self.logger.info(
                "Single non-ideal run uses deterministic sampled parameters with seed=%d",
                self.single_run_seed,
            )
        self._setup_sensors()
        self._setup_estimator()

    @staticmethod
    def _property_value(
        items: Iterable[Dict], target_name: str, default: float | int | str
    ) -> float | int | str:
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

    @classmethod
    def _property_bool(
        cls, items: Iterable[Dict], target_name: str, default: bool = True
    ) -> bool:
        value = cls._property_value(items, target_name, default)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _validated_option(
        value: object, valid_options: set[str], field_name: str
    ) -> str:
        option = str(value).strip().lower()
        if option not in valid_options:
            valid_list = ", ".join(sorted(valid_options))
            raise ValueError(f"{field_name} must be one of: {valid_list}")
        return option

    @staticmethod
    def _config_bool(value: object, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _measurement_array(measurement: np.ndarray | None) -> np.ndarray:
        if measurement is None:
            return np.full(3, np.nan, dtype=float)
        return np.asarray(measurement, dtype=float).reshape(-1)

    def _sensor_update_period(
        self, sensor_name: str, sensor_cfg: dict, default_rate_hz: float
    ) -> float:
        update_rate_hz = float(sensor_cfg.get("update_rate_hz", default_rate_hz))
        if update_rate_hz <= 0.0:
            raise ValueError(
                f"sensor_properties.{sensor_name}.update_rate_hz must be positive"
            )
        return 1.0 / update_rate_hz

    def _setup_sensors(self) -> None:
        sensor_properties = self.cfg.get("sensor_properties", {}) or {}
        self.sensor_models: dict[str, object] = {}
        self.sensor_update_periods: dict[str, float] = {}
        self.sensor_next_update_times: dict[str, float] = {}
        self.sensor_records: dict[str, dict[str, list]] = {}
        self.sensor_targets: dict[str, np.ndarray] = {}

        if not isinstance(sensor_properties, dict):
            return
        if not self._config_bool(sensor_properties.get("enabled"), False):
            return

        default_rate_hz = float(sensor_properties.get("update_rate_hz", 1.0))
        if default_rate_hz <= 0.0:
            raise ValueError("sensor_properties.update_rate_hz must be positive")
        base_seed = int(
            sensor_properties.get(
                "seed", 42 if self.single_run_seed is None else self.single_run_seed
            )
        )

        magnetometer_cfg = sensor_properties.get("magnetometer", {}) or {}
        if self._config_bool(magnetometer_cfg.get("enabled"), True):
            self.sensor_models["magnetometer"] = Magnetometer(
                covariance=magnetometer_cfg.get("covariance"),
                bias=magnetometer_cfg.get("bias"),
                rng=np.random.default_rng(base_seed + 1),
            )
            self.sensor_update_periods["magnetometer"] = self._sensor_update_period(
                "magnetometer", magnetometer_cfg, default_rate_hz
            )

        gyroscope_cfg = sensor_properties.get("gyroscope", {}) or {}
        if self._config_bool(gyroscope_cfg.get("enabled"), True):
            self.sensor_models["gyroscope"] = Gyroscope(
                covariance=gyroscope_cfg.get("covariance"),
                bias=gyroscope_cfg.get("bias"),
                bias_random_walk_sigma=gyroscope_cfg.get("bias_random_walk_sigma"),
                rng=np.random.default_rng(base_seed + 2),
            )
            self.sensor_update_periods["gyroscope"] = self._sensor_update_period(
                "gyroscope", gyroscope_cfg, default_rate_hz
            )

        accelerometer_cfg = sensor_properties.get("accelerometer", {}) or {}
        if self._config_bool(accelerometer_cfg.get("enabled"), True):
            self.sensor_models["accelerometer"] = Accelerometer(
                covariance=accelerometer_cfg.get("covariance"),
                bias=accelerometer_cfg.get("bias"),
                rng=np.random.default_rng(base_seed + 3),
            )
            self.sensor_update_periods["accelerometer"] = self._sensor_update_period(
                "accelerometer", accelerometer_cfg, default_rate_hz
            )

        sun_sensor_cfg = sensor_properties.get("sun_sensor", {}) or {}
        if self._config_bool(sun_sensor_cfg.get("enabled"), True):
            sun_model = SunModel(
                direction_eci=self.spacecraft.sun_direction_eci,
                use_spice=self._config_bool(sun_sensor_cfg.get("use_spice"), False),
                require_spice=self._config_bool(
                    sun_sensor_cfg.get("require_spice"), False
                ),
            )
            self.sensor_models["sun_sensor"] = SunSensor(
                sun_model=sun_model,
                covariance=sun_sensor_cfg.get("covariance"),
                bias=sun_sensor_cfg.get("bias"),
                rng=np.random.default_rng(base_seed + 4),
                return_none_if_eclipsed=self._config_bool(
                    sun_sensor_cfg.get("return_none_if_eclipsed"), True
                ),
            )
            self.sensor_update_periods["sun_sensor"] = self._sensor_update_period(
                "sun_sensor", sun_sensor_cfg, default_rate_hz
            )

        visual_camera_cfg = sensor_properties.get("visual_camera", {}) or {}
        if self._config_bool(visual_camera_cfg.get("enabled"), True):
            self.sensor_models["visual_camera"] = VisualCamera(
                covariance=visual_camera_cfg.get("covariance"),
                bias=visual_camera_cfg.get("bias"),
                rng=np.random.default_rng(base_seed + 5),
            )
            # For now the target is just the center of the Earth
            self.sensor_targets["visual_camera"] = np.asarray(
                visual_camera_cfg.get("target_position_eci", [0.0, 0.0, 0.0]),
                dtype=float,
            )
            self.sensor_update_periods["visual_camera"] = self._sensor_update_period(
                "visual_camera", visual_camera_cfg, default_rate_hz
            )

        enabled = ", ".join(self.sensor_models) if self.sensor_models else "none"
        self.logger.info("Enabled sensors: %s", enabled)

    def _setup_estimator(self) -> None:
        estimator_cfg = self.cfg.get("estimator_properties", {}) or {}
        self.estimator_enabled = False
        self.estimator: MEKF | None = None
        self.estimator_records: dict[str, list] = {
            "times_s": [],
            "states": [],
            "sigmas": [],
        }
        self.latest_gyro_measurement: np.ndarray | None = None

        if not isinstance(estimator_cfg, dict):
            return
        if not self._config_bool(estimator_cfg.get("enabled"), False):
            return

        self.estimator_enabled = True
        self.estimator = MEKF(
            sigma_initial_attitude=float(
                estimator_cfg.get("sigma_initial_attitude", 0.0)
            ),
            sigma_initial_gyro_bias=float(
                estimator_cfg.get("sigma_initial_gyro_bias", 0.0)
            ),
            sigma_gyro_white=float(estimator_cfg.get("sigma_gyro_white", 0.0)),
            sigma_gyro_bias_deriv=float(
                estimator_cfg.get("sigma_gyro_bias_deriv", 0.0)
            ),
            sigma_sunsensor_direction=float(
                estimator_cfg.get("sigma_sunsensor_direction", 0.0)
            ),
            sigma_magnetometer_direction=float(
                estimator_cfg.get("sigma_magnetometer_direction", 0.0)
            ),
        )

        current_state = self.spacecraft.get_state()
        estimator_state = np.zeros(7, dtype=float)
        estimator_state[0:4] = np.asarray(
            estimator_cfg.get(
                "initial_attitude",
                current_state[self.idx["ATTITUDE"]],
            ),
            dtype=float,
        )
        estimator_state[4:7] = np.asarray(
            estimator_cfg.get("initial_gyro_bias", [0.0, 0.0, 0.0]),
            dtype=float,
        )
        self.estimator.set_state(estimator_state)
        self.logger.info("MEKF estimator enabled")

    def _reset_sensor_records(self) -> None:
        self.sensor_records = {
            name: {"times_s": [], "measurements": []} for name in self.sensor_models
        }
        self.sensor_next_update_times = {name: 0.0 for name in self.sensor_models}

    def _reset_estimator_records(self) -> None:
        self.estimator_records = {"times_s": [], "states": [], "sigmas": []}
        self.latest_gyro_measurement = None

    def _sensor_measurement(
        self, sensor_name: str, sensor_model: object, state: np.ndarray, time_s: float
    ) -> np.ndarray | None:
        if sensor_name == "accelerometer":
            acceleration_eci = gravity.acceleration(state[self.idx["POS_ECI"]])
            return sensor_model.get_measurement(
                state, self.idx, time_s, acceleration_eci=acceleration_eci
            )
        if sensor_name == "visual_camera":
            return sensor_model.get_measurement(
                state,
                self.idx,
                time_s,
                target_position_eci=self.sensor_targets["visual_camera"],
            )
        return sensor_model.get_measurement(state, self.idx, time_s)

    def _update_estimator(
        self,
        sensor_name: str,
        sensor_model: object,
        measurement: np.ndarray | None,
        state: np.ndarray,
        time_s: float,
    ) -> bool:
        if not self.estimator_enabled or self.estimator is None:
            return False
        if measurement is None:
            return False

        measurement_array = np.asarray(measurement, dtype=float).reshape(-1)
        if not np.isfinite(measurement_array).all():
            return False

        position = state[self.idx["POS_ECI"]]
        if sensor_name == "gyroscope":
            self.latest_gyro_measurement = measurement_array.copy()
            self.estimator.predict(measurement_array, time_s)
            return True

        if self.latest_gyro_measurement is not None:
            self.estimator.predict(self.latest_gyro_measurement, time_s)

        # The vector measurements are instantaneous at time_s, so the attitude
        # must be propagated to time_s before applying their correction.
        if sensor_name == "magnetometer":
            b_eci = sensor_model.magnetic_field_model.field_eci(position, time_s)
            self.estimator.Bfield_update(measurement_array, b_eci)
            return True
        elif sensor_name == "sun_sensor":
            sun_eci = sensor_model.sun_model.direction_eci(position, time_s)
            self.estimator.sun_sensor_update(measurement_array, sun_eci)
            return True
        elif sensor_name == "visual_camera":
            target_eci = self.sensor_targets["visual_camera"]
            self.estimator.vector_update(measurement_array, target_eci - position)
            return True

        return False

    def _record_due_sensor_measurements(self, state: np.ndarray, time_s: float) -> bool:
        due_names = [
            sensor_name
            for sensor_name in self.sensor_models
            if time_s + 1e-12 >= self.sensor_next_update_times[sensor_name]
        ]
        due_names.sort(key=lambda name: 0 if name == "gyroscope" else 1)

        estimator_updated = False
        for sensor_name in due_names:
            sensor_model = self.sensor_models[sensor_name]
            measurement = self._sensor_measurement(
                sensor_name, sensor_model, state, time_s
            )
            measurement_array = self._measurement_array(measurement)

            self.sensor_records[sensor_name]["times_s"].append(float(time_s))
            self.sensor_records[sensor_name]["measurements"].append(measurement_array)
            estimator_updated = (
                self._update_estimator(
                    sensor_name, sensor_model, measurement, state, time_s
                )
                or estimator_updated
            )

            while self.sensor_next_update_times[sensor_name] <= time_s + 1e-12:
                self.sensor_next_update_times[sensor_name] += (
                    self.sensor_update_periods[sensor_name]
                )

        return estimator_updated

    def _record_estimator_state(self, time_s: float) -> None:
        if not self.estimator_enabled or self.estimator is None:
            return
        self.estimator_records["times_s"].append(float(time_s))
        self.estimator_records["states"].append(self.estimator.get_state())
        self.estimator_records["sigmas"].append(self.estimator.get_uncertainty_sigma())

    def _sensor_history_arrays(self) -> dict[str, dict[str, np.ndarray]]:
        sensor_history = {}
        for sensor_name, records in self.sensor_records.items():
            measurements = records["measurements"]
            if measurements:
                measurement_array = np.asarray(measurements, dtype=float)
            else:
                measurement_array = np.empty((0, 3), dtype=float)
            sensor_history[sensor_name] = {
                "times_s": np.asarray(records["times_s"], dtype=float),
                "measurements": measurement_array,
            }
        return sensor_history

    def _estimator_history_arrays(self) -> dict[str, np.ndarray]:
        if not self.estimator_enabled:
            return {}
        states = self.estimator_records["states"]
        sigmas = self.estimator_records["sigmas"]
        return {
            "times_s": np.asarray(self.estimator_records["times_s"], dtype=float),
            "state_estimates": (
                np.asarray(states, dtype=float)
                if states
                else np.empty((0, 7), dtype=float)
            ),
            "error_sigmas": (
                np.asarray(sigmas, dtype=float)
                if sigmas
                else np.empty((0, 6), dtype=float)
            ),
        }

    def _save_sensor_history(
        self, sensor_history: dict[str, dict[str, np.ndarray]]
    ) -> Path | None:
        if not sensor_history:
            return None

        payload = {}
        for sensor_name, sensor_data in sensor_history.items():
            payload[f"{sensor_name}_times_s"] = sensor_data["times_s"]
            payload[f"{sensor_name}_measurements"] = sensor_data["measurements"]

        sensor_file = self.output_dir / "sensor_history.npz"
        np.savez_compressed(sensor_file, **payload)
        self.logger.info("Sensor history saved: %s", sensor_file)
        return sensor_file

    def _save_estimator_history(
        self, estimator_history: dict[str, np.ndarray]
    ) -> Path | None:
        if not estimator_history:
            return None

        estimator_file = self.output_dir / "estimator_history.npz"
        np.savez_compressed(estimator_file, **estimator_history)
        self.logger.info("Estimator history saved: %s", estimator_file)
        return estimator_file

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
        """Create one trial config with sampled initial-condition uncertainties."""
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

            if (
                deviation is None
                and perturbation is None
                and "nominal_value" not in item
            ):
                continue

            sampled_value = self._sample_with_uncertainty(
                rng, base, deviation, perturbation
            )

            if item_name == "attitude":
                q = np.asarray(sampled_value, dtype=float)
                if q.size == 4:
                    norm_q = float(np.linalg.norm(q))
                    if norm_q > 0.0:
                        sampled_value = (q / norm_q).tolist()

            item["value"] = sampled_value

        inertia_properties = trial_cfg.get("inertia_properties", []) or []
        inertia_seed = self._property_item(inertia_properties, "inertia_seed")
        if inertia_seed is None:
            inertia_properties.append(
                {"name": "inertia_seed", "value": seed + trial_index}
            )
            trial_cfg["inertia_properties"] = inertia_properties
        else:
            inertia_seed["value"] = seed + trial_index

        sensor_properties = trial_cfg.get("sensor_properties", {}) or {}
        if isinstance(sensor_properties, dict) and "seed" in sensor_properties:
            sensor_properties["seed"] = seed + trial_index

        return trial_cfg

    @staticmethod
    def _orbit_period_seconds(position_m: np.ndarray) -> float:
        r = np.linalg.norm(position_m)
        return (
            2.0 * np.pi * np.sqrt(r**3 / MU_EARTH)
        )  # TODO Change for non-circular orbits later

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
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        self.logger.addHandler(handler)

        self.logger.info("Logger initialized")
        self.logger.info("Config file: %s", self.config_path)
        return log_file

    @staticmethod
    def _vector_to_string(vec: np.ndarray, precision: int = 6) -> str:
        """Format a vector for readable component logging."""
        return np.array2string(
            np.asarray(vec, dtype=float), precision=precision, separator=", "
        )

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
        self.logger.info(
            "  position [m]: %s", self._vector_to_string(state[self.idx["POS_ECI"]])
        )
        self.logger.info(
            "  velocity [m/s]: %s", self._vector_to_string(state[self.idx["VEL_ECI"]])
        )
        self.logger.info(
            "  attitude [-]: %s", self._vector_to_string(state[self.idx["ATTITUDE"]])
        )
        self.logger.info(
            "  omega [rad/s]: %s",
            self._vector_to_string(state[self.idx["ATTITUDE_RATE"]]),
        )
        self.logger.info(
            "  rho [kg m^2/s]: %s", self._vector_to_string(state[self.idx["RHO"]])
        )

    @staticmethod
    def _progress_fraction(current: int, total: int) -> float:
        if total <= 0:
            return 1.0
        return min(max(current / total, 0.0), 1.0)

    @classmethod
    def _progress_line(
        cls, label: str, current: int, total: int, unit: str, width: int = 28
    ) -> str:
        fraction = cls._progress_fraction(current, total)
        filled = int(width * fraction)
        bar = "#" * filled + "-" * (width - filled)
        return (
            f"{label:<11} [{bar}] {current}/{total} {unit} ({100.0 * fraction:5.1f}%)"
        )

    @classmethod
    def _print_progress(cls, label: str, current: int, total: int, unit: str) -> None:
        line = cls._progress_line(label, current, total, unit)
        sys.stdout.write(f"\r{line}")
        if current >= total:
            sys.stdout.write("\n")
        sys.stdout.flush()

    def run(self, show_progress: bool = True) -> dict[str, object]:
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
        rho = state[self.idx["RHO"]]
        if self.spacecraft.safe_mode_enabled:
            rho_message = (
                f"Dynamic balance rho (Jeff = J_33 * {self.spacecraft.J_33_multiplier:.6g}): "
                f"{self._vector_to_string(rho)} [kg m^2/s] "
                f" | rho magnitude: {np.linalg.norm(rho):.6g} [kg m^2/s]"
            )
        else:
            rho_message = (
                f"Safe mode disabled; gyrostat rho: {self._vector_to_string(rho)} "
                f"[kg m^2/s] | rho magnitude: {np.linalg.norm(rho):.6g} [kg m^2/s]"
            )
        self.logger.info(rho_message)
        if show_progress or self.spacecraft.debug:
            print(rho_message)
        if self.spacecraft.safe_mode_enabled:
            desired_omega = (
                self.spacecraft.desired_spin_rate * self.spacecraft.desired_spin_axis
            )
            initial_omega = state[self.idx["ATTITUDE_RATE"]]
            omega_error = initial_omega - desired_omega
            if np.linalg.norm(omega_error) > 1e-9:
                self.logger.warning(
                    "Initial omega differs from dynamic-balance omega by %s rad/s; "
                    "rho is balanced for desired omega %s rad/s, not the current initial omega %s rad/s",
                    self._vector_to_string(omega_error),
                    self._vector_to_string(desired_omega),
                    self._vector_to_string(initial_omega),
                )
        self._log_state_components(
            "Initial state", state, step=0, total_steps=num_steps, time_s=0.0
        )

        times = np.zeros(num_steps + 1, dtype=float)
        history = np.zeros((num_steps + 1, state.size), dtype=float)
        history[0] = state
        self._reset_sensor_records()
        self._reset_estimator_records()
        self._record_due_sensor_measurements(state, 0.0)
        self._record_estimator_state(0.0)

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
            if self._record_due_sensor_measurements(state, t):
                self._record_estimator_state(t)

            if show_progress:
                self._print_progress("Simulation", k, num_steps, "steps")

            if self.log_interval_steps > 0 and (
                k % self.log_interval_steps == 0 or k == num_steps
            ):
                self._log_state_components(
                    "Progress", state, step=k, total_steps=num_steps, time_s=t
                )

        final_state = history[-1]

        final_pos_m = final_state[self.idx["POS_ECI"]]
        final_vel_ms = final_state[self.idx["VEL_ECI"]]
        updated_state = self.spacecraft.get_state()
        updated_state[self.idx["POS_ECI"]] = final_pos_m
        updated_state[self.idx["VEL_ECI"]] = final_vel_ms
        updated_state[self.idx["RHO"]] = final_state[self.idx["RHO"]]
        self.spacecraft.set_state(updated_state)

        self.logger.info("Simulation complete")
        self._log_state_components(
            "Final state",
            updated_state,
            step=num_steps,
            total_steps=num_steps,
            time_s=t,
        )
        self.logger.info("Log file saved: %s", self.log_file)
        sensor_history = self._sensor_history_arrays()
        sensor_history_file = self._save_sensor_history(sensor_history)
        estimator_history = self._estimator_history_arrays()
        estimator_history_file = self._save_estimator_history(estimator_history)

        return {
            "times_s": times,
            "state_history_si": history,
            "orbit_period_s": orbit_period,
            "sim_duration_s": sim_duration,
            "num_steps": num_steps,
            "log_file": str(self.log_file),
            "dynamic_balance_rho": rho,
            "sensor_measurements": sensor_history,
            "sensor_history_file": (
                None if sensor_history_file is None else str(sensor_history_file)
            ),
            "estimator_history": estimator_history,
            "estimator_history_file": (
                None if estimator_history_file is None else str(estimator_history_file)
            ),
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
        default_root = (
            self.output_dir
            if self._output_dir_explicit
            else (self.config_path.parent / "results")
        )
        mc_root = default_root / "monte_carlo"
        root_dir = mc_root / datetime.now().strftime("%Y%m%d_%H%M%S")
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
            self._print_progress(
                "Monte Carlo", completed_trials, total_trials, "trials"
            )

        if max_workers == 1:
            for job in trial_jobs:
                summaries.append(_run_single_monte_carlo_trial(*job))
                completed_trials += 1
                if show_progress:
                    self._print_progress(
                        "Monte Carlo", completed_trials, total_trials, "trials"
                    )
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_run_single_monte_carlo_trial, *job)
                    for job in trial_jobs
                ]
                for future in as_completed(futures):
                    summaries.append(future.result())
                    completed_trials += 1
                    if show_progress:
                        self._print_progress(
                            "Monte Carlo", completed_trials, total_trials, "trials"
                        )

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
