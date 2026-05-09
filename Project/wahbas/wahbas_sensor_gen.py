"""Generate Wahba bearing-vector samples from direct sensor getter calls."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from world.math_utils import add_bearing_noise, add_noise, unit_vector
from world.models.sun import SunModel
from world.rotations_and_transformations import inertial_to_body
from world.sensors import Magnetometer, SunSensor, VisualCamera
from world.spacecraft import Spacecraft


def config_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def random_quaternion(rng: np.random.Generator) -> np.ndarray:
    q = rng.standard_normal(4)
    return q / np.linalg.norm(q)


def enabled_wahba_sensors(  # Only the vector sensors can be used, so all but the gyro for ARGUS
    cfg: dict[str, Any], spacecraft: Spacecraft, rng: np.random.Generator
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    sensor_cfg = cfg.get("sensor_properties", {}) or {}
    sensors: dict[str, object] = {}
    targets: dict[str, np.ndarray] = {}

    if not isinstance(sensor_cfg, dict) or not config_bool(sensor_cfg.get("enabled")):
        return sensors, targets

    # Is mag enabled
    magnetometer_cfg = sensor_cfg.get("magnetometer", {}) or {}
    if config_bool(magnetometer_cfg.get("enabled"), True):
        sensors["magnetometer"] = Magnetometer(
            covariance=magnetometer_cfg.get("covariance"),
            bias=magnetometer_cfg.get("bias"),
            rng=rng,
        )

    # Is sun sensor enabled
    sun_sensor_cfg = sensor_cfg.get("sun_sensor", {}) or {}
    if config_bool(sun_sensor_cfg.get("enabled"), True):
        sensors["sun_sensor"] = SunSensor(
            sun_model=SunModel(kernel_paths=sun_sensor_cfg.get("kernel_paths", [])),
            sigma_angle_deg=float(sun_sensor_cfg.get("sigma_angle_deg", 0.0)),
            bias=sun_sensor_cfg.get("bias"),
            rng=rng,
            return_none_if_eclipsed=False,
        )

    # Is camera enabled
    camera_cfg = sensor_cfg.get("visual_camera", {}) or {}
    if config_bool(camera_cfg.get("enabled"), True):
        sensors["visual_camera"] = VisualCamera(
            sigma_angle_deg=float(camera_cfg.get("sigma_angle_deg", 0.0)),
            bias=camera_cfg.get("bias"),
            rng=rng,
        )
        targets["visual_camera"] = np.asarray(
            camera_cfg.get("target_position_eci", [0.0, 0.0, 0.0]),
            dtype=float,
        )

    return sensors, targets


def reference_vector_eci(
    sensor_name: str,
    sensor: object,
    state: np.ndarray,
    idx: dict[str, slice],
    targets: dict[str, np.ndarray],
    time_s: float,
) -> np.ndarray:
    position = state[idx["POS_ECI"]]
    if sensor_name == "magnetometer":
        return unit_vector(sensor.magnetic_field_model.field_eci(position, time_s))
    if sensor_name == "sun_sensor":
        return unit_vector(sensor.sun_model.direction_eci(position, time_s))
    if sensor_name == "visual_camera":
        return unit_vector(targets["visual_camera"] - position)
    raise ValueError(f"{sensor_name} is not configured as a Wahba bearing sensor")


def sensor_measurement(
    sensor_name: str,
    sensor: object,
    state: np.ndarray,
    idx: dict[str, slice],
    targets: dict[str, np.ndarray],
    time_s: float,
) -> np.ndarray | None:
    reference_eci = reference_vector_eci(
        sensor_name, sensor, state, idx, targets, time_s
    )
    q = state[idx["ATTITUDE"]]
    if sensor_name == "visual_camera":
        return add_bearing_noise(
            unit_vector(inertial_to_body(q, reference_eci)) + sensor.bias,
            sensor.sigma_angle_rad,
            sensor.rng,
        )
    if sensor_name == "sun_sensor":
        return add_bearing_noise(
            unit_vector(inertial_to_body(q, reference_eci)) + sensor.bias,
            sensor.sigma_angle_rad,
            sensor.rng,
        )
    if sensor_name == "magnetometer":
        position = state[idx["POS_ECI"]]
        field_eci = sensor.magnetic_field_model.field_eci(position, time_s)
        return add_noise(
            inertial_to_body(q, field_eci) + sensor.bias,
            sensor.covariance,
            sensor.rng,
        )
    return None


def generate_wahba_sensor_sample(
    config_path: Path,
    seed: int,
    time_s: float,
    min_vectors: int,
    max_attempts: int,
) -> dict[str, object]:
    with config_path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file) or {}

    rng = np.random.default_rng(seed)
    spacecraft = Spacecraft(config_path)
    sensors, targets = enabled_wahba_sensors(cfg, spacecraft, rng)
    if not sensors:
        raise ValueError("No enabled Wahba-capable sensors found")
    if len(sensors) < 2:
        raise ValueError("At least two Wahba-capable sensors must be enabled")

    required_vectors = min(max(2, int(min_vectors)), len(sensors))
    idx = spacecraft.Idx["X"]
    base_state = spacecraft.get_state().astype(float, copy=True)

    for _ in range(max_attempts):
        q_true = random_quaternion(rng)
        state = base_state.copy()
        state[idx["ATTITUDE"]] = q_true

        names = []
        body_vectors = []
        reference_vectors = []
        for sensor_name, sensor in sensors.items():
            measurement = sensor_measurement(
                sensor_name, sensor, state, idx, targets, time_s
            )
            if measurement is None or not np.isfinite(measurement).all():
                continue
            names.append(sensor_name)
            body_vectors.append(unit_vector(measurement))
            reference_vectors.append(
                reference_vector_eci(sensor_name, sensor, state, idx, targets, time_s)
            )

        if len(body_vectors) >= required_vectors:
            return {
                "sensor_names": names,
                "body_vectors": np.asarray(body_vectors, dtype=float),
                "reference_vectors_eci": np.asarray(reference_vectors, dtype=float),
                "attitude_true": q_true,
            }

    enabled_names = ", ".join(sensors.keys())
    raise RuntimeError(
        f"Could not collect {required_vectors} valid bearing vectors "
        f"from enabled Wahba sensors: {enabled_names}"
    )


def generate_wahba_monte_carlo_samples(
    config_path: Path,
    seed: int,
    time_s: float,
    min_vectors: int,
    max_attempts: int,
    trials: int,
) -> list[dict[str, object]]:
    samples = []
    for trial in range(trials):
        samples.append(
            generate_wahba_sensor_sample(
                config_path=config_path,
                seed=seed + trial,
                time_s=time_s,
                min_vectors=min_vectors,
                max_attempts=max_attempts,
            )
        )
    return samples
