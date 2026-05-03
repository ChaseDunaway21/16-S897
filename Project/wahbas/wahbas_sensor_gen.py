"""Generate Wahba bearing-vector samples from direct sensor getter calls."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from world.math import unit
from world.models.sun import SunModel
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


def enabled_wahba_sensors(
    cfg: dict[str, Any], spacecraft: Spacecraft, rng: np.random.Generator
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    sensor_cfg = cfg.get("sensor_properties", {}) or {}
    sensors: dict[str, object] = {}
    targets: dict[str, np.ndarray] = {}

    if not isinstance(sensor_cfg, dict) or not config_bool(sensor_cfg.get("enabled")):
        return sensors, targets

    magnetometer_cfg = sensor_cfg.get("magnetometer", {}) or {}
    if config_bool(magnetometer_cfg.get("enabled"), True):
        sensors["magnetometer"] = Magnetometer(
            covariance=magnetometer_cfg.get("covariance"),
            rng=rng,
        )

    sun_sensor_cfg = sensor_cfg.get("sun_sensor", {}) or {}
    if config_bool(sun_sensor_cfg.get("enabled"), True):
        sensors["sun_sensor"] = SunSensor(
            sun_model=SunModel(
                direction_eci=spacecraft.sun_direction_eci,
                use_spice=config_bool(sun_sensor_cfg.get("use_spice")),
                require_spice=config_bool(sun_sensor_cfg.get("require_spice")),
            ),
            covariance=sun_sensor_cfg.get("covariance"),
            rng=rng,
            return_none_if_eclipsed=config_bool(
                sun_sensor_cfg.get("return_none_if_eclipsed"),
                True,
            ),
        )

    camera_cfg = sensor_cfg.get("visual_camera", {}) or {}
    if config_bool(camera_cfg.get("enabled"), True):
        sensors["visual_camera"] = VisualCamera(
            boresight_body=np.asarray(
                camera_cfg.get("boresight_body", [1.0, 0.0, 0.0]),
                dtype=float,
            ),
            field_of_view_rad=np.deg2rad(
                float(camera_cfg.get("field_of_view_deg", 75.0))
            ),
            covariance=camera_cfg.get("covariance"),
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
        return unit(sensor.magnetic_field_model.field_eci(position, time_s))
    if sensor_name == "sun_sensor":
        return unit(sensor.sun_model.direction_eci(position, time_s))
    if sensor_name == "visual_camera":
        return unit(targets["visual_camera"] - position)
    raise ValueError(f"{sensor_name} is not configured as a Wahba bearing sensor")


def sensor_measurement(
    sensor_name: str,
    sensor: object,
    state: np.ndarray,
    idx: dict[str, slice],
    targets: dict[str, np.ndarray],
    time_s: float,
) -> np.ndarray | None:
    if sensor_name == "visual_camera":
        return sensor.get_measurement(
            state,
            idx,
            time_s,
            target_position_eci=targets["visual_camera"],
        )
    return sensor.get_measurement(state, idx, time_s)


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
            body_vectors.append(unit(measurement))
            reference_vectors.append(
                reference_vector_eci(sensor_name, sensor, state, idx, targets, time_s)
            )

        if len(body_vectors) >= min_vectors:
            return {
                "sensor_names": names,
                "body_vectors": np.asarray(body_vectors, dtype=float),
                "reference_vectors_eci": np.asarray(reference_vectors, dtype=float),
                "attitude_true": q_true,
            }

    raise RuntimeError("Could not collect enough valid bearing vectors")


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
