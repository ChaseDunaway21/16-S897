"""Simulation plotting helpers extracted from the simulator driver."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Protocol

import matplotlib.pyplot as plt
import numpy as np

from .common import (
    AXIS_FACE_COLOR,
    FIGURE_FACE_COLOR,
    default_plot_dir,
    save_figure,
    set_equal_orbit_axes,
    style_time_axis,
)
from world.rotations_and_transformations import (
    L,
    R_body_to_inertial,
    normalize_quaternion,
    quaternion_to_euler,
)
from world.models.constants import EARTH_RADIUS_KM


class SimulationPlotContext(Protocol):
    idx: Mapping[str, Any]
    plot_layout: str
    attitude_plot_layout: str
    sensor_plot_layout: str
    attitude_plot_mode: str
    show_simulation_overview: bool
    show_trajectory_plot: bool
    show_velocity_plot: bool
    show_attitude_plot: bool
    show_angular_velocity_plot: bool
    show_gyrostat_components: bool
    show_sun_safe_mode_axis_plot: bool
    show_sensor_plot: bool
    show_camera_measurement_plot: bool
    show_estimator_plot: bool
    sensor_targets: Mapping[str, np.ndarray]
    spacecraft: Any
    output_dir: Path | None
    config_path: Path
    logger: logging.Logger


SENSOR_PLOT_SPEC = {
    "magnetometer": (
        "Magnetometer Measurements",
        "magnetic field [uT]",
        ["Bx [uT]", "By [uT]", "Bz [uT]"],
    ),
    "gyroscope": (
        "Gyroscope Measurements",
        "angular velocity [rad/s]",
        ["wx [rad/s]", "wy [rad/s]", "wz [rad/s]"],
    ),
    "accelerometer": (
        "Accelerometer Measurements",
        "specific force [m/s^2]",
        ["ax [m/s^2]", "ay [m/s^2]", "az [m/s^2]"],
    ),
    "sun_sensor": (
        "Sun Sensor Measurements",
        "unit vector [-]",
        ["sx [-]", "sy [-]", "sz [-]"],
    ),
    "visual_camera": (
        "Visual Camera Bearing Measurements",
        "bearing [-]",
        ["bx [-]", "by [-]", "bz [-]"],
    ),
}
SENSOR_COLORS = ["#2563eb", "#f97316", "#059669", "#7c3aed"]


def attitude_plot_values(
    ctx: SimulationPlotContext, attitudes: np.ndarray
) -> np.ndarray:
    attitude_history = np.asarray(attitudes, dtype=float)
    if ctx.attitude_plot_mode == "quaternion":
        return attitude_history

    norms = np.linalg.norm(attitude_history, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized_attitude = attitude_history / norms
    return np.rad2deg(
        np.asarray([quaternion_to_euler(q) for q in normalized_attitude], dtype=float)
    )


def attitude_plot_spec(
    ctx: SimulationPlotContext, attitudes: np.ndarray
) -> dict[str, object]:
    if ctx.attitude_plot_mode == "quaternion":
        return {
            "values": attitude_plot_values(ctx, attitudes),
            "labels": ["q0 [-]", "q1 [-]", "q2 [-]", "q3 [-]"],
            "colors": ["#6d28d9", "#db2777", "#0ea5e9", "#16a34a"],
            "title": "Quaternion Components",
            "filename": "simulation_attitude_quaternion.png",
        }

    return {
        "values": attitude_plot_values(ctx, attitudes),
        "labels": ["roll [deg]", "pitch [deg]", "yaw [deg]"],
        "colors": ["#6d28d9", "#db2777", "#0ea5e9"],
        "title": "Euler Angle Components",
        "filename": "simulation_attitude_euler.png",
    }


def simulation_plot_paths(
    ctx: SimulationPlotContext,
    save_path: str | Path | None,
    attitude_filename: str,
) -> dict[str, Path]:
    if save_path is None:
        root_dir = default_plot_dir(ctx.output_dir, ctx.config_path)
        return {
            "overview": root_dir / "simulation_plot.png",
            "trajectory": root_dir / "simulation_trajectory.png",
            "velocity": root_dir / "simulation_velocity.png",
            "attitude": root_dir / attitude_filename,
            "angular_velocity": root_dir / "simulation_angular_velocity.png",
            "rho": root_dir / "simulation_rho.png",
            "sun_safe_mode_axis": root_dir / "simulation_sun_safe_mode_axis.png",
            "sensors": root_dir / "simulation_sensors.png",
            "camera_measurements": root_dir / "simulation_camera_measurements.png",
            "estimator": root_dir / "simulation_estimator.png",
        }

    base_path = Path(save_path)
    if ctx.plot_layout == "together":
        if base_path.suffix:
            return {
                "overview": base_path,
                "sun_safe_mode_axis": base_path.with_name(
                    f"{base_path.stem}_sun_safe_mode_axis{base_path.suffix}"
                ),
                "sensors": base_path.with_name(
                    f"{base_path.stem}_sensors{base_path.suffix}"
                ),
                "camera_measurements": base_path.with_name(
                    f"{base_path.stem}_camera_measurements{base_path.suffix}"
                ),
                "estimator": base_path.with_name(
                    f"{base_path.stem}_estimator{base_path.suffix}"
                ),
            }
        return {
            "overview": base_path,
            "sun_safe_mode_axis": base_path / "simulation_sun_safe_mode_axis.png",
            "sensors": base_path / "simulation_sensors.png",
            "camera_measurements": base_path / "simulation_camera_measurements.png",
            "estimator": base_path / "simulation_estimator.png",
        }

    if base_path.suffix:
        stem_path = base_path.with_suffix("")
        prefix = stem_path.name
        root_dir = stem_path.parent
        return {
            "trajectory": root_dir / f"{prefix}_trajectory.png",
            "velocity": root_dir / f"{prefix}_velocity.png",
            "attitude": root_dir / f"{prefix}_{Path(attitude_filename).stem}.png",
            "angular_velocity": root_dir / f"{prefix}_angular_velocity.png",
            "rho": root_dir / f"{prefix}_rho.png",
            "sun_safe_mode_axis": root_dir / f"{prefix}_sun_safe_mode_axis.png",
            "sensors": root_dir / f"{prefix}_sensors.png",
            "camera_measurements": root_dir / f"{prefix}_camera_measurements.png",
            "estimator": root_dir / f"{prefix}_estimator.png",
        }

    return {
        "trajectory": base_path / "simulation_trajectory.png",
        "velocity": base_path / "simulation_velocity.png",
        "attitude": base_path / attitude_filename,
        "angular_velocity": base_path / "simulation_angular_velocity.png",
        "rho": base_path / "simulation_rho.png",
        "sun_safe_mode_axis": base_path / "simulation_sun_safe_mode_axis.png",
        "sensors": base_path / "simulation_sensors.png",
        "camera_measurements": base_path / "simulation_camera_measurements.png",
        "estimator": base_path / "simulation_estimator.png",
    }


def orbit_extent_points(pos_km: np.ndarray) -> np.ndarray:
    earth_extent_points = np.array(
        [
            [EARTH_RADIUS_KM, 0.0, 0.0],
            [-EARTH_RADIUS_KM, 0.0, 0.0],
            [0.0, EARTH_RADIUS_KM, 0.0],
            [0.0, -EARTH_RADIUS_KM, 0.0],
            [0.0, 0.0, EARTH_RADIUS_KM],
            [0.0, 0.0, -EARTH_RADIUS_KM],
        ],
        dtype=float,
    )
    return np.vstack((np.asarray(pos_km, dtype=float), earth_extent_points))


def plot_earth_sphere(ax: plt.Axes, alpha: float = 1.0) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 80)
    v = np.linspace(0.0, np.pi, 40)
    uu, vv = np.meshgrid(u, v)
    x = EARTH_RADIUS_KM * np.cos(uu) * np.sin(vv)
    y = EARTH_RADIUS_KM * np.sin(uu) * np.sin(vv)
    z = EARTH_RADIUS_KM * np.cos(vv)
    ax.plot_surface(
        x,
        y,
        z,
        color="#2563eb",
        alpha=alpha,
        edgecolor="none",
        antialiased=True,
        shade=True,
        zorder=0,
    )


def plot_orbit_figure(pos_km: np.ndarray) -> plt.Figure:
    fig = plt.figure(figsize=(10, 9), facecolor=FIGURE_FACE_COLOR)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_facecolor(AXIS_FACE_COLOR)
    # plot_earth_sphere(ax)
    ax.plot(
        pos_km[:, 0],
        pos_km[:, 1],
        pos_km[:, 2],
        linewidth=1.4,
        color="#FF781F",
        alpha=1.00,
    )
    ax.scatter(
        pos_km[0, 0],
        pos_km[0, 1],
        pos_km[0, 2],
        color="#25d32e",
        edgecolors="white",
        alpha=0.6,
        linewidths=0.8,
        s=55,
        label="start",
        zorder=3,
    )
    ax.scatter(
        pos_km[-1, 0],
        pos_km[-1, 1],
        pos_km[-1, 2],
        color="#f33d3d",
        edgecolors="white",
        alpha=0.6,
        linewidths=0.8,
        s=55,
        label="end",
        zorder=3,
    )
    ax.set_title("Trajectory (ECI)")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")
    ax.legend(loc="upper right")
    set_equal_orbit_axes(ax, orbit_extent_points(pos_km))
    fig.tight_layout()
    return fig


def camera_measurement_samples(
    times: np.ndarray,
    sensor_history: Mapping[str, Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    camera_data = sensor_history.get("visual_camera")
    if camera_data is None:
        return np.empty(0, dtype=int), np.empty((0, 3), dtype=float)

    camera_times = np.asarray(camera_data["times_s"], dtype=float)
    camera_measurements = np.asarray(camera_data["measurements"], dtype=float)
    if camera_times.size == 0 or camera_measurements.size == 0:
        return np.empty(0, dtype=int), np.empty((0, 3), dtype=float)
    if camera_measurements.ndim == 1:
        camera_measurements = camera_measurements.reshape(1, -1)

    valid = np.isfinite(camera_measurements).all(axis=1)
    valid_times = camera_times[valid]
    valid_measurements = camera_measurements[valid]
    if valid_times.size == 0:
        return np.empty(0, dtype=int), np.empty((0, 3), dtype=float)

    indices = np.searchsorted(times, valid_times)
    indices = np.clip(indices, 0, times.size - 1)
    previous_indices = np.maximum(indices - 1, 0)
    use_previous = np.abs(times[previous_indices] - valid_times) < np.abs(
        times[indices] - valid_times
    )
    indices[use_previous] = previous_indices[use_previous]
    return indices, valid_measurements


def plot_camera_measurement_figure(
    times: np.ndarray,
    pos_km: np.ndarray,
    attitudes: np.ndarray,
    sensor_history: Mapping[str, Mapping[str, object]],
    target_position_eci_km: np.ndarray,
) -> plt.Figure | None:
    measurement_indices, measured_bearings_body = camera_measurement_samples(
        times, sensor_history
    )
    if measurement_indices.size == 0:
        return None

    max_vectors = 150
    if measurement_indices.size > max_vectors:
        selected = np.linspace(0, measurement_indices.size - 1, max_vectors, dtype=int)
        measurement_indices = measurement_indices[selected]
        measured_bearings_body = measured_bearings_body[selected]

    camera_positions = pos_km[measurement_indices]
    camera_attitudes = np.asarray(attitudes, dtype=float)[measurement_indices]

    bearing_norms = np.linalg.norm(measured_bearings_body, axis=1, keepdims=True)
    bearing_norms[bearing_norms == 0.0] = 1.0
    measured_bearings_body = measured_bearings_body / bearing_norms
    measured_directions_eci = np.asarray(
        [
            R_body_to_inertial(q) @ bearing_body
            for q, bearing_body in zip(camera_attitudes, measured_bearings_body)
        ],
        dtype=float,
    )
    measured_direction_norms = np.linalg.norm(
        measured_directions_eci, axis=1, keepdims=True
    )
    measured_direction_norms[measured_direction_norms == 0.0] = 1.0
    measured_directions_eci /= measured_direction_norms

    target_km = np.asarray(target_position_eci_km, dtype=float)
    directions = target_km - camera_positions
    direction_norms = np.linalg.norm(directions, axis=1, keepdims=True)
    direction_norms[direction_norms == 0.0] = 1.0
    unit_directions = directions / direction_norms
    vector_length = 0.25 * EARTH_RADIUS_KM

    fig = plt.figure(figsize=(10, 9), facecolor=FIGURE_FACE_COLOR)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_facecolor(AXIS_FACE_COLOR)
    plot_earth_sphere(ax, alpha=0.12)
    ax.plot(
        pos_km[:, 0],
        pos_km[:, 1],
        pos_km[:, 2],
        linewidth=1.2,
        color="#f97316",
        alpha=0.65,
        label="orbit",
    )
    ax.scatter(
        camera_positions[:, 0],
        camera_positions[:, 1],
        camera_positions[:, 2],
        color="#facc15",
        edgecolors="#78350f",
        linewidths=0.5,
        s=22,
        label="camera measurement",
        zorder=4,
    )
    ax.quiver(
        camera_positions[:, 0],
        camera_positions[:, 1],
        camera_positions[:, 2],
        unit_directions[:, 0],
        unit_directions[:, 1],
        unit_directions[:, 2],
        length=vector_length,
        normalize=False,
        color="#6b7280",
        linewidth=0.7,
        alpha=0.35,
        label="ideal target bearing",
    )
    ax.quiver(
        camera_positions[:, 0],
        camera_positions[:, 1],
        camera_positions[:, 2],
        measured_directions_eci[:, 0],
        measured_directions_eci[:, 1],
        measured_directions_eci[:, 2],
        length=vector_length,
        normalize=False,
        color="#dc2626",
        linewidth=0.9,
        alpha=0.85,
        label="measured bearing",
    )
    ax.scatter(
        target_km[0],
        target_km[1],
        target_km[2],
        color="#111827",
        edgecolors="white",
        linewidths=0.7,
        s=60,
        label="target",
        zorder=5,
    )
    ax.set_title("Camera Bearing Measurements (ECI)")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")
    ax.legend(loc="upper right")
    extent_points = np.vstack((pos_km, target_km[np.newaxis, :]))
    set_equal_orbit_axes(ax, orbit_extent_points(extent_points))
    fig.tight_layout()
    return fig


def plot_velocity_figure(times: np.ndarray, vel_kms: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=FIGURE_FACE_COLOR)
    style_time_axis(ax)
    ax.plot(times, vel_kms[:, 0], label="vx", linewidth=2.0, color="#2563eb")
    ax.plot(times, vel_kms[:, 1], label="vy", linewidth=2.0, color="#f59e0b")
    ax.plot(times, vel_kms[:, 2], label="vz", linewidth=2.0, color="#14b8a6")
    ax.set_title("Velocity Components")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("velocity [km/s]")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def sun_safe_mode_axis_values(
    ctx: SimulationPlotContext,
    attitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sun_eci = np.asarray(ctx.spacecraft.sun_vector_eci(), dtype=float)
    sun_eci = sun_eci / np.linalg.norm(sun_eci)

    safe_axis_body = np.asarray(ctx.spacecraft.desired_spin_axis, dtype=float)
    safe_axis_body = safe_axis_body / np.linalg.norm(safe_axis_body)

    safe_axis_eci = np.asarray(
        [
            R_body_to_inertial(q / np.linalg.norm(q)) @ safe_axis_body
            for q in np.asarray(attitudes, dtype=float)
        ],
        dtype=float,
    )
    safe_axis_eci /= np.linalg.norm(safe_axis_eci, axis=1, keepdims=True)

    alignment = np.clip(safe_axis_eci @ sun_eci, -1.0, 1.0)
    return sun_eci, safe_axis_eci, np.rad2deg(np.arccos(alignment))


def plot_sun_safe_mode_axis_figure(
    times: np.ndarray,
    sun_eci: np.ndarray,
    safe_axis_eci: np.ndarray,
    alignment_angle_deg: np.ndarray,
) -> plt.Figure:
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7), facecolor=FIGURE_FACE_COLOR, sharex=True
    )
    safe_colors = ["#dc2626", "#16a34a", "#2563eb"]
    sun_colors = ["#f97316", "#84cc16", "#38bdf8"]
    labels = ["x", "y", "z"]
    sun_history = np.tile(sun_eci, (times.size, 1))

    style_time_axis(axes[0])
    for i, label in enumerate(labels):
        axes[0].plot(
            times,
            safe_axis_eci[:, i],
            color=safe_colors[i],
            linewidth=1.8,
            label=f"safe {label}",
        )
        axes[0].plot(
            times,
            sun_history[:, i],
            color=sun_colors[i],
            linewidth=1.4,
            linestyle="--",
            alpha=0.9,
            label=f"sun {label}",
        )
    axes[0].set_title("Sun Direction and Safe-Mode Axis in ECI")
    axes[0].set_ylabel("component [-]")
    axes[0].legend(loc="upper right", ncol=3)

    style_time_axis(axes[1])
    axes[1].plot(times, alignment_angle_deg, color="#0f172a", linewidth=1.8)
    axes[1].set_title("Sun/Safe-Mode Axis Separation")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("angle [deg]")

    fig.tight_layout()
    return fig


def plot_component_stack(
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
        facecolor=FIGURE_FACE_COLOR,
        sharex=True,
    )
    axes_array = np.atleast_1d(axes)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)

    for i, ax in enumerate(axes_array):
        style_time_axis(ax)
        ax.plot(
            times, component_values[:, i], linewidth=1.6, color=colors[i % len(colors)]
        )
        ax.set_ylabel(labels[i])

    axes_array[-1].set_xlabel("time [s]")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    return fig


def plot_component_overlay(
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

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=FIGURE_FACE_COLOR)
    style_time_axis(ax)
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


def sensor_plot_items(
    sensor_history: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    items = []
    for sensor_name, sensor_data in sensor_history.items():
        times = np.asarray(sensor_data["times_s"], dtype=float)
        measurements = np.asarray(sensor_data["measurements"], dtype=float)
        if times.size == 0 or measurements.size == 0:
            continue
        if measurements.ndim == 1:
            measurements = measurements[:, np.newaxis]

        title, ylabel, labels = SENSOR_PLOT_SPEC.get(
            sensor_name,
            (
                sensor_name.replace("_", " ").title(),
                "measurement [-]",
                [f"component {i + 1}" for i in range(measurements.shape[1])],
            ),
        )
        component_labels = list(labels[: measurements.shape[1]])
        component_labels.extend(
            f"component {i + 1}"
            for i in range(len(component_labels), measurements.shape[1])
        )
        items.append(
            {
                "times": times,
                "measurements": measurements,
                "labels": component_labels,
                "title": title,
                "ylabel": ylabel,
            }
        )
    return items


def plot_sensor_measurements(
    sensor_history: Mapping[str, Mapping[str, object]],
    layout: str,
) -> plt.Figure | None:
    items = sensor_plot_items(sensor_history)
    if not items:
        return None

    if layout == "overlay":
        fig, axes = plt.subplots(
            len(items),
            1,
            figsize=(12, max(4.5, 2.8 * len(items))),
            facecolor=FIGURE_FACE_COLOR,
            sharex=False,
        )
        axes_array = np.atleast_1d(axes)
        fig.suptitle("Sensor Measurements", fontsize=15, fontweight="bold", y=0.99)
        for ax, item in zip(axes_array, items):
            times = np.asarray(item["times"], dtype=float)
            measurements = np.asarray(item["measurements"], dtype=float)
            labels = list(item["labels"])
            style_time_axis(ax)
            for i in range(measurements.shape[1]):
                ax.plot(
                    times,
                    measurements[:, i],
                    linewidth=1.5,
                    color=SENSOR_COLORS[i % len(SENSOR_COLORS)],
                    label=labels[i],
                )
            ax.set_title(str(item["title"]))
            ax.set_ylabel(str(item["ylabel"]))
            ax.legend(loc="best")
        axes_array[-1].set_xlabel("time [s]")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        return fig

    total_rows = sum(np.asarray(item["measurements"]).shape[1] for item in items)
    fig, axes = plt.subplots(
        total_rows,
        1,
        figsize=(12, max(4.5, 2.0 * total_rows)),
        facecolor=FIGURE_FACE_COLOR,
        sharex=False,
    )
    axes_array = np.atleast_1d(axes)
    fig.suptitle("Sensor Measurements", fontsize=15, fontweight="bold", y=0.99)

    row = 0
    for item in items:
        times = np.asarray(item["times"], dtype=float)
        measurements = np.asarray(item["measurements"], dtype=float)
        labels = list(item["labels"])
        for i in range(measurements.shape[1]):
            ax = axes_array[row]
            style_time_axis(ax)
            ax.plot(
                times,
                measurements[:, i],
                linewidth=1.4,
                color=SENSOR_COLORS[i % len(SENSOR_COLORS)],
            )
            ax.set_ylabel(labels[i])
            if i == 0:
                ax.set_title(str(item["title"]))
            row += 1
    axes_array[-1].set_xlabel("time [s]")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def nearest_time_indices(
    reference_times: np.ndarray, query_times: np.ndarray
) -> np.ndarray:
    """Return nearest reference index for each query time."""
    reference_times = np.asarray(reference_times, dtype=float)
    query_times = np.asarray(query_times, dtype=float)
    indices = np.searchsorted(reference_times, query_times)
    indices = np.clip(indices, 0, reference_times.size - 1)
    previous_indices = np.maximum(indices - 1, 0)
    use_previous = np.abs(reference_times[previous_indices] - query_times) < np.abs(
        reference_times[indices] - query_times
    )
    indices[use_previous] = previous_indices[use_previous]
    return indices


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Return the conjugate of one [w, x, y, z] quaternion."""
    q = normalize_quaternion(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def attitude_error_vectors(
    true_attitudes: np.ndarray, estimated_attitudes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return sign-aligned estimated quaternions and MEKF attitude-error vectors."""
    true_q = np.asarray([normalize_quaternion(q) for q in true_attitudes], dtype=float)
    est_q = np.asarray(
        [normalize_quaternion(q) for q in estimated_attitudes], dtype=float
    )

    signs = np.sign(np.sum(true_q * est_q, axis=1, keepdims=True))
    signs[signs == 0.0] = 1.0
    est_q = est_q * signs

    errors = []
    for q_true, q_est in zip(true_q, est_q):
        q_error = normalize_quaternion(L(quaternion_conjugate(q_true)) @ q_est)
        if q_error[0] < 0.0:
            q_error = -q_error
        errors.append(q_error[1:4])

    return est_q, np.asarray(errors, dtype=float)


def configured_gyro_bias(ctx: SimulationPlotContext) -> np.ndarray | None:
    """Return configured gyroscope bias when the simulator exposes it."""
    sensor_models = getattr(ctx, "sensor_models", {})
    gyroscope = (
        sensor_models.get("gyroscope") if isinstance(sensor_models, dict) else None
    )
    if gyroscope is None or not hasattr(gyroscope, "bias"):
        return None
    bias_random_walk_sigma = getattr(gyroscope, "bias_random_walk_sigma", np.zeros(3))
    if np.any(bias_random_walk_sigma):
        return None
    return np.asarray(gyroscope.bias, dtype=float).reshape(3)


def plot_estimator_figure(
    ctx: SimulationPlotContext,
    truth_times: np.ndarray,
    true_attitudes: np.ndarray,
    estimator_history: Mapping[str, object],
) -> plt.Figure | None:
    """Plot MEKF attitude estimate, attitude error, and 3-sigma bounds."""
    if not estimator_history:
        return None

    est_times = np.asarray(estimator_history.get("times_s", []), dtype=float)
    est_states = np.asarray(estimator_history.get("state_estimates", []), dtype=float)
    sigmas = np.asarray(estimator_history.get("error_sigmas", []), dtype=float)
    if est_times.size == 0 or est_states.size == 0 or sigmas.size == 0:
        return None
    if est_states.ndim != 2 or est_states.shape[1] < 7:
        return None
    if sigmas.ndim != 2 or sigmas.shape[1] < 6:
        return None

    truth_times = np.asarray(truth_times, dtype=float)
    truth_q = np.asarray(true_attitudes, dtype=float)
    if truth_times.size == 0 or truth_q.size == 0:
        return None
    if truth_q.ndim != 2 or truth_q.shape[1] < 4:
        return None
    if truth_times.shape[0] != truth_q.shape[0]:
        return None

    truth_q = np.asarray(
        [normalize_quaternion(q) for q in truth_q[:, 0:4]], dtype=float
    )
    truth_indices = nearest_time_indices(truth_times, est_times)
    true_q_at_estimator_times = truth_q[truth_indices]
    est_q, attitude_error = attitude_error_vectors(
        true_q_at_estimator_times, est_states[:, 0:4]
    )
    attitude_bounds = 3.0 * sigmas[:, 0:3]
    bias_estimates = est_states[:, 4:7]
    bias_bounds = 3.0 * sigmas[:, 3:6]
    bias_truth = configured_gyro_bias(ctx)

    fig, axes = plt.subplots(
        3, 1, figsize=(13, 11), facecolor=FIGURE_FACE_COLOR, sharex=True
    )
    fig.suptitle("MEKF Estimate and 3-Sigma Bounds", fontsize=15, fontweight="bold")

    q_colors = ["#6d28d9", "#db2777", "#0ea5e9", "#16a34a"]
    q_labels = ["q0", "q1", "q2", "q3"]
    style_time_axis(axes[0])
    for i, label in enumerate(q_labels):
        axes[0].plot(
            truth_times,
            truth_q[:, i],
            color=q_colors[i],
            linewidth=1.5,
            linestyle="--",
            alpha=0.8,
            label=f"true {label}",
        )
        axes[0].plot(
            est_times,
            est_q[:, i],
            color=q_colors[i],
            linewidth=1.4,
            label=f"est {label}",
        )
    axes[0].set_title("Attitude Quaternion Mean vs Truth")
    axes[0].set_ylabel("quaternion [-]")
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)

    component_colors = ["#2563eb", "#f97316", "#059669"]
    bound_colors = ["#1d4ed8", "#c2410c", "#047857"]
    component_labels = ["x", "y", "z"]
    style_time_axis(axes[1])
    for i, label in enumerate(component_labels):
        color = component_colors[i]
        bound_color = bound_colors[i]
        axes[1].fill_between(
            est_times,
            -attitude_bounds[:, i],
            attitude_bounds[:, i],
            color=bound_color,
            alpha=0.10,
            linewidth=0.0,
            label=f"±3sigma {label}",
        )
        axes[1].plot(
            est_times,
            attitude_error[:, i],
            color=color,
            linewidth=1.4,
            label=f"error {label}",
        )
        axes[1].plot(
            est_times,
            attitude_bounds[:, i],
            color=bound_color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.45,
        )
        axes[1].plot(
            est_times,
            -attitude_bounds[:, i],
            color=bound_color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.45,
        )
    axes[1].set_title("MEKF Quaternion-Vector Error with 3-Sigma Bounds")
    axes[1].set_ylabel("quaternion-vector error [-]")
    axes[1].legend(loc="upper right", ncol=3, fontsize=8)

    style_time_axis(axes[2])
    for i, label in enumerate(component_labels):
        color = component_colors[i]
        bound_color = bound_colors[i]
        axes[2].fill_between(
            est_times,
            -bias_bounds[:, i],
            bias_bounds[:, i],
            color=bound_color,
            alpha=0.10,
            linewidth=0.0,
            label=f"±3sigma {label}",
        )
        axes[2].plot(
            est_times,
            bias_estimates[:, i],
            color=color,
            linewidth=1.4,
            label=f"bias {label}",
        )
        axes[2].plot(
            est_times,
            bias_bounds[:, i],
            color=bound_color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.45,
        )
        axes[2].plot(
            est_times,
            -bias_bounds[:, i],
            color=bound_color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.45,
        )
        if bias_truth is not None:
            axes[2].axhline(
                bias_truth[i],
                color=color,
                linewidth=1.0,
                linestyle=":",
                alpha=0.85,
            )
    axes[2].set_title("Gyroscope Bias Estimate with 3-Sigma Bounds")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("bias [rad/s]")
    axes[2].legend(loc="upper right", ncol=3, fontsize=8)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    return fig


def plot_simulation_overview(
    ctx: SimulationPlotContext,
    times: np.ndarray,
    pos_km: np.ndarray,
    vel_kms: np.ndarray,
    attitude_spec: dict[str, object],
    att_rate: np.ndarray,
    rho: np.ndarray,
) -> plt.Figure:
    att_values = np.asarray(attitude_spec["values"], dtype=float)
    att_labels = list(attitude_spec["labels"])
    att_colors = list(attitude_spec["colors"])
    angle_ylabel = (
        "quaternion [-]" if ctx.attitude_plot_mode == "quaternion" else "angle [deg]"
    )
    omega_colors = ["#2563eb", "#f97316", "#059669"]
    omega_labels = ["wx [rad/s]", "wy [rad/s]", "wz [rad/s]"]
    rho_colors = ["#7c2d12", "#be123c", "#4338ca"]
    rho_labels = ["rho_x [kg m^2/s]", "rho_y [kg m^2/s]", "rho_z [kg m^2/s]"]
    overlay_attitude_plots = ctx.attitude_plot_layout == "overlay"
    show_rho = ctx.show_gyrostat_components

    if overlay_attitude_plots:
        fig_height = 17.0 if show_rho else 14.5
    else:
        rho_rows = rho.shape[1] if show_rho else 0
        fig_height = 11.5 + 0.65 * (att_values.shape[1] + att_rate.shape[1] + rho_rows)
    fig = plt.figure(figsize=(17, fig_height), facecolor=FIGURE_FACE_COLOR)
    fig.suptitle("ARGUS Simulation Overview", fontsize=16, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 2.2])

    ax_orbit = fig.add_subplot(gs[0, 0], projection="3d")
    ax_orbit.set_facecolor(AXIS_FACE_COLOR)
    # plot_earth_sphere(ax_orbit)
    ax_orbit.plot(
        pos_km[:, 0],
        pos_km[:, 1],
        pos_km[:, 2],
        linewidth=2.2,
        color="#FF781F",
        alpha=0.95,
    )
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
    ax_orbit.set_title("Trajectory (ECI)")
    ax_orbit.set_xlabel("x [km]")
    ax_orbit.set_ylabel("y [km]")
    ax_orbit.set_zlabel("z [km]")
    ax_orbit.legend(loc="best")
    set_equal_orbit_axes(ax_orbit, orbit_extent_points(pos_km))

    ax_vel = fig.add_subplot(gs[0, 1])
    style_time_axis(ax_vel)
    ax_vel.plot(times, vel_kms[:, 0], label="vx", linewidth=2.0, color="#2563eb")
    ax_vel.plot(times, vel_kms[:, 1], label="vy", linewidth=2.0, color="#f59e0b")
    ax_vel.plot(times, vel_kms[:, 2], label="vz", linewidth=2.0, color="#14b8a6")
    ax_vel.set_title("Velocity Components")
    ax_vel.set_xlabel("time [s]")
    ax_vel.set_ylabel("velocity [km/s]")
    ax_vel.legend(loc="best")
    ax_vel.set_box_aspect(1)

    if overlay_attitude_plots:
        component_rows = 3 if show_rho else 2
        component_gs = gs[1, :].subgridspec(component_rows, 1, hspace=0.25)

        ax_att = fig.add_subplot(component_gs[0, 0])
        style_time_axis(ax_att)
        for i in range(att_values.shape[1]):
            ax_att.plot(
                times,
                att_values[:, i],
                linewidth=1.5,
                color=att_colors[i],
                label=att_labels[i],
            )
        ax_att.set_title(str(attitude_spec["title"]))
        ax_att.set_ylabel(angle_ylabel)
        ax_att.legend(loc="upper right")
        ax_att.tick_params(labelbottom=False)

        ax_omega = fig.add_subplot(component_gs[1, 0])
        style_time_axis(ax_omega)
        for i in range(att_rate.shape[1]):
            ax_omega.plot(
                times,
                att_rate[:, i],
                linewidth=1.5,
                color=omega_colors[i % len(omega_colors)],
                label=omega_labels[i],
            )
        ax_omega.set_title("Angular Velocity Components")
        ax_omega.set_ylabel("angular velocity [rad/s]")
        ax_omega.legend(loc="upper right")
        if show_rho:
            ax_omega.tick_params(labelbottom=False)
        else:
            ax_omega.set_xlabel("time [s]")

        if show_rho:
            ax_rho = fig.add_subplot(component_gs[2, 0])
            style_time_axis(ax_rho)
            for i in range(rho.shape[1]):
                ax_rho.plot(
                    times,
                    rho[:, i],
                    linewidth=1.5,
                    color=rho_colors[i % len(rho_colors)],
                    label=rho_labels[i],
                )
            ax_rho.set_title("Gyrostat Momentum Components")
            ax_rho.set_xlabel("time [s]")
            ax_rho.set_ylabel("rho [kg m^2/s]")
            ax_rho.legend(loc="upper right")
    else:
        rho_rows = rho.shape[1] if show_rho else 0
        total_rows = att_values.shape[1] + att_rate.shape[1] + rho_rows
        component_gs = gs[1, :].subgridspec(total_rows, 1, hspace=0.22)

        for i in range(att_values.shape[1]):
            ax = fig.add_subplot(component_gs[i, 0])
            style_time_axis(ax)
            ax.plot(
                times,
                att_values[:, i],
                linewidth=1.4,
                color=att_colors[i],
                label=att_labels[i],
            )
            ax.set_ylabel(att_labels[i])
            if i == 0:
                ax.set_title(str(attitude_spec["title"]))
            ax.tick_params(labelbottom=False)

        for i in range(att_rate.shape[1]):
            ax = fig.add_subplot(component_gs[att_values.shape[1] + i, 0])
            style_time_axis(ax)
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
            if show_rho or i < (att_rate.shape[1] - 1):
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("time [s]")

        if show_rho:
            rho_row_start = att_values.shape[1] + att_rate.shape[1]
            for i in range(rho.shape[1]):
                ax = fig.add_subplot(component_gs[rho_row_start + i, 0])
                style_time_axis(ax)
                ax.plot(
                    times,
                    rho[:, i],
                    linewidth=1.4,
                    color=rho_colors[i % len(rho_colors)],
                    label=rho_labels[i],
                )
                ax.set_ylabel(rho_labels[i])
                if i == 0:
                    ax.set_title("Gyrostat Momentum Components")
                if i < (rho.shape[1] - 1):
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel("time [s]")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    return fig


def plot_simulation(
    ctx: SimulationPlotContext,
    result: dict[str, object],
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure | dict[str, plt.Figure]:
    """Plot trajectory and state histories from a simulation result."""

    times = np.asarray(result["times_s"], dtype=float)
    history = np.asarray(result["state_history_si"], dtype=float)

    pos_km = history[:, ctx.idx["POS_ECI"]] / 1_000.0
    vel_kms = history[:, ctx.idx["VEL_ECI"]] / 1_000.0
    att = history[:, ctx.idx["ATTITUDE"]]
    att_rate = history[:, ctx.idx["ATTITUDE_RATE"]]
    rho = history[:, ctx.idx["RHO"]]
    current_attitude_spec = attitude_plot_spec(ctx, att)
    plot_paths = simulation_plot_paths(
        ctx, save_path, str(current_attitude_spec["filename"])
    )
    sun_safe_mode_axis_fig = None
    if ctx.show_sun_safe_mode_axis_plot:
        sun_eci, safe_axis_eci, alignment_angle_deg = sun_safe_mode_axis_values(
            ctx, att
        )
        sun_safe_mode_axis_fig = plot_sun_safe_mode_axis_figure(
            times,
            sun_eci,
            safe_axis_eci,
            alignment_angle_deg,
        )

    sensor_fig = None
    sensor_history = result.get("sensor_measurements", {})
    if ctx.show_sensor_plot and isinstance(sensor_history, Mapping):
        sensor_fig = plot_sensor_measurements(sensor_history, ctx.sensor_plot_layout)

    camera_measurement_fig = None
    if ctx.show_camera_measurement_plot and isinstance(sensor_history, Mapping):
        target_position_eci_km = (
            np.asarray(
                ctx.sensor_targets.get("visual_camera", np.zeros(3)), dtype=float
            )
            / 1_000.0
        )
        camera_measurement_fig = plot_camera_measurement_figure(
            times,
            pos_km,
            att,
            sensor_history,
            target_position_eci_km,
        )

    estimator_fig = None
    estimator_history = result.get("estimator_history", {})
    if getattr(ctx, "show_estimator_plot", True) and isinstance(
        estimator_history, Mapping
    ):
        estimator_fig = plot_estimator_figure(ctx, times, att, estimator_history)

    if ctx.plot_layout == "together":
        fig = None
        if ctx.show_simulation_overview:
            fig = plot_simulation_overview(
                ctx, times, pos_km, vel_kms, current_attitude_spec, att_rate, rho
            )
            save_figure(
                ctx.logger, fig, plot_paths["overview"], "Simulation plot saved"
            )
        if sun_safe_mode_axis_fig is not None:
            save_figure(
                ctx.logger,
                sun_safe_mode_axis_fig,
                plot_paths["sun_safe_mode_axis"],
                "Sun safe-mode axis plot saved",
            )
        if sensor_fig is not None:
            save_figure(
                ctx.logger,
                sensor_fig,
                plot_paths["sensors"],
                "Sensor plot saved",
            )
        if camera_measurement_fig is not None:
            save_figure(
                ctx.logger,
                camera_measurement_fig,
                plot_paths["camera_measurements"],
                "Camera measurement plot saved",
            )
        if estimator_fig is not None:
            save_figure(
                ctx.logger,
                estimator_fig,
                plot_paths["estimator"],
                "Estimator plot saved",
            )

        if show:
            plt.show()

        return (
            fig
            if fig is not None
            else (
                sun_safe_mode_axis_fig
                or sensor_fig
                or camera_measurement_fig
                or estimator_fig
                or {}
            )
        )

    figures = {}
    if ctx.show_trajectory_plot:
        figures["trajectory"] = plot_orbit_figure(pos_km)
    if ctx.show_velocity_plot:
        figures["velocity"] = plot_velocity_figure(times, vel_kms)
    if ctx.show_attitude_plot:
        figures["attitude"] = (
            plot_component_overlay(
                times,
                np.asarray(current_attitude_spec["values"], dtype=float),
                list(current_attitude_spec["labels"]),
                list(current_attitude_spec["colors"]),
                str(current_attitude_spec["title"]),
                "quaternion [-]"
                if ctx.attitude_plot_mode == "quaternion"
                else "angle [deg]",
            )
            if ctx.attitude_plot_layout == "overlay"
            else plot_component_stack(
                times,
                np.asarray(current_attitude_spec["values"], dtype=float),
                list(current_attitude_spec["labels"]),
                list(current_attitude_spec["colors"]),
                str(current_attitude_spec["title"]),
            )
        )
    if ctx.show_angular_velocity_plot:
        figures["angular_velocity"] = (
            plot_component_overlay(
                times,
                att_rate,
                ["wx [rad/s]", "wy [rad/s]", "wz [rad/s]"],
                ["#2563eb", "#f97316", "#059669"],
                "Angular Velocity Components",
                "angular velocity [rad/s]",
            )
            if ctx.attitude_plot_layout == "overlay"
            else plot_component_stack(
                times,
                att_rate,
                ["wx [rad/s]", "wy [rad/s]", "wz [rad/s]"],
                ["#2563eb", "#f97316", "#059669"],
                "Angular Velocity Components",
            )
        )
    if sun_safe_mode_axis_fig is not None:
        figures["sun_safe_mode_axis"] = sun_safe_mode_axis_fig
    if sensor_fig is not None:
        figures["sensors"] = sensor_fig
    if camera_measurement_fig is not None:
        figures["camera_measurements"] = camera_measurement_fig
    if estimator_fig is not None:
        figures["estimator"] = estimator_fig
    if ctx.show_gyrostat_components:
        figures["rho"] = (
            plot_component_overlay(
                times,
                rho,
                ["rho_x [kg m^2/s]", "rho_y [kg m^2/s]", "rho_z [kg m^2/s]"],
                ["#7c2d12", "#be123c", "#4338ca"],
                "Gyrostat Momentum Components",
                "rho [kg m^2/s]",
            )
            if ctx.attitude_plot_layout == "overlay"
            else plot_component_stack(
                times,
                rho,
                ["rho_x [kg m^2/s]", "rho_y [kg m^2/s]", "rho_z [kg m^2/s]"],
                ["#7c2d12", "#be123c", "#4338ca"],
                "Gyrostat Momentum Components",
            )
        )
    if "trajectory" in figures:
        save_figure(
            ctx.logger,
            figures["trajectory"],
            plot_paths["trajectory"],
            "Trajectory plot saved",
        )
    if "velocity" in figures:
        save_figure(
            ctx.logger,
            figures["velocity"],
            plot_paths["velocity"],
            "Velocity plot saved",
        )
    if "attitude" in figures:
        save_figure(
            ctx.logger,
            figures["attitude"],
            plot_paths["attitude"],
            "Attitude plot saved",
        )
    if "angular_velocity" in figures:
        save_figure(
            ctx.logger,
            figures["angular_velocity"],
            plot_paths["angular_velocity"],
            "Angular velocity plot saved",
        )
    if "sun_safe_mode_axis" in figures:
        save_figure(
            ctx.logger,
            figures["sun_safe_mode_axis"],
            plot_paths["sun_safe_mode_axis"],
            "Sun safe-mode axis plot saved",
        )
    if "rho" in figures:
        save_figure(
            ctx.logger,
            figures["rho"],
            plot_paths["rho"],
            "Gyrostat momentum plot saved",
        )
    if "sensors" in figures:
        save_figure(
            ctx.logger,
            figures["sensors"],
            plot_paths["sensors"],
            "Sensor plot saved",
        )
    if "camera_measurements" in figures:
        save_figure(
            ctx.logger,
            figures["camera_measurements"],
            plot_paths["camera_measurements"],
            "Camera measurement plot saved",
        )
    if "estimator" in figures:
        save_figure(
            ctx.logger,
            figures["estimator"],
            plot_paths["estimator"],
            "Estimator plot saved",
        )

    if show:
        plt.show()

    return figures
