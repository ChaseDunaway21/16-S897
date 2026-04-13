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
from world.math import quaternion_to_euler
from world.models.constants import RADIUS_EARTH


class SimulationPlotContext(Protocol):
    idx: Mapping[str, Any]
    plot_layout: str
    attitude_plot_layout: str
    attitude_plot_mode: str
    output_dir: Path | None
    config_path: Path
    logger: logging.Logger


EARTH_RADIUS_KM = RADIUS_EARTH / 1_000.0


def attitude_plot_values(ctx: SimulationPlotContext, attitudes: np.ndarray) -> np.ndarray:
    attitude_history = np.asarray(attitudes, dtype=float)
    if ctx.attitude_plot_mode == "quaternion":
        return attitude_history

    norms = np.linalg.norm(attitude_history, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized_attitude = attitude_history / norms
    return np.rad2deg(
        np.asarray([quaternion_to_euler(q) for q in normalized_attitude], dtype=float)
    )


def attitude_plot_spec(ctx: SimulationPlotContext, attitudes: np.ndarray) -> dict[str, object]:
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
        }

    base_path = Path(save_path)
    if ctx.plot_layout == "together":
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


def plot_earth_sphere(ax: plt.Axes) -> None:
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
        alpha=0.20,
        edgecolor="none",
        antialiased=True,
        shade=True,
        zorder=0,
    )


def plot_orbit_figure(pos_km: np.ndarray) -> plt.Figure:
    fig = plt.figure(figsize=(10, 9), facecolor=FIGURE_FACE_COLOR)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_facecolor(AXIS_FACE_COLOR)
    plot_earth_sphere(ax)
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
    ax.set_title("Trajectory (ECI)")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")
    ax.legend(loc="best")
    set_equal_orbit_axes(ax, orbit_extent_points(pos_km))
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
        ax.plot(times, component_values[:, i], linewidth=1.6, color=colors[i % len(colors)])
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


def plot_simulation_overview(
    ctx: SimulationPlotContext,
    times: np.ndarray,
    pos_km: np.ndarray,
    vel_kms: np.ndarray,
    attitude_spec: dict[str, object],
    att_rate: np.ndarray,
) -> plt.Figure:
    att_values = np.asarray(attitude_spec["values"], dtype=float)
    att_labels = list(attitude_spec["labels"])
    att_colors = list(attitude_spec["colors"])
    angle_ylabel = "quaternion [-]" if ctx.attitude_plot_mode == "quaternion" else "angle [deg]"
    omega_colors = ["#2563eb", "#f97316", "#059669"]
    omega_labels = ["wx [rad/s]", "wy [rad/s]", "wz [rad/s]"]
    overlay_attitude_plots = ctx.attitude_plot_layout == "overlay"

    if overlay_attitude_plots:
        fig_height = 14.5
    else:
        fig_height = 11.5 + 0.65 * (att_values.shape[1] + att_rate.shape[1])
    fig = plt.figure(figsize=(17, fig_height), facecolor=FIGURE_FACE_COLOR)
    fig.suptitle("ARGUS Simulation Overview", fontsize=16, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 2.2])

    ax_orbit = fig.add_subplot(gs[0, 0], projection="3d")
    ax_orbit.set_facecolor(AXIS_FACE_COLOR)
    plot_earth_sphere(ax_orbit)
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
        component_gs = gs[1, :].subgridspec(2, 1, hspace=0.25)

        ax_att = fig.add_subplot(component_gs[0, 0])
        style_time_axis(ax_att)
        for i in range(att_values.shape[1]):
            ax_att.plot(times, att_values[:, i], linewidth=1.5, color=att_colors[i], label=att_labels[i])
        ax_att.set_title(str(attitude_spec["title"]))
        ax_att.set_ylabel(angle_ylabel)
        ax_att.legend(loc="best")
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
        ax_omega.set_xlabel("time [s]")
        ax_omega.set_ylabel("angular velocity [rad/s]")
        ax_omega.legend(loc="best")
    else:
        total_rows = att_values.shape[1] + att_rate.shape[1]
        component_gs = gs[1, :].subgridspec(total_rows, 1, hspace=0.22)

        for i in range(att_values.shape[1]):
            ax = fig.add_subplot(component_gs[i, 0])
            style_time_axis(ax)
            ax.plot(times, att_values[:, i], linewidth=1.4, color=att_colors[i], label=att_labels[i])
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
            if i < (att_rate.shape[1] - 1):
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("time [s]")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    return fig


def plot_simulation(
    ctx: SimulationPlotContext,
    result: dict[str, np.ndarray | float | int],
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
    current_attitude_spec = attitude_plot_spec(ctx, att)
    plot_paths = simulation_plot_paths(ctx, save_path, str(current_attitude_spec["filename"]))

    if ctx.plot_layout == "together":
        fig = plot_simulation_overview(ctx, times, pos_km, vel_kms, current_attitude_spec, att_rate)
        save_figure(ctx.logger, fig, plot_paths["overview"], "Simulation plot saved")

        if show:
            plt.show()

        return fig

    figures = {
        "trajectory": plot_orbit_figure(pos_km),
        "velocity": plot_velocity_figure(times, vel_kms),
        "attitude": (
            plot_component_overlay(
                times,
                np.asarray(current_attitude_spec["values"], dtype=float),
                list(current_attitude_spec["labels"]),
                list(current_attitude_spec["colors"]),
                str(current_attitude_spec["title"]),
                "quaternion [-]" if ctx.attitude_plot_mode == "quaternion" else "angle [deg]",
            )
            if ctx.attitude_plot_layout == "overlay"
            else plot_component_stack(
                times,
                np.asarray(current_attitude_spec["values"], dtype=float),
                list(current_attitude_spec["labels"]),
                list(current_attitude_spec["colors"]),
                str(current_attitude_spec["title"]),
            )
        ),
        "angular_velocity": (
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
        ),
    }
    save_figure(ctx.logger, figures["trajectory"], plot_paths["trajectory"], "Trajectory plot saved")
    save_figure(ctx.logger, figures["velocity"], plot_paths["velocity"], "Velocity plot saved")
    save_figure(ctx.logger, figures["attitude"], plot_paths["attitude"], "Attitude plot saved")
    save_figure(
        ctx.logger,
        figures["angular_velocity"],
        plot_paths["angular_velocity"],
        "Angular velocity plot saved",
    )

    if show:
        plt.show()

    return figures
