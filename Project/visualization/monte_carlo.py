"""Monte Carlo plotting helpers extracted from the simulator driver."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Protocol

import matplotlib.pyplot as plt
import numpy as np

from .common import FIGURE_FACE_COLOR, save_figure, style_time_axis
from .simulation import attitude_plot_values


class MonteCarloPlotContext(Protocol):
    idx: Mapping[str, Any]
    plot_layout: str
    attitude_plot_layout: str
    attitude_plot_mode: str
    config_path: Path
    logger: logging.Logger


def plot_monte_carlo_component_stack(
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
        facecolor=FIGURE_FACE_COLOR,
        sharex=True,
    )
    axes_array = np.atleast_1d(axes)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)

    for i, ax in enumerate(axes_array):
        style_time_axis(ax)
        for times, values in trial_series:
            if i < values.shape[1]:
                ax.plot(times, values[:, i], color=colors[i % len(colors)], alpha=line_alpha, linewidth=1.0)
        ax.set_ylabel(labels[i])

    axes_array[-1].set_xlabel("time [s]")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    return fig


def plot_monte_carlo_component_overlay(
    trial_series: list[tuple[np.ndarray, np.ndarray]],
    labels: list[str],
    colors: list[str],
    title: str,
    ylabel: str,
    line_alpha: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=FIGURE_FACE_COLOR)
    style_time_axis(ax)

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


def plot_monte_carlo_overview(
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
        facecolor=FIGURE_FACE_COLOR,
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
            style_time_axis(ax)
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
            style_time_axis(ax)
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


def monte_carlo_plot_paths(
    ctx: MonteCarloPlotContext,
    root_dir: str | Path,
    save_path: str | Path | None,
    attitude_plot_suffix: str,
) -> dict[str, Path]:
    if save_path is None:
        output_root = Path(root_dir)
        return {
            "overview": output_root / "monte_carlo_components.png",
            "position": output_root / "monte_carlo_position.png",
            "velocity": output_root / "monte_carlo_velocity.png",
            "attitude": output_root / f"monte_carlo_attitude_{attitude_plot_suffix}.png",
            "angular_velocity": output_root / "monte_carlo_angular_velocity.png",
        }

    base_path = Path(save_path)
    if ctx.plot_layout == "together":
        return {"overview": base_path}

    if base_path.suffix:
        stem_path = base_path.with_suffix("")
        prefix = stem_path.name
        output_root = stem_path.parent
        return {
            "position": output_root / f"{prefix}_position.png",
            "velocity": output_root / f"{prefix}_velocity.png",
            "attitude": output_root / f"{prefix}_attitude_{attitude_plot_suffix}.png",
            "angular_velocity": output_root / f"{prefix}_angular_velocity.png",
        }

    return {
        "position": base_path / "monte_carlo_position.png",
        "velocity": base_path / "monte_carlo_velocity.png",
        "attitude": base_path / f"monte_carlo_attitude_{attitude_plot_suffix}.png",
        "angular_velocity": base_path / "monte_carlo_angular_velocity.png",
    }


def plot_monte_carlo_trials(
    ctx: MonteCarloPlotContext,
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
        with np.load(state_path) as data:
            times = np.asarray(data["times_s"], dtype=float)
            history = np.asarray(data["state_history_si"], dtype=float)
        position_trials.append((times, history[:, ctx.idx["POS_ECEF"]]))
        velocity_trials.append((times, history[:, ctx.idx["VEL_ECEF"]]))
        attitude_trials.append((times, attitude_plot_values(ctx, history[:, ctx.idx["ATTITUDE"]])))
        omega_trials.append((times, history[:, ctx.idx["ATTITUDE_RATE"]]))

    attitude_suffix = "quaternion" if ctx.attitude_plot_mode == "quaternion" else "euler"
    summary_root_dir = Path(str(summary.get("root_dir", ctx.config_path.parent / "results")))
    plot_paths = monte_carlo_plot_paths(ctx, summary_root_dir, save_path, attitude_suffix)
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
            "title": "Quaternion Components" if ctx.attitude_plot_mode == "quaternion" else "Euler Angle Components",
            "labels": (
                ["q0 [-]", "q1 [-]", "q2 [-]", "q3 [-]"]
                if ctx.attitude_plot_mode == "quaternion"
                else ["roll [deg]", "pitch [deg]", "yaw [deg]"]
            ),
            "colors": ["#7c3aed", "#db2777", "#0ea5e9", "#16a34a"],
            "trial_series": attitude_trials,
            "overlay": ctx.attitude_plot_layout == "overlay",
            "ylabel": "quaternion [-]" if ctx.attitude_plot_mode == "quaternion" else "angle [deg]",
        },
        {
            "title": "Angular Velocity Components",
            "labels": ["wx [rad/s]", "wy [rad/s]", "wz [rad/s]"],
            "colors": ["#2563eb", "#f97316", "#059669"],
            "trial_series": omega_trials,
            "overlay": ctx.attitude_plot_layout == "overlay",
            "ylabel": "angular velocity [rad/s]",
        },
    ]

    if ctx.plot_layout == "together":
        fig = plot_monte_carlo_overview(plot_groups, line_alpha)
        save_figure(ctx.logger, fig, plot_paths["overview"], "Monte Carlo component plot saved", dpi=180)

        if show:
            plt.show()

        return fig

    figures = {
        "position": plot_monte_carlo_component_stack(
            position_trials,
            ["x [m]", "y [m]", "z [m]"],
            ["#2563eb", "#1d4ed8", "#1e40af"],
            "Position Components",
            line_alpha,
        ),
        "velocity": plot_monte_carlo_component_stack(
            velocity_trials,
            ["vx [m/s]", "vy [m/s]", "vz [m/s]"],
            ["#f59e0b", "#d97706", "#b45309"],
            "Velocity Components",
            line_alpha,
        ),
        "attitude": (
            plot_monte_carlo_component_overlay(
                attitude_trials,
                (
                    ["q0 [-]", "q1 [-]", "q2 [-]", "q3 [-]"]
                    if ctx.attitude_plot_mode == "quaternion"
                    else ["roll [deg]", "pitch [deg]", "yaw [deg]"]
                ),
                ["#7c3aed", "#db2777", "#0ea5e9", "#16a34a"],
                "Quaternion Components" if ctx.attitude_plot_mode == "quaternion" else "Euler Angle Components",
                "quaternion [-]" if ctx.attitude_plot_mode == "quaternion" else "angle [deg]",
                line_alpha,
            )
            if ctx.attitude_plot_layout == "overlay"
            else plot_monte_carlo_component_stack(
                attitude_trials,
                (
                    ["q0 [-]", "q1 [-]", "q2 [-]", "q3 [-]"]
                    if ctx.attitude_plot_mode == "quaternion"
                    else ["roll [deg]", "pitch [deg]", "yaw [deg]"]
                ),
                ["#7c3aed", "#db2777", "#0ea5e9", "#16a34a"],
                "Quaternion Components" if ctx.attitude_plot_mode == "quaternion" else "Euler Angle Components",
                line_alpha,
            )
        ),
        "angular_velocity": (
            plot_monte_carlo_component_overlay(
                omega_trials,
                ["wx [rad/s]", "wy [rad/s]", "wz [rad/s]"],
                ["#2563eb", "#f97316", "#059669"],
                "Angular Velocity Components",
                "angular velocity [rad/s]",
                line_alpha,
            )
            if ctx.attitude_plot_layout == "overlay"
            else plot_monte_carlo_component_stack(
                omega_trials,
                ["wx [rad/s]", "wy [rad/s]", "wz [rad/s]"],
                ["#2563eb", "#f97316", "#059669"],
                "Angular Velocity Components",
                line_alpha,
            )
        ),
    }
    save_figure(ctx.logger, figures["position"], plot_paths["position"], "Monte Carlo position plot saved", dpi=180)
    save_figure(ctx.logger, figures["velocity"], plot_paths["velocity"], "Monte Carlo velocity plot saved", dpi=180)
    save_figure(ctx.logger, figures["attitude"], plot_paths["attitude"], "Monte Carlo attitude plot saved", dpi=180)
    save_figure(
        ctx.logger,
        figures["angular_velocity"],
        plot_paths["angular_velocity"],
        "Monte Carlo angular velocity plot saved",
        dpi=180,
    )

    if show:
        plt.show()

    return figures
