"""Momentum sphere plotting helper extracted from the simulator driver."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from .common import AXIS_FACE_COLOR, FIGURE_FACE_COLOR, default_plot_dir, save_figure


class MomentumPlotContext(Protocol):
    idx: Mapping[str, Any]
    output_dir: Path | None
    config_path: Path
    logger: logging.Logger
    spacecraft: Any


def plot_momentum_sphere(
    ctx: MomentumPlotContext,
    result: dict[str, np.ndarray | float | int],
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the spacecraft's normalized body angular momentum on the unit sphere."""

    history = np.asarray(result["state_history_si"], dtype=float)
    w = history[:, ctx.idx["ATTITUDE_RATE"]]
    inertia_tensor = ctx.spacecraft.inertia_tensor

    h_body = (inertia_tensor @ w.T).T # From lecture notes
    h_magnitude = np.linalg.norm(h_body, axis=1)
    h_norm = h_magnitude[:, np.newaxis]
    h_norm[h_norm == 0.0] = 1.0
    momentum_path = h_body / h_norm 
    momentum_history = momentum_path.T
    principal_moments, principal_axes = np.linalg.eigh(inertia_tensor)
    inertia_inverse = np.linalg.inv(inertia_tensor)
    nonzero_magnitudes = h_magnitude[h_magnitude > 0.0]
    reference_h_magnitude = float(np.mean(nonzero_magnitudes)) if nonzero_magnitudes.size else 1.0

    surface_resolution = 100
    u = np.linspace(-np.pi, np.pi, 1000)
    v = np.linspace(0, np.pi, 1000)
    U, V = np.meshgrid(u, v)

    # The factor 0.95 plots the path floating above the sphere slightly
    x = 0.95 * np.cos(U) * np.sin(V)
    y = 0.95 * np.sin(U) * np.sin(V)
    z = 0.95 * np.cos(V)
    sphere_directions = np.stack((x, y, z), axis=-1)
    energy_field = 0.5 * reference_h_magnitude**2 * np.einsum(
        "...i,ij,...j->...",
        sphere_directions,
        inertia_inverse,
        sphere_directions,
    )
    energy_min = float(np.min(energy_field))
    energy_max = float(np.max(energy_field))
    if np.isclose(energy_min, energy_max):
        energy_max = energy_min + 1.0
    energy_norm = Normalize(vmin=energy_min, vmax=energy_max)
    energy_cmap = plt.get_cmap("cividis")
    sphere_facecolors = energy_cmap(energy_norm(energy_field))
    sphere_facecolors[..., -1] = 0.45

    fig = plt.figure(figsize=(9, 8), facecolor=FIGURE_FACE_COLOR)
    ax = plt.subplot(1, 1, 1, projection="3d")
    ax.set_facecolor(AXIS_FACE_COLOR)
    ax.plot_surface(
        x,
        y,
        z,
        facecolors=sphere_facecolors,
        rcount=surface_resolution,
        ccount=surface_resolution,
        shade=False,
        edgecolor="none",
        antialiased=True,
    )
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
    for i, (_, color) in enumerate(zip(principal_moments, axis_colors), start=1):
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

    ax.set_title("Normalized Body Angular Momentum Sphere with Energy Gradient")
    ax.set_xlabel("Lx / ||L|| [-]")
    ax.set_ylabel("Ly / ||L|| [-]")
    ax.set_zlabel("Lz / ||L|| [-]")
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=8)
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=energy_norm, cmap=energy_cmap),
        ax=ax,
        shrink=0.72,
        pad=0.08,
    )
    colorbar.set_label("Rotational kinetic energy [J]")
    fig.tight_layout()

    output_path = save_path
    if output_path is None:
        output_path = default_plot_dir(ctx.output_dir, ctx.config_path) / "momentum_sphere.png"

    save_figure(ctx.logger, fig, output_path, "Momentum sphere plot saved")

    if show:
        plt.show(block=True)

    return fig
