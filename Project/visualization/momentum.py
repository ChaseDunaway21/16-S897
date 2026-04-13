"""Momentum sphere plotting helper extracted from the simulator driver."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Protocol

import matplotlib.pyplot as plt
import numpy as np
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
    inertia_tensor = ctx.spacecraft.compute_inertia_tensor()

    h_body = (inertia_tensor @ w.T).T
    h_norm = np.linalg.norm(h_body, axis=1, keepdims=True)
    h_norm[h_norm == 0.0] = 1.0
    momentum_history = (h_body / h_norm).T
    principal_moments, principal_axes = np.linalg.eigh(inertia_tensor)
    principal_moments = principal_moments[::-1]
    principal_axes = principal_axes[:, ::-1]

    u = np.linspace(-np.pi, np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    U, V = np.meshgrid(u, v)

    x = np.cos(U) * np.sin(V)
    y = np.sin(U) * np.sin(V)
    z = np.cos(V)

    fig = plt.figure(figsize=(9, 8), facecolor=FIGURE_FACE_COLOR)
    ax = plt.subplot(1, 1, 1, projection="3d")
    ax.set_facecolor(AXIS_FACE_COLOR)
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

    ax.set_title("Normalized Body Angular Momentum Sphere")
    ax.set_xlabel("Lx / ||L|| [-]")
    ax.set_ylabel("Ly / ||L|| [-]")
    ax.set_zlabel("Lz / ||L|| [-]")
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=8)
    fig.tight_layout()

    output_path = save_path
    if output_path is None:
        output_path = default_plot_dir(ctx.output_dir, ctx.config_path) / "momentum_sphere.png"

    save_figure(ctx.logger, fig, output_path, "Momentum sphere plot saved")

    if show:
        plt.show(block=True)

    return fig
