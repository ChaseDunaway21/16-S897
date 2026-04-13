"""Shared helpers for simulator plotting modules."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURE_FACE_COLOR = "#f7f8fa"
AXIS_FACE_COLOR = "#f2f4f8"


def default_plot_dir(output_dir: Path | None, config_path: Path) -> Path:
    return output_dir if output_dir is not None else (config_path.parent / "results")


def style_time_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(AXIS_FACE_COLOR)
    ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.7)


def set_equal_orbit_axes(ax: plt.Axes, pos_km: np.ndarray) -> None:
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


def save_figure(
    logger: logging.Logger,
    fig: plt.Figure,
    save_path: str | Path,
    log_message: str,
    dpi: int = 150,
) -> Path:
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info("%s: %s", log_message, output_path)
    return output_path
