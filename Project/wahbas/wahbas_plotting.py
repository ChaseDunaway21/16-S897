"""Plotting helpers for Wahba Monte Carlo runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_wahba_monte_carlo(
    attitude_errors_deg: np.ndarray,
    vector_counts: np.ndarray,
    save_path: Path,
    show: bool,
):
    import matplotlib.pyplot as plt

    from visualization.common import FIGURE_FACE_COLOR, style_time_axis

    trials = np.arange(1, attitude_errors_deg.size + 1)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        facecolor=FIGURE_FACE_COLOR,
        sharex=False,
    )

    style_time_axis(axes[0])
    axes[0].plot(trials, attitude_errors_deg, color="#2563eb", linewidth=1.1)
    axes[0].scatter(trials, attitude_errors_deg, s=14, color="#1d4ed8", alpha=0.8)
    axes[0].axhline(
        np.median(attitude_errors_deg),
        color="#dc2626",
        linestyle="--",
        linewidth=1.0,
        label="median",
    )
    axes[0].set_title("Wahba Monte Carlo Attitude Error")
    axes[0].set_xlabel("trial")
    axes[0].set_ylabel("error [deg]")
    axes[0].legend(loc="upper right")

    style_time_axis(axes[1])
    bins = min(30, max(5, attitude_errors_deg.size // 5))
    axes[1].hist(
        attitude_errors_deg,
        bins=bins,
        color="#14b8a6",
        edgecolor="white",
        alpha=0.9,
    )
    axes[1].set_title("Error Distribution")
    axes[1].set_xlabel("error [deg]")
    axes[1].set_ylabel("trials")

    axis_text = (
        f"mean={np.mean(attitude_errors_deg):.4g} deg\n"
        f"median={np.median(attitude_errors_deg):.4g} deg\n"
        f"max={np.max(attitude_errors_deg):.4g} deg\n"
        f"vectors/trial={np.mean(vector_counts):.2f}"
    )
    axes[1].text(
        0.98,
        0.96,
        axis_text,
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return fig
