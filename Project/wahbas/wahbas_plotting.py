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
        1,
        2,
        figsize=(14, 4.8),
        facecolor=FIGURE_FACE_COLOR,
        gridspec_kw={"width_ratios": [2.2, 1.0]},
    )
    ax, hist_ax = axes

    style_time_axis(ax)
    ax.plot(trials, attitude_errors_deg, color="#2563eb", linewidth=1.1)
    ax.scatter(trials, attitude_errors_deg, s=14, color="#1d4ed8", alpha=0.8)
    ax.axhline(
        np.median(attitude_errors_deg),
        color="#dc2626",
        linestyle="--",
        linewidth=1.0,
        label="median",
    )
    ax.set_title("Wahba Monte Carlo Attitude Error")
    ax.set_xlabel("trial")
    ax.set_ylabel("error [deg]")
    ax.legend(loc="upper right")

    axis_text = (
        f"mean={np.mean(attitude_errors_deg):.4g} deg\n"
        f"median={np.median(attitude_errors_deg):.4g} deg\n"
        f"max={np.max(attitude_errors_deg):.4g} deg\n"
        f"vectors/trial={np.mean(vector_counts):.2f}"
    )
    ax.text(
        0.98,
        0.88,
        axis_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )

    style_time_axis(hist_ax)
    finite_errors = attitude_errors_deg[np.isfinite(attitude_errors_deg)]
    if finite_errors.size > 0:
        bins = min(30, max(8, int(np.sqrt(finite_errors.size))))
        hist_ax.hist(
            finite_errors,
            bins=bins,
            color="#93c5fd",
            edgecolor="#1d4ed8",
            linewidth=0.8,
            alpha=0.9,
        )
        hist_ax.axvline(
            np.mean(finite_errors),
            color="#f59e0b",
            linestyle="-",
            linewidth=1.4,
            label="mean",
        )
        hist_ax.axvline(
            np.median(finite_errors),
            color="#dc2626",
            linestyle="--",
            linewidth=1.2,
            label="median",
        )
        hist_ax.legend(loc="upper right")
    hist_ax.set_title("Error Distribution")
    hist_ax.set_xlabel("error [deg]")
    hist_ax.set_ylabel("count")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _rotation_matrix_to_euler_deg(rotation_matrix: np.ndarray) -> np.ndarray:
    R = np.asarray(rotation_matrix, dtype=float).reshape(3, 3)
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    cos_pitch = np.cos(pitch)

    if abs(cos_pitch) > 1e-12:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])

    return np.rad2deg(np.array([roll, pitch, yaw], dtype=float))


def plot_wahba_attitude_trials(
    results: list[dict[str, object]],
    save_path: Path,
    show: bool,
):
    import matplotlib.pyplot as plt

    from visualization.common import FIGURE_FACE_COLOR, style_time_axis

    trials = np.arange(1, len(results) + 1)
    true_euler = np.vstack(
        [_rotation_matrix_to_euler_deg(result["R_true"]) for result in results]
    )
    svd_euler = np.vstack(
        [_rotation_matrix_to_euler_deg(result["R_svd"]) for result in results]
    )
    sdp_euler = np.vstack(
        [_rotation_matrix_to_euler_deg(result["R_sdp"]) for result in results]
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 8),
        facecolor=FIGURE_FACE_COLOR,
        sharex=True,
    )
    component_labels = ("roll", "pitch", "yaw")

    for i, axis in enumerate(axes):
        style_time_axis(axis)
        axis.scatter(
            trials,
            true_euler[:, i],
            s=24,
            color="#dc2626",
            marker="o",
            label="true" if i == 0 else None,
            zorder=4,
        )
        axis.scatter(
            trials,
            svd_euler[:, i],
            s=62,
            facecolors="none",
            edgecolors="#2563eb",
            linewidths=1.3,
            marker="s",
            label="SVD" if i == 0 else None,
            zorder=5,
        )
        axis.scatter(
            trials,
            sdp_euler[:, i],
            s=86,
            facecolors="none",
            edgecolors="#16a34a",
            linewidths=1.3,
            marker="^",
            label="SDP" if i == 0 else None,
            zorder=6,
        )
        axis.set_ylabel(f"{component_labels[i]} [deg]")

    axes[0].set_title("Wahba Attitude Estimates by Trial")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("trial")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return fig
