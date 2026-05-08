#!/usr/bin/env python3
"""Plot smoothed IMU drift and drift rate for gyroscope and magnetometer CSVs."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
PLOTS_DIR = SCRIPT_DIR / "plots"
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "argus_matplotlib_cache"))

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError as error:
    venv_python = SCRIPT_DIR / ".venv" / "bin" / "python"
    if venv_python.exists() and not os.environ.get("ARGUS_DRIFT_PLOT_REEXECED"):
        os.environ["ARGUS_DRIFT_PLOT_REEXECED"] = "1"
        os.execv(str(venv_python), [str(venv_python), *sys.argv])
    raise SystemExit(
        "numpy and matplotlib are required. Try running:\n"
        "  .venv/bin/python plot_imu_drift_rate.py"
    ) from error


def newest_matching_file(pattern: str) -> Path:
    matches = [
        path
        for base_dir in (DATA_DIR, SCRIPT_DIR)
        for path in base_dir.glob(pattern)
        if not path.stem.endswith("_old")
    ]
    matches.sort(key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No files found matching {pattern!r} in {DATA_DIR} or {SCRIPT_DIR}")
    return matches[-1]


def resolve_input_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    script_candidate = SCRIPT_DIR / path
    if script_candidate.exists():
        return script_candidate

    data_candidate = DATA_DIR / path
    if data_candidate.exists():
        return data_candidate

    return script_candidate


def resolve_plot_output_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return SCRIPT_DIR / path
    return PLOTS_DIR / path


def read_sensor_csv(
    path: Path,
    columns: tuple[str, str, str],
    collection_period_s: float,
    use_timestamps: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    timestamps_ns: list[float] = []

    with path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = set(columns)
        if use_timestamps:
            required_columns.add("timestamp_ns")

        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"{path.name} is missing required column(s): {missing_list}")

        for row in reader:
            if use_timestamps:
                timestamps_ns.append(float(row["timestamp_ns"]))
            rows.append([float(row[column]) for column in columns])

    if len(rows) < 2:
        raise ValueError(f"{path.name} needs at least 2 rows")

    data = np.asarray(rows, dtype=float)
    if use_timestamps:
        timestamps = np.asarray(timestamps_ns, dtype=float)
        time_s = (timestamps - timestamps[0]) / 1e9
    else:
        time_s = np.arange(len(data), dtype=float) * collection_period_s

    return time_s, data


def timing_from_args(args: argparse.Namespace) -> float:
    if args.collection_frequency is not None:
        if args.collection_frequency <= 0:
            raise SystemExit("--collection-frequency must be greater than 0")
        return 1.0 / args.collection_frequency

    if args.collection_period <= 0:
        raise SystemExit("--collection-period must be greater than 0")
    return args.collection_period


def rolling_mean(values: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 1:
        return values.copy()

    window_samples = min(window_samples, len(values))
    left_pad = window_samples // 2
    right_pad = window_samples - 1 - left_pad
    padded = np.pad(values, (left_pad, right_pad), mode="edge")
    cumulative_sum = np.concatenate(([0.0], np.cumsum(padded)))
    return (cumulative_sum[window_samples:] - cumulative_sum[:-window_samples]) / window_samples


def compute_drift_and_rate(
    time_s: np.ndarray,
    data: np.ndarray,
    window_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smoothed = np.column_stack(
        [rolling_mean(data[:, axis_index], window_samples) for axis_index in range(data.shape[1])]
    )
    baseline = smoothed[0]
    drift = smoothed - baseline
    drift_rate = np.column_stack(
        [np.gradient(smoothed[:, axis_index], time_s) for axis_index in range(data.shape[1])]
    )
    linear_rate = np.asarray(
        [np.polyfit(time_s, smoothed[:, axis_index], 1)[0] for axis_index in range(data.shape[1])]
    )
    return drift, drift_rate, linear_rate


def plot_axes(ax, time_s: np.ndarray, data: np.ndarray, labels: tuple[str, str, str]) -> None:
    for axis_index, label in enumerate(labels):
        ax.plot(time_s, data[:, axis_index], linewidth=1.0, label=label)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def plot_drift_rate(
    gyro_path: Path,
    mag_path: Path,
    output_path: Path,
    collection_period_s: float,
    use_timestamps: bool,
    window_s: float,
    show_plot: bool,
) -> None:
    gyro_time_s, gyro_data = read_sensor_csv(
        gyro_path,
        ("gyro_x", "gyro_y", "gyro_z"),
        collection_period_s,
        use_timestamps,
    )
    mag_time_s, mag_data = read_sensor_csv(
        mag_path,
        ("mag_x", "mag_y", "mag_z"),
        collection_period_s,
        use_timestamps,
    )

    gyro_window = max(1, int(round(window_s / np.median(np.diff(gyro_time_s)))))
    mag_window = max(1, int(round(window_s / np.median(np.diff(mag_time_s)))))

    gyro_drift, gyro_rate, gyro_linear_rate = compute_drift_and_rate(gyro_time_s, gyro_data, gyro_window)
    mag_drift, mag_rate, mag_linear_rate = compute_drift_and_rate(mag_time_s, mag_data, mag_window)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex="col")
    fig.suptitle(f"IMU Drift and Drift Rate, {window_s:g}s smoothing window")

    plot_axes(axes[0, 0], gyro_time_s, gyro_drift, ("gyro_x", "gyro_y", "gyro_z"))
    axes[0, 0].set_title("Gyroscope Drift")
    axes[0, 0].set_ylabel("Drift (rad/s)")

    plot_axes(axes[1, 0], gyro_time_s, gyro_rate, ("gyro_x", "gyro_y", "gyro_z"))
    axes[1, 0].set_title("Gyroscope Drift Rate")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Drift rate (rad/s^2)")

    plot_axes(axes[0, 1], mag_time_s, mag_drift, ("mag_x", "mag_y", "mag_z"))
    axes[0, 1].set_title("Magnetometer Drift")
    axes[0, 1].set_ylabel("Drift (uT)")

    plot_axes(axes[1, 1], mag_time_s, mag_rate, ("mag_x", "mag_y", "mag_z"))
    axes[1, 1].set_title("Magnetometer Drift Rate")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Drift rate (uT/s)")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    print(f"Gyroscope file: {gyro_path}")
    print(f"Magnetometer file: {mag_path}")
    if use_timestamps:
        print("Time axis: timestamp_ns column")
    else:
        print(f"Time axis: {collection_period_s:g} seconds/sample")
    print(f"Smoothing window: {window_s:g}s")
    print(f"Saved plot: {output_path}")
    print("Estimated linear drift rates:")
    print(f"  gyro_x, gyro_y, gyro_z: {', '.join(f'{value:.6g}' for value in gyro_linear_rate)} rad/s^2")
    print(f"  mag_x, mag_y, mag_z: {', '.join(f'{value:.6g}' for value in mag_linear_rate)} uT/s")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot smoothed IMU drift and drift rate.")
    parser.add_argument(
        "--gyro",
        type=Path,
        default=None,
        help="Path to gyroscope CSV. Defaults to newest data/gyroscope_data_*.csv.",
    )
    parser.add_argument(
        "--mag",
        type=Path,
        default=None,
        help="Path to magnetometer CSV. Defaults to newest data/magnetometer_data_*.csv.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=PLOTS_DIR / "imu_drift_rate_plot.png",
        help="Output PNG path. Default: plots/imu_drift_rate_plot.png.",
    )
    timing_group = parser.add_mutually_exclusive_group()
    timing_group.add_argument(
        "--collection-period",
        "--sample-period",
        type=float,
        default=0.08,
        help="Seconds between samples. Default: 0.08.",
    )
    timing_group.add_argument(
        "--collection-frequency",
        "--sample-rate",
        type=float,
        default=None,
        help="Samples per second in Hz. For example, 12.5 is equivalent to 0.08 seconds/sample.",
    )
    parser.add_argument(
        "--use-timestamps",
        action="store_true",
        help="Use the timestamp_ns column for the time axis instead of a fixed collection period.",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=60.0,
        help="Smoothing window in seconds before differentiating. Default: 60.",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot window after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collection_period_s = timing_from_args(args)

    if args.window <= 0:
        raise SystemExit("--window must be greater than 0")

    gyro_path = resolve_input_path(args.gyro) if args.gyro else newest_matching_file("gyroscope_data_*.csv")
    mag_path = resolve_input_path(args.mag) if args.mag else newest_matching_file("magnetometer_data_*.csv")
    output_path = resolve_plot_output_path(args.output)

    plot_drift_rate(
        gyro_path,
        mag_path,
        output_path,
        collection_period_s,
        args.use_timestamps,
        args.window,
        args.show,
    )


if __name__ == "__main__":
    main()
