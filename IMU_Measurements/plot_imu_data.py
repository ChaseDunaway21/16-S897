#!/usr/bin/env python3
"""
Plot gyroscope_data and magnetometer_data CSV files from the data directory.

By default this script auto-selects the newest files named:
  gyroscope_data_*.csv
  magnetometer_data_*.csv

It saves a combined time-series plot to plots/imu_sensor_plot.png.

The time axis defaults to 0.08 seconds per sample.
"""

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
except ModuleNotFoundError as error:
    venv_python = SCRIPT_DIR / ".venv" / "bin" / "python"
    if error.name == "matplotlib" and venv_python.exists() and not os.environ.get("ARGUS_IMU_PLOT_REEXECED"):
        os.environ["ARGUS_IMU_PLOT_REEXECED"] = "1"
        os.execv(str(venv_python), [str(venv_python), *sys.argv])
    raise SystemExit(
        "matplotlib is required to plot these CSVs. Try running:\n"
        "  .venv/bin/python plot_imu_data.py\n"
        "or install matplotlib for your current Python environment."
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
) -> tuple[list[float], list[list[float]]]:
    times_ns: list[float] = []
    axes = [[], [], []]

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
                times_ns.append(float(row["timestamp_ns"]))
            for index, column in enumerate(columns):
                axes[index].append(float(row[column]))

    if not axes[0]:
        raise ValueError(f"{path.name} does not contain any data rows")

    if use_timestamps:
        start_ns = times_ns[0]
        times_s = [(timestamp - start_ns) / 1e9 for timestamp in times_ns]
    else:
        times_s = [index * collection_period_s for index in range(len(axes[0]))]

    return times_s, axes


def add_axis_plot(ax, times_s: list[float], axes: list[list[float]], labels: tuple[str, str, str]) -> None:
    markers = ("o", "s", "^")
    for values, label, marker in zip(axes, labels, markers):
        ax.plot(times_s, values, label=label, linewidth=1.2, marker=marker, markersize=2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def plot_imu_data(
    gyro_path: Path,
    mag_path: Path,
    output_path: Path,
    show_plot: bool,
    collection_period_s: float,
    use_timestamps: bool,
) -> None:
    gyro_time_s, gyro_axes = read_sensor_csv(
        gyro_path,
        ("gyro_x", "gyro_y", "gyro_z"),
        collection_period_s,
        use_timestamps,
    )
    mag_time_s, mag_axes = read_sensor_csv(
        mag_path,
        ("mag_x", "mag_y", "mag_z"),
        collection_period_s,
        use_timestamps,
    )

    fig, (gyro_ax, mag_ax) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    fig.suptitle("IMU Sensor Data")

    add_axis_plot(gyro_ax, gyro_time_s, gyro_axes, ("gyro_x", "gyro_y", "gyro_z"))
    gyro_ax.set_title("Gyroscope")
    gyro_ax.set_xlabel("Time (s)")
    gyro_ax.set_ylabel("Angular velocity (rad/s)")

    add_axis_plot(mag_ax, mag_time_s, mag_axes, ("mag_x", "mag_y", "mag_z"))
    mag_ax.set_title("Magnetometer")
    mag_ax.set_xlabel("Time (s)")
    mag_ax.set_ylabel("Magnetic field (uT)")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    print(f"Gyroscope file: {gyro_path}")
    print(f"Magnetometer file: {mag_path}")
    if use_timestamps:
        print("Time axis: timestamp_ns column")
    else:
        print(f"Time axis: {collection_period_s:g} seconds/sample")
    print(f"Saved plot: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot gyroscope and magnetometer CSV data.")
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
        default=PLOTS_DIR / "imu_sensor_plot.png",
        help="Output PNG path. Defaults to plots/imu_sensor_plot.png.",
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
    parser.add_argument("--show", action="store_true", help="Display the plot window after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gyro_path = args.gyro if args.gyro else newest_matching_file("gyroscope_data_*.csv")
    mag_path = args.mag if args.mag else newest_matching_file("magnetometer_data_*.csv")
    output_path = args.output

    gyro_path = resolve_input_path(gyro_path)
    mag_path = resolve_input_path(mag_path)
    output_path = resolve_plot_output_path(output_path)

    if args.collection_frequency is not None:
        if args.collection_frequency <= 0:
            raise SystemExit("--collection-frequency must be greater than 0")
        collection_period_s = 1.0 / args.collection_frequency
    else:
        if args.collection_period <= 0:
            raise SystemExit("--collection-period must be greater than 0")
        collection_period_s = args.collection_period

    plot_imu_data(
        gyro_path,
        mag_path,
        output_path,
        args.show,
        collection_period_s,
        args.use_timestamps,
    )


if __name__ == "__main__":
    main()
