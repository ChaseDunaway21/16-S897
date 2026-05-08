#!/usr/bin/env python3
"""
Compute and plot overlapping Allan variance for IMU CSV files with AllanTools.
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
    import allantools
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError as error:
    venv_python = SCRIPT_DIR / ".venv" / "bin" / "python"
    if venv_python.exists() and not os.environ.get("ARGUS_ALLAN_REEXECED"):
        os.environ["ARGUS_ALLAN_REEXECED"] = "1"
        os.execv(str(venv_python), [str(venv_python), *sys.argv])
    raise SystemExit(
        "allantools, numpy, and matplotlib are required. Try running:\n"
        "  .venv/bin/python allan_variance.py"
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


def resolve_csv_output_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return SCRIPT_DIR / path
    return DATA_DIR / path


def read_sensor_csv(
    path: Path,
    columns: tuple[str, str, str],
    collection_period_s: float,
    use_timestamps: bool,
) -> tuple[float, np.ndarray]:
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

    if len(rows) < 3:
        raise ValueError(f"{path.name} needs at least 3 rows for Allan variance")

    data = np.asarray(rows, dtype=float)
    if use_timestamps:
        timestamps = np.asarray(timestamps_ns, dtype=float)
        dt_s = float(np.median(np.diff(timestamps)) / 1e9)
    else:
        dt_s = collection_period_s

    if dt_s <= 0:
        raise ValueError(f"{path.name} has invalid sample timing")

    return dt_s, data


def timing_from_args(args: argparse.Namespace) -> float:
    if args.collection_frequency is not None:
        if args.collection_frequency <= 0:
            raise SystemExit("--collection-frequency must be greater than 0")
        return 1.0 / args.collection_frequency

    if args.collection_period <= 0:
        raise SystemExit("--collection-period must be greater than 0")
    return args.collection_period


def cluster_sizes(sample_count: int, points_per_decade: int, max_tau_s: float | None, dt_s: float) -> np.ndarray:
    max_cluster = (sample_count - 1) // 2
    if max_tau_s is not None:
        max_cluster = min(max_cluster, int(max_tau_s / dt_s))

    if max_cluster < 1:
        raise ValueError("Not enough data for the requested Allan variance range")

    decade_count = np.log10(max_cluster) if max_cluster > 1 else 1.0
    point_count = max(2, int(np.ceil(decade_count * points_per_decade)) + 1)
    return np.unique(np.logspace(0, np.log10(max_cluster), point_count).astype(int))


def compute_allan_table(
    sensor_name: str,
    labels: tuple[str, str, str],
    dt_s: float,
    data: np.ndarray,
    points_per_decade: int,
    max_tau_s: float | None,
) -> list[dict[str, float | int | str]]:
    sizes = cluster_sizes(len(data), points_per_decade, max_tau_s, dt_s)
    taus_s = sizes * dt_s
    sample_rate_hz = 1.0 / dt_s
    rows: list[dict[str, float | int | str]] = []

    for axis_index, label in enumerate(labels):
        taus_out, deviations, errors, pair_counts = allantools.oadev(
            data[:, axis_index],
            rate=sample_rate_hz,
            data_type="freq",
            taus=taus_s,
        )
        variances = deviations * deviations
        cluster_sizes_out = np.rint(taus_out / dt_s).astype(int)

        for tau_s, cluster_size, variance, deviation, error, pair_count in zip(
            taus_out,
            cluster_sizes_out,
            variances,
            deviations,
            errors,
            pair_counts,
        ):
            rows.append(
                {
                    "sensor": sensor_name,
                    "axis": label,
                    "tau_s": float(tau_s),
                    "cluster_size": int(cluster_size),
                    "allan_variance": float(variance),
                    "allan_deviation": float(deviation),
                    "allan_deviation_error": float(error),
                    "pair_count": int(pair_count),
                }
            )

    return rows


def write_results_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fieldnames = [
        "sensor",
        "axis",
        "tau_s",
        "cluster_size",
        "allan_variance",
        "allan_deviation",
        "allan_deviation_error",
        "pair_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_allan_results(
    rows: list[dict[str, float | int | str]],
    output_path: Path,
    plot_kind: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    sensors = [("gyroscope", axes[0]), ("magnetometer", axes[1])]
    value_key = "allan_variance" if plot_kind == "variance" else "allan_deviation"
    y_label = "Allan variance" if plot_kind == "variance" else "Allan deviation"

    for sensor_name, ax in sensors:
        axes_for_sensor = sorted({row["axis"] for row in rows if row["sensor"] == sensor_name})
        for axis_name in axes_for_sensor:
            axis_rows = [
                row
                for row in rows
                if row["sensor"] == sensor_name and row["axis"] == axis_name
            ]
            tau_s = [float(row["tau_s"]) for row in axis_rows]
            values = [float(row[value_key]) for row in axis_rows]
            ax.loglog(tau_s, values, marker="o", markersize=3, linewidth=1.2, label=str(axis_name))

        ax.set_title(sensor_name.title())
        ax.set_ylabel(y_label)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")

    axes[1].set_xlabel("Averaging time tau (s)")
    fig.suptitle(f"IMU {y_label}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and plot overlapping Allan variance with AllanTools.")
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
        default=PLOTS_DIR / "imu_allan_deviation.png",
        help="Output plot path. Default: plots/imu_allan_deviation.png.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DATA_DIR / "imu_allan_variance.csv",
        help="Output CSV path. Default: data/imu_allan_variance.csv.",
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
        help="Use the median timestamp_ns spacing instead of a fixed collection period.",
    )
    parser.add_argument(
        "--points-per-decade",
        type=int,
        default=10,
        help="Number of tau samples per decade. Default: 10.",
    )
    parser.add_argument(
        "--max-tau",
        type=float,
        default=None,
        help="Maximum averaging time tau in seconds.",
    )
    parser.add_argument(
        "--plot",
        choices=("deviation", "variance"),
        default="deviation",
        help="Plot Allan deviation or Allan variance. Default: deviation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collection_period_s = timing_from_args(args)

    if args.points_per_decade <= 0:
        raise SystemExit("--points-per-decade must be greater than 0")
    if args.max_tau is not None and args.max_tau <= 0:
        raise SystemExit("--max-tau must be greater than 0")

    gyro_path = resolve_input_path(args.gyro) if args.gyro else newest_matching_file("gyroscope_data_*.csv")
    mag_path = resolve_input_path(args.mag) if args.mag else newest_matching_file("magnetometer_data_*.csv")
    output_path = resolve_plot_output_path(args.output)
    csv_output_path = resolve_csv_output_path(args.csv_output)

    gyro_dt_s, gyro_data = read_sensor_csv(
        gyro_path,
        ("gyro_x", "gyro_y", "gyro_z"),
        collection_period_s,
        args.use_timestamps,
    )
    mag_dt_s, mag_data = read_sensor_csv(
        mag_path,
        ("mag_x", "mag_y", "mag_z"),
        collection_period_s,
        args.use_timestamps,
    )

    rows = []
    rows.extend(
        compute_allan_table(
            "gyroscope",
            ("gyro_x", "gyro_y", "gyro_z"),
            gyro_dt_s,
            gyro_data,
            args.points_per_decade,
            args.max_tau,
        )
    )
    rows.extend(
        compute_allan_table(
            "magnetometer",
            ("mag_x", "mag_y", "mag_z"),
            mag_dt_s,
            mag_data,
            args.points_per_decade,
            args.max_tau,
        )
    )

    write_results_csv(csv_output_path, rows)
    plot_allan_results(rows, output_path, args.plot)

    print(f"Gyroscope file: {gyro_path}")
    print(f"Magnetometer file: {mag_path}")
    if args.use_timestamps:
        print(f"Gyroscope median dt: {gyro_dt_s:g}s")
        print(f"Magnetometer median dt: {mag_dt_s:g}s")
    else:
        print(f"Time axis: {collection_period_s:g} seconds/sample")
    print(f"Saved Allan {args.plot} plot: {output_path}")
    print(f"Saved Allan variance CSV: {csv_output_path}")


if __name__ == "__main__":
    main()
