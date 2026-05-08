#!/usr/bin/env python3
"""Compute covariance matrices for gyroscope and magnetometer CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"


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


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    script_candidate = SCRIPT_DIR / path
    if script_candidate.exists():
        return script_candidate

    data_candidate = DATA_DIR / path
    if data_candidate.exists():
        return data_candidate

    return script_candidate


def read_axis_data(path: Path, columns: tuple[str, str, str]) -> list[list[float]]:
    rows: list[list[float]] = []

    with path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = set(columns) - set(reader.fieldnames or [])
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"{path.name} is missing required column(s): {missing_list}")

        for row in reader:
            rows.append([float(row[column]) for column in columns])

    if len(rows) < 2:
        raise ValueError(f"{path.name} needs at least 2 data rows for sample covariance")

    return rows


def covariance_matrix(rows: list[list[float]], sample: bool = True) -> tuple[list[float], list[list[float]]]:
    sample_count = len(rows)
    axis_count = len(rows[0])
    means = [sum(row[index] for row in rows) / sample_count for index in range(axis_count)]
    denominator = sample_count - 1 if sample else sample_count

    covariance = []
    for row_index in range(axis_count):
        covariance_row = []
        for column_index in range(axis_count):
            value = sum(
                (row[row_index] - means[row_index]) * (row[column_index] - means[column_index])
                for row in rows
            ) / denominator
            covariance_row.append(value)
        covariance.append(covariance_row)

    return means, covariance


def print_matrix(matrix: list[list[float]], precision: int) -> None:
    for row in matrix:
        formatted = ", ".join(f"{value:.{precision}g}" for value in row)
        print(f"  [{formatted}]")


def print_sensor_covariance(
    label: str,
    path: Path,
    columns: tuple[str, str, str],
    sample: bool,
    precision: int,
) -> None:
    rows = read_axis_data(path, columns)
    means, covariance = covariance_matrix(rows, sample=sample)

    print(f"{label}: {path}")
    print(f"Axis order: {', '.join(columns)}")
    print(f"Samples: {len(rows)}")
    print("Means:")
    print(f"  {', '.join(f'{value:.{precision}g}' for value in means)}")
    print("Covariance matrix:")
    print_matrix(covariance, precision)
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute IMU axis covariance matrices.")
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
        "--population",
        action="store_true",
        help="Use population covariance with denominator n. Default is sample covariance with denominator n - 1.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=12,
        help="Significant digits to print. Default: 12.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gyro_path = resolve_path(args.gyro) if args.gyro else newest_matching_file("gyroscope_data_*.csv")
    mag_path = resolve_path(args.mag) if args.mag else newest_matching_file("magnetometer_data_*.csv")
    sample = not args.population

    covariance_type = "sample" if sample else "population"
    print(f"Using {covariance_type} covariance")
    print()

    print_sensor_covariance("Gyroscope", gyro_path, ("gyro_x", "gyro_y", "gyro_z"), sample, args.precision)
    print_sensor_covariance("Magnetometer", mag_path, ("mag_x", "mag_y", "mag_z"), sample, args.precision)


if __name__ == "__main__":
    main()
