"""
Helper math functions for simulation and visualization.
"""

from __future__ import annotations

import numpy as np


def unit_vector(v: np.ndarray) -> np.ndarray:
    vector = np.asarray(v, dtype=float)
    return vector / np.linalg.norm(vector)


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    """Return a copy of vectors with each row normalized."""
    array = np.asarray(vectors, dtype=float)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


def scalar_value(value: float) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def covariance_matrix(covariance: np.ndarray | None, size: int = 3) -> np.ndarray:
    if covariance is None:
        return np.zeros((size, size), dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim == 0:
        return float(cov) * np.eye(size)
    return cov


def matrix_from_config(value: object, shape: tuple[int, int]) -> np.ndarray:
    """Return a diagonal matrix from input vector."""
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim == 1:
        matrix = np.diag(matrix)
    return matrix.reshape(shape)


def add_noise(
    value: np.ndarray, covariance: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    return np.asarray(value, dtype=float) + rng.multivariate_normal(
        np.zeros(covariance.shape[0]), covariance
    )


def bearing_covariance(
    bearing: np.ndarray,
    sigma_angle_rad: float,
) -> np.ndarray:
    """Return sigma^2 (I - b b.T) for a unit bearing vector."""
    b = unit_vector(bearing)
    return float(sigma_angle_rad) ** 2 * (np.eye(3) - np.outer(b, b))


def add_bearing_noise(
    bearing: np.ndarray,
    sigma_angle_rad: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample bearing noise in the tangent plane and renormalize."""
    b = unit_vector(bearing)
    noise_covariance = bearing_covariance(b, sigma_angle_rad)
    noisy = b + rng.multivariate_normal(np.zeros(3), noise_covariance)
    return unit_vector(noisy)


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix for cross product."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def unskew(S: np.ndarray) -> np.ndarray:
    """Return the vector of a skew-symmetric matrix."""
    S = np.asarray(S, dtype=float).reshape(3, 3)
    return 0.5 * np.array(
        [
            S[2, 1] - S[1, 2],
            S[0, 2] - S[2, 0],
            S[1, 0] - S[0, 1],
        ]
    )
