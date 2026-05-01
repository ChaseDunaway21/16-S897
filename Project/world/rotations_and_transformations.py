"""Rotation and frame transformation helpers.

References:
[1] F. L. Markley and J. L. Crassidis, Fundamentals of Spacecraft Attitude
    Determination and Control, Springer, 2014.
[2] SciPy scipy.linalg.expm documentation.
[3] NAIF CSPICE recgeo documentation.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

try:
    import spiceypy as spice
except ImportError:
    spice = None

from world.math import skew_symmetric
from world.models.constants import RADIUS_EARTH, WGS84_FLATTENING


def R_z(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def enu_to_ecef(vector_enu: np.ndarray, lon_rad: float, lat_rad: float) -> np.ndarray:
    east = np.array([-np.sin(lon_rad), np.cos(lon_rad), 0.0])
    north = np.array(
        [
            -np.sin(lat_rad) * np.cos(lon_rad),
            -np.sin(lat_rad) * np.sin(lon_rad),
            np.cos(lat_rad),
        ]
    )
    up = np.array(
        [
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ]
    )
    return vector_enu[0] * east + vector_enu[1] * north + vector_enu[2] * up


def geodetic_from_ecef(position_ecef_m: np.ndarray) -> tuple[float, float, float]:
    if spice is not None:
        lon, lat, alt = spice.recgeo(
            np.asarray(position_ecef_m, dtype=float), RADIUS_EARTH, WGS84_FLATTENING
        )  # [3]
        return np.rad2deg(lon), np.rad2deg(lat), alt / 1000.0

    a = RADIUS_EARTH
    e2 = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)
    x, y, z = position_ecef_m
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - e2))

    for _ in range(5):
        sin_lat = np.sin(lat)
        N = a / np.sqrt(1.0 - e2 * sin_lat**2)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1.0 - e2 * N / (N + alt)))

    sin_lat = np.sin(lat)
    N = a / np.sqrt(1.0 - e2 * sin_lat**2)
    alt = p / np.cos(lat) - N
    return np.rad2deg(lon), np.rad2deg(lat), alt / 1000.0


def rotation_vector_exponential(rotation_vector: np.ndarray) -> np.ndarray:
    """Compute exp(v_hat) for a rotation vector."""
    return expm(skew_symmetric(rotation_vector))  # [2]


def axis_angle_to_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Convert an axis-angle vector into its respective rotation matrix."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    return (
        c * np.eye(3) + (1.0 - c) * np.outer(axis, axis) + s * skew_symmetric(axis)
    )  # [1]


def attitude_jacobian(q: np.ndarray) -> np.ndarray:
    """Return G(q) Attitude Jacobian."""
    q_scalar = q[0]
    q_vector = q[1:4]
    return np.vstack(
        (
            -q_vector[np.newaxis, :],
            q_scalar * np.eye(3) + skew_symmetric(q_vector),
        )
    )


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion to a rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
        ]
    )


def quaternion_from_two_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the quaternion that rotates from a to b."""
    a = np.asarray(source, dtype=float)
    b = np.asarray(target, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    rotation_axis = np.cross(a, b)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    theta = np.arccos(dot)

    if dot > 1 - 10e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])

    if dot < -1 + 10e-6:
        rotation_axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(rotation_axis) < 1e-6:
            rotation_axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        theta = np.pi

    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    q = np.hstack([np.cos(theta / 2), np.sin(theta / 2) * rotation_axis])  # [1]
    return q / np.linalg.norm(q)


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion to roll, pitch, yaw."""
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    s = 2 * (w * y - z * x)
    s = np.clip(s, -1, 1)
    pitch = np.arcsin(s)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return np.array([roll, pitch, yaw])


def R_body_to_inertial(q: np.ndarray) -> np.ndarray:
    """Convert a body-frame vector to inertial frame using the quaternion."""
    return quaternion_to_rotation_matrix(q)


def R_inertial_to_body(q: np.ndarray) -> np.ndarray:
    """Convert an inertial-frame vector to body frame using the quaternion."""
    return quaternion_to_rotation_matrix(q).T


def inertial_to_body(q: np.ndarray, vector_eci: np.ndarray) -> np.ndarray:
    return R_inertial_to_body(q) @ np.asarray(vector_eci, dtype=float)
