"""
Author: Chase Dunaway

Helper math functions for simulation and visualization.
"""

from __future__ import annotations

import numpy as np

def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix for cross product."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion to a rotation matrix."""
    w, x, y, z = q
    return np.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                     [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
                     [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]])

def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion to roll, pitch, yaw."""
    w, x, y, z = q
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    s = 2*(w*y - z*x)
    s = np.clip(s, -1, 1)
    pitch = np.arcsin(s)
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    return np.array([roll, pitch, yaw])