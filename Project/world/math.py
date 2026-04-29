"""
Author: Chase Dunaway

Helper math functions for simulation and visualization.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix for cross product."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def rotation_vector_exponential(rotation_vector: np.ndarray) -> np.ndarray:
    """Compute exp(v_hat) for a rotation vector."""
    return expm(skew_symmetric(rotation_vector))

def attitude_jacobian(q: np.ndarray) -> np.ndarray:
    """Return G(q) Attitude Jacobian."""
    q_scalar = q[0]
    q_vector = q[1:4]
    return np.vstack((
        -q_vector[np.newaxis, :],
        q_scalar * np.eye(3) + skew_symmetric(q_vector),
    ))

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion to a rotation matrix."""
    w, x, y, z = q
    return np.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                     [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
                     [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]])

def quaternion_from_two_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the quaternion that rotates from a to b."""
    a = np.asarray(source, dtype=float)
    b = np.asarray(target, dtype=float)
    # Unit vectors
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    # Get the angle between the vectors
    rotation_axis = np.cross(a, b)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    theta = np.arccos(dot)

    if dot > 1-10e-6:
        # Vectors are parallel so return identity and dont divide by zero
        return np.array([1.0, 0.0, 0.0, 0.0])

    elif dot < -1+10e-6:
        # The vectors are collinear but opposite directions, so I have to rotate it 
        # around an axis perpendicular to a. I cross a with two arbitrary vectors to make sure 
        # one of them is not already perpendicular to a.
        arbitrary_vector_1 = np.array([1.0, 0.0, 0.0])
        rotation_axis = np.cross(a, arbitrary_vector_1)
        if np.linalg.norm(rotation_axis) < 1e-6:
            arbitrary_vector_2 = np.array([0.0, 1.0, 0.0])
            rotation_axis = np.cross(a, arbitrary_vector_2)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        theta = np.pi

    q = np.hstack([np.cos(theta/2), np.sin(theta/2) * rotation_axis]) # This is covered in the lecture notes 2-3
    return q / np.linalg.norm(q)

def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion to roll, pitch, yaw."""
    w, x, y, z = q
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    s = 2*(w*y - z*x)
    s = np.clip(s, -1, 1)
    pitch = np.arcsin(s)
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    return np.array([roll, pitch, yaw])

def R_body_to_inertial(q: np.ndarray) -> np.ndarray:
    """Convert a body-frame vector to inertial frame using the quaternion."""
    return quaternion_to_rotation_matrix(q)

def R_inertial_to_body(q: np.ndarray) -> np.ndarray:
    """Convert an inertial-frame vector to body frame using the quaternion."""
    return quaternion_to_rotation_matrix(q).T
