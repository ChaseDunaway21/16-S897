"""
Author: Chase Dunaway

Helper math functions for simulation.
"""

from __future__ import annotations

import numpy as np

def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix for cross product."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])