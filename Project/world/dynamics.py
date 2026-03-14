"""
Author: Chase Dunaway

Integrates the orbital dynamics of the ARGUS Satellite using RK4 and spherical acceleration.

OUTPUT:
    Time
    State
"""

from __future__ import annotations

import numpy as np

def gravitational_acceleration(position: np.ndarray, mu: float) -> np.ndarray:
    """Use a spherical acceleration model to compute the ODEs"""

def RK4(spacecraft: Spacecraft, state: np.ndarray, dt: float, mu: float, j2000: float) -> np.ndarray:
    """Integrate the state using the RK4 method"""

