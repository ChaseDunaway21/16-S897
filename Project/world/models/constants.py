"""
This file contains the constants used in the world model.

These constants are immutable, the config.yaml file contains varying parameters.

References:
-   National Geospatial-Intelligence Agency.
    Department of Defense World Geodetic System 1984
    NGA.STND.0036_1.0.0_WGS84, 8 July 2014.

-   Jet Propulsion Laboratory.
    “Astrodynamic Parameters.”
    Solar System Dynamics, NASA / California Institute of Technology.

-   Park, Ryan S., William M. Folkner, James G. Williams, and Dale H. Boggs.
    “The JPL Planetary and Lunar Ephemerides DE440 and DE441.”
    The Astronomical Journal, vol. 161, no. 3, 2021, article 105.
    https://doi.org/10.3847/1538-3881/abd414.

-   Prša, Andrej, et al.
    “Nominal Values for Selected Solar and Planetary Quantities: IAU 2015 Resolution B3.”
    The Astronomical Journal, vol. 152, no. 2, 2016, article 41.
    https://doi.org/10.3847/0004-6256/152/2/41.

-   United States Naval Observatory.
    “Terrestrial Time (TT).”
    Astronomical Applications Department.

-   Montenbruck, Oliver, and Eberhard Gill.
    Satellite Orbits: Models, Methods and Applications.
    Springer, 2000.
    https://doi.org/10.1007/978-3-642-58351-3.
"""

from datetime import datetime
import numpy as np

# Timing Constants
J2000_UTC = datetime(2000, 1, 1, 12, 0, 0)

# Gravity Constants
MU_EARTH = 3.986004418e14  # m^3/s^2
MU_MOON = 4.902801e12  # m^3/s^2
MU_SUN = 1.3271244004127942e20  # m^3/s^2

J2 = 1.0826261738522227e-3

# Earth Constants
RADIUS_EARTH = 6.378137e6  # m
EARTH_RADIUS_KM = RADIUS_EARTH / 1_000.0
EARTH_ROTATION_RATE = 7.2921150e-5  # rad/s
GMST_J2000 = 4.894961212823059  # rad
WGS84_FLATTENING = 1.0 / 298.257223563

# Solar Constants
ASTRONOMICAL_UNIT = 149_597_870_700.0  # m
RADIUS_SUN = 695_700_000.0  # m
SOLAR_CONSTANT_1_AU = 1361.0  # W/m^2
SPEED_OF_LIGHT = 299_792_458.0  # m/s

# Drag Constants
HP_ALTITUDES_KM = np.array(
    [
        100.0,
        120.0,
        130.0,
        140.0,
        150.0,
        160.0,
        170.0,
        180.0,
        190.0,
        200.0,
        210.0,
        220.0,
        230.0,
        240.0,
        250.0,
        260.0,
        270.0,
        280.0,
        290.0,
        300.0,
        320.0,
        340.0,
        360.0,
        380.0,
        400.0,
        420.0,
        440.0,
        460.0,
        480.0,
        500.0,
        520.0,
        540.0,
        560.0,
        580.0,
        600.0,
        620.0,
        640.0,
        660.0,
        680.0,
        700.0,
        720.0,
        740.0,
        760.0,
        780.0,
        800.0,
        840.0,
        880.0,
        920.0,
        960.0,
        1000.0,
    ],
    dtype=float,
)

HP_RHO_MIN = np.array(
    [
        4.974e05,
        2.490e04,
        8.377e03,
        3.899e03,
        2.122e03,
        1.263e03,
        8.008e02,
        5.283e02,
        3.617e02,
        2.557e02,
        1.839e02,
        1.341e02,
        9.949e01,
        7.488e01,
        5.709e01,
        4.403e01,
        3.430e01,
        2.697e01,
        2.139e01,
        1.708e01,
        1.099e01,
        7.214e00,
        4.824e00,
        3.274e00,
        2.249e00,
        1.558e00,
        1.091e00,
        7.701e-01,
        5.474e-01,
        3.916e-01,
        2.819e-01,
        2.042e-01,
        1.488e-01,
        1.092e-01,
        8.070e-02,
        6.012e-02,
        4.519e-02,
        3.430e-02,
        2.632e-02,
        2.043e-02,
        1.607e-02,
        1.281e-02,
        1.036e-02,
        8.496e-03,
        7.069e-03,
        4.680e-03,
        3.200e-03,
        2.210e-03,
        1.560e-03,
        1.150e-03,
    ],
    dtype=float,
)

HP_RHO_MAX = np.array(
    [
        4.974e05,
        2.490e04,
        8.710e03,
        4.059e03,
        2.215e03,
        1.344e03,
        8.758e02,
        6.010e02,
        4.297e02,
        3.162e02,
        2.396e02,
        1.853e02,
        1.455e02,
        1.157e02,
        9.308e01,
        7.555e01,
        6.182e01,
        5.095e01,
        4.226e01,
        3.526e01,
        2.511e01,
        1.819e01,
        1.337e01,
        9.955e00,
        7.492e00,
        5.684e00,
        4.355e00,
        3.362e00,
        2.612e00,
        2.042e00,
        1.605e00,
        1.267e00,
        1.005e00,
        7.997e-01,
        6.390e-01,
        5.123e-01,
        4.121e-01,
        3.325e-01,
        2.691e-01,
        2.185e-01,
        1.779e-01,
        1.452e-01,
        1.190e-01,
        9.776e-02,
        8.059e-02,
        5.741e-02,
        4.210e-02,
        3.130e-02,
        2.360e-02,
        1.810e-02,
    ],
    dtype=float,
)

HP_PARAMETER = 3.0
RA_LAG_RAD = 0.523599
