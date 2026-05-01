"""
This file contains the constants used in the world model.

These constants are immutable, the config.yaml file contains varying parameters.
"""

from datetime import datetime

# Timing Constants
J2000_UTC = datetime(2000, 1, 1, 12, 0, 0)

# Gravity Constants
MU_EARTH = 3.986004418e14  # m^3/s^2
MU_MOON = 4.902801e12  # m^3/s^2
MU_SUN = 1.32712440018e20  # m^3/s^2

J2 = 1.08262668e-3

# Physical Constants
RADIUS_EARTH = 6.378137e6  # m
EARTH_RADIUS_KM = RADIUS_EARTH / 1_000.0
EARTH_ROTATION_RATE = 7.2921150e-5  # rad/s
GMST_J2000 = 4.894961212823059  # rad
WGS84_FLATTENING = 1.0 / 298.257223563

# Solar Constants
ASTRONOMICAL_UNIT = 149_597_870_700.0  # m
RADIUS_SUN = 696_340_000.0  # m
SOLAR_CONSTANT_1_AU = 1367.0  # W/m^2
SPEED_OF_LIGHT = 299_792_458.0  # m/s
