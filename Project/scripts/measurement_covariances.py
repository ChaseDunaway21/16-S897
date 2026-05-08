"""
Script helper for computing bearing measurement covariances
given a rough pointing error.
"""

import numpy as np


def camera_covariance(pointing_error_deg: float) -> np.ndarray:
    """
    Compute the covariance matrix for bearing measurements given a pointing error.
    We assume that the covariance is isotropic and that the variance is equal to the square of the pointing error.
    """
    pointing_error_rad = np.deg2rad(pointing_error_deg)
    variance = pointing_error_rad**2
    covariance_matrix = np.diag([variance, variance])
    return covariance_matrix


def main():
    camera_pointing_error_deg = 5.0  # degrees, this is a rough estimate of the pointing error for the bearing measurements
    covariance_matrix = camera_covariance(camera_pointing_error_deg)
    print(
        "Covariance matrix for bearing measurements given a pointing error of {} degrees:".format(
            camera_pointing_error_deg
        )
    )
    print(covariance_matrix)


if __name__ == "__main__":
    main()
