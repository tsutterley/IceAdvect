#!/usr/bin/env python
"""
spatial.py
Written by Tyler Sutterley (01/2026)

Spatial routines

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html

UPDATE HISTORY:
    Updated 04/2026: updated scale factors to add case where reference
        latitude is at the pole
    Written 01/2026
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "data_type",
    "scale_factors",
]


def data_type(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> str:
    """
    Determines input data type based on variable dimensions

    Parameters
    ----------
    x: np.ndarray
        x-dimension coordinates
    y: np.ndarray
        y-dimension coordinates
    t: np.ndarray
        time-dimension coordinates

    Returns
    -------
    string denoting input data type

        - ``'time series'``
        - ``'drift'``
        - ``'grid'``
    """
    xsize = np.size(x)
    ysize = np.size(y)
    tsize = np.size(t)
    if (xsize == 1) and (ysize == 1) and (tsize >= 1):
        return "time series"
    elif (xsize == ysize) & (xsize == tsize):
        return "drift"
    elif (np.ndim(x) > 1) & (xsize == ysize):
        return "grid"
    elif xsize != ysize:
        return "grid"
    else:
        raise ValueError("Unknown data type")


def scale_factors(
    lat: np.ndarray,
    flat: float = 1.0 / 298.257223563,
    reference_latitude: float = 70.0,
    metric: str = "area",
):
    """
    Calculates scaling factors to account for polar stereographic
    distortion including special case of at the exact pole
    :cite:p:`Snyder:1982gf`

    Parameters
    ----------
    lat: np.ndarray
        Latitude (degrees north)
    flat: float, default 1.0/298.257223563
        Ellipsoidal flattening
    reference_latitude: float, default 70.0
        Reference latitude (true scale latitude)
    metric: str, default 'area'
        Metric to calculate scaling factors

            - ``'distance'``: scale factors for distance
            - ``'area'``: scale factors for area

    Returns
    -------
    scale: np.ndarray
        Scaling factors at input latitudes
    """
    assert metric.lower() in ["distance", "area"], "Unknown metric"
    # power for scaling factors
    power = 1.0 if metric.lower() == "distance" else 2.0
    # convert latitude to positive radians
    phi = np.radians(np.abs(lat))
    # convert reference latitude to positive radians
    phi_ref = np.radians(np.abs(reference_latitude))
    # square of the eccentricity of the ellipsoid
    # ecc2 = (1-b**2/a**2) = 2.0*flat - flat^2
    ecc2 = 2.0 * flat - flat**2
    # eccentricity of the ellipsoid
    ecc = np.sqrt(ecc2)
    # get p values following equations 17.33 and 17.35
    p = np.sqrt(np.power(1.0 + ecc, 1.0 + ecc) * np.power(1.0 - ecc, 1.0 - ecc))
    # calculate m factors using equation 12.15
    m = np.cos(phi) / np.sqrt(1.0 - ecc2 * np.sin(phi) ** 2)
    m_ref = np.cos(phi_ref) / np.sqrt(1.0 - ecc2 * np.sin(phi_ref) ** 2)
    # calculate t factors using equation 13.9
    t = np.tan(np.pi / 4.0 - phi / 2.0) / np.power(
        (1.0 - ecc * np.sin(phi)) / (1.0 + ecc * np.sin(phi)), ecc / 2.0
    )
    t_ref = np.tan(np.pi / 4.0 - phi_ref / 2.0) / np.power(
        (1.0 - ecc * np.sin(phi_ref)) / (1.0 + ecc * np.sin(phi_ref)), ecc / 2.0
    )
    # calculate scaling factors following Snyder (1982)
    # ignore divide by zero and invalid value warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        # check if reference latitude is at the pole
        if np.isclose(phi_ref, np.pi / 2.0):
            # equations 17.32 and 17.33
            k = 2.0 * t / (p * m)
            # at the pole (true scale)
            k_pole = 1.0
        else:
            # equations 17.32 and 17.34
            k = (m_ref / m) * (t / t_ref)
            # at the pole from equation 17.35
            k_pole = 0.5 * m_ref * p / t_ref
        # distance and area scaling factors with special case at the pole
        scale = np.where(
            np.isclose(phi, np.pi / 2.0),
            np.power(1.0 / k_pole, power),
            np.power(1.0 / k, power),
        )
    # return the scaling factors
    return scale
