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
        latitude (degrees north)
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening
    reference_latitude: float, default 70.0
        reference latitude (true scale latitude)
    metric: str, default 'area'
        metric to calculate scaling factors

            - ``'distance'``: scale factors for distance
            - ``'area'``: scale factors for area

    Returns
    -------
    scale: np.ndarray
        scaling factors at input latitudes
    """
    assert metric.lower() in ["distance", "area"], "Unknown metric"
    # convert latitude from degrees to positive radians
    theta = np.radians(np.abs(lat))
    # convert reference latitude from degrees to positive radians
    theta_ref = np.radians(np.abs(reference_latitude))
    # square of the eccentricity of the ellipsoid
    # ecc2 = (1-b**2/a**2) = 2.0*flat - flat^2
    ecc2 = 2.0 * flat - flat**2
    # eccentricity of the ellipsoid
    ecc = np.sqrt(ecc2)
    # calculate ratio at input latitudes
    m = np.cos(theta) / np.sqrt(1.0 - ecc2 * np.sin(theta) ** 2)
    t = np.tan(np.pi / 4.0 - theta / 2.0) / (
        (1.0 - ecc * np.sin(theta)) / (1.0 + ecc * np.sin(theta))
    ) ** (ecc / 2.0)
    # calculate ratio at reference latitude
    mref = np.cos(theta_ref) / np.sqrt(1.0 - ecc2 * np.sin(theta_ref) ** 2)
    tref = np.tan(np.pi / 4.0 - theta_ref / 2.0) / (
        (1.0 - ecc * np.sin(theta_ref)) / (1.0 + ecc * np.sin(theta_ref))
    ) ** (ecc / 2.0)
    # distance scaling
    k = (mref / m) * (t / tref)
    kp = (
        0.5
        * mref
        * np.sqrt(((1.0 + ecc) ** (1.0 + ecc)) * ((1.0 - ecc) ** (1.0 - ecc)))
        / tref
    )
    if metric.lower() == "distance":
        # distance scaling
        scale = np.where(np.isclose(theta, np.pi / 2.0), 1.0 / kp, 1.0 / k)
    elif metric.lower() == "area":
        # area scaling
        scale = np.where(
            np.isclose(theta, np.pi / 2.0), 1.0 / (kp**2), 1.0 / (k**2)
        )
    return scale
