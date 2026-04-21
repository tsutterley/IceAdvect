#!/usr/bin/env python
"""
interpolate.py
Written by Tyler Sutterley (04/2026)
Interpolators for spatial data

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    xarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

UPDATE HISTORY:
    Updated 04/2026: add 1st and 2nd order barycentric interpolation function
    Written 01/2026
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import scipy.fftpack
import scipy.spatial

__all__ = [
    "inpaint",
    "barycentric",
    "_to_barycentric",
    "_inside_triangle",
    "_shape_functions",
]


def inpaint(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    N: int = 0,
    s0: int = 3,
    power: int = 2,
    epsilon: float = 2.0,
    **kwargs,
):
    """
    Inpaint over missing data in a two-dimensional array using a
    penalized least-squares method based on discrete cosine transforms
    :cite:p:`Garcia:2010hn,Wang:2012ei`

    Parameters
    ----------
    xs: np.ndarray
        x-coordinates
    ys: np.ndarray
        y-coordinates
    zs: np.ndarray
        Data with masked values
    N: int, default 0
        Number of iterations (0 for nearest neighbors)
    s0: int, default 3
        Smoothing factor
    power: int, default 2
        Power for lambda function
    epsilon: float, default 2.0
        Relaxation factor

    Returns
    -------
    z0: np.ndarray
        Data with inpainted (filled) values
    """
    # find masked values
    if isinstance(zs, np.ma.MaskedArray):
        W = np.logical_not(zs.mask)
    else:
        W = np.isfinite(zs)
    # no valid values can be found
    if not np.any(W):
        raise ValueError("No valid values found")

    # dimensions of input grid
    ny, nx = np.shape(zs)

    # calculate initial values using nearest neighbors
    # computation of distance Matrix
    # use scipy spatial KDTree routines
    xgrid, ygrid = np.meshgrid(xs, ys)
    tree = scipy.spatial.cKDTree(np.c_[xgrid[W], ygrid[W]])
    # find nearest neighbors
    masked = np.logical_not(W)
    _, ii = tree.query(np.c_[xgrid[masked], ygrid[masked]], k=1)
    # copy valid original values
    z0 = np.zeros((ny, nx), dtype=zs.dtype)
    z0[W] = np.copy(zs[W])
    # copy nearest neighbors
    z0[masked] = zs[W][ii]
    # return nearest neighbors interpolation
    if N == 0:
        return z0

    # copy data to new array with 0 values for mask
    ZI = np.zeros((ny, nx), dtype=zs.dtype)
    ZI[W] = np.copy(z0[W])

    # calculate lambda function
    L = np.zeros((ny, nx))
    L += np.broadcast_to(np.cos(np.pi * np.arange(ny) / ny)[:, None], (ny, nx))
    L += np.broadcast_to(np.cos(np.pi * np.arange(nx) / nx)[None, :], (ny, nx))
    LAMBDA = np.power(2.0 * (2.0 - L), power)

    # smoothness parameters
    s = np.logspace(s0, -6, N)
    for i in range(N):
        # calculate discrete cosine transform
        GAMMA = 1.0 / (1.0 + s[i] * LAMBDA)
        DISCOS = GAMMA * scipy.fftpack.dctn(W * (ZI - z0) + z0, norm="ortho")
        # update interpolated grid
        z0 = (
            epsilon * scipy.fftpack.idctn(DISCOS, norm="ortho")
            + (1.0 - epsilon) * z0
        )

    # reset original values
    z0[W] = np.copy(zs[W])
    # return the inpainted grid
    return z0


def barycentric(
    xv: np.ndarray,
    yv: np.ndarray,
    ze: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    order: int = 1,
    **kwargs,
):
    """
    Interpolation of unstructured model data using a barycentric
    method

    Parameters
    ----------
    xv: np.ndarray
        x-coordinates of triangle vertices
    yv: np.ndarray
        y-coordinates of triangle vertices
    ze: np.ndarray
        Unstructured model data at elements
    x: np.ndarray
        Output x-coordinates
    y: np.ndarray
        Output y-coordinates
    order: int, default 1
        Polynomial order of the triangular elements

        - ``1``: linear
        - ``2``: quadratic

    Returns
    -------
    data: xr.DataArray
        Interpolated data
    """
    # set default data type
    dtype = kwargs.get("dtype", ze.dtype)
    # convert to barycentric coordinates
    xi, eta = _to_barycentric(xv, yv, x, y)
    # check if inside polygon
    valid = _inside_triangle(xi, eta)
    # allocate to output extrapolate data array
    data = np.zeros_like(x, dtype=dtype)
    if not np.any(valid):
        return xr.DataArray(data)
    # get shape functions for order
    N = _shape_functions(xi, eta, order)
    # calculate interpolation
    for p, sf in enumerate(N):
        data += sf * valid * ze[..., p]
    # return the interpolated value
    return xr.DataArray(data)


def _to_barycentric(
    xv: np.ndarray,
    yv: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
):
    """
    Convert coordinates to barycentric space

    Parameters
    ----------
    xv: np.ndarray
        x-coordinates of triangle vertices
    yv: np.ndarray
        y-coordinates of triangle vertices
    x: np.ndarray
        Output x-coordinates
    y: np.ndarray
        Output y-coordinates

    Returns
    -------
    xi: np.ndarray
        Normalized barycentric (areal) xi-coordinates
    eta: np.ndarray
        Normalized barycentric (areal) eta-coordinates
    """
    # calculate triangle area
    A = 0.5 * (
        xv[0] * (yv[1] - yv[2])
        + xv[1] * (yv[2] - yv[0])
        + xv[2] * (yv[0] - yv[1])
    )
    # calculate Jacobian
    # ignore divide by zero and invalid value warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        J = 1.0 / (2.0 * A)
    # mapping into barycentric coordinates
    xi = J * (
        (xv[1] * yv[2] - xv[2] * yv[1])
        + (yv[1] - yv[2]) * x
        + (xv[2] - xv[1]) * y
    )
    eta = J * (
        (xv[2] * yv[0] - xv[0] * yv[2])
        + (yv[2] - yv[0]) * x
        + (xv[0] - xv[2]) * y
    )
    # return the barycentric coordinates
    return xi, eta


def _inside_triangle(
    xi: np.ndarray,
    eta: np.ndarray,
    atol: float = 1e-8,
):
    """
    Check if point is within the triangular area

    Parameters
    ----------
    xi: np.ndarray
        Normalized barycentric (areal) xi-coordinates
    eta: np.ndarray
        Normalized barycentric (areal) eta-coordinates
    atol: float = 1e-8
        Absolute tolerance parameter

    Returns
    -------
    valid: np.ndarray
        Mask for coordinates
    """
    # simple check to see if areas are valid
    la = 1.0 - eta - xi
    # all barycentric coordinates should be within 0 to 1
    # and have valid Jacobians (not dividing by 0)
    valid = (
        (np.isfinite(xi) & np.isfinite(eta))
        & (la >= (0.0 - atol))
        & (la <= (1.0 + atol))
        & (xi >= (0.0 - atol))
        & (xi <= (1.0 + atol))
        & (eta >= (0.0 - atol))
        & (eta <= (1.0 + atol))
    )
    return valid


def _shape_functions(xi: np.ndarray, eta: np.ndarray, order: int):
    """
    Get the interpolating shape functions for a polynomial order

    Parameters
    ----------
    xi: np.ndarray
        Normalized barycentric (areal) xi-coordinates
    eta: np.ndarray
        Normalized barycentric (areal) eta-coordinates
    order: int
        Polynomial order of the triangular elements

        - ``1``: linear
        - ``2``: quadratic

    Returns
    -------
    N: list
        Shape functions in barycentric space
    """
    # shape functions in barycentric space
    N = [None] * (3 * order)
    if order == 1:
        # 1st order terms: linear triangular elements
        N[0] = xi
        N[1] = eta
        N[2] = 1.0 - eta - xi
    elif order == 2:
        # 2nd order terms: quadratic triangular elements
        N[0] = xi * (2.0 * xi - 1.0)
        N[1] = 4.0 * xi * eta
        N[2] = eta * (2.0 * eta - 1.0)
        N[3] = 4.0 * eta * (1.0 - xi - eta)
        N[4] = (1.0 - xi - eta) * (1.0 - 2.0 * xi - 2.0 * eta)
        N[5] = 4.0 * xi * (1.0 - xi - eta)
    else:
        raise ValueError(f"Unsupported polynomial order {order}")
    # return the shape functions
    return N
