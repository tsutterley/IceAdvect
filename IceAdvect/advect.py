#!/usr/bin/env python3
"""
advect.py
Written by Tyler Sutterley (02/2026)
Routines for advecting ice parcels using velocity estimates

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    xarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

UPDATE HISTORY:
    Updated 02/2026: adjust logging levels for advection steps
    Written 01/2026
"""

from __future__ import annotations

import copy
import logging
import numpy as np
import xarray as xr
import timescale.time


# default epoch for time conversions
__epoch__ = timescale.time._j2000_epoch


class Advect:
    """
    Data class for advecting ice parcels using velocity estimates

    Attributes
    ----------
    ds: xarray.DataTree
        xarray DataTree of velocity data
    x: np.ndarray
        x-coordinates
    y: np.ndarray
        y-coordinates
    t: np.ndarray
        time coordinates
    x0: np.ndarray or NoneType, default None
        Final x-coordinate after advection
    y0: np.ndarray or NoneType, default None
        Final y-coordinate after advection
    t0: np.ndarray or float, default 0.0
        Ending time for advection
    ds: xarray.Dataset
        xarray Dataset of velocity data
    integrator: str
        Advection function
            - ``'euler'``
            - ``'RK4'``
            - ``'RKF45'``
    method: str, default 'linear'
        Interpolation method for velocities
            - ``'linear'``, ``'nearest'``: xarray interpolation method
    time_units: str, default 'seconds'
        Units for input time coordinates
    fill_value: float or NoneType, default np.nan
        invalid value for output data
    """

    np.seterr(invalid="ignore")

    def __init__(self, ds, **kwargs):
        # set default keyword arguments
        kwargs.setdefault("x", None)
        kwargs.setdefault("y", None)
        kwargs.setdefault("t", None)
        kwargs.setdefault("t0", 0.0)
        kwargs.setdefault("integrator", "RK4")
        kwargs.setdefault("method", "linear")
        kwargs.setdefault("time_units", "seconds since 2018-01-01T00:00:00")
        # parse time units
        epoch, to_sec = timescale.time.parse_date_string(kwargs["time_units"])
        # set default class attributes
        self.x = copy.deepcopy(kwargs["x"])
        self.y = copy.deepcopy(kwargs["y"])
        # convert times to deltatime in seconds since J2000
        self._time = timescale.from_deltatime(
            to_sec * np.array(kwargs["t"], dtype="f8"), epoch=epoch
        )
        self.t = self._time.to_deltatime(epoch=__epoch__, scale=86400.0)
        self._time0 = timescale.from_deltatime(
            to_sec * np.array(kwargs["t0"], dtype="f8"), epoch=epoch
        )
        self.t0 = self._time0.to_deltatime(epoch=__epoch__, scale=86400.0)
        self.velocity = ds
        self.integrator = copy.deepcopy(kwargs["integrator"])
        self.method = copy.deepcopy(kwargs["method"])

    def run(self, **kwargs):
        """
        Runs the advection of parcels using specified parameters

        Returns
        -------
        x0: np.ndarray
            Final x-coordinate after advection
        y0: np.ndarray
            Final y-coordinate after advection
        """
        # advect the parcels
        self.translate(**kwargs)
        # return final coordinates
        return self.x0, self.y0

    def interp(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: float | np.ndarray = 0.0,
        **kwargs,
    ):
        """
        Interpolates velocity data to specified coordinates

        Parameters
        ----------
        x: np.ndarray
            x-coordinates
        y: np.ndarray
            y-coordinates
        t: float or np.ndarray, default 0.0
            time coordinates
        kwargs: dict
            keyword arguments for xarray interpolation
        """
        # create xarray Dataset for interpolated velocities
        ds = xr.Dataset()
        # interpolate to specified coordinates
        if "t" in self.velocity:
            # clip time to within range of velocity dataset
            clipped = np.clip(
                copy.deepcopy(t),
                np.min(self.velocity.t.values),
                np.max(self.velocity.t.values),
            )
            # interpolate to x, y, and t coordinates
            for v in ("U", "V"):
                ds[v] = self.velocity[v].interp(x=x, y=y, t=clipped, **kwargs)
        else:
            # interpolate to x and y coordinates
            for v in ("U", "V"):
                ds[v] = self.velocity[v].interp(x=x, y=y, **kwargs)
        # return the dataset
        return ds

    # PURPOSE: translate a parcel between two times using an advection function
    def translate(self, **kwargs):
        """
        Translates a parcel using an advection function

        Parameters
        ----------
        integrator: str
            Advection function

                - ``'euler'``
                - ``'RK4'``
                - ``'RKF45'``
        method: str
            Interpolation method from xarray

                - ``'linear'``: linear interpolation for regular grids
                - ``'nearest'``: nearest-neighbor interpolation
        step: int or float, default 86400
            Temporal step size for advection (in seconds)
        N: int or NoneType, default None
            Number of integration steps

            Default is determined based on the temporal step size
        t0: float, default 0.0
            Ending time for advection
        """
        # set default keyword arguments
        kwargs.setdefault("integrator", self.integrator)
        kwargs.setdefault("method", self.method)
        kwargs.setdefault("step", 86400)
        kwargs.setdefault("N", None)
        kwargs.setdefault("t0", self.t0)
        # update advection class attributes
        if kwargs["integrator"] != self.integrator:
            self.integrator = copy.deepcopy(kwargs["integrator"])
        if kwargs["method"] != self.method:
            self.method = copy.deepcopy(kwargs["method"])
        if kwargs["t0"] != self.t0:
            self.t0 = copy.deepcopy(kwargs["t0"])
        # advect the parcel every step
        # (using closest number of iterations)
        step = np.float64(kwargs["step"])
        # set or calculate the number of steps to advect the dataset
        if kwargs["N"] is not None:
            n_steps = copy.deepcopy(kwargs["N"])
        elif np.min(self.t0) < np.min(self.t):
            # maximum number of steps to advect backwards in time
            n_steps = np.abs(np.max(self.t) - np.min(self.t0)) / step
        elif np.max(self.t0) > np.max(self.t):
            # maximum number of steps to advect forward in time
            n_steps = np.abs(np.max(self.t0) - np.min(self.t)) / step
        elif (np.ndim(self.t0) == 0) or (np.ndim(self.t) == 0):
            # maximum number of steps between the two datasets
            n_steps = np.max(np.abs(self.t0 - self.t)) / step
        else:
            # average number of steps between the two datasets
            n_steps = np.abs(np.mean(self.t0) - np.mean(self.t)) / step
        # check input advection functions
        kwargs.update({"N": np.int64(n_steps)})
        logging.info(f"Advecting {kwargs['N']} steps")
        if self.integrator == "euler":
            # euler: Explicit Euler method
            return self.euler(**kwargs)
        elif self.integrator == "RK4":
            # RK4: Fourth-order Runge-Kutta method
            return self.RK4(**kwargs)
        elif self.integrator == "RKF45":
            # RKF45: adaptive Runge-Kutta-Fehlberg 4(5) method
            return self.RKF45(**kwargs)
        else:
            raise ValueError("Invalid advection function")

    # PURPOSE: Advects parcels using an Explicit Euler integration
    def euler(self, **kwargs):
        """
        Advects parcels using an Explicit Euler integration

        Parameters
        ----------
        N: int, default 1
            Number of integration steps
        """
        # set default keyword options
        kwargs.setdefault("N", 1)
        # translate parcel from t to t0 at time step
        dt = (self.t0 - self.t) / np.float64(kwargs["N"])
        self.x0 = copy.deepcopy(self.x)
        self.y0 = copy.deepcopy(self.y)
        # keep track of time for 3-dimensional interpolations
        t = copy.deepcopy(self.t)
        for i in range(kwargs["N"]):
            logging.debug(f"Euler step {i + 1} of {kwargs['N']}")
            ds = self.interp(x=self.x0, y=self.y0, t=t)
            # add displacements to x0 and y0
            self.x0 += ds.U.values * dt
            self.y0 += ds.V.values * dt
            # add to time
            t += dt
        # return the translated coordinates
        return self

    # PURPOSE: Advects parcels using a fourth-order Runge-Kutta integration
    def RK4(self, **kwargs):
        """
        Advects parcels using a fourth-order Runge-Kutta integration

        Parameters
        ----------
        N: int, default 1
            Number of integration steps
        """
        # set default keyword options
        kwargs.setdefault("N", 1)
        # translate parcel from t to t0 at time step
        dt = np.squeeze(self.t0 - self.t) / np.float64(kwargs["N"])
        self.x0 = copy.deepcopy(self.x)
        self.y0 = copy.deepcopy(self.y)
        # keep track of time for 3-dimensional interpolations
        t = copy.deepcopy(self.t)
        for i in range(kwargs["N"]):
            logging.debug(f"RK4 step {i + 1} of {kwargs['N']}")
            ds1 = self.interp(x=self.x0, y=self.y0, t=t)
            x2 = self.x0 + 0.5 * ds1.U.values * dt
            y2 = self.y0 + 0.5 * ds1.V.values * dt
            ds2 = self.interp(x=x2, y=y2, t=t)
            x3 = self.x0 + 0.5 * ds2.U.values * dt
            y3 = self.y0 + 0.5 * ds2.V.values * dt
            ds3 = self.interp(x=x3, y=y3, t=t)
            x4 = self.x0 + ds3.U.values * dt
            y4 = self.y0 + ds3.V.values * dt
            ds4 = self.interp(x=x4, y=y4, t=t)
            # add displacements to x0 and y0
            self.x0 += (
                dt
                * (
                    ds1.U.values
                    + 2.0 * ds2.U.values
                    + 2.0 * ds3.U.values
                    + ds4.U.values
                )
                / 6.0
            )
            self.y0 += (
                dt
                * (
                    ds1.V.values
                    + 2.0 * ds2.V.values
                    + 2.0 * ds3.V.values
                    + ds4.V.values
                )
                / 6.0
            )
            # add to time
            t += dt
        # return the translated coordinates
        return self

    # PURPOSE: Advects parcels using a Runge-Kutta-Fehlberg integration
    def RKF45(self, **kwargs):
        """
        Advects parcels using a Runge-Kutta-Fehlberg 4(5) integration

        Parameters
        ----------
        N: int, default 1
            Number of integration steps
        """
        # set default keyword options
        kwargs.setdefault("N", 1)
        # coefficients in Butcher tableau for Runge-Kutta-Fehlberg 4(5) method
        b4 = [
            25.0 / 216.0,
            0.0,
            1408.0 / 2565.0,
            2197.0 / 4104.0,
            -1.0 / 5.0,
            0.0,
        ]
        b5 = [
            16.0 / 135.0,
            0.0,
            6656.0 / 12825.0,
            28561.0 / 56430.0,
            -9.0 / 50.0,
            2.0 / 55.0,
        ]
        # using an adaptive step size:
        # iterate solution until the difference is less than the tolerance
        # difference between the 4th and 5th order solutions
        sigma = np.inf
        # tolerance for solutions
        tolerance = 5e-2
        # multiply scale by factors of 2 until iteration reaches tolerance level
        scale = 1
        self.x0 = copy.deepcopy(self.x)
        self.y0 = copy.deepcopy(self.y)
        # while the difference (sigma) is greater than the tolerance
        while (sigma > tolerance) or np.isnan(sigma):
            # translate parcel from t to t0 at time step
            dt = (self.t0 - self.t) / np.float64(scale * kwargs["N"])
            X4OA = copy.deepcopy(self.x)
            Y4OA = copy.deepcopy(self.y)
            X5OA = copy.deepcopy(self.x)
            Y5OA = copy.deepcopy(self.y)
            # keep track of time for 3-dimensional interpolations
            t = copy.deepcopy(self.t)
            for i in range(scale * kwargs["N"]):
                logging.debug(f"RKF45 step {i + 1} of {scale * kwargs['N']}")
                # calculate fourth order accurate solutions
                u4, v4 = self.RFK45_interp(X4OA, Y4OA, dt, t=t)
                # add displacements to X40A and Y40A
                X4OA += dt * np.dot(b4, u4)
                Y4OA += dt * np.dot(b4, v4)
                # calculate fifth order accurate solutions
                u5, v5 = self.RFK45_interp(X5OA, Y5OA, dt, t=t)
                # add displacements to X50A and Y50A
                X5OA += dt * np.dot(b5, u5)
                Y5OA += dt * np.dot(b5, v5)
                # add to time
                t += dt
            # calculate difference between 4th and 5th order accurate solutions
            variance = (X5OA - X4OA) ** 2 + (Y5OA - Y4OA) ** 2
            sigma = np.sqrt(np.nanmean(variance))
            logging.info(f"RKF45 sigma: {sigma:.4f} (tolerance: {tolerance})")
            # if sigma is less than the tolerance: save x and y coordinates
            # else: multiply scale by factors of 2 and re-run iteration
            if (sigma <= tolerance) or np.isnan(sigma):
                self.x0 = copy.deepcopy(X4OA)
                self.y0 = copy.deepcopy(Y4OA)
            else:
                scale *= 2
        # return the translated coordinates
        return self

    # PURPOSE: calculates X and Y velocities for Runge-Kutta-Fehlberg 4(5) method
    def RFK45_interp(
        self, xi: np.ndarray, yi: np.ndarray, dt: np.ndarray, **kwargs
    ):
        """
        Calculates X and Y velocities for Runge-Kutta-Fehlberg 4(5) method

        Parameters
        ----------
        xi: np.ndarray
            x-coordinates
        yi: np.ndarray
            y-coordinates
        dt: np.ndarray
            integration time step size
        t: np.ndarray or NoneType, default None
            time coordinates
        """
        kwargs.setdefault("t", None)
        # Butcher tableau for Runge-Kutta-Fehlberg 4(5) method
        A = np.array(
            [
                [1.0 / 4.0, 0.0, 0.0, 0.0, 0.0],
                [3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0],
                [1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 0.0, 0.0],
                [439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0.0],
                [
                    -8.0 / 27.0,
                    2.0,
                    -3544.0 / 2565.0,
                    1859.0 / 4104.0,
                    -11.0 / 40.0,
                ],
            ]
        )
        # calculate velocities and parameters for iteration
        ds1 = self.interp(x=xi, y=yi, t=kwargs["t"])
        x2 = xi + A[0, 0] * ds1.U.values * dt
        y2 = yi + A[0, 0] * ds1.V.values * dt
        ds2 = self.interp(x=x2, y=y2, t=kwargs["t"])
        x3 = xi + (A[1, 0] * ds1.U.values + A[1, 1] * ds2.U.values) * dt
        y3 = yi + (A[1, 0] * ds1.V.values + A[1, 1] * ds2.V.values) * dt
        ds3 = self.interp(x=x3, y=y3, t=kwargs["t"])
        x4 = (
            xi
            + (
                A[2, 0] * ds1.U.values
                + A[2, 1] * ds2.U.values
                + A[2, 2] * ds3.U.values
            )
            * dt
        )
        y4 = (
            yi
            + (
                A[2, 0] * ds1.V.values
                + A[2, 1] * ds2.V.values
                + A[2, 2] * ds3.V.values
            )
            * dt
        )
        ds4 = self.interp(x=x4, y=y4, t=kwargs["t"])
        x5 = (
            xi
            + (
                A[3, 0] * ds1.U.values
                + A[3, 1] * ds2.U.values
                + A[3, 2] * ds3.U.values
                + A[3, 3] * ds4.U.values
            )
            * dt
        )
        y5 = (
            yi
            + (
                A[3, 0] * ds1.V.values
                + A[3, 1] * ds2.V.values
                + A[3, 2] * ds3.V.values
                + A[3, 3] * ds4.V.values
            )
            * dt
        )
        ds5 = self.interp(x=x5, y=y5, t=kwargs["t"])
        x6 = (
            xi
            + (
                A[4, 0] * ds1.U.values
                + A[4, 1] * ds2.U.values
                + A[4, 2] * ds3.U.values
                + A[4, 3] * ds4.U.values
                + A[4, 4] * ds5.U.values
            )
            * dt
        )
        y6 = (
            yi
            + (
                A[4, 0] * ds1.V.values
                + A[4, 1] * ds2.V.values
                + A[4, 2] * ds3.V.values
                + A[4, 3] * ds4.V.values
                + A[4, 4] * ds5.V.values
            )
            * dt
        )
        ds6 = self.interp(x=x6, y=y6, t=kwargs["t"])
        U = np.array(
            [
                ds1.U.values,
                ds2.U.values,
                ds3.U.values,
                ds4.U.values,
                ds5.U.values,
                ds6.U.values,
            ]
        )
        V = np.array(
            [
                ds1.V.values,
                ds2.V.values,
                ds3.V.values,
                ds4.V.values,
                ds5.V.values,
                ds6.V.values,
            ]
        )
        return (U, V)

    @property
    def distance(self):
        """
        Calculates displacement between the start and end coordinates

        Returns
        -------
        dist: np.ndarray
            Eulerian distance between start and end points
        """
        try:
            dist = np.sqrt((self.x0 - self.x) ** 2 + (self.y0 - self.y) ** 2)
        except Exception as exc:
            return None
        else:
            return dist

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
