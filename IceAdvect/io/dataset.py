#!/usr/bin/env python
"""
dataset.py
Written by Tyler Sutterley (04/2026)
An xarray.Dataset extension for velocity data

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
        https://pyproj4.github.io/pyproj/
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    xarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

UPDATE HISTORY:
    Updated 04/2026: add barycentric interpolation for unstructured grids
        add support for unstructured (e.g. finite element) grids
    Updated 02/2026: create subaccessor registration functions
    Written 01/2026
"""

import pint
import pyproj
import warnings
import numpy as np
import xarray as xr
import timescale.time

# suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

__all__ = [
    "Dataset",
    "DataArray",
    "register_dataset_subaccessor",
    "register_dataarray_subaccessor",
    "_transform",
    "_coords",
]

# pint unit registry
__ureg__ = pint.UnitRegistry()
# default epoch for time conversions
__epoch__ = timescale.time._j2000_epoch


@xr.register_dataset_accessor("advect")
class Dataset:
    """Accessor for extending an ``xarray.Dataset`` for velocity data"""

    def __init__(self, ds):
        # initialize Dataset
        self._ds = ds

    def assign_coords(
        self,
        x: np.ndarray,
        y: np.ndarray,
        crs: str | int | dict = 4326,
        **kwargs,
    ):
        """
        Assign new coordinates to the ``Dataset``

        Parameters
        ----------
        x: np.ndarray
            Updated x-coordinates
        y: np.ndarray
            Updated y-coordinates
        crs: str, int, or dict, default 4326 (WGS84 Latitude/Longitude)
            Coordinate reference system of coordinates
        kwargs: dict
            Keyword arguments for ``xarray.Dataset.assign_coords``

        Returns
        -------
        ds: xarray.Dataset
            ``Dataset`` with updated coordinates
        """
        # assign new coordinates to dataset
        ds = self._ds.assign_coords(dict(x=x, y=y), **kwargs)
        ds.attrs["crs"] = crs
        # return the dataset
        return ds

    def barycentric_interp(
        self,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ):
        """
        Interpolate unstructured ``Datasets`` using a barycentric
        method with first or second order triangular finite elements

        Parameters
        ----------
        x: np.ndarray
            Interpolation x-coordinates
        y: np.ndarray
            Interpolation y-coordinates
        order: int
            Polynomial order of the triangular elements

            - ``1``: linear
            - ``2``: quadratic
        cutoff: int or float, default np.inf
            Maximum distance to check for elements

        Returns
        -------
        other: xarray.Dataset
            Interpolated ``Dataset``
        """
        # import barycentric interpolation functions
        from IceAdvect.interpolate import (
            _to_barycentric,
            _inside_triangle,
            _shape_functions,
        )

        # get the polynomial order of the finite elements
        order = self._ds["element"].attrs.get("order", 1)
        # default order is same as the tide model
        kwargs.setdefault("order", order)

        # get cutoff distance to crop elements to bounding box
        cutoff = kwargs.get("cutoff", np.inf)
        # crop dataset to bounding box of other dataset plus buffer
        if np.isfinite(cutoff):
            # use the cutoff distance as a buffer
            cutoff_km = cutoff * __ureg__.parse_units("km")
            buffer = cutoff_km.to(self.axis_units).magnitude
            # bounds of interpolation coordinates
            bounds = [np.min(x), np.max(x), np.min(y), np.max(y)]
            # crop dataset to bounding box of other dataset plus buffer
            ds = self.crop(bounds=bounds, buffer=buffer)
        else:
            # copy dataset without cropping
            ds = self._ds.copy()

        # allocate for barycentric coordinates
        xi = xr.full_like(x, np.nan)
        eta = xr.full_like(x, np.nan)
        null_points = xi.isnull()
        # allocate for indices of valid elements
        element = xr.zeros_like(x, dtype="i")
        # find the valid elements and barycentric coordinates
        for i, elem in enumerate(ds.element):
            # x and y coordinates of element vertices
            x_elem = ds.x.isel(element=i).drop_vars("element")
            y_elem = ds.y.isel(element=i).drop_vars("element")
            # convert model coordinates to barycentric
            xi_elem, eta_elem = _to_barycentric(x_elem, y_elem, x, y)
            # drop dimensions
            xi_elem = xi_elem.drop_vars("vertex", errors="ignore")
            eta_elem = eta_elem.drop_vars("vertex", errors="ignore")
            # determine if points are within element and need values
            inside_element = _inside_triangle(xi_elem, eta_elem)
            # skip if nothing is inside the element
            if not np.any(inside_element & null_points):
                continue
            # save barycentric coordinates and indices
            update_element = np.logical_not(inside_element & null_points)
            xi = xi.where(update_element, xi_elem, drop=False)
            eta = eta.where(update_element, eta_elem, drop=False)
            element = element.where(update_element, i, drop=False)
            # can quit search if all interpolation points have values
            null_points = xi.isnull()
            if not null_points.any():
                break
        # get shape functions and convert to DataArray
        N = _shape_functions(xi, eta, kwargs["order"])
        beta = xr.concat(N, dim="node")
        # allocate for output dataset
        other = xr.Dataset()
        # copy attributes
        for att_name, att_val in self._ds.attrs.items():
            other.attrs[att_name] = att_val
        # iterate over variables in dataset
        for i, v in enumerate(ds.data_vars.keys()):
            # tide model variable for valid elements
            var = ds[v].isel(element=element)
            # calculate dot product over elements and nodes
            other[v] = var.dot(beta, dim="node")
            # copy variable attributes
            for att_name, att_val in self._ds[v].attrs.items():
                other[v].attrs[att_name] = att_val
        # add coordinates to output dataset
        other.coords["x"] = x
        other.coords["y"] = y
        # return the interpolated dataset
        # drop empty vertex coordinates
        return other.drop_vars("vertex", errors="ignore").compute()

    def coords_as(
        self,
        x: np.ndarray,
        y: np.ndarray,
        crs: str | int | dict = 4326,
        **kwargs,
    ):
        """
        Transform coordinates into ``DataArrays`` in the ``Dataset``
        coordinate reference system

        Parameters
        ----------
        x: np.ndarray
            Input x-coordinates
        y: np.ndarray
            Input y-coordinates
        crs: str, int, or dict, default 4326 (WGS84 Latitude/Longitude)
            Coordinate reference system of input coordinates

        Returns
        -------
        X: xarray.DataArray
            Transformed x-coordinates
        Y: xarray.DataArray
            Transformed y-coordinates
        """
        # convert coordinate reference system to that of the dataset
        # and format as xarray DataArray with appropriate dimensions
        X, Y = _coords(x, y, source_crs=crs, target_crs=self.crs, **kwargs)
        # return the transformed coordinates
        return X, Y

    def crop(
        self,
        bounds: list | tuple,
        buffer: int | float = 0,
    ):
        """
        Crop ``Dataset`` to input bounding box

        Parameters
        ----------
        bounds: list, tuple
            Bounding box [min_x, max_x, min_y, max_y]
        buffer: int or float, default 0
            Buffer to add to bounds for cropping
        """
        # create copy of dataset
        ds = self._ds.copy()
        # check if chunks are present
        if hasattr(ds, "chunks") and ds.chunks is not None:
            ds = ds.chunk(-1).compute()
        # unpack bounds and buffer
        xmin = bounds[0] - buffer
        xmax = bounds[1] + buffer
        ymin = bounds[2] - buffer
        ymax = bounds[3] + buffer
        # crop dataset to bounding box
        if self.grid_type == "unstructured":
            # crop unstructured datasets
            # include elements that cross the bounding box
            ds = ds.where(
                (ds.x.max(dim="vertex") >= xmin)
                & (ds.x.min(dim="vertex") <= xmax)
                & (ds.y.max(dim="vertex") >= ymin)
                & (ds.y.min(dim="vertex") <= ymax),
                drop=True,
            )
        else:
            # crop gridded datasets
            ds = ds.where(
                (ds.x >= xmin)
                & (ds.x <= xmax)
                & (ds.y >= ymin)
                & (ds.y <= ymax),
                drop=True,
            )
        # return the cropped dataset
        return ds

    def grid_interp(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method="linear",
        **kwargs,
    ):
        """
        Interpolate a regular or rectilinear ``Dataset`` to new coordinates

        Parameters
        ----------
        x: np.ndarray
            Interpolation x-coordinates
        y: np.ndarray
            Interpolation y-coordinates
        method: str, default 'linear'
            Interpolation method

        Returns
        -------
        other: xarray.Dataset
            Interpolated ``Dataset``
        """
        # interpolate dataset using built-in xarray methods
        other = self._ds.interp(x=x, y=y, method=method)
        # return xarray dataset
        return other

    def inpaint(self, **kwargs):
        """
        Inpaint over missing data in ``Dataset``

        Parameters
        ----------
        kwargs: dict
            Keyword arguments for :py:func:`IceAdvect.interpolate.inpaint`

        Returns
        -------
        ds: xarray.Dataset
            Interpolated ``Dataset``
        """
        # import inpaint function
        from IceAdvect.interpolate import inpaint

        # create copy of dataset
        ds = self._ds.copy()
        # inpaint each variable in the dataset
        for v in ds.data_vars.keys():
            ds[v].values = inpaint(
                self._x, self._y, self._ds[v].values, **kwargs
            )
        # return the dataset
        return ds

    def interp(
        self,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ):
        """
        Interpolate ``Dataset`` to new coordinates

        Parameters
        ----------
        x: np.ndarray
            Interpolation x-coordinates
        y: np.ndarray
            Interpolation y-coordinates
        kwargs: dict
            Additional keyword arguments for interpolation functions

        Returns
        -------
        other: xarray.Dataset
            Interpolated ``Dataset``
        """
        # set default keyword arguments
        kwargs.setdefault("method", "linear")
        kwargs.setdefault("extrapolate", False)
        kwargs.setdefault("cutoff", np.inf)
        # check if interpolating from a grid or mesh
        if self.grid_type == "unstructured":
            # use barycentric interpolation if data is unstructured
            other = self.barycentric_interp(x, y, **kwargs)
        else:
            # use built-in xarray interpolation methods
            other = self.grid_interp(x, y, **kwargs)
        # return xarray dataset
        return other

    def run(self, **kwargs):
        """
        Advect coordinates using the velocity field in the ``Dataset``

        Parameters
        ----------
        kwargs: dict
            keyword arguments for :py:class:`IceAdvect.advect`

        Returns
        -------
        x0: np.ndarray
            Advected x-coordinates
        y0: np.ndarray
            Advected y-coordinates
        """
        from IceAdvect.advect import Advect

        # convert dataset to base units
        ds = self.to_base_units()
        # extract keyword arguments for run function
        run_keywords = ["integrator", "method", "step", "N"]
        run_kwargs = {k: kwargs.pop(k) for k in run_keywords if k in kwargs}
        # run advection model on dataset
        x0, y0 = Advect(ds, **kwargs).run(**run_kwargs)
        # return the advected coordinates
        return x0, y0

    def transform_as(
        self,
        x: np.ndarray,
        y: np.ndarray,
        crs: str | int | dict = 4326,
        **kwargs,
    ):
        """
        Transform coordinates to/from the ``Dataset`` coordinate reference system

        Parameters
        ----------
        x: np.ndarray
            Input x-coordinates
        y: np.ndarray
            Input y-coordinates
        crs: str, int, or dict, default 4326 (WGS84 Latitude/Longitude)
            Coordinate reference system of input coordinates
        direction: str, default 'FORWARD'
            Direction of transformation

            - ``'FORWARD'``: from input crs to model crs
            - ``'INVERSE'``: from model crs to input crs

        Returns
        -------
        X: np.ndarray
            Transformed x-coordinates
        Y: np.ndarray
            Transformed y-coordinates
        """
        # convert coordinate reference system to that of the dataset
        X, Y = _transform(x, y, source_crs=crs, target_crs=self.crs, **kwargs)
        # return the transformed coordinates
        return (X, Y)

    def to_units(
        self,
        units: str,
        value: float = 1.0,
    ):
        """Convert ``Dataset`` to specified velocity units

        Parameters
        ----------
        units: str
            Output units
        value: float, default 1.0
            Scaling factor to apply
        """
        # create copy of dataset
        ds = self._ds.copy()
        # convert velocities to specified units
        ds.U = ds.U.advect.to_units(units, value=value)
        ds.V = ds.V.advect.to_units(units, value=value)
        # return the dataset
        return ds

    def to_base_units(self):
        """Convert ``Dataset`` to base units"""
        # create copy of dataset
        ds = self._ds.copy()
        # convert velocities to base units
        ds["U"] = ds.U.advect.to_base_units()
        ds["V"] = ds.V.advect.to_base_units()
        # convert time coordinate to deltatime in seconds
        if "time" in ds:
            ts = timescale.from_datetime(ds["time"])
            ds["t"] = xr.DataArray(
                ts.to_deltatime(epoch=__epoch__, scale=86400.0), dims="time"
            )
            ds = ds.swap_dims({"time": "t"}).drop_vars("time")
        # return the dataset
        return ds

    @property
    def area_of_use(self) -> str | None:
        """Area of use from the ``Dataset`` CRS"""
        if self.crs.area_of_use is not None:
            return self.crs.area_of_use.name.replace(".", "").lower()

    @property
    def axis_units(self) -> str:
        """Units of the coordinate axes from the ``Dataset`` CRS"""
        return self.crs.axis_info[0].unit_name

    @property
    def crs(self):
        """Coordinate reference system of the ``Dataset``"""
        # return the CRS of the dataset
        # default is EPSG:4326 (WGS84)
        CRS = self._ds.attrs.get("crs", 4326)
        return pyproj.CRS.from_user_input(CRS)

    @property
    def divergence(self):
        """
        Calculate the divergence of a velocity field
        """
        # calculate divergence
        dU = self._ds.U.differentiate("x")
        dV = self._ds.V.differentiate("y")
        return dU + dV

    @property
    def grid_type(self) -> str:
        """Spatial structure of the ``Dataset``"""
        return self._ds.attrs.get("grid_type", "grid")

    @property
    def speed(self):
        """
        Calculate the speed from a velocity field
        """
        amp = np.sqrt(self._ds.U**2 + self._ds.V**2)
        return amp

    @property
    def _x(self):
        """x-coordinates of the ``Dataset``"""
        return self._ds.x.values

    @property
    def _y(self):
        """y-coordinates of the ``Dataset``"""
        return self._ds.y.values


@xr.register_dataarray_accessor("advect")
class DataArray:
    """Accessor for extending an ``xarray.DataArray`` for velocity data"""

    def __init__(self, da):
        # initialize DataArray
        self._da = da

    def crop(self, bounds: list | tuple, buffer: int | float = 0):
        """
        Crop ``DataArray`` to input bounding box

        Parameters
        ----------
        bounds: list, tuple
            bounding box [min_x, max_x, min_y, max_y]
        buffer: int or float, default 0
            buffer to add to bounds for cropping
        """
        # create copy of dataarray
        da = self._da.copy()
        # unpack bounds and buffer
        xmin = bounds[0] - buffer
        xmax = bounds[1] + buffer
        ymin = bounds[2] - buffer
        ymax = bounds[3] + buffer
        # crop dataset to bounding box
        da = da.where(
            (da.x >= xmin) & (da.x <= xmax) & (da.y >= ymin) & (da.y <= ymax),
            drop=True,
        )
        # return the cropped dataarray
        return da

    def to_units(
        self,
        units: str,
        value: float = 1.0,
    ):
        """Convert ``DataArray`` to specified units

        Parameters
        ----------
        units: str
            Output units
        value: float, default 1.0
            Scaling factor to apply
        """
        # convert to specified units
        conversion = value * self.quantity.to(units)
        da = self._da * conversion.magnitude
        da.attrs["units"] = str(conversion.units)
        return da

    def to_base_units(self, value=1.0):
        """Convert ``DataArray`` to base units

        Parameters
        ----------
        value: float, default 1.0
            Scaling factor to apply
        """
        # convert to base units
        conversion = value * self.quantity.to_base_units()
        da = self._da * conversion.magnitude
        da.attrs["units"] = str(conversion.units)
        return da

    @property
    def units(self):
        """Units of the ``DataArray``"""
        try:
            return __ureg__.parse_units(self._units)
        except TypeError as exc:
            raise ValueError(f"Unknown units: {self._units}") from exc
        except AttributeError as exc:
            raise AttributeError("DataArray has no attribute 'units'") from exc

    @property
    def quantity(self):
        """``Pint`` Quantity of the ``DataArray``"""
        return 1.0 * self.units

    @property
    def _units(self):
        """Units attribute of the ``DataArray`` as a string"""
        return self._da.attrs.get("units")


def register_dataset_subaccessor(name):
    """Register a custom subaccessor on ``Dataset`` objects

    Parameters
    ----------
    name: str
        Name of the subaccessor
    """
    return xr.core.extensions._register_accessor(name, Dataset)


def register_dataarray_subaccessor(name):
    """Register a custom subaccessor on ``DataArray`` objects

    Parameters
    ----------
    name: str
        Name of the subaccessor
    """
    return xr.core.extensions._register_accessor(name, DataArray)


def _transform(
    i1: np.ndarray,
    i2: np.ndarray,
    source_crs: str | int | dict = 4326,
    target_crs: str | int | dict = None,
    **kwargs,
):
    """
    Transform coordinates to/from the dataset coordinate reference system

    Parameters
    ----------
    i1: np.ndarray
        Input x-coordinates
    i2: np.ndarray
        Input y-coordinates
    source_crs: str, int, or dict, default 4326 (WGS84 Latitude/Longitude)
        Coordinate reference system of input coordinates
    target_crs: str, int, or dict, default None
        Coordinate reference system of output coordinates
    direction: str, default 'FORWARD'
        Direction of transformation

        - ``'FORWARD'``: from input crs to model crs
        - ``'INVERSE'``: from model crs to input crs

    Returns
    -------
    o1: np.ndarray
        Transformed x-coordinates
    o2: np.ndarray
        Transformed y-coordinates
    """
    # set the direction of the transformation
    kwargs.setdefault("direction", "FORWARD")
    assert kwargs["direction"] in ("FORWARD", "INVERSE", "IDENT")
    # get the coordinate reference system and transform
    source_crs = pyproj.CRS.from_user_input(source_crs)
    transformer = pyproj.Transformer.from_crs(
        source_crs, target_crs, always_xy=True
    )
    # convert coordinate reference system
    o1, o2 = transformer.transform(i1, i2, **kwargs)
    # return the transformed coordinates
    return (o1, o2)


def _coords(
    x: np.ndarray,
    y: np.ndarray,
    source_crs: str | int | dict = 4326,
    target_crs: str | int | dict = None,
    **kwargs,
):
    """
    Transform coordinates into DataArrays in a new
    coordinate reference system

    Parameters
    ----------
    x: np.ndarray
        Input x-coordinates
    y: np.ndarray
        Input y-coordinates
    source_crs: str, int, or dict, default 4326 (WGS84 Latitude/Longitude)
        Coordinate reference system of input coordinates
    target_crs: str, int, or dict, default None
        Coordinate reference system of output coordinates
    type: str or None, default None
        Coordinate data type

        If not provided: must specify ``time`` parameter to auto-detect

        - ``None``: determined from input variable dimensions
        - ``'drift'``: drift buoys or satellite/airborne altimetry
        - ``'grid'``: spatial grids or images
        - ``'time series'``: time series at a single point
    time: np.ndarray or None, default None
        Time variable for determining coordinate data type

    Returns
    -------
    X: xarray.DataArray
        Transformed x-coordinates
    Y: xarray.DataArray
        Transformed y-coordinates
    """
    from IceAdvect.spatial import data_type

    # set default keyword arguments
    kwargs.setdefault("type", None)
    kwargs.setdefault("time", None)
    # determine coordinate data type if possible
    if (np.ndim(x) == 0) and (np.ndim(y) == 0):
        coord_type = "time series"
    elif kwargs["type"] is None:
        # must provide time variable to determine data type
        assert kwargs["time"] is not None, (
            "Must provide time parameter when type is not specified"
        )
        coord_type = data_type(x, y, np.ravel(kwargs["time"]))
    else:
        # use provided coordinate data type
        # and verify that it is lowercase
        coord_type = kwargs.get("type").lower()
    # convert coordinates to a new coordinate reference system
    if (coord_type == "grid") and (np.size(x) != np.size(y)):
        gridx, gridy = np.meshgrid(x, y)
        mx, my = _transform(
            gridx,
            gridy,
            source_crs=source_crs,
            target_crs=target_crs,
            direction="FORWARD",
        )
    else:
        mx, my = _transform(
            x,
            y,
            source_crs=source_crs,
            target_crs=target_crs,
            direction="FORWARD",
        )
    # convert to xarray DataArray with appropriate dimensions
    if (np.ndim(x) == 0) and (np.ndim(y) == 0):
        X = xr.DataArray(mx)
        Y = xr.DataArray(my)
    elif coord_type == "grid":
        X = xr.DataArray(mx, dims=("y", "x"))
        Y = xr.DataArray(my, dims=("y", "x"))
    elif coord_type == "drift":
        X = xr.DataArray(mx, dims=("time"))
        Y = xr.DataArray(my, dims=("time"))
    elif coord_type == "time series":
        X = xr.DataArray(mx, dims=("station"))
        Y = xr.DataArray(my, dims=("station"))
    else:
        raise ValueError(f"Unknown coordinate data type: {coord_type}")
    # return the transformed coordinates
    return (X, Y)
