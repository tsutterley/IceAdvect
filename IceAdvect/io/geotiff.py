#!/usr/bin/env python
"""
geotiff.py
Written by Tyler Sutterley (02/2026)

Reads geotiff files as xarray Datasets

PYTHON DEPENDENCIES:
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
        https://pyproj4.github.io/pyproj/
    rioxarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

UPDATE HISTORY:
    Updated 02/2026: added logging information when opening files
    Written 01/2026
"""

from __future__ import division, annotations

import os
import re
import pyproj
import logging
import pathlib
import warnings
import numpy as np
import xarray as xr
import IceAdvect.utilities
import timescale.time

# attempt imports
dask = IceAdvect.utilities.import_dependency("dask")
dask_available = IceAdvect.utilities.dependency_available("dask")
rioxarray = IceAdvect.utilities.import_dependency("rioxarray")
rioxarray.merge = IceAdvect.utilities.import_dependency("rioxarray.merge")

# set environmental variable for anonymous s3 access
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
# suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


def open_mfdataset(
    filename: list,
    mapping: dict,
    **kwargs,
) -> xr.Dataset:
    """Open a geotiff file as an xarray Dataset

    Parameters
    ----------
    filename: str
        Path to geotiff file
    mapping: dict
        Dictionary mapping standard variable names to patterns for the file
    chunks: int, dict, str, or None, default None
        variable chunk sizes for dask (see ``rioxarray.open_rasterio``)

    Returns
    -------
    ds: xr.Dataset
        xarray Dataset
    """
    # read the geotiff files as an xarray Datasets
    datasets = []
    for f in filename:
        # determine variable name from mapping
        try:
            (variable,) = [
                k for k, v in mapping.items() if re.search(v, str(f), re.I)
            ]
        except ValueError:
            pattern = None
            variable = "variable"
        else:
            # determine pattern for extracting time information
            pattern = mapping[variable]
        # append Dataset to list
        datasets.append(
            open_dataset(
                f,
                variable=variable,
                pattern=pattern,
                **kwargs,
            )
        )
    # merge Datasets
    darr = xr.merge(datasets, compat="override")
    # return xarray Dataset
    return darr


def open_dataset(
    filename: list,
    variable: str = "variable",
    **kwargs,
) -> xr.Dataset:
    """Open a geotiff file as an xarray Dataset

    Parameters
    ----------
    filename: str
        Path to geotiff file
    variable: str, default "variable"
        variable name for the file
    mapping: dict or None, default None
        Dictionary mapping standard variable names to those in the file
    chunks: int, dict, str, or None, default None
        variable chunk sizes for dask (see ``rioxarray.open_rasterio``)

    Returns
    -------
    ds: xr.Dataset
        xarray Dataset
    """
    kwargs.setdefault("mapping", None)
    if kwargs["mapping"] is not None:
        (variable,) = [
            k
            for k, v in kwargs["mapping"].items()
            if re.search(v, str(filename), re.I)
        ]
        # determine pattern for extracting time information
        kwargs["pattern"] = kwargs["mapping"][variable]
    # read the geotiff file as an xarray DataArray
    darr = open_dataarray(
        filename,
        **kwargs,
    )
    # convert DataArray to Dataset
    ds = darr.to_dataset(name=variable)
    # return the xarray Dataset
    return ds


# PURPOSE: read a list of model files
def open_mfdataarray(filename: list[str] | list[pathlib.Path], **kwargs):
    """
    Open multiple geotiff files

    Parameters
    ----------
    filename: list of str or pathlib.Path
        list of files
    parallel: bool, default False
        Open files in parallel using ``dask.delayed``
    kwargs: dict
        additional keyword arguments for opening files

    Returns
    -------
    darr: xarray.DataArray
        xarray DataArray
    """
    # set default keyword arguments
    kwargs.setdefault("parallel", False)
    parallel = kwargs.get("parallel") and dask_available
    # read each file as xarray DataArray and append to list
    if parallel:
        opener = dask.delayed(open_dataarray)
        (d,) = dask.compute([opener(f, **kwargs) for f in filename])
    else:
        d = [open_dataarray(f, **kwargs) for f in filename]
    # merge DataArray
    darr = xr.merge(d, compat="override")
    # return xarray DataArray
    return darr


def open_dataarray(
    filename: str,
    longterm: bool = False,
    pattern: str | None = None,
    chunks: int | dict | str | None = None,
    **kwargs,
) -> xr.DataArray:
    """Open a geotiff file as an xarray DataArray

    Parameters
    ----------
    filename: str
        Path to geotiff file
    longterm: bool, default False
        Datafile is a long-term average
    pattern: str or None, default None
        Regular expression pattern for extracting time information
    chunks: int, dict, str, or None, default None
        variable chunk sizes for dask (see ``rioxarray.open_rasterio``)

    Returns
    -------
    darr: xr.DataArray
        xarray DataArray
    """
    # get coordinate reference system (CRS) information from kwargs
    crs = kwargs.get("crs", None)
    # verbose logging
    logging.debug(f"Opening GeoTIFF file: {filename}")
    # open the geotiff file using rioxarray
    darr = rioxarray.open_rasterio(
        filename, masked=True, chunks=chunks, **kwargs
    )
    # name of the input file
    name = IceAdvect.utilities.Path(filename).name
    # assign time dimension for long-term averages or from filename pattern
    if longterm:
        pass
    elif pattern and re.search(pattern, name, re.I):
        # extract start and end time from filename
        _, start, end, _ = re.findall(pattern, name, re.I).pop()
        # parse strings into datetime objects
        start_time = timescale.time.parse(start)
        end_time = timescale.time.parse(end)
        time_array = np.array([start_time, end_time], dtype="datetime64[D]")
        # convert to timescale objects and take the mean
        ts = timescale.from_datetime(time_array)
        darr["time"] = xr.DataArray(ts.mean().to_datetime(), dims="band")
        darr = darr.swap_dims({"band": "time"})
    # attach coordinate reference system (CRS) information
    if crs is not None:
        darr.attrs["crs"] = pyproj.CRS.from_user_input(crs).to_dict()
    else:
        crs_wkt = darr.spatial_ref.attrs["crs_wkt"]
        darr.attrs["crs"] = pyproj.CRS.from_user_input(crs_wkt).to_dict()
    # return xarray DataArray
    return darr
