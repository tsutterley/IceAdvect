#!/usr/bin/env python
"""
geotiff.py
Written by Tyler Sutterley (04/2026)

Reads geotiff files as xarray Datasets

PYTHON DEPENDENCIES:
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
        https://pyproj4.github.io/pyproj/
    rioxarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

UPDATE HISTORY:
    Updated 04/2026: added lineage attribute to save filename(s)
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
import timescale.time
import IceAdvect.utilities
from IceAdvect.io.dataset import combine_attrs

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
    filenames: list,
    mapping: dict,
    **kwargs,
) -> xr.Dataset:
    """Open a geotiff file as an xarray Dataset

    Parameters
    ----------
    filenames: str
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
    for f in filenames:
        # determine variable name from mapping
        variable, pattern = parse_file(f, mapping)
        # append dataset to list
        datasets.append(
            open_dataset(f, variable=variable, pattern=pattern, **kwargs)
        )
    # merge Datasets
    darr = xr.merge(
        datasets,
        combine_attrs=combine_attrs,
        compat="override",
        join="override",
    )
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
    mapping = kwargs.get("mapping", None)
    if mapping is not None:
        variable, pattern = parse_file(filename, mapping)
        kwargs["pattern"] = pattern
    # read the geotiff file as an xarray DataArray
    darr = open_dataarray(filename, **kwargs)
    # convert DataArray to Dataset
    ds = darr.to_dataset(name=variable)
    # add lineage attribute
    ds.attrs["lineage"] = pathlib.Path(filename).name
    # return the xarray Dataset
    return ds


# PURPOSE: read a list of model files
def open_mfdataarray(
    filenames: list[str] | list[pathlib.Path],
    parallel: bool = False,
    **kwargs,
):
    """
    Open multiple geotiff files

    Parameters
    ----------
    filenames: list of str or pathlib.Path
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
    # read each file as xarray DataArray and append to list
    if parallel and dask_available:
        opener = dask.delayed(open_dataarray)
    else:
        opener = open_dataarray
    # read each file as xarray dataset and append to list
    dataarrays = [opener(f, **kwargs) for f in filenames]
    # read datasets as dask arrays
    if parallel and dask_available:
        (dataarrays,) = dask.compute(dataarrays)
    # merge DataArray
    darr = xr.merge(dataarrays, compat="override", join="override")
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


def parse_file(filename: str | pathlib.Path, mapping: dict):
    """
    Determine variable name from filename using mapping

    Parameters
    ----------
    filename: str or pathlib.Path
        Path to file
    mapping: dict
        Dictionary mapping standard variable names to patterns for the file

    Returns
    -------
    variable: str
        Variable name for the file
    pattern: str or None
        Regular expression pattern for extracting time information
    """
    # default variable name and pattern
    variable = "variable"
    pattern = None
    # determine pattern for extracting time information
    for k, v in mapping.items():
        if re.search(v, str(filename), re.IGNORECASE):
            variable, pattern = k, v
            break
    # return variable name and pattern
    return variable, pattern
