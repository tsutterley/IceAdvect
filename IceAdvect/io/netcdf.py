#!/usr/bin/env python
"""
netcdf.py
Written by Tyler Sutterley (02/2026)

Reads netCDF4 files as xarray Datasets with variable mapping

PYTHON DEPENDENCIES:
    h5netcdf: Python interface to HDF5 and netCDF4
        https://pypi.org/project/h5netcdf/
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
        https://pyproj4.github.io/pyproj/
    xarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

UPDATE HISTORY:
    Updated 02/2026: added logging information when opening files
    Written 01/2026
"""

from __future__ import division, annotations

import os
import pyproj
import pathlib
import logging
import warnings
import numpy as np
import xarray as xr
import IceAdvect.utilities
import timescale.time

# attempt imports
dask = IceAdvect.utilities.import_dependency("dask")
dask_available = IceAdvect.utilities.dependency_available("dask")

# set environmental variable for anonymous s3 access
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
# suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


# PURPOSE: read a list of files
def open_mfdataset(filename: list[str] | list[pathlib.Path], **kwargs):
    """
    Open multiple netCDF4 files

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
    ds: xarray.Dataset
        xarray Dataset
    """
    # set default keyword arguments
    kwargs.setdefault("parallel", False)
    parallel = kwargs.get("parallel") and dask_available
    # read each file as xarray dataset and append to list
    if parallel:
        opener = dask.delayed(open_dataset)
        (d,) = dask.compute([opener(f, **kwargs) for f in filename])
    else:
        d = [open_dataset(f, **kwargs) for f in filename]
    # merge datasets
    ds = xr.merge(d, compat="override")
    # return xarray dataset
    return ds


def open_dataset(
    filename: str,
    mapping: dict | None = None,
    chunks: int | dict | str | None = None,
    **kwargs,
) -> xr.Dataset:
    """Open a netCDF4 file as an xarray Dataset and remap variables

    Parameters
    ----------
    filename: str
        Path to netCDF4 file
    mapping: dict or None, default None
        Dictionary mapping standard variable names to those in the file
    chunks: int, dict, str, or None, default None
        variable chunk sizes for dask (see ``xarray.open_dataset``)

    Returns
    -------
    ds: xr.Dataset
        xarray Dataset
    """
    # set default keyword arguments
    kwargs.setdefault("longterm", False)
    # get coordinate reference system (CRS) information from kwargs
    crs = kwargs.get("crs", None)
    # verbose logging
    logging.debug(f"Opening netCDF4 file: {filename}")
    # open the netCDF4 file using xarray
    tmp = xr.open_dataset(filename, mask_and_scale=True, chunks=chunks)
    tmp = tmp.drop_vars(["lon", "lat"], errors="ignore")
    # apply variable mapping if provided
    if mapping is not None:
        # create xarray dataset
        ds = xr.Dataset()
        for key, value in mapping.items():
            ds[key] = tmp[value]
        # copy attributes
        ds.attrs = tmp.attrs.copy()
    else:
        ds = tmp.copy()
    # assign time dimension for long-term averages or from attributes
    if kwargs["longterm"]:
        pass
    elif "time_coverage_start" in ds.attrs and "time_coverage_end" in ds.attrs:
        ds = ds.expand_dims(dim="time", axis=2)
        # parse strings into datetime objects
        start_time = timescale.time.parse(ds.attrs["time_coverage_start"])
        end_time = timescale.time.parse(ds.attrs["time_coverage_end"])
        time_array = np.array([start_time, end_time], dtype="datetime64[D]")
        # convert to timescale objects and take the mean
        ts = timescale.from_datetime(time_array)
        ds["time"] = ts.mean().to_datetime()
    # attach coordinate reference system (CRS) information
    if crs is not None:
        ds.attrs["crs"] = pyproj.CRS.from_user_input(crs).to_dict()
    # return the xarray dataset
    return ds
