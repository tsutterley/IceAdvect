"""
Input/output functions for reading velocity data
"""

import os
import pathlib
import xarray as xr
from . import dataset
from . import netcdf
from . import geotiff

# set environmental variable for anonymous s3 access
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"


def open_dataset(
    filename: str,
    mapping: dict | None = None,
    chunks: int | dict | str | None = None,
    format: str | None = None,
    **kwargs,
) -> xr.Dataset:
    """Open a file as an xarray Dataset

    Parameters
    ----------
    filename: str
        Path to file
    mapping: dict or None, default None
        Dictionary mapping standard variable names to those in the file
    chunks: int, dict, str, or None, default None
        variable chunk sizes for dask (see ``xarray.open_dataset``)
    format: str or None, default None
        File format for xarray to use when opening the dataset

        - ``'netCDF4'``: netCDF4 file
        - ``'geotiff'``: geoTIFF file
        - ``None``: infer from file extension
    **kwargs: dict
        additional keyword arguments for opening files

    Returns
    -------
    ds: xr.Dataset
        xarray Dataset
    """
    nc = (
        ".nc",
        ".nc4",
        ".h5",
        ".hdf5",
    )
    tiff = (
        ".tif",
        ".tiff",
        ".geotiff",
        ".cog",
    )
    if format == "netCDF4" and isinstance(filename, list):
        return netcdf.open_mfdataset(
            filename=filename,
            mapping=mapping,
            chunks=chunks,
            **kwargs,
        )
    elif format == "netCDF4" or pathlib.Path(filename).suffix in nc:
        return netcdf.open_dataset(
            filename=filename,
            mapping=mapping,
            chunks=chunks,
            **kwargs,
        )
    elif format == "geotiff" and isinstance(filename, list):
        return geotiff.open_mfdataset(
            filename=filename,
            mapping=mapping,
            chunks=chunks,
            **kwargs,
        )
    elif format == "geotiff" or pathlib.Path(filename).suffix in tiff:
        return geotiff.open_dataset(
            filename=filename,
            chunks=chunks,
            **kwargs,
        )
    elif isinstance(filename, list):
        return xr.open_mfdataset(filename, chunks=chunks, **kwargs)
    else:
        return xr.open_dataset(filename, chunks=chunks, **kwargs)
