# IceAdvect

Utilities for advecting point data for use in a Lagrangian reference frame

## About

<table>
  <tr>
    <td><b>Tests:</b></td>
    <td>
        <a href="https://IceAdvect.readthedocs.io/en/latest/?badge=latest" alt="Documentation Status"><img src="https://readthedocs.org/projects/IceAdvect/badge/?version=latest"></a>
        <a href="https://github.com/tsutterley/IceAdvect/actions/workflows/python-request.yml" alt="Build"><img src="https://github.com/tsutterley/IceAdvect/actions/workflows/python-request.yml/badge.svg"></a>
        <a href="https://github.com/tsutterley/IceAdvect/actions/workflows/ruff-format.yml" alt="Ruff"><img src="https://github.com/tsutterley/IceAdvect/actions/workflows/ruff-format.yml/badge.svg"></a>
    </td>
  </tr>
  <tr>
    <td><b>License:</b></td>
    <td>
        <a href="https://github.com/tsutterley/IceAdvect/blob/main/LICENSE" alt="License"><img src="https://img.shields.io/github/license/tsutterley/IceAdvect"></a>
    </td>
  </tr>
</table>

For more information: see the documentation at [IceAdvect.readthedocs.io](https://IceAdvect.readthedocs.io/)

## Installation

Development version from GitHub:

```bash
python3 -m pip install git+https://github.com/tsutterley/IceAdvect.git
```

### Running with Pixi

Alternatively, you can use [Pixi](https://pixi.sh/) for a streamlined workspace environment:

1. Install Pixi following the [installation instructions](https://pixi.sh/latest/#installation)
2. Clone the project repository:

```bash
git clone https://github.com/tsutterley/IceAdvect.git
```

3. Move into the `IceAdvect` directory

```bash
cd IceAdvect
```

4. Install dependencies and start JupyterLab:

```bash
pixi run start
```

This will automatically create the environment, install all dependencies, and launch JupyterLab in the [notebooks](./doc/source/notebooks/) directory.

## Dependencies

- [h5netcdf: Pythonic interface to netCDF4 via h5py](https://h5netcdf.org/)
- [lxml: processing XML and HTML in Python](https://pypi.python.org/pypi/lxml)
- [numpy: Scientific Computing Tools For Python](https://www.numpy.org)
- [platformdirs: Python module for determining platform-specific directories](https://pypi.org/project/platformdirs/)
- [pyproj: Python interface to PROJ library](https://pypi.org/project/pyproj/)
- [scipy: Scientific Tools for Python](https://www.scipy.org/)
- [timescale: Python tools for time and astronomical calculations](https://pypi.org/project/timescale/)
- [xarray: N-D labeled arrays and datasets in Python](https://docs.xarray.dev/en/stable/) 

## References

> T. C. Sutterley, T. Markus, T. A. Neumann, M. R. van den Broeke, J. M. van Wessem, and S. R. M. Ligtenberg,
> "Antarctic ice shelf thickness change from multimission lidar mapping", *The Cryosphere*,
> 13, 1801-1817, (2019). [doi: 10.5194/tc-13-1801-2019](https://doi.org/10.5194/tc-13-1801-2019)

## Download

The program homepage is:  
<https://github.com/tsutterley/IceAdvect>

A zip archive of the latest version is available directly at:  
<https://github.com/tsutterley/IceAdvect/archive/main.zip>

## Alternative Software

Advection tools built upon [`pointCollection`](https://github.com/SmithB/pointCollection):  
<https://github.com/tsutterley/pointAdvection>


## Disclaimer

This package includes software developed at NASA Goddard Space Flight Center (GSFC) and the University of Washington Applied Physics Laboratory (UW-APL).
It is not sponsored or maintained by the Universities Space Research Association (USRA), AVISO or NASA.
The software is provided here for your convenience but *with no guarantees whatsoever*.

## Contributing

This project contains work and contributions from the [scientific community](./CONTRIBUTORS.md).
If you would like to contribute to the project, please have a look at the [contribution guidelines](./doc/source/getting_started/Contributing.rst), [open issues](https://github.com/tsutterley/IceAdvect/issues) and [discussions board](https://github.com/tsutterley/IceAdvect/discussions).

## License

The content of this project is licensed under the [Creative Commons Attribution 4.0 Attribution license](https://creativecommons.org/licenses/by/4.0/) and the source code is licensed under the [MIT license](LICENSE).
