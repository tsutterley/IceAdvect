#!/usr/bin/env python
"""
test_advect.py (11/2020)
Verify advection operations
"""

import pytest
import numpy as np
import xarray as xr
import IceAdvect


# parametrize over advection method
@pytest.mark.parametrize("INTEGRATOR", ["euler", "RK4", "RKF45"])
# PURPOSE: test the advection module
def test_advect(INTEGRATOR):
    # create test data
    N = 100
    x = xr.DataArray(5.0 * np.random.rand(N), dims="points")
    y = xr.DataArray(5.0 * np.random.rand(N), dims="points")
    t, t0 = 0.0, 20.0
    # create a dataset with uniform velocity fields
    u, v = 0.1, 0.1  # m/s
    ny, nx = 101, 101
    ds = xr.Dataset()
    ds["x"] = (("x",), np.linspace(0, 10, nx))
    ds["y"] = (("y",), np.linspace(0, 10, ny))
    ds["U"] = (("y", "x"), u * np.ones((ny, nx)))
    ds["V"] = (("y", "x"), v * np.ones((ny, nx)))
    ds["U"].attrs["units"] = "m/s"
    ds["V"].attrs["units"] = "m/s"
    # perform advection
    x_new, y_new = ds.advect.run(
        x=x, y=y, t=t, t0=t0, step=1, integrator=INTEGRATOR
    )
    # expected results for a uniform velocity field
    x_expected = x + u * (t0 - t)
    y_expected = y + v * (t0 - t)
    # verify results
    assert np.allclose(x_new, x_expected)
    assert np.allclose(y_new, y_expected)
