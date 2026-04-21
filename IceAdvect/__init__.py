#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IceAdvect
=======

Python tools for advecting point data for use in a
Lagrangian reference frame powered by xarray

Documentation is available at https://IceAdvect.readthedocs.io
"""

# base modules
import IceAdvect.interpolate
import IceAdvect.spatial
import IceAdvect.tools
import IceAdvect.utilities
from IceAdvect.advect import Advect
from IceAdvect import io, datasets

import IceAdvect.version

# get version number
__version__ = IceAdvect.version.version
