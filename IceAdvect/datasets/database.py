#!/usr/bin/env python3
"""
database.py
Written by Tyler Sutterley (01/2026)
Load and maintain the JSON database of velocity datasets

UPDATE HISTORY:
    Written 01/2026
"""

import copy
import json
import pathlib
from IceAdvect.utilities import get_data_path


# PURPOSE: load the JSON database of velocity datasets
def load_database(extra_databases: list = []):
    """
    Load the JSON database of velocity datasets

    Parameters
    ----------
    extra_databases: list, default []
        A list of additional databases to load, as either
        JSON file paths or dictionaries

    Returns
    -------
    parameters: dict
        Database of velocity datasets parameters
    """
    # path to velocity dataset database
    database = get_data_path(["datasets", "database.json"])
    # extract JSON data
    with database.open(mode="r", encoding="utf-8") as fid:
        parameters = json.load(fid)
    # verify that extra_databases is iterable
    if isinstance(extra_databases, (str, pathlib.Path, dict)):
        extra_databases = [extra_databases]
    # load any additional databases
    for db in extra_databases:
        # use database parameters directly if a dictionary
        if isinstance(db, dict):
            extra_database = copy.copy(db)
        # otherwise load parameters from JSON file path
        else:
            # verify that extra database file exists
            db = pathlib.Path(db)
            if not db.exists():
                raise FileNotFoundError(db)
            # extract JSON data
            with db.open(mode="r", encoding="utf-8") as fid:
                extra_database = json.load(fid)
        # Add additional datasets to database
        parameters.update(extra_database)
    return parameters
