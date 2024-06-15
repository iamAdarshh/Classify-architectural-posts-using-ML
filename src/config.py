"""
This module provides a constant for the default data folder path.

The `DEFAULT_DATA_FOLDER` is set to a path that points to a 'data' directory
located in the parent directory of the current file.

Attributes:
    DEFAULT_DATA_FOLDER (str): The default path to the data folder.
"""

import os

DEFAULT_DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'data')
