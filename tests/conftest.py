"""
This module contains shared fixtures, hooks, and configuration settings for pytest.
"""

import sys
import os

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
