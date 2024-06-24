import pandas as pd

"""
This module contains utility functions for the LSTM model.
"""


def determine_labels(row: pd.Series) -> str:
    """
    Determine the label for a given row based on its content.

    Parameters:
    row (pd.Series): A pandas Series representing a row of data.

    Returns:
    str: The label for the row ('evaluation', 'analysis', 'synthesis', or 'other').
    """
    if row['is_evaluation']:
        return 'evaluation'
    if row['is_analysis']:
        return 'analysis'
    if row['is_synthesis']:
        return 'synthesis'
    return 'other'
