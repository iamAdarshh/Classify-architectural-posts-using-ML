"""
This module provides following functions:

Functions:
    extract_post_id(df: pd.DataFrame) -> pd.DataFrame:
        Extracts post IDs from the 'URL ' column.

    save_as_excel(df: pd.DataFrame, filepath: str):
        Saves a DataFrame to an Excel file.
"""
import os
import pandas as pd


def extract_post_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts post IDs from the 'URL ' column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with 'postId' column.
    """
    clone_df = df.copy()
    clone_df['postId'] = clone_df['URL '].str.extract(r'questions/(\d+)', expand=False)
    return clone_df


def save_as_excel(df: pd.DataFrame, filepath: str):
    """
    Saves a DataFrame to an Excel file.

    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        filepath (str): Path to save the Excel file.
    """
    # Extract directory path from the filepath
    directory = os.path.dirname(filepath)

    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the DataFrame to Excel
    df.to_excel(filepath, index=False)
