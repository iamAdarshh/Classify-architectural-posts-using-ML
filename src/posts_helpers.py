"""
This module provides functions to clean data, extract post IDs, save and load data,
and fetch Stack Overflow post details.

Functions:
    clean_input(filename: str, new_filename: str) -> pd.DataFrame:
        Cleans input data and returns a DataFrame.

    load_data(filepath: str) -> pd.DataFrame:
        Loads data from an Excel file.

    extract_post_id(df: pd.DataFrame) -> pd.DataFrame:
        Extracts post IDs from the 'URL ' column.

    save_data(df: pd.DataFrame, filepath: str):
        Saves a DataFrame to an Excel file.

    get_stackoverflow_posts(post_ids: list) -> list:
        Fetches Stack Overflow posts by their IDs.
"""

import os
import pandas as pd
import requests
from config import DEFAULT_DATA_FOLDER


def clean_input(filename: str, new_filename: str) -> pd.DataFrame:
    """
    Cleans input data and returns a DataFrame.

    Parameters:
        filename (str): Name of the input file.
        new_filename (str): Name of the new file to save cleaned data.

    Returns:
        pd.DataFrame: The cleaned data.
    """
    new_filepath = os.path.join(DEFAULT_DATA_FOLDER, new_filename)

    if os.path.exists(new_filepath):
        return load_data(new_filepath)

    file_path = os.path.join(DEFAULT_DATA_FOLDER, filename)
    df = pd.read_excel(file_path, header=0)

    df = extract_post_id(df)
    save_data(df, new_filepath)

    return df


def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads data from an Excel file.

    Parameters:
        filepath (str): Path to the Excel file.

    Returns:
        pd.DataFrame: The loaded data.
    """
    return pd.read_excel(filepath, header=0)


def extract_post_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts post IDs from the 'URL ' column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with 'postId' column.
    """
    df['postId'] = df['URL '].str.extract(r'questions/(\d+)', expand=False)
    return df


def save_data(df: pd.DataFrame, filepath: str):
    """
    Saves a DataFrame to an Excel file.

    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        filepath (str): Path to save the Excel file.
    """
    df.to_excel(filepath, index=False)


def get_stackoverflow_posts(post_ids):
    """
    Fetches Stack Overflow posts by their IDs.

    Parameters:
        post_ids (list): List of Stack Overflow post IDs.

    Returns:
        list: List of post details.
    """
    url = "https://api.stackexchange.com/2.3/questions/{}"
    ids = ";".join(map(str, post_ids))
    params = {
        'order': 'desc',
        'sort': 'activity',
        'site': 'stackoverflow'
    }

    response = requests.get(url.format(ids), params=params, timeout=10)

    if response.status_code == 200:
        data = response.json()
        return data.get('items', [])

    print(f"Error fetching data: {response.status_code}")
    return []


# # Example usage
# post_ids = [67890345, 67890487, 67890525]  # Replace with your list of post IDs
# posts = get_stackoverflow_posts(post_ids)
#
# # Print the details of the posts
# for post in posts:
#     print(f"Title: {post['title']}")
#     print(f"Link: {post['link']}")
#     print(f"Score: {post['score']}")
#     print(f"Creation Date: {post['creation_date']}")
#     print("Tags:", ", ".join(post['tags']))
#     print("-" * 80)
