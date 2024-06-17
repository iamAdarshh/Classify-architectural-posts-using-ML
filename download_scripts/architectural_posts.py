"""
This script processes architectural posts data by performing the following steps:
1. Reads the original data from an Excel file.
2. Cleans the data by extracting post IDs.
3. Saves the cleaned data to a new Excel file.
4. Fetches additional post details in batches using the post IDs.
5. Saves the fetched post details to another Excel file.
"""

import os
import pandas as pd

from src.config import DEFAULT_DATA_FOLDER
from src.data.utils import extract_post_id, save_as_excel
from src.utils.posts_helpers import fetch_posts_in_batches


def main():
    """
    Main function to process architectural posts data.

    This script performs the following steps:
    1. Reads the original data from an Excel file.
    2. Cleans the data by extracting post IDs.
    3. Saves the cleaned data to a new Excel file.
    4. Fetches additional post details in batches using the post IDs.
    5. Saves the fetched post details to another Excel file.
    """
    # Read the original data from an Excel file
    filepath = os.path.join(DEFAULT_DATA_FOLDER, 'raw', 'Architectural Posts.xlsx')
    original_df = pd.read_excel(filepath)

    # Clean the data by extracting post IDs
    cleaned_df = extract_post_id(original_df)

    # Save the cleaned data to a new Excel file
    cleaned_filepath = os.path.join(DEFAULT_DATA_FOLDER,
                                    'processed',
                                    'cleaned_architectural_posts.xlsx')
    save_as_excel(cleaned_df, cleaned_filepath)

    # Fetch additional post details in batches using the post IDs
    post_ids = cleaned_df['postId'].to_list()
    posts, not_found_ids = fetch_posts_in_batches(post_ids)
    print("Total not found posts: ", len(not_found_ids))

    # Convert the fetched post details to a DataFrame
    architectural_posts_df = pd.DataFrame(posts)

    # Save the fetched post details to another Excel file
    new_filepath = os.path.join(DEFAULT_DATA_FOLDER,
                                'processed',
                                'architectural_posts_details.xlsx')
    save_as_excel(architectural_posts_df, new_filepath)
    print("Saved file")


if __name__ == '__main__':
    main()
