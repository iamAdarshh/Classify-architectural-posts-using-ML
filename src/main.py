"""
This script processes an Excel file containing post data, cleans the data,
extracts post IDs, and fetches details of Stack Overflow posts using those IDs.

Functions imported from posts_helpers:
    clean_input(filename: str, new_filename: str) -> pd.DataFrame:
        Cleans input data and returns a DataFrame.
    get_stackoverflow_posts(post_ids: list) -> list:
        Fetches Stack Overflow posts by their IDs.

Usage:
    Run this script to process the input Excel file and fetch post details.
"""

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Hello world")