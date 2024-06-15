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

from posts_helpers import clean_input, get_stackoverflow_posts

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    FILENAME = "Architectural Posts.xlsx"
    NEW_FILENAME = "cleaned architectural posts.xlsx"

    df = clean_input(FILENAME, NEW_FILENAME)

    posts = get_stackoverflow_posts(df['postId'].head(10))

    for post in posts:
        print(post)
