import os.path

import pandas as pd

from src.config import DEFAULT_DATA_FOLDER
from src.data.utils import extract_post_id, save_as_excel
from src.utils.posts_helpers import fetch_posts_in_batches



if __name__ == '__main__':
    filepath = os.path.join(f"{DEFAULT_DATA_FOLDER}/raw/Architectural Posts.xlsx")
    original_df = pd.read_excel(filepath)

    cleaned_df = extract_post_id(original_df)

    save_as_excel(cleaned_df, f"{DEFAULT_DATA_FOLDER}/processed/cleaned_architectural_posts.xlsx")

    posts: list = fetch_posts_in_batches(cleaned_df['postId'].to_list())

    architectural_posts_df = pd.DataFrame(posts)
    save_as_excel(architectural_posts_df, f"{DEFAULT_DATA_FOLDER}/processed/architectural_posts_details.xlsx")
    print("saved file")