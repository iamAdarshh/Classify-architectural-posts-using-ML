import os.path

import pandas as pd

from src.config import DEFAULT_DATA_FOLDER
from src.data.utils import extract_post_id, save_as_excel
from src.utils.posts_helpers import get_stackoverflow_posts

if __name__ == '__main__':
    filepath = os.path.join(f"{DEFAULT_DATA_FOLDER}/raw/Architectural Posts.xlsx")
    original_df = pd.read_excel(filepath)

    cleaned_df = extract_post_id(original_df)

    save_as_excel(cleaned_df, f"{DEFAULT_DATA_FOLDER}/processed/cleaned_architectural_posts.xlsx")

    total_posts_found = 0
    total_posts_not_found = 0
    posts: list = []

    for index,row in cleaned_df.iterrows():
        post_id  = row['postId']
        print("Fetching post", post_id)

        response = get_stackoverflow_posts(post_id)

        if response.status_code == 200:
            total_posts_found += 1
            items = response.json().get('items', [])

            for item in items:
                post = {
                    'id': item['id'],
                    'title': item['title'],
                    'score': item['score'],
                    'created_date': item['created_date'],
                    'tags': ','.join(item['tags']),

                }

            posts.append(response.json())
            break

        total_posts_not_found += 1

    print("Total posts found:", total_posts_found)
    print("Total posts not found:", total_posts_not_found)

    architectural_posts_df = pd.DataFrame(posts)
    save_as_excel(architectural_posts_df, f"{DEFAULT_DATA_FOLDER}/processed/architectural_posts_details.xlsx")
    #print(posts[0])
    print("saved file")