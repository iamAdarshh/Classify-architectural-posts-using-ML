import os
import pandas as pd
import requests
from config import DEFAULT_DATA_FOLDER


def clean_input(filename: str, new_filename: str) -> pd.DataFrame:
    new_filepath = os.path.join(DEFAULT_DATA_FOLDER, new_filename)

    if os.path.exists(new_filepath):
        return load_data(new_filepath)

    file_path = os.path.join(DEFAULT_DATA_FOLDER, filename)
    df = pd.read_excel(file_path, header=0)

    df = extract_post_id(df)
    save_data(df, new_filepath)

    return df


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_excel(filepath, header=0)


def extract_post_id(df: pd.DataFrame) -> pd.DataFrame:
    df['postId'] = df['URL '].str.extract(r'questions/(\d+)', expand=False)
    return df


def save_data(df: pd.DataFrame, filepath: str):
    df.to_excel(filepath, index=False)


def get_stackoverflow_posts(post_ids):
    """
    Fetch details of Stack Overflow posts using their IDs.

    Parameters:
        post_ids (list): A list of Stack Overflow post IDs.

    Returns:
        list: A list of dictionaries containing post details.
    """
    # Stack Exchange API URL
    url = "https://api.stackexchange.com/2.3/questions/{}"

    # Join the list of post IDs into a comma-separated string
    ids = ";".join(map(str, post_ids))

    # Define the parameters for the API request
    params = {
        'order': 'desc',
        'sort': 'activity',
        'site': 'stackoverflow'
    }

    # Make the API request
    response = requests.get(url.format(ids), params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
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
