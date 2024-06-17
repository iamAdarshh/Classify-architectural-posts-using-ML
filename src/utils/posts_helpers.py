"""
This module provides functionality to fetch Stack Overflow posts in batches.

It contains the following functions:
1. fetch_posts_in_batches: Fetches Stack Overflow posts in batches.
2. get_stackoverflow_posts: Fetches Stack Overflow posts by their IDs.

The main steps include:
1. Dividing the post IDs into batches.
2. Fetching the posts for each batch.
3. Returning the combined list of posts.
"""
from typing import List, Dict
import time
import requests


def fetch_posts_in_batches(post_ids: List[int], batch_size: int = 30) -> (List[Dict], list[int]):
    """
    Fetches Stack Overflow posts in batches.

    Args:
        post_ids (List[int]): List of Stack Overflow post IDs.
        batch_size (int, optional): Number of post IDs to fetch in each batch. Defaults to 50.

    Returns:
        List[Dict]: A list of dictionaries containing post details.
    """
    batches = [post_ids[i:i + batch_size] for i in range(0, len(post_ids), batch_size)]
    all_posts = []
    not_found_ids = []

    for batch in batches:
        response = get_stackoverflow_posts(batch)

        if response.status_code == 200:
            items = response.json().get('items', [])
            for item in items:
                post = {
                    'id': item.get('question_id'),
                    'title': item.get('title', ''),
                    'body': item.get('body', ''),
                    'is_answered': item.get('is_answered', False),
                    'view_count': item.get('view_count', 0),
                    'accepted_answer_id': item.get('accepted_answer_id', None),
                    'answer_count': item.get('answer_count', 0),
                    'score': item.get('score', 0),
                    'tags': ','.join(item.get('tags', [])),
                }
                all_posts.append(post)
            batch_ids = [post['id'] for post in all_posts]
            not_found_ids = list(set(post_ids) - set(batch_ids))
        else:
            not_found_ids.extend(batch)
            print(f"Failed to fetch posts for batch: {batch}")
            print("Response: ", response.text)

        # adding 30 seconds delay to avoid getting ip blocked
        time.sleep(30)

    return all_posts, not_found_ids


def get_stackoverflow_posts(post_ids: List[int]) -> requests.Response:
    """
    Fetches Stack Overflow posts by their IDs.

    Args:
        post_ids (List[int]): List of Stack Overflow post IDs.

    Returns:
        requests.Response: The response object containing post details.
    """
    url = "https://api.stackexchange.com/2.3/questions/{}"
    ids = ";".join(map(str, post_ids))
    params = {
        'order': 'desc',
        'sort': 'activity',
        'site': 'stackoverflow',
        'filter': 'withbody'
    }

    return requests.get(url.format(ids), params=params, timeout=10)
