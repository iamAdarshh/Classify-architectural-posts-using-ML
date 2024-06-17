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

from typing import List, Dict, Tuple, Optional
import time
import requests

POST_BASE_URL = "https://api.stackexchange.com/2.3/questions/{}"
ANSWER_BASE_URL = "https://api.stackexchange.com/2.2/questions/{}/answers"
PARAMS = {
    "site": "stackoverflow",
    "key": "rl_cMc4Gvbc6zeCSRtpyw18cVHQK",
    "filter": "withbody"  # Include the post body in the response
}


def fetch_post_answers(question_id: str, accepted_answer_id: Optional[str]) -> Optional[Dict]:
    """Fetch answers for a given Stack Overflow question."""
    response = requests.get(ANSWER_BASE_URL.format(question_id), params=PARAMS, timeout=30)

    if response.status_code == 200:
        items = response.json().get("items", [])

        if not items:
            print(f"No answers found for question {question_id}")
            return None

        selected_item = None
        if accepted_answer_id is not None:
            for item in items:
                if item.get('answer_id') == accepted_answer_id:
                    selected_item = item
                    break
        else:
            selected_item = max(items, key=lambda x: x.get("score", 0))

        if selected_item is None:
            # That means this question has not been answered
            return None

        return {
            'question_id': question_id,
            'answer_id': selected_item.get('answer_id'),
            'body': selected_item.get('body', ''),
            'is_accepted': selected_item.get('is_accepted', False),
            'score': selected_item.get('score', 0),
        }

    time.sleep(5)
    print(f"Failed to find answers for question {question_id}")
    return None


def fetch_posts_in_batches(post_ids: List[int], batch_size: int = 30)\
        -> Tuple[List[Dict], List[Dict]]:
    """
    Fetches Stack Overflow posts in batches.

    Args:
        post_ids (List[int]): List of Stack Overflow post IDs.
        batch_size (int, optional): Number of post IDs to fetch in each batch. Defaults to 50.

    Returns:
        Tuple[List[Dict], List[int], List[Dict]]:
        A tuple containing a list of dictionaries with post details,
        a list of post IDs not found, and a list of dictionaries with answer details.
    """
    batches = [post_ids[i:i + batch_size] for i in range(0, len(post_ids), batch_size)]
    print(f"Total batches: {len(batches)}")
    all_posts = []
    answers = []

    for index, batch in enumerate(batches):
        print(f"Executing batch: {index}")
        response = get_stackoverflow_posts(batch)

        if response.status_code == 200:
            items = response.json().get('items', [])
            for item in items:
                question_id = item.get('question_id')
                accepted_answer_id = item.get('accepted_answer_id', None)
                post = {
                    'id': question_id,
                    'title': item.get('title', ''),
                    'body': item.get('body', ''),
                    'is_answered': item.get('is_answered', False),
                    'view_count': item.get('view_count', 0),
                    'accepted_answer_id': accepted_answer_id,
                    'answer_count': item.get('answer_count', 0),
                    'score': item.get('score', 0),
                    'tags': ','.join(item.get('tags', [])),
                }

                answer = fetch_post_answers(question_id, accepted_answer_id)
                if answer is not None:
                    answers.append(answer)

                all_posts.append(post)
        else:
            print(f"Failed to fetch posts for batch: {batch}")
            print("Response: ", response.text)

        # Adding 5 seconds delay to avoid getting IP blocked
        time.sleep(5)
        print(f"Remaining Quota: {response.json().get('quota_remaining', 0)}")
        print()

    return all_posts, answers


def get_stackoverflow_posts(post_ids: List[int]) -> requests.Response:
    """
    Fetches Stack Overflow posts by their IDs.

    Args:
        post_ids (List[int]): List of Stack Overflow post IDs.

    Returns:
        requests.Response: The response object containing post details.
    """
    ids = ";".join(map(str, post_ids))
    return requests.get(POST_BASE_URL.format(ids), params=PARAMS, timeout=30)
