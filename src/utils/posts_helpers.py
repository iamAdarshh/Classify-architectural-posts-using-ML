import requests


def fetch_posts_in_batches(post_ids: list, batch_size=50):
    batches = [post_ids[i:i + batch_size] for i in range(0, len(post_ids), batch_size)]
    all_posts = []

    for batch in batches:

        response = get_stackoverflow_posts(batch)

        if response.status_code == 200:
            items = response.json().get('items', [])
            for item in items:
                post = {
                    'id': item['post_id'],  # use 'post_id' instead of 'id'
                    'title': item['title'],
                    'score': item['score'],
                    'created_date': item['creation_date'],  # use 'creation_date' instead of 'created_date'
                    'tags': ','.join(item['tags']),
                }
                all_posts.append(post)
        else:
            print(f"Failed to fetch posts for batch: {batch}")
            print("Response: ", response.text)

    return all_posts



def get_stackoverflow_posts(post_ids) -> requests.Response:
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

    return requests.get(url.format(ids), params=params, timeout=10)
