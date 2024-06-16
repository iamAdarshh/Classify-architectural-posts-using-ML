import requests



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
