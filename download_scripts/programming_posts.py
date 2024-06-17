"""
This script processes programming post IDs from an RTF file by performing the following steps:
1. Reads and extracts text from the RTF file.
2. Processes the extracted text to obtain a list of integers.
3. Fetches post details in batches using the post IDs.
4. Saves the post IDs and fetched details to Excel files.
"""

import pandas as pd

from striprtf.striprtf import rtf_to_text
from src.config import DEFAULT_DATA_FOLDER
from src.data.utils import save_as_excel
from src.utils.posts_helpers import fetch_posts_in_batches


def read_rtf_with_striprtf(path: str) -> str:
    """
    Reads an RTF file and converts its content to plain text.

    Args:
        path (str): The path to the RTF file.

    Returns:
        str: The plain text extracted from the RTF file.
    """
    with open(path, 'r', encoding='utf-8') as file:
        rtf_content = file.read()
        text = rtf_to_text(rtf_content)
    return text


def process_text_to_int_list(text: str) -> list[int]:
    """
    Processes a string of comma-separated numbers into a list of integers.

    Args:
        text (str): The string containing comma-separated numbers.

    Returns:
        List[int]: A list of integers.
    """
    string_list = text.split(',')
    int_list = [int(item.strip()) for item in string_list]
    return int_list


if __name__ == '__main__':
    file_path = f'{DEFAULT_DATA_FOLDER}/raw/Programming posts.rtf'
    raw_text = read_rtf_with_striprtf(file_path)
    post_ids = process_text_to_int_list(raw_text)

    post_ids_df = pd.DataFrame(post_ids, columns=['post_id'])
    save_as_excel(post_ids_df, f"{DEFAULT_DATA_FOLDER}/processed/programming_posts_keys.xlsx")

    posts = fetch_posts_in_batches(post_ids, 100)

    posts_df = pd.DataFrame(posts)
    save_as_excel(posts_df, f"{DEFAULT_DATA_FOLDER}/processed/programming_posts.xlsx")
