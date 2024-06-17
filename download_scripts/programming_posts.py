from striprtf.striprtf import rtf_to_text

from src.config import DEFAULT_DATA_FOLDER
from src.utils.posts_helpers import fetch_posts_in_batches


def read_rtf_with_striprtf(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        rtf_content = file.read()
        text = rtf_to_text(rtf_content)
    return text


def process_text_to_int_list(text):
    # Split the text by commas
    string_list = text.split(',')
    # Strip any whitespace and convert to integers
    int_list = [int(item.strip()) for item in string_list]
    return int_list


if __name__ == '__main__':
    file_path = f'{DEFAULT_DATA_FOLDER}/raw/Programming posts.rtf'
    text = read_rtf_with_striprtf(file_path)
    post_ids = process_text_to_int_list(text)

    posts = fetch_posts_in_batches(post_ids, 100)
    print(post_ids)
