"""
This is the main module script.

It prints "Hello world" when executed.
"""
import pandas as pd
from src.config import DEFAULT_DATA_FOLDER


if __name__ == '__main__':
    cleaned_architectural_posts = pd.read_excel(
        f"{DEFAULT_DATA_FOLDER}/processed/cleaned_architectural_posts.xlsx")
