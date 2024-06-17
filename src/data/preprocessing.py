"""
This module provides functions for preprocessing text data.
It includes normalization, contraction expansion, lowercasing, 
special character removal, lemmatization, stop word removal, 
punctuation removal, number removal, whitespace removal, 
and HTML tag removal.
"""

import re
import unicodedata
import string
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import contractions

# Download necessary NLTK resources at the start of your script
nltk_resources = ['stopwords', 'wordnet',
                  'punkt', 'averaged_perceptron_tagger']
for resource in nltk_resources:
    nltk.download(resource, quiet=True)


def normalize_text(text: str) -> str:
    """Normalize unicode text to ASCII."""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def expand_contractions(text: str) -> str:
    """Expand contractions in text."""
    return contractions.fix(text)


def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def remove_special_characters(text: str) -> str:
    """Remove special characters from text, but keep hyphens within words."""
    text = re.sub(r'\s*-\s*', '-', text)  # Ensure hyphens are properly spaced
    text = text.strip('-')
    return re.sub(r'[^a-zA-Z0-9\s-]', '', text)


def get_wordnet_pos(word: str) -> str:
    """Map POS tag to first character lemmatize() accepts."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN,
                "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_text(words):
    """Lemmatize words in text using POS tagging."""
    lemmatizer = WordNetLemmatizer()
    pos_tags = nltk.pos_tag(words)
    return [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word, tag in pos_tags]


def remove_stop_words(words: list) -> list:
    """Remove stop words from text."""
    stop_words = set(stopwords.words('english'))
    return [word for word in words if word.lower() not in stop_words]


def remove_punctuation(text: str) -> str:
    """Remove punctuation from text."""
    words = nltk.word_tokenize(text)
    return ' '.join(word for word in words if word not in string.punctuation)


def remove_numbers(text: str) -> str:
    """Remove numbers from text."""
    return re.sub(r'\d+', '', text)


def remove_extra_whitespace(text: str) -> str:
    """Remove extra whitespace from text."""
    return re.sub(r'\s+', ' ', text).strip()


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text."""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def preprocess_text(text: str|None) -> str|None:
    """Preprocess text by applying a series of text normalization steps."""

    try:
        text_no_html = remove_html_tags(text)
        normalized_text = normalize_text(text_no_html)
        expanded_text = expand_contractions(normalized_text)
        lowercase_text = to_lowercase(expanded_text)
        no_special_chars_text = remove_special_characters(lowercase_text)
        no_numbers_text = remove_numbers(no_special_chars_text)
        no_punctuation_text = remove_punctuation(no_numbers_text)
        words = nltk.word_tokenize(no_punctuation_text)
        lemmatized_words = lemmatize_text(words)
        no_stop_words_text = ' '.join(remove_stop_words(lemmatized_words))
        final_text = remove_extra_whitespace(no_stop_words_text)
    except Exception as e:
        print("Exception", e)
        print('Text: ', text)
        print()

    return final_text
