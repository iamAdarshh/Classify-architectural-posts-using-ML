import re
import nltk
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import string
import contractions
from bs4 import BeautifulSoup

# Download necessary NLTK resources at the start of your script
nltk_resources = ['stopwords', 'wordnet', 'punkt', 'averaged_perceptron_tagger']

for resource in nltk_resources:
    nltk.download(resource, quiet=True)

def normalize_text(text: str) -> str:
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def expand_contractions(text: str) -> str:
    return contractions.fix(text)

def to_lowercase(text: str) -> str:
    return text.lower()

def remove_special_characters(text: str) -> str:
    text = re.sub(r'\s*-\s*', '-', text)  # Ensure hyphens are properly spaced
    text = text.strip('-')
    return re.sub(r'[^a-zA-Z0-9\s-]', '', text)

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_text(words):
    lemmatizer = WordNetLemmatizer()
    pos_tags = nltk.pos_tag(words)
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word, tag in pos_tags]
    return lemmatized_words

def remove_stop_words(words: list) -> list:
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return filtered_words

def remove_punctuation(text: str) -> str:
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in string.punctuation]
    return ' '.join(words)

def remove_numbers(text: str) -> str:
    return re.sub(r'\d+', '', text)

def remove_extra_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def remove_html_tags(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def preprocess_text(text: str) -> str:
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

    return final_text
