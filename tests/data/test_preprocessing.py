import pytest
from src.data.preprocessing import preprocess_text

def test_preprocess_text():
    text = "Python is a great programming language for data science!"
    expected = "python great program language data science"
    assert preprocess_text(text) == expected

def test_preprocess_text_empty():
    text = ""
    expected = ""
    assert preprocess_text(text) == expected

def test_preprocess_text_html():
    text = "<p>Object-oriented programming (OOP) is a programming paradigm.</p>"
    expected = "object-oriented program oop program paradigm"
    assert preprocess_text(text) == expected

def test_preprocess_text_contractions():
    text = "Machine learning algorithms can't be overlooked."
    expected = "machine learn algorithm overlook"
    assert preprocess_text(text) == expected

def test_preprocess_text_numbers():
    text = "There are 3 main types of machine learning: supervised, unsupervised, and reinforcement."
    expected = "main type machine learn supervise unsupervised reinforcement"
    assert preprocess_text(text) == expected

def test_preprocess_text_special_characters():
    text = "The Agile methodology emphasizes iterative development."
    expected = "agile methodology emphasizes iterative development"
    assert preprocess_text(text) == expected

def test_preprocess_text_stop_words():
    text = "This is a detailed guide to software design patterns."
    expected = "detailed guide software design pattern"
    assert preprocess_text(text) == expected

def test_preprocess_text_lemmatization():
    text = "Developers are writing unit tests for their codebases."
    expected = "developer write unit test codebases"
    assert preprocess_text(text) == expected

def test_preprocess_text_special_characters_extended():
    text = "C++ is a powerful language! #coding @developer"
    expected = "c powerful language cod developer"
    assert preprocess_text(text) == expected

def test_preprocess_text_mixed_case():
    text = "Test-Driven Development (TDD) is essential."
    expected = "test-driven development tdd essential"
    assert preprocess_text(text) == expected

def test_preprocess_text_non_ascii():
    text = "Déjà vu is a common feeling."
    expected = "deja vu common feel"
    assert preprocess_text(text) == expected

def test_preprocess_text_combined():
    text = "<html>Testing 1, 2, 3... Python's versatility & power!!!</html>"
    expected = "test python versatility power"
    assert preprocess_text(text) == expected

def test_preprocess_text_complex_html():
    text = """
    <html>
        <head>
            <title>Test Page</title>
        </head>
        <body>
            <h1>This is a heading</h1>
            <p>This is a <b>bold</b> paragraph.</p>
            <div>
                <p>Another paragraph with <a href="http://example.com">a link</a> and some <span>nested <b>bold</b> text</span>.</p>
            </div>
        </body>
    </html>
    """
    expected = "test page head bold paragraph another paragraph link nest bold text"
    assert preprocess_text(text) == expected
