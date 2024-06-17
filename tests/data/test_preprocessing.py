"""
This module contains tests for the text preprocessing functions in the preprocessing.py module.
"""

from src.data.preprocessing import preprocess_text


def test_preprocess_text():
    """Test preprocessing of a general sentence."""
    text = "Python is a great programming language for data science!"
    expected = "python great program language data science"
    assert preprocess_text(text) == expected


def test_preprocess_text_empty():
    """Test preprocessing of an empty string."""
    text = ""
    expected = ""
    assert preprocess_text(text) == expected


def test_preprocess_text_html():
    """Test preprocessing of a string containing HTML tags."""
    text = "<p>Object-oriented programming (OOP) is a programming paradigm.</p>"
    expected = "object-oriented program oop program paradigm"
    assert preprocess_text(text) == expected


def test_preprocess_text_contractions():
    """Test preprocessing of a string with contractions."""
    text = "Machine learning algorithms can't be overlooked."
    expected = "machine learn algorithm overlook"
    assert preprocess_text(text) == expected


def test_preprocess_text_numbers():
    """Test preprocessing of a string with numbers."""
    text = (
        "There are 3 main types of machine learning: supervised, unsupervised, and reinforcement."
    )
    expected = "main type machine learn supervise unsupervised reinforcement"
    assert preprocess_text(text) == expected


def test_preprocess_text_special_characters():
    """Test preprocessing of a string with special characters."""
    text = "The Agile methodology emphasizes iterative development."
    expected = "agile methodology emphasizes iterative development"
    assert preprocess_text(text) == expected


def test_preprocess_text_stop_words():
    """Test preprocessing of a string with stop words."""
    text = "This is a detailed guide to software design patterns."
    expected = "detailed guide software design pattern"
    assert preprocess_text(text) == expected


def test_preprocess_text_lemmatization():
    """Test preprocessing of a string requiring lemmatization."""
    text = "Developers are writing unit tests for their codebases."
    expected = "developer write unit test codebases"
    assert preprocess_text(text) == expected


def test_preprocess_text_special_characters_extended():
    """Test preprocessing of a string with extended special characters."""
    text = "C++ is a powerful language! #coding @developer"
    expected = "c powerful language cod developer"
    assert preprocess_text(text) == expected


def test_preprocess_text_mixed_case():
    """Test preprocessing of a string with mixed case."""
    text = "Test-Driven Development (TDD) is essential."
    expected = "test-driven development tdd essential"
    assert preprocess_text(text) == expected


def test_preprocess_text_non_ascii():
    """Test preprocessing of a string with non-ASCII characters."""
    text = "Déjà vu is a common feeling."
    expected = "deja vu common feel"
    assert preprocess_text(text) == expected


def test_preprocess_text_combined():
    """Test preprocessing of a string with combined edge cases."""
    text = "<html>Testing 1, 2, 3... Python's versatility & power!!!</html>"
    expected = "test python versatility power"
    assert preprocess_text(text) == expected


def test_preprocess_text_complex_html():
    """Test preprocessing of a complex HTML string."""
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
    expected = (
        "test page head bold paragraph another paragraph link nest bold text"
    )
    assert preprocess_text(text) == expected
