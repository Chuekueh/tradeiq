from src.ingestion.processors.text_cleaner import TextCleaner


def test_normalizes_whitespace():
    cleaner = TextCleaner()
    result = cleaner.clean("hello   world\r\n\r\ntest")
    assert "   " not in result
    assert "\r\n" not in result


def test_removes_html_entities():
    cleaner = TextCleaner()
    result = cleaner.clean("AT&amp;T &lt;strong&gt;")
    assert "&amp;" not in result
    assert "AT&T" in result


def test_removes_page_numbers():
    cleaner = TextCleaner()
    result = cleaner.clean("Some content\n Page 5 of 120 \nMore content")
    assert "Page 5 of 120" not in result


def test_strips_whitespace():
    cleaner = TextCleaner()
    result = cleaner.clean("  content  ")
    assert result == "content"
