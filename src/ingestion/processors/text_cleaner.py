import re


class TextCleaner:
    """Cleans and normalizes text from financial documents."""

    def clean(self, text: str) -> str:
        # Normalize whitespace
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove common HTML artifacts
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)

        # Remove page numbers and headers/footers patterns
        text = re.sub(r"\n\s*Page \d+ of \d+\s*\n", "\n", text)
        text = re.sub(r"\n\s*-\s*\d+\s*-\s*\n", "\n", text)

        return text.strip()
