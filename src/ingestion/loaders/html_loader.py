from pathlib import Path

from bs4 import BeautifulSoup

from src.ingestion.loaders.base import BaseLoader, Document


class HTMLLoader(BaseLoader):
    """Loads HTML files and extracts text content."""

    def supports(self, path: str) -> bool:
        return Path(path).suffix.lower() in (".html", ".htm")

    def load(self, path: str) -> list[Document]:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        raw_html = file_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(raw_html, "lxml")

        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        metadata = {"file_name": file_path.name, "file_type": "html"}

        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            metadata["title"] = title_tag.string.strip()

        return [Document(content=text, metadata=metadata, source=str(file_path))]
