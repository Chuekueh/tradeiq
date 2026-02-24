from pathlib import Path

from src.ingestion.loaders.base import BaseLoader, Document


class MarkdownLoader(BaseLoader):
    def supports(self, path: str) -> bool:
        return Path(path).suffix.lower() in (".md", ".markdown")

    def load(self, path: str) -> list[Document]:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = file_path.read_text(encoding="utf-8")
        metadata = {"file_name": file_path.name, "file_type": "markdown"}

        return [Document(content=content, metadata=metadata, source=str(file_path))]
