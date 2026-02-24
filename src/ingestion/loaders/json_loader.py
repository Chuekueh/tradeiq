import json
from pathlib import Path

from src.ingestion.loaders.base import BaseLoader, Document


class JSONLoader(BaseLoader):
    """Loads structured JSON documents (e.g., SEC filing metadata + content)."""

    def __init__(self, content_key: str = "content", metadata_keys: list[str] | None = None):
        self._content_key = content_key
        self._metadata_keys = metadata_keys

    def supports(self, path: str) -> bool:
        return Path(path).suffix.lower() == ".json"

    def load(self, path: str) -> list[Document]:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        raw = json.loads(file_path.read_text(encoding="utf-8"))
        items = raw if isinstance(raw, list) else [raw]
        documents: list[Document] = []

        for item in items:
            content = item.get(self._content_key, "")
            if not content:
                continue

            metadata: dict[str, str | int | float | bool] = {
                "file_name": file_path.name,
                "file_type": "json",
            }
            keys = self._metadata_keys or [k for k in item if k != self._content_key]
            for key in keys:
                if key in item and isinstance(item[key], (str, int, float, bool)):
                    metadata[key] = item[key]

            documents.append(Document(content=content, metadata=metadata, source=str(file_path)))

        return documents
