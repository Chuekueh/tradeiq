from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.ingestion.loaders.base import Document


@dataclass
class Chunk:
    content: str
    metadata: dict[str, str | int | float | bool]
    source: str
    chunk_id: str
    chunk_index: int
    doc_id: str
    parent_chunk_id: str | None = None


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks."""
        ...
