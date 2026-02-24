from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SearchResult:
    content: str
    metadata: dict[str, str | int | float | bool]
    source: str
    chunk_id: str
    score: float


class BaseVectorStore(ABC):
    @abstractmethod
    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str | int | float | bool]],
    ) -> None:
        """Add documents with embeddings to the store."""
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return total number of documents in the store."""
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        ...
