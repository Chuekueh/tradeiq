from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Document:
    content: str
    metadata: dict[str, str | int | float | bool]
    source: str
    doc_id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.doc_id:
            import hashlib

            self.doc_id = hashlib.sha256(
                f"{self.source}:{self.content[:200]}".encode()
            ).hexdigest()[:16]


class BaseLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> list[Document]:
        """Load documents from a given path."""
        ...

    @abstractmethod
    def supports(self, path: str) -> bool:
        """Check if this loader supports the given file type."""
        ...
