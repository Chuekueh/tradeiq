import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.chunkers.base import BaseChunker, Chunk
from src.ingestion.loaders.base import Document


class RecursiveChunker(BaseChunker):
    """Splits documents using recursive character splitting, tuned for financial documents."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n## ",  # Markdown H2 (section headers in filings)
                "\n### ",  # Markdown H3
                "\n\n",  # Paragraph breaks
                "\n",  # Line breaks
                ". ",  # Sentences
                " ",  # Words
            ],
            keep_separator=True,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        texts = self._splitter.split_text(document.content)
        chunks: list[Chunk] = []

        for i, text in enumerate(texts):
            chunk_id = hashlib.sha256(
                f"{document.doc_id}:{i}:{text[:100]}".encode()
            ).hexdigest()[:16]

            chunks.append(
                Chunk(
                    content=text,
                    metadata={**document.metadata, "chunk_index": i, "total_chunks": len(texts)},
                    source=document.source,
                    chunk_id=chunk_id,
                    chunk_index=i,
                    doc_id=document.doc_id,
                )
            )

        return chunks
