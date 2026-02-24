from src.ingestion.chunkers.recursive_chunker import RecursiveChunker
from src.ingestion.loaders.base import Document


def test_chunker_splits_document():
    doc = Document(
        content="This is a test document. " * 100,
        metadata={"file_name": "test.md"},
        source="test.md",
    )
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 1
    assert all(chunk.doc_id == doc.doc_id for chunk in chunks)
    assert all(chunk.chunk_index == i for i, chunk in enumerate(chunks))


def test_chunker_preserves_metadata():
    doc = Document(
        content="Short document content.",
        metadata={"ticker": "AAPL", "filing_type": "10-K"},
        source="test.md",
    )
    chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=0)
    chunks = chunker.chunk(doc)

    assert len(chunks) == 1
    assert chunks[0].metadata["ticker"] == "AAPL"
    assert chunks[0].metadata["filing_type"] == "10-K"


def test_chunker_empty_document():
    doc = Document(content="", metadata={}, source="empty.md")
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(doc)

    assert len(chunks) == 0
