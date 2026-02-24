import numpy as np
import pytest

from src.ingestion.chunkers.semantic_chunker import SemanticChunker
from src.ingestion.loaders.base import Document


@pytest.fixture
def sample_document():
    """Create a document with distinct semantic sections."""
    content = (
        "Apple Inc. reported record revenue of $94.9 billion for Q3 2024. "
        "The iPhone segment contributed $42.3 billion to total revenue. "
        "Mac revenue reached $7.7 billion, up 2% year over year. "
        "The company announced a new AI initiative called Apple Intelligence. "
        "This platform integrates generative AI across all Apple devices. "
        "Apple Intelligence will be available in iOS 18 and macOS Sequoia. "
        "In other news, Apple opened three new retail stores in Asia. "
        "The stores are located in Shanghai, Mumbai, and Kuala Lumpur. "
        "Each store features Apple's latest architectural design."
    )
    return Document(
        content=content,
        metadata={"ticker": "AAPL", "filing_type": "10-Q"},
        source="test_doc.md",
        doc_id="test123",
    )


@pytest.fixture
def short_document():
    """Create a document with too few sentences for semantic chunking."""
    return Document(
        content="Short document. Only two sentences.",
        metadata={},
        source="short.md",
        doc_id="short123",
    )


class TestSemanticChunker:
    def test_chunks_at_semantic_boundaries(self, sample_document):
        """Semantic chunker should split at topic transitions."""
        chunker = SemanticChunker(breakpoint_percentile_threshold=25)
        chunks = chunker.chunk(sample_document)

        assert len(chunks) >= 2, "Should split into at least 2 semantic groups"
        # Verify all content is preserved
        combined = " ".join(c.content for c in chunks)
        for sentence in ["Apple Inc.", "Apple Intelligence", "retail stores"]:
            assert sentence in combined

    def test_preserves_metadata(self, sample_document):
        """Chunker should carry forward document metadata."""
        chunker = SemanticChunker(breakpoint_percentile_threshold=25)
        chunks = chunker.chunk(sample_document)

        for chunk in chunks:
            assert chunk.metadata["ticker"] == "AAPL"
            assert chunk.metadata["filing_type"] == "10-Q"
            assert chunk.metadata["chunking_strategy"] == "semantic"
            assert "chunk_index" in chunk.metadata
            assert "total_chunks" in chunk.metadata
            assert chunk.doc_id == "test123"
            assert chunk.source == "test_doc.md"

    def test_fallback_for_short_documents(self, short_document):
        """Short documents should fall back to recursive chunking."""
        chunker = SemanticChunker(min_sentences_for_semantic=3)
        chunks = chunker.chunk(short_document)

        assert len(chunks) >= 1
        assert chunks[0].metadata["chunking_strategy"] == "recursive_fallback"

    def test_chunk_ids_are_unique(self, sample_document):
        """Each chunk should have a unique ID."""
        chunker = SemanticChunker(breakpoint_percentile_threshold=25)
        chunks = chunker.chunk(sample_document)

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_indices_are_sequential(self, sample_document):
        """Chunk indices should be sequential starting from 0."""
        chunker = SemanticChunker(breakpoint_percentile_threshold=25)
        chunks = chunker.chunk(sample_document)

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_large_chunks_are_split(self):
        """Chunks exceeding max_chunk_size should be split with fallback."""
        # Create a document with one very long semantic section
        long_section = "This is a detailed financial analysis. " * 100
        short_section = "The company is headquartered in Cupertino, California."
        doc = Document(
            content=f"{long_section} {short_section}",
            metadata={},
            source="long.md",
            doc_id="long123",
        )
        chunker = SemanticChunker(
            breakpoint_percentile_threshold=25,
            max_chunk_size=500,
        )
        chunks = chunker.chunk(doc)

        # Large section should be split into multiple sub-chunks
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.content) <= 1500  # Reasonable upper bound

    def test_cosine_similarity_computation(self):
        """Test the static cosine similarity method."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert SemanticChunker._cosine_similarity(a, b) == pytest.approx(1.0)

        c = np.array([0.0, 1.0, 0.0])
        assert SemanticChunker._cosine_similarity(a, c) == pytest.approx(0.0)

        zero = np.array([0.0, 0.0, 0.0])
        assert SemanticChunker._cosine_similarity(a, zero) == 0.0

    def test_group_sentences(self):
        """Test sentence grouping by breakpoints."""
        sentences = ["A", "B", "C", "D", "E"]
        groups = SemanticChunker._group_sentences(sentences, [2, 4])
        assert groups == [["A", "B"], ["C", "D"], ["E"]]

    def test_group_sentences_no_breakpoints(self):
        """No breakpoints means all sentences in one group."""
        sentences = ["A", "B", "C"]
        groups = SemanticChunker._group_sentences(sentences, [])
        assert groups == [["A", "B", "C"]]
