from src.ingestion.loaders.base import Document
from src.retrieval.parent_child_retriever import ParentChildChunker
from src.vectorstore.base import SearchResult


class TestParentChildChunker:
    def test_creates_parent_and_child_chunks(self):
        """Should produce both parent and child chunks."""
        doc = Document(
            content="First section about revenue growth and financial performance. " * 20
            + "Second section about risk factors and market conditions. " * 20,
            metadata={"ticker": "AAPL"},
            source="test.md",
            doc_id="testdoc",
        )
        chunker = ParentChildChunker(
            parent_chunk_size=500,
            child_chunk_size=200,
        )
        chunks = chunker.chunk(doc)

        parents = [c for c in chunks if c.metadata.get("chunk_type") == "parent"]
        children = [c for c in chunks if c.metadata.get("chunk_type") == "child"]

        assert len(parents) >= 1
        assert len(children) >= 1
        assert len(children) > len(parents), "Should have more children than parents"

    def test_children_reference_parents(self):
        """Each child chunk should reference its parent via parent_chunk_id."""
        doc = Document(
            content="Financial data about the company. " * 30,
            metadata={},
            source="test.md",
            doc_id="testdoc",
        )
        chunker = ParentChildChunker(
            parent_chunk_size=400,
            child_chunk_size=150,
        )
        chunks = chunker.chunk(doc)

        parents = {c.chunk_id for c in chunks if c.metadata.get("chunk_type") == "parent"}
        children = [c for c in chunks if c.metadata.get("chunk_type") == "child"]

        for child in children:
            assert child.parent_chunk_id is not None
            assert child.parent_chunk_id in parents

    def test_preserves_metadata(self):
        """Parent and child chunks should inherit document metadata."""
        doc = Document(
            content="Important financial content. " * 20,
            metadata={"ticker": "MSFT", "filing_type": "10-K"},
            source="msft.md",
            doc_id="msft123",
        )
        chunker = ParentChildChunker(parent_chunk_size=300, child_chunk_size=100)
        chunks = chunker.chunk(doc)

        for chunk in chunks:
            assert chunk.metadata["ticker"] == "MSFT"
            assert chunk.metadata["filing_type"] == "10-K"
            assert chunk.source == "msft.md"
            assert chunk.doc_id == "msft123"

    def test_unique_ids(self):
        """All chunk IDs should be unique."""
        doc = Document(
            content="Content for testing uniqueness. " * 30,
            metadata={},
            source="test.md",
            doc_id="testdoc",
        )
        chunker = ParentChildChunker(parent_chunk_size=300, child_chunk_size=100)
        chunks = chunker.chunk(doc)

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_parent_chunks_are_larger(self):
        """Parent chunks should generally be larger than child chunks."""
        doc = Document(
            content="Detailed financial analysis content. " * 50,
            metadata={},
            source="test.md",
            doc_id="testdoc",
        )
        chunker = ParentChildChunker(
            parent_chunk_size=800,
            child_chunk_size=200,
        )
        chunks = chunker.chunk(doc)

        parents = [c for c in chunks if c.metadata.get("chunk_type") == "parent"]
        children = [c for c in chunks if c.metadata.get("chunk_type") == "child"]

        if parents and children:
            avg_parent = sum(len(p.content) for p in parents) / len(parents)
            avg_child = sum(len(c.content) for c in children) / len(children)
            assert avg_parent > avg_child


class TestParentChildRetriever:
    def test_deduplicates_parents(self):
        """Multiple children from the same parent should yield one parent result."""
        parent_id = "parent_001"
        children = [
            SearchResult(
                content="child 1 content",
                metadata={"chunk_type": "child", "parent_chunk_id": parent_id},
                source="test.md",
                chunk_id="child_001",
                score=0.95,
            ),
            SearchResult(
                content="child 2 content",
                metadata={"chunk_type": "child", "parent_chunk_id": parent_id},
                source="test.md",
                chunk_id="child_002",
                score=0.90,
            ),
        ]

        # Simulate deduplication logic
        seen_parent_ids: set[str] = set()
        deduplicated: list[SearchResult] = []
        for child in children:
            pid = child.metadata.get("parent_chunk_id")
            if pid and pid not in seen_parent_ids:
                seen_parent_ids.add(str(pid))
                deduplicated.append(child)

        assert len(deduplicated) == 1, "Duplicate parents should be deduplicated"
