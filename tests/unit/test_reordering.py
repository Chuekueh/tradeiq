
from src.generation.chain import RAGChain
from src.vectorstore.base import SearchResult


def _make_result(score: float, label: str) -> SearchResult:
    """Helper to create a SearchResult with a specific score and label."""
    return SearchResult(
        content=f"Content for {label}",
        metadata={"label": label},
        source=f"{label}.md",
        chunk_id=label,
        score=score,
    )


class TestLostInMiddleReordering:
    def test_reorder_5_results(self):
        """5 results should be reordered: [1st, 3rd, 5th, 4th, 2nd]."""
        results = [
            _make_result(0.95, "1st"),
            _make_result(0.90, "2nd"),
            _make_result(0.85, "3rd"),
            _make_result(0.80, "4th"),
            _make_result(0.75, "5th"),
        ]
        reordered = RAGChain._reorder_for_attention(results)

        labels = [r.chunk_id for r in reordered]
        assert labels == ["1st", "3rd", "5th", "4th", "2nd"]

    def test_reorder_4_results(self):
        """4 results should be reordered: [1st, 3rd, 4th, 2nd]."""
        results = [
            _make_result(0.95, "1st"),
            _make_result(0.90, "2nd"),
            _make_result(0.85, "3rd"),
            _make_result(0.80, "4th"),
        ]
        reordered = RAGChain._reorder_for_attention(results)

        labels = [r.chunk_id for r in reordered]
        assert labels == ["1st", "3rd", "4th", "2nd"]

    def test_reorder_3_results(self):
        """3 results should be reordered: [1st, 3rd, 2nd]."""
        results = [
            _make_result(0.95, "1st"),
            _make_result(0.90, "2nd"),
            _make_result(0.85, "3rd"),
        ]
        reordered = RAGChain._reorder_for_attention(results)

        labels = [r.chunk_id for r in reordered]
        assert labels == ["1st", "3rd", "2nd"]

    def test_no_reorder_2_results(self):
        """2 or fewer results should not be reordered."""
        results = [
            _make_result(0.95, "1st"),
            _make_result(0.90, "2nd"),
        ]
        reordered = RAGChain._reorder_for_attention(results)

        labels = [r.chunk_id for r in reordered]
        assert labels == ["1st", "2nd"]

    def test_no_reorder_1_result(self):
        """Single result should be returned as-is."""
        results = [_make_result(0.95, "1st")]
        reordered = RAGChain._reorder_for_attention(results)
        assert len(reordered) == 1
        assert reordered[0].chunk_id == "1st"

    def test_no_reorder_empty(self):
        """Empty list should return empty."""
        assert RAGChain._reorder_for_attention([]) == []

    def test_preserves_all_results(self):
        """Reordering should not lose or duplicate any results."""
        results = [_make_result(1.0 - i * 0.1, f"doc{i}") for i in range(7)]
        reordered = RAGChain._reorder_for_attention(results)

        assert len(reordered) == len(results)
        original_ids = sorted(r.chunk_id for r in results)
        reordered_ids = sorted(r.chunk_id for r in reordered)
        assert original_ids == reordered_ids

    def test_most_relevant_at_edges(self):
        """Most relevant (#1) should be first, #2 should be last."""
        results = [_make_result(1.0 - i * 0.1, f"doc{i}") for i in range(5)]
        reordered = RAGChain._reorder_for_attention(results)

        # First and last positions should have highest relevance
        assert reordered[0].chunk_id == "doc0"  # Most relevant = first
        assert reordered[-1].chunk_id == "doc1"  # Second most relevant = last
