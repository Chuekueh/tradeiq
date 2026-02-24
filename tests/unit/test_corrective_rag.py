from unittest.mock import MagicMock

from src.retrieval.corrective_rag import RelevanceGrade, RetrievalGrader
from src.vectorstore.base import SearchResult


def _make_result(chunk_id: str, content: str = "Test content") -> SearchResult:
    return SearchResult(
        content=content,
        metadata={},
        source="test.md",
        chunk_id=chunk_id,
        score=0.9,
    )


class TestRetrievalGrader:
    def test_all_relevant(self):
        """When all documents are relevant, all should be returned."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "relevant"
        mock_llm.generate.return_value = mock_response

        grader = RetrievalGrader(mock_llm)
        results = [_make_result("doc1"), _make_result("doc2"), _make_result("doc3")]
        grading = grader.grade("What is AAPL revenue?", results)

        assert not grading.all_irrelevant
        assert len(grading.relevant_results) == 3

    def test_mixed_relevance(self):
        """Mixed relevance should filter to only relevant documents."""
        mock_llm = MagicMock()
        responses = [
            MagicMock(content="relevant"),
            MagicMock(content="irrelevant"),
            MagicMock(content="relevant"),
        ]
        mock_llm.generate.side_effect = responses

        grader = RetrievalGrader(mock_llm)
        results = [_make_result("doc1"), _make_result("doc2"), _make_result("doc3")]
        grading = grader.grade("What is AAPL revenue?", results)

        assert not grading.all_irrelevant
        assert len(grading.relevant_results) == 2
        assert grading.grades["doc1"] == RelevanceGrade.RELEVANT
        assert grading.grades["doc2"] == RelevanceGrade.IRRELEVANT
        assert grading.grades["doc3"] == RelevanceGrade.RELEVANT

    def test_all_irrelevant(self):
        """When all are irrelevant, all_irrelevant flag should be True."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "irrelevant"
        mock_llm.generate.return_value = mock_response

        grader = RetrievalGrader(mock_llm)
        results = [_make_result("doc1"), _make_result("doc2")]
        grading = grader.grade("What is the weather?", results)

        assert grading.all_irrelevant
        assert len(grading.relevant_results) == 0

    def test_empty_results(self):
        """Empty results should return all_irrelevant=True."""
        mock_llm = MagicMock()
        grader = RetrievalGrader(mock_llm)
        grading = grader.grade("test query", [])

        assert grading.all_irrelevant
        assert len(grading.relevant_results) == 0

    def test_api_error_fails_open(self):
        """On API error, should assume document is relevant (fail-open)."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("API Error")

        grader = RetrievalGrader(mock_llm)
        results = [_make_result("doc1")]
        grading = grader.grade("test query", results)

        assert not grading.all_irrelevant
        assert len(grading.relevant_results) == 1
        assert grading.grades["doc1"] == RelevanceGrade.RELEVANT

    def test_grades_stored_per_chunk(self):
        """Grades should be stored in a dict keyed by chunk_id."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "relevant"
        mock_llm.generate.return_value = mock_response

        grader = RetrievalGrader(mock_llm)
        results = [_make_result("alpha"), _make_result("beta")]
        grading = grader.grade("test", results)

        assert "alpha" in grading.grades
        assert "beta" in grading.grades
