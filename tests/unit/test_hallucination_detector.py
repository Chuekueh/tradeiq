from unittest.mock import MagicMock

import pytest

from src.generation.hallucination_detector import (
    HallucinationDetector,
)


class TestHallucinationDetector:
    def test_all_claims_supported(self):
        """All claims supported = faithfulness 1.0."""
        mock_llm = MagicMock()
        # First call: decompose claims
        mock_llm.generate.side_effect = [
            MagicMock(content="Apple revenue was $94.9B\nMac revenue was $7.7B"),
            MagicMock(content="supported"),  # Verify claim 1
            MagicMock(content="supported"),  # Verify claim 2
        ]

        detector = HallucinationDetector(mock_llm)
        result = detector.verify(
            "Apple revenue was $94.9B. Mac revenue was $7.7B.",
            ["Apple reported $94.9B revenue. Mac segment earned $7.7B."],
            ["aapl_10q.md"],
        )

        assert result.faithfulness_score == 1.0
        assert len(result.flagged_claims) == 0
        assert len(result.claims) == 2

    def test_some_claims_unsupported(self):
        """Unsupported claims should be flagged with correct score."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            MagicMock(content="Revenue was $94.9B\nEmployee count is 164000\nStock price hit $200"),
            MagicMock(content="supported"),
            MagicMock(content="unsupported"),
            MagicMock(content="unsupported"),
        ]

        detector = HallucinationDetector(mock_llm)
        result = detector.verify(
            "Test answer",
            ["Revenue was $94.9B"],
            ["test.md"],
        )

        assert result.faithfulness_score == pytest.approx(1 / 3, abs=0.01)
        assert len(result.flagged_claims) == 2
        assert "Employee count is 164000" in result.flagged_claims
        assert "Stock price hit $200" in result.flagged_claims

    def test_no_claims_returns_perfect_score(self):
        """Answer with no verifiable claims should return 1.0."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = MagicMock(content="NO_CLAIMS")

        detector = HallucinationDetector(mock_llm)
        result = detector.verify(
            "I don't have enough information to answer.",
            ["Some context"],
            ["test.md"],
        )

        assert result.faithfulness_score == 1.0
        assert len(result.claims) == 0
        assert len(result.flagged_claims) == 0

    def test_decomposition_error_returns_empty(self):
        """Decomposition failure should return empty claims with 1.0 score."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("API Error")

        detector = HallucinationDetector(mock_llm)
        result = detector.verify("Test", ["context"], ["src.md"])

        assert result.faithfulness_score == 1.0
        assert len(result.claims) == 0

    def test_verification_error_fails_open(self):
        """Verification error should assume supported (fail-open)."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            MagicMock(content="A single claim"),  # Decompose
            Exception("API Error"),  # Verify fails
        ]

        detector = HallucinationDetector(mock_llm)
        result = detector.verify("Test", ["context"], ["src.md"])

        assert result.faithfulness_score == 1.0
        assert len(result.flagged_claims) == 0

    def test_claim_structure(self):
        """Each claim should have text, supported flag, and optional source."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            MagicMock(content="Revenue was $94.9B"),
            MagicMock(content="supported"),
        ]

        detector = HallucinationDetector(mock_llm)
        result = detector.verify("Test", ["Revenue was $94.9B"], ["aapl.md"])

        assert len(result.claims) == 1
        claim = result.claims[0]
        assert claim.text == "Revenue was $94.9B"
        assert claim.supported is True
        assert claim.supporting_source == "aapl.md"
