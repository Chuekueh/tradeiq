from unittest.mock import MagicMock

from src.retrieval.query_router import QueryComplexity, QueryRouter


class TestQueryRouter:
    def test_simple_query(self):
        """Simple factual queries should get minimal retrieval."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = MagicMock(content="SIMPLE")

        router = QueryRouter(mock_llm)
        decision = router.route("What is Apple's revenue?")

        assert decision.complexity == QueryComplexity.SIMPLE
        assert decision.retrieval_top_k == 3
        assert decision.rerank_top_k == 2
        assert decision.use_query_expansion is False

    def test_moderate_query(self):
        """Analytical queries should get standard retrieval."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = MagicMock(content="MODERATE")

        router = QueryRouter(mock_llm)
        decision = router.route("Explain Apple's risk factors")

        assert decision.complexity == QueryComplexity.MODERATE
        assert decision.retrieval_top_k == 5
        assert decision.rerank_top_k == 3
        assert decision.use_query_expansion is False

    def test_complex_query(self):
        """Complex queries should get expanded retrieval."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = MagicMock(content="COMPLEX")

        router = QueryRouter(mock_llm)
        decision = router.route("Compare Apple and Microsoft risk profiles")

        assert decision.complexity == QueryComplexity.COMPLEX
        assert decision.retrieval_top_k == 8
        assert decision.rerank_top_k == 5
        assert decision.use_query_expansion is True

    def test_error_defaults_to_moderate(self):
        """On classification error, should default to moderate."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("API Error")

        router = QueryRouter(mock_llm)
        decision = router.route("test query")

        assert decision.complexity == QueryComplexity.MODERATE

    def test_tier_configs_exist(self):
        """All complexity tiers should have valid configs."""
        for tier in QueryComplexity:
            assert tier in QueryRouter.TIER_CONFIG
            config = QueryRouter.TIER_CONFIG[tier]
            assert "retrieval_top_k" in config
            assert "rerank_top_k" in config
            assert "use_query_expansion" in config

    def test_complex_has_most_candidates(self):
        """Complex tier should retrieve more candidates than simple."""
        simple_k = QueryRouter.TIER_CONFIG[QueryComplexity.SIMPLE]["retrieval_top_k"]
        complex_k = QueryRouter.TIER_CONFIG[QueryComplexity.COMPLEX]["retrieval_top_k"]
        assert complex_k > simple_k
