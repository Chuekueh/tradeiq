from dataclasses import dataclass
from enum import StrEnum

import structlog

from src.generation.llm_service import LLMService

logger = structlog.get_logger()

ROUTING_PROMPT = """You are a query complexity classifier for a financial research system.
Classify the user's question into one of three complexity levels:

- SIMPLE: Direct factual lookups with a single, clear answer.
  Examples: "What is Apple's revenue?", "Who is the CEO of Tesla?", "What was MSFT's EPS?"

- MODERATE: Questions requiring analysis, explanation, or synthesis of a single topic.
  Examples: "Explain Apple's risk factors", "What drove revenue growth for NVDA?",
  "Summarize the key points from Tesla's 10-K"

- COMPLEX: Comparative analysis, multi-company questions, trend analysis, or multi-hop reasoning.
  Examples: "Compare Apple and Microsoft's risk profiles",
  "Which tech companies had declining margins in 2024?",
  "How do AAPL's risk factors relate to their revenue decline?"

Respond with ONLY one word: SIMPLE, MODERATE, or COMPLEX."""


class QueryComplexity(StrEnum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class RoutingDecision:
    """Routing decision with parameters for the retrieval pipeline."""

    complexity: QueryComplexity
    retrieval_top_k: int
    rerank_top_k: int
    use_query_expansion: bool
    expanded_queries: list[str] | None = None


class QueryRouter:
    """Routes queries to appropriate retrieval strategies based on complexity.

    Simple queries get lightweight retrieval (fewer candidates).
    Complex queries get expanded retrieval (more candidates, query expansion).
    This optimizes both latency and recall based on query needs.
    """

    # Retrieval parameters per complexity tier
    TIER_CONFIG = {
        QueryComplexity.SIMPLE: {
            "retrieval_top_k": 3,
            "rerank_top_k": 2,
            "use_query_expansion": False,
        },
        QueryComplexity.MODERATE: {
            "retrieval_top_k": 5,
            "rerank_top_k": 3,
            "use_query_expansion": False,
        },
        QueryComplexity.COMPLEX: {
            "retrieval_top_k": 8,
            "rerank_top_k": 5,
            "use_query_expansion": True,
        },
    }

    def __init__(self, llm: LLMService):
        self._llm = llm

    def route(self, query: str) -> RoutingDecision:
        """Classify query complexity and return retrieval parameters."""
        complexity = self._classify(query)
        config = self.TIER_CONFIG[complexity]

        decision = RoutingDecision(
            complexity=complexity,
            retrieval_top_k=config["retrieval_top_k"],
            rerank_top_k=config["rerank_top_k"],
            use_query_expansion=config["use_query_expansion"],
        )

        logger.info(
            "query_routed",
            query=query[:100],
            complexity=complexity.value,
            top_k=decision.retrieval_top_k,
            expansion=decision.use_query_expansion,
        )
        return decision

    def _classify(self, query: str) -> QueryComplexity:
        """Use LLM to classify query complexity."""
        try:
            response = self._llm.generate(
                messages=[
                    {"role": "system", "content": ROUTING_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=10,
            )
            result = response.content.strip().upper()

            if "COMPLEX" in result:
                return QueryComplexity.COMPLEX
            if "MODERATE" in result:
                return QueryComplexity.MODERATE
            return QueryComplexity.SIMPLE
        except Exception:
            logger.warning("query_classification_failed", exc_info=True)
            return QueryComplexity.MODERATE  # Default to moderate on error
