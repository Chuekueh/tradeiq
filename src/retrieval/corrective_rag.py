from dataclasses import dataclass
from enum import StrEnum

import structlog

from src.generation.llm_service import LLMService
from src.vectorstore.base import SearchResult

logger = structlog.get_logger()

GRADING_PROMPT = """You are a relevance grader for a financial research knowledge base.

Given a user question and a retrieved document, determine if the document contains
information relevant to answering the question.

Rules:
- A document is "relevant" if it contains ANY information useful for answering the question
- Consider partial relevance as "relevant" (even tangential financial context helps)
- Only grade as "irrelevant" if the document is completely off-topic

Respond with ONLY one word: "relevant" or "irrelevant"."""

GRADING_USER_TEMPLATE = """Question: {question}

Document:
{document}

Grade:"""


class RelevanceGrade(StrEnum):
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"


@dataclass
class GradingResult:
    """Result of grading retrieved documents for relevance."""

    results: list[SearchResult]
    grades: dict[str, RelevanceGrade]  # chunk_id -> grade
    relevant_results: list[SearchResult]
    all_irrelevant: bool


class RetrievalGrader:
    """Grades retrieved documents for relevance using an LLM.

    Implements the Corrective RAG (CRAG) pattern: after retrieval,
    each document is evaluated for relevance. Irrelevant documents
    are filtered out. If ALL documents are irrelevant, the system
    signals that it lacks sufficient information.
    """

    def __init__(self, llm: LLMService):
        self._llm = llm

    def grade(self, query: str, results: list[SearchResult]) -> GradingResult:
        """Grade each retrieved document for relevance to the query."""
        if not results:
            return GradingResult(
                results=results,
                grades={},
                relevant_results=[],
                all_irrelevant=True,
            )

        grades: dict[str, RelevanceGrade] = {}
        relevant: list[SearchResult] = []

        for result in results:
            grade = self._grade_single(query, result)
            grades[result.chunk_id] = grade
            if grade == RelevanceGrade.RELEVANT:
                relevant.append(result)

        all_irrelevant = len(relevant) == 0

        logger.info(
            "corrective_rag_grading",
            total=len(results),
            relevant=len(relevant),
            irrelevant=len(results) - len(relevant),
            all_irrelevant=all_irrelevant,
        )

        return GradingResult(
            results=results,
            grades=grades,
            relevant_results=relevant,
            all_irrelevant=all_irrelevant,
        )

    def _grade_single(self, query: str, result: SearchResult) -> RelevanceGrade:
        """Grade a single document for relevance."""
        try:
            response = self._llm.generate(
                messages=[
                    {"role": "system", "content": GRADING_PROMPT},
                    {
                        "role": "user",
                        "content": GRADING_USER_TEMPLATE.format(
                            question=query,
                            document=result.content[:1000],
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=10,
            )
            grade_text = response.content.strip().lower()
            if "irrelevant" in grade_text:
                return RelevanceGrade.IRRELEVANT
            return RelevanceGrade.RELEVANT
        except Exception:
            logger.warning(
                "grading_failed",
                chunk_id=result.chunk_id,
                exc_info=True,
            )
            # On error, assume relevant (fail-open)
            return RelevanceGrade.RELEVANT
