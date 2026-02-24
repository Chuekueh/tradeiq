import time
from dataclasses import dataclass

import structlog

from src.generation.hallucination_detector import HallucinationDetector
from src.generation.llm_service import LLMService
from src.generation.prompts import RAG_PROMPT_TEMPLATE, SYSTEM_PROMPT
from src.generation.response_parser import ParsedResponse, ResponseParser
from src.guardrails.pipeline import (
    ADVICE_BLOCKED_RESPONSE,
    BLOCKED_RESPONSE,
    GuardrailAction,
    GuardrailsPipeline,
)
from src.memory.conversation_memory import ConversationMemory
from src.retrieval.corrective_rag import RetrievalGrader
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.query_router import QueryRouter
from src.retrieval.query_transformer import QueryTransformer
from src.retrieval.reranker import Reranker
from src.vectorstore.base import SearchResult

logger = structlog.get_logger()


@dataclass
class RAGResponse:
    answer: str
    sources: list[SearchResult]
    cited_sources: list[str]
    session_id: str
    query_time_ms: float
    retrieval_time_ms: float
    generation_time_ms: float
    model_used: str
    faithfulness_score: float | None = None
    flagged_claims: list[str] | None = None


class RAGChain:
    """Orchestrates the full RAG pipeline: guardrails -> retrieve -> rerank -> generate."""

    def __init__(
        self,
        searcher: HybridSearcher,
        reranker: Reranker,
        llm_service: LLMService,
        memory: ConversationMemory,
        response_parser: ResponseParser,
        retrieval_top_k: int = 5,
        rerank_top_k: int = 3,
        retrieval_grader: RetrievalGrader | None = None,
        query_router: QueryRouter | None = None,
        query_transformer: QueryTransformer | None = None,
        hallucination_detector: HallucinationDetector | None = None,
        guardrails_pipeline: GuardrailsPipeline | None = None,
    ):
        self._searcher = searcher
        self._reranker = reranker
        self._llm = llm_service
        self._memory = memory
        self._parser = response_parser
        self._retrieval_top_k = retrieval_top_k
        self._rerank_top_k = rerank_top_k
        self._grader = retrieval_grader
        self._router = query_router
        self._transformer = query_transformer
        self._hallucination_detector = hallucination_detector
        self._guardrails = guardrails_pipeline

    def invoke(
        self,
        query: str,
        session_id: str = "default",
        where: dict | None = None,
    ) -> RAGResponse:
        start = time.perf_counter()

        # -1. Input guardrails (prompt injection, topic filter, PII redaction)
        if self._guardrails:
            input_result = self._guardrails.run_input_guards(query)
            if input_result.action == GuardrailAction.BLOCK:
                total_ms = (time.perf_counter() - start) * 1000
                logger.warning(
                    "guardrail_blocked_input",
                    guardrail=input_result.guardrail_name,
                    reason=input_result.reason,
                )
                return RAGResponse(
                    answer=BLOCKED_RESPONSE,
                    sources=[],
                    cited_sources=[],
                    session_id=session_id,
                    query_time_ms=round(total_ms, 1),
                    retrieval_time_ms=0.0,
                    generation_time_ms=0.0,
                    model_used=self._llm.model_name,
                )
            if input_result.action == GuardrailAction.MODIFY and input_result.modified_text:
                query = input_result.modified_text

        # 0. Adaptive query routing
        effective_retrieval_k = self._retrieval_top_k
        effective_rerank_k = self._rerank_top_k
        search_queries = [query]

        if self._router:
            routing = self._router.route(query)
            effective_retrieval_k = routing.retrieval_top_k
            effective_rerank_k = routing.rerank_top_k
            if routing.use_query_expansion and self._transformer:
                search_queries = self._transformer.expand_query(query)

        # 1. Retrieve (with possible multi-query from routing)
        retrieval_start = time.perf_counter()
        if len(search_queries) > 1:
            # Multi-query: search with each query variant, merge results
            all_candidates: dict[str, SearchResult] = {}
            for sq in search_queries:
                for result in self._searcher.search(sq, top_k=effective_retrieval_k, where=where):
                    if result.chunk_id not in all_candidates:
                        all_candidates[result.chunk_id] = result
                    else:
                        # Keep highest score
                        existing = all_candidates[result.chunk_id]
                        if result.score > existing.score:
                            all_candidates[result.chunk_id] = result
            candidates = sorted(all_candidates.values(), key=lambda r: r.score, reverse=True)[
                :effective_retrieval_k
            ]
        else:
            candidates = self._searcher.search(query, top_k=effective_retrieval_k, where=where)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # 2. Rerank
        reranked = self._reranker.rerank(query, candidates, top_k=effective_rerank_k)

        # 2.5. Corrective RAG — grade relevance
        if self._grader:
            grading = self._grader.grade(query, reranked)
            if grading.all_irrelevant:
                total_ms = (time.perf_counter() - start) * 1000
                logger.info("crag_all_irrelevant", query=query[:100])
                return RAGResponse(
                    answer=(
                        "I don't have sufficient information in the knowledge base to answer "
                        "this question accurately. The retrieved documents don't appear to be "
                        "relevant to your query. Please try rephrasing your question or ask "
                        "about a topic covered in the financial documents."
                    ),
                    sources=reranked,
                    cited_sources=[],
                    session_id=session_id,
                    query_time_ms=round(total_ms, 1),
                    retrieval_time_ms=round(retrieval_ms, 1),
                    generation_time_ms=0.0,
                    model_used=self._llm.model_name,
                )
            reranked = grading.relevant_results

        # 2.6. Reorder for attention (lost-in-the-middle mitigation)
        reranked = self._reorder_for_attention(reranked)

        # 3. Build context
        context = self._format_context(reranked)

        # 4. Get conversation history
        history = self._memory.get_history(session_id)
        chat_history = self._format_history(history)

        # 5. Generate
        gen_start = time.perf_counter()
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context, chat_history=chat_history, question=query
        )
        response = self._llm.generate(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        gen_ms = (time.perf_counter() - gen_start) * 1000

        # 6. Parse response
        parsed: ParsedResponse = self._parser.parse(response.content)

        # 6.5. Hallucination detection
        faithfulness_score = None
        flagged_claims = None
        if self._hallucination_detector and reranked:
            contexts = [r.content for r in reranked]
            source_names = [
                r.metadata.get("file_name", r.source) if r.metadata else r.source for r in reranked
            ]
            verification = self._hallucination_detector.verify(
                parsed.answer, contexts, source_names
            )
            faithfulness_score = verification.faithfulness_score
            flagged_claims = verification.flagged_claims

        # 6.8. Output guardrails (advice detection, PII, content safety, disclaimer)
        if self._guardrails:
            output_result = self._guardrails.run_output_guards(parsed.answer)
            if output_result.action == GuardrailAction.BLOCK:
                total_ms = (time.perf_counter() - start) * 1000
                logger.warning(
                    "guardrail_blocked_output",
                    guardrail=output_result.guardrail_name,
                    reason=output_result.reason,
                )
                return RAGResponse(
                    answer=ADVICE_BLOCKED_RESPONSE,
                    sources=reranked,
                    cited_sources=[],
                    session_id=session_id,
                    query_time_ms=round(total_ms, 1),
                    retrieval_time_ms=round(retrieval_ms, 1),
                    generation_time_ms=round(gen_ms, 1),
                    model_used=response.model,
                )
            if output_result.action == GuardrailAction.MODIFY and output_result.modified_text:
                parsed = ParsedResponse(
                    answer=output_result.modified_text,
                    cited_sources=parsed.cited_sources,
                )

        # 7. Save to memory
        self._memory.add_message(session_id, "user", query)
        self._memory.add_message(session_id, "assistant", parsed.answer)

        total_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "rag_query",
            query=query[:100],
            retrieval_ms=round(retrieval_ms, 1),
            generation_ms=round(gen_ms, 1),
            total_ms=round(total_ms, 1),
            sources=len(reranked),
        )

        return RAGResponse(
            answer=parsed.answer,
            sources=reranked,
            cited_sources=parsed.cited_sources,
            session_id=session_id,
            query_time_ms=round(total_ms, 1),
            retrieval_time_ms=round(retrieval_ms, 1),
            generation_time_ms=round(gen_ms, 1),
            model_used=response.model,
            faithfulness_score=faithfulness_score,
            flagged_claims=flagged_claims,
        )

    @staticmethod
    def _reorder_for_attention(results: list[SearchResult]) -> list[SearchResult]:
        """Reorder documents to mitigate the 'lost in the middle' problem.

        LLMs attend best to content at the beginning and end of the context.
        This places the most relevant documents at these positions.

        Input (by relevance):  [1st, 2nd, 3rd, 4th, 5th]
        Output (by position):  [1st, 3rd, 5th, 4th, 2nd]
        """
        if len(results) < 3:
            return results

        reordered: list[SearchResult] = []
        for i in range(0, len(results), 2):
            reordered.append(results[i])
        for i in range(len(results) - 1 if len(results) % 2 == 0 else len(results) - 2, 0, -2):
            reordered.append(results[i])

        return reordered

    def _format_context(self, results: list[SearchResult]) -> str:
        parts: list[str] = []
        for i, r in enumerate(results, 1):
            source = r.metadata.get("file_name", r.source) if r.metadata else r.source
            parts.append(f"[{i}] Source: {source} (relevance: {r.score:.3f})\n{r.content}")
        return "\n\n".join(parts)

    def _format_history(self, history: list[dict[str, str]]) -> str:
        if not history:
            return "No previous conversation."
        lines: list[str] = []
        for msg in history[-6:]:  # Last 3 exchanges
            role = msg["role"].capitalize()
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)
