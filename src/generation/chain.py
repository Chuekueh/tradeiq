import time
from dataclasses import dataclass

import structlog

from src.generation.llm_service import LLMService
from src.generation.prompts import RAG_PROMPT_TEMPLATE, SYSTEM_PROMPT
from src.generation.response_parser import ParsedResponse, ResponseParser
from src.memory.conversation_memory import ConversationMemory
from src.retrieval.hybrid_search import HybridSearcher
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


class RAGChain:
    """Orchestrates the full RAG pipeline: retrieve -> rerank -> generate."""

    def __init__(
        self,
        searcher: HybridSearcher,
        reranker: Reranker,
        llm_service: LLMService,
        memory: ConversationMemory,
        response_parser: ResponseParser,
        retrieval_top_k: int = 5,
        rerank_top_k: int = 3,
    ):
        self._searcher = searcher
        self._reranker = reranker
        self._llm = llm_service
        self._memory = memory
        self._parser = response_parser
        self._retrieval_top_k = retrieval_top_k
        self._rerank_top_k = rerank_top_k

    def invoke(
        self,
        query: str,
        session_id: str = "default",
        where: dict | None = None,
    ) -> RAGResponse:
        start = time.perf_counter()

        # 1. Retrieve
        retrieval_start = time.perf_counter()
        candidates = self._searcher.search(query, top_k=self._retrieval_top_k, where=where)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # 2. Rerank
        reranked = self._reranker.rerank(query, candidates, top_k=self._rerank_top_k)

        # 2.5. Reorder for attention (lost-in-the-middle mitigation)
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
