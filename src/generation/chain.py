import time
from dataclasses import dataclass

import structlog

from src.generation.llm_service import LLMService
from src.generation.prompts import RAG_PROMPT_TEMPLATE, SYSTEM_PROMPT
from src.generation.response_parser import ParsedResponse, ResponseParser
from src.memory.conversation_memory import ConversationMemory
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
        query_router: QueryRouter | None = None,
        query_transformer: QueryTransformer | None = None,
    ):
        self._searcher = searcher
        self._reranker = reranker
        self._llm = llm_service
        self._memory = memory
        self._parser = response_parser
        self._retrieval_top_k = retrieval_top_k
        self._rerank_top_k = rerank_top_k
        self._router = query_router
        self._transformer = query_transformer

    def invoke(
        self,
        query: str,
        session_id: str = "default",
        where: dict | None = None,
    ) -> RAGResponse:
        start = time.perf_counter()

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
