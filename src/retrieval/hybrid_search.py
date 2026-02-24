import structlog
from rank_bm25 import BM25Okapi

from src.embeddings.embedding_service import EmbeddingService
from src.vectorstore.base import SearchResult
from src.vectorstore.chroma_store import ChromaVectorStore

logger = structlog.get_logger()


class HybridSearcher:
    """Combines dense vector search with sparse BM25 using Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_service: EmbeddingService,
        alpha: float = 0.7,
    ):
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._alpha = alpha
        self._bm25: BM25Okapi | None = None
        self._corpus_ids: list[str] = []
        self._corpus_docs: list[str] = []

    def build_bm25_index(self) -> None:
        """Build BM25 index from all documents in the vector store."""
        self._corpus_docs = self._vector_store.get_all_documents()
        self._corpus_ids = self._vector_store.get_all_ids()

        if not self._corpus_docs:
            logger.warning("empty_corpus_for_bm25")
            return

        tokenized = [doc.lower().split() for doc in self._corpus_docs]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("bm25_index_built", documents=len(self._corpus_docs))

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """
        Hybrid search:
        1. Dense vector search via ChromaDB
        2. Sparse BM25 search
        3. Reciprocal Rank Fusion to merge results
        """
        # Fetch more candidates for fusion
        fetch_k = top_k * 3

        # Dense search
        query_embedding = self._embedding_service.embed_query(query)
        vector_results = self._vector_store.search(query_embedding, top_k=fetch_k, where=where)

        # If no BM25 index, fall back to vector-only
        if self._bm25 is None or not self._corpus_docs:
            return vector_results[:top_k]

        # BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query)

        # Build BM25 results ranked by score
        scored_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:fetch_k]

        # Reciprocal Rank Fusion
        rrf_scores: dict[str, float] = {}
        rrf_docs: dict[str, SearchResult] = {}
        k = 60  # RRF constant

        # Score vector results
        for rank, result in enumerate(vector_results):
            rrf_scores[result.chunk_id] = self._alpha * (1.0 / (k + rank + 1))
            rrf_docs[result.chunk_id] = result

        # Score BM25 results
        for rank, idx in enumerate(scored_indices):
            if idx < len(self._corpus_ids):
                doc_id = self._corpus_ids[idx]
                bm25_score = (1.0 - self._alpha) * (1.0 / (k + rank + 1))
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + bm25_score

                if doc_id not in rrf_docs and idx < len(self._corpus_docs):
                    rrf_docs[doc_id] = SearchResult(
                        content=self._corpus_docs[idx],
                        metadata={},
                        source="",
                        chunk_id=doc_id,
                        score=0.0,
                    )

        # Sort by fused score and return top_k
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results: list[SearchResult] = []
        for doc_id, score in ranked:
            if doc_id in rrf_docs:
                result = rrf_docs[doc_id]
                result.score = score
                results.append(result)

        return results
