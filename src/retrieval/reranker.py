import structlog
from sentence_transformers import CrossEncoder

from src.vectorstore.base import SearchResult

logger = structlog.get_logger()


class Reranker:
    """Reranks retrieved documents using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info("loading_reranker", model=model_name)
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[SearchResult], top_k: int = 3) -> list[SearchResult]:
        if not results:
            return []

        pairs = [(query, r.content) for r in results]
        scores = self._model.predict(pairs)

        for result, score in zip(results, scores, strict=True):
            result.score = float(score)

        reranked = sorted(results, key=lambda r: r.score, reverse=True)
        return reranked[:top_k]
