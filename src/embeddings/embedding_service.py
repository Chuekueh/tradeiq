import structlog
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger()


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info("loading_embedding_model", model=model_name)
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()  # type: ignore[return-value]

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Embed a batch of documents."""
        embeddings = self._model.encode(
            texts, batch_size=batch_size, show_progress_bar=len(texts) > 100
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        embedding = self._model.encode(query)
        return embedding.tolist()
