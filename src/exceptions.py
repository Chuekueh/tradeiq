class TradeIQError(Exception):
    """Base exception for TradeIQ."""


class IngestionError(TradeIQError):
    """Error during document ingestion."""


class EmbeddingError(TradeIQError):
    """Error during embedding generation."""


class RetrievalError(TradeIQError):
    """Error during document retrieval."""


class GenerationError(TradeIQError):
    """Error during LLM response generation."""


class VectorStoreError(TradeIQError):
    """Error interacting with the vector store."""
