from enum import StrEnum

from pydantic import Field
from pydantic_settings import BaseSettings


class EmbeddingModel(StrEnum):
    MINILM = "all-MiniLM-L6-v2"
    BGE_SMALL = "BAAI/bge-small-en-v1.5"


class ChunkingStrategy(StrEnum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Embeddings
    embedding_model: EmbeddingModel = EmbeddingModel.MINILM
    embedding_dimension: int = 384

    # Vector store
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "financial_knowledge_base"

    # Retrieval
    retrieval_top_k: int = 5
    rerank_top_k: int = 3
    hybrid_search_alpha: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Weight for vector vs BM25 (1.0 = pure vector)"
    )
    corrective_rag_enabled: bool = True

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE

    # Contextual retrieval
    contextual_retrieval_enabled: bool = False

    # Parent-child retrieval
    parent_child_enabled: bool = True
    parent_chunk_size: int = 1024
    child_chunk_size: int = 256

    # Contextual retrieval
    contextual_retrieval_enabled: bool = False

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # SEC EDGAR
    sec_edgar_user_agent: str = "TradeIQ research@example.com"

    model_config = {"env_file": ".env", "env_prefix": "RAG_"}


def get_settings() -> Settings:
    return Settings()
