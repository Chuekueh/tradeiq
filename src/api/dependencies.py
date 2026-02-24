
from src.config import Settings, get_settings
from src.embeddings.embedding_service import EmbeddingService
from src.generation.chain import RAGChain
from src.generation.llm_service import LLMService
from src.generation.response_parser import ResponseParser
from src.memory.conversation_memory import ConversationMemory
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import Reranker
from src.vectorstore.chroma_store import ChromaVectorStore


class AppState:
    """Holds initialized application components."""

    def __init__(self) -> None:
        self.settings: Settings | None = None
        self.rag_chain: RAGChain | None = None
        self.vector_store: ChromaVectorStore | None = None
        self.embedding_service: EmbeddingService | None = None
        self.memory: ConversationMemory | None = None

    def initialize(self) -> None:
        self.settings = get_settings()

        self.embedding_service = EmbeddingService(
            model_name=self.settings.embedding_model.value
        )
        self.vector_store = ChromaVectorStore(self.settings)
        self.memory = ConversationMemory()

        searcher = HybridSearcher(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            alpha=self.settings.hybrid_search_alpha,
        )
        searcher.build_bm25_index()

        reranker = Reranker()
        llm_service = LLMService(self.settings)
        response_parser = ResponseParser()

        self.rag_chain = RAGChain(
            searcher=searcher,
            reranker=reranker,
            llm_service=llm_service,
            memory=self.memory,
            response_parser=response_parser,
            retrieval_top_k=self.settings.retrieval_top_k,
            rerank_top_k=self.settings.rerank_top_k,
        )

    def shutdown(self) -> None:
        if self.memory:
            self.memory.close()


# Global singleton
app_state = AppState()
