import structlog

from src.config import Settings, get_settings
from src.embeddings.embedding_service import EmbeddingService
from src.generation.chain import RAGChain
from src.generation.hallucination_detector import HallucinationDetector
from src.generation.llm_service import LLMService
from src.generation.response_parser import ResponseParser
from src.guardrails.content_safety import ContentSafetyGuardrail
from src.guardrails.financial_advice_detector import FinancialAdviceDetector
from src.guardrails.pii_detector import PIIDetector
from src.guardrails.pipeline import GuardrailsPipeline
from src.guardrails.prompt_injection import PromptInjectionDetector
from src.guardrails.topic_filter import TopicFilter
from src.memory.conversation_memory import ConversationMemory
from src.retrieval.corrective_rag import RetrievalGrader
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.query_router import QueryRouter
from src.retrieval.query_transformer import QueryTransformer
from src.retrieval.reranker import Reranker
from src.vectorstore.chroma_store import ChromaVectorStore

logger = structlog.get_logger()


def _build_guardrails(settings: Settings, llm: LLMService) -> GuardrailsPipeline | None:
    """Build the guardrails pipeline from settings."""
    if not settings.guardrails_enabled:
        return None

    injection_detector = None
    if settings.prompt_injection_detection_enabled:
        injection_detector = PromptInjectionDetector(
            threshold=settings.prompt_injection_threshold,
        )

    topic_filter = None
    if settings.topic_filter_enabled:
        topic_filter = TopicFilter(llm=llm)

    pii_detector = None
    if settings.pii_detection_enabled:
        pii_detector = PIIDetector()

    advice_detector = None
    if settings.financial_advice_detection_enabled:
        advice_detector = FinancialAdviceDetector(llm=llm)

    content_safety = None
    if settings.content_safety_enabled and settings.openai_api_key:
        content_safety = ContentSafetyGuardrail(api_key=settings.openai_api_key)

    pipeline = GuardrailsPipeline(
        injection_detector=injection_detector,
        topic_filter=topic_filter,
        pii_detector=pii_detector,
        advice_detector=advice_detector,
        content_safety=content_safety,
    )
    logger.info(
        "guardrails_initialized",
        injection=injection_detector is not None,
        topic=topic_filter is not None,
        pii=pii_detector is not None,
        advice=advice_detector is not None,
        safety=content_safety is not None,
    )
    return pipeline


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

        self.embedding_service = EmbeddingService(model_name=self.settings.embedding_model.value)
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

        # Advanced RAG components
        retrieval_grader = (
            RetrievalGrader(llm=llm_service) if self.settings.corrective_rag_enabled else None
        )
        query_router = (
            QueryRouter(llm=llm_service) if self.settings.adaptive_routing_enabled else None
        )
        query_transformer = (
            QueryTransformer(llm=llm_service) if self.settings.adaptive_routing_enabled else None
        )
        hallucination_detector = (
            HallucinationDetector(llm=llm_service)
            if self.settings.hallucination_detection_enabled
            else None
        )

        # Guardrails pipeline
        guardrails_pipeline = _build_guardrails(self.settings, llm_service)

        self.rag_chain = RAGChain(
            searcher=searcher,
            reranker=reranker,
            llm_service=llm_service,
            memory=self.memory,
            response_parser=response_parser,
            retrieval_top_k=self.settings.retrieval_top_k,
            rerank_top_k=self.settings.rerank_top_k,
            retrieval_grader=retrieval_grader,
            query_router=query_router,
            query_transformer=query_transformer,
            hallucination_detector=hallucination_detector,
            guardrails_pipeline=guardrails_pipeline,
        )

    def shutdown(self) -> None:
        if self.memory:
            self.memory.close()


# Global singleton
app_state = AppState()
