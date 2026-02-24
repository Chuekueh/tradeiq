import structlog
from openai import OpenAI

from src.config import Settings
from src.ingestion.chunkers.base import Chunk

logger = structlog.get_logger()

CONTEXT_PROMPT = """You are an expert at understanding financial documents. Given the full document
and a specific chunk extracted from it, provide a brief 1-2 sentence context that explains:
- What document this chunk is from (company, filing type, date if available)
- What section or topic this chunk relates to within the document

Be concise and factual. Do not repeat the chunk content."""

CONTEXT_USER_TEMPLATE = """<document_summary>
{doc_summary}
</document_summary>

<chunk>
{chunk_content}
</chunk>

Provide the contextual summary for this chunk:"""


class ContextEnricher:
    """Enriches chunks with document-level context before embedding.

    Based on Anthropic's Contextual Retrieval technique, which prepends
    a short context summary to each chunk, reducing retrieval failures
    by up to 35% when combined with reranking.
    """

    def __init__(self, settings: Settings):
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model
        self._enabled = settings.contextual_retrieval_enabled
        logger.info("context_enricher_initialized", enabled=self._enabled)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enrich(self, doc_content: str, chunk: Chunk) -> Chunk:
        """Add document-level context to a chunk.

        Prepends a short context summary to the chunk content.
        The original content is preserved in metadata for display.
        """
        if not self._enabled:
            return chunk

        # Truncate document for prompt efficiency (first 2000 chars)
        doc_summary = doc_content[:2000]
        if len(doc_content) > 2000:
            doc_summary += "\n... [document continues]"

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": CONTEXT_PROMPT},
                    {
                        "role": "user",
                        "content": CONTEXT_USER_TEMPLATE.format(
                            doc_summary=doc_summary,
                            chunk_content=chunk.content[:500],
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=100,
            )
            context = response.choices[0].message.content or ""
            context = context.strip()
        except Exception:
            logger.warning(
                "context_enrichment_failed",
                chunk_id=chunk.chunk_id,
                exc_info=True,
            )
            return chunk

        # Store original content in metadata, prepend context
        chunk.metadata["original_content"] = chunk.content
        chunk.metadata["context_summary"] = context
        chunk.content = f"{context}\n\n{chunk.content}"

        return chunk

    def enrich_batch(self, doc_content: str, chunks: list[Chunk]) -> list[Chunk]:
        """Enrich a batch of chunks from the same document."""
        if not self._enabled:
            return chunks

        enriched: list[Chunk] = []
        for chunk in chunks:
            enriched.append(self.enrich(doc_content, chunk))

        logger.info(
            "batch_enrichment_complete",
            chunks_enriched=len(enriched),
        )
        return enriched
