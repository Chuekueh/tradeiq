from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.chunkers.base import Chunk
from src.ingestion.processors.context_enricher import ContextEnricher


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.openai_api_key = "test-key"
    settings.openai_model = "gpt-4o-mini"
    settings.contextual_retrieval_enabled = True
    return settings


@pytest.fixture
def sample_chunk():
    return Chunk(
        content="Revenue grew by 15% year over year to $94.9 billion.",
        metadata={"ticker": "AAPL", "filing_type": "10-Q"},
        source="aapl_10q.md",
        chunk_id="abc123",
        chunk_index=0,
        doc_id="doc456",
    )


@pytest.fixture
def sample_doc_content():
    return (
        "Apple Inc. Form 10-Q for Q3 2024\n\n"
        "## Financial Highlights\n"
        "Revenue grew by 15% year over year to $94.9 billion.\n"
        "iPhone revenue was $42.3 billion.\n\n"
        "## Risk Factors\n"
        "Supply chain disruptions remain a concern."
    )


class TestContextEnricher:
    @patch("src.ingestion.processors.context_enricher.OpenAI")
    def test_enrich_prepends_context(
        self, mock_openai_class, mock_settings, sample_chunk, sample_doc_content
    ):
        """Enricher should prepend LLM-generated context to the chunk."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = (
            "This chunk is from Apple's Q3 2024 10-Q filing, discussing financial highlights."
        )
        mock_client.chat.completions.create.return_value = mock_response

        enricher = ContextEnricher(mock_settings)
        enriched = enricher.enrich(sample_doc_content, sample_chunk)

        assert enriched.content.startswith("This chunk is from Apple's Q3 2024 10-Q filing")
        assert "Revenue grew by 15%" in enriched.content
        assert (
            enriched.metadata["original_content"]
            == "Revenue grew by 15% year over year to $94.9 billion."
        )
        assert "context_summary" in enriched.metadata

    @patch("src.ingestion.processors.context_enricher.OpenAI")
    def test_disabled_returns_unchanged(self, mock_openai_class, sample_chunk, sample_doc_content):
        """When disabled, enricher should return chunk unchanged."""
        settings = MagicMock()
        settings.openai_api_key = "test-key"
        settings.openai_model = "gpt-4o-mini"
        settings.contextual_retrieval_enabled = False

        enricher = ContextEnricher(settings)
        result = enricher.enrich(sample_doc_content, sample_chunk)

        assert result.content == "Revenue grew by 15% year over year to $94.9 billion."
        assert "original_content" not in result.metadata

    @patch("src.ingestion.processors.context_enricher.OpenAI")
    def test_handles_api_error_gracefully(
        self, mock_openai_class, mock_settings, sample_chunk, sample_doc_content
    ):
        """On API error, enricher should return chunk unchanged."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        enricher = ContextEnricher(mock_settings)
        result = enricher.enrich(sample_doc_content, sample_chunk)

        assert result.content == "Revenue grew by 15% year over year to $94.9 billion."

    @patch("src.ingestion.processors.context_enricher.OpenAI")
    def test_enrich_batch(self, mock_openai_class, mock_settings, sample_doc_content):
        """Batch enrichment should process all chunks."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Context for this chunk."
        mock_client.chat.completions.create.return_value = mock_response

        chunks = [
            Chunk(
                content=f"Chunk {i}",
                metadata={},
                source="test.md",
                chunk_id=f"id{i}",
                chunk_index=i,
                doc_id="doc1",
            )
            for i in range(3)
        ]

        enricher = ContextEnricher(mock_settings)
        enriched = enricher.enrich_batch(sample_doc_content, chunks)

        assert len(enriched) == 3
        for chunk in enriched:
            assert chunk.content.startswith("Context for this chunk.")

    @patch("src.ingestion.processors.context_enricher.OpenAI")
    def test_batch_disabled_returns_unchanged(self, mock_openai_class, sample_doc_content):
        """Disabled batch enrichment should return chunks unchanged."""
        settings = MagicMock()
        settings.openai_api_key = "test-key"
        settings.openai_model = "gpt-4o-mini"
        settings.contextual_retrieval_enabled = False

        chunks = [
            Chunk(
                content=f"Chunk {i}",
                metadata={},
                source="test.md",
                chunk_id=f"id{i}",
                chunk_index=i,
                doc_id="doc1",
            )
            for i in range(3)
        ]

        enricher = ContextEnricher(settings)
        result = enricher.enrich_batch(sample_doc_content, chunks)

        assert len(result) == 3
        assert result[0].content == "Chunk 0"
