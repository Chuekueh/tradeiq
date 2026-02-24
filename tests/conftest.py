import pytest

from src.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        openai_api_key="test-key",
        chroma_persist_dir="./data/test_chroma_db",
        chroma_collection_name="test_collection",
    )
