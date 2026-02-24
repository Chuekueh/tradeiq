from src.ingestion.loaders.base import Document
from src.ingestion.processors.metadata_extractor import MetadataExtractor


def test_extracts_ticker():
    doc = Document(
        content="AAPL reported strong quarterly results with revenue growth.",
        metadata={},
        source="test.md",
    )
    extractor = MetadataExtractor()
    result = extractor.extract(doc)

    assert result.metadata["ticker"] == "AAPL"


def test_extracts_filing_type():
    doc = Document(
        content="This 10-K annual report contains the following information.",
        metadata={},
        source="test.md",
    )
    extractor = MetadataExtractor()
    result = extractor.extract(doc)

    assert result.metadata["filing_type"] == "10-K"


def test_extracts_date():
    doc = Document(
        content="Filing date: 2024-03-15. Revenue grew significantly.",
        metadata={},
        source="test.md",
    )
    extractor = MetadataExtractor()
    result = extractor.extract(doc)

    assert result.metadata["filing_date"] == "2024-03-15"


def test_extracts_sector():
    doc = Document(
        content="The company develops cloud computing and SaaS solutions.",
        metadata={},
        source="test.md",
    )
    extractor = MetadataExtractor()
    result = extractor.extract(doc)

    assert result.metadata["sector"] == "technology"


def test_no_metadata_from_generic_text():
    doc = Document(
        content="This is a generic document with no financial terms.",
        metadata={},
        source="test.md",
    )
    extractor = MetadataExtractor()
    result = extractor.extract(doc)

    assert "ticker" not in result.metadata
    assert "filing_type" not in result.metadata
