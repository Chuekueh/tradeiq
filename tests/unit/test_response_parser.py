from src.generation.response_parser import ResponseParser


def test_extracts_citations():
    parser = ResponseParser()
    response = (
        "Apple reported strong results [Source: AAPL_10K.md]. "
        "Revenue grew significantly [Source: AAPL_Q3_earnings.md]."
    )
    parsed = parser.parse(response)

    assert len(parsed.cited_sources) == 2
    assert "AAPL_10K.md" in parsed.cited_sources
    assert "AAPL_Q3_earnings.md" in parsed.cited_sources


def test_no_citations():
    parser = ResponseParser()
    parsed = parser.parse("A response with no source citations.")

    assert len(parsed.cited_sources) == 0
    assert parsed.answer == "A response with no source citations."


def test_duplicate_citations():
    parser = ResponseParser()
    response = "First point [Source: doc.md]. Second point [Source: doc.md]."
    parsed = parser.parse(response)

    assert len(parsed.cited_sources) == 1
