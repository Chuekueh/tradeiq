import re

from src.ingestion.loaders.base import Document


class MetadataExtractor:
    """Extracts structured financial metadata from documents using patterns."""

    TICKER_PATTERN = re.compile(r"\b([A-Z]{1,5})\b")
    FILING_TYPE_PATTERN = re.compile(r"\b(10-K|10-Q|8-K|S-1|DEF 14A)\b", re.IGNORECASE)
    DATE_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

    SECTOR_KEYWORDS: dict[str, list[str]] = {
        "technology": ["software", "cloud", "SaaS", "computing", "semiconductor", "AI"],
        "finance": ["banking", "insurance", "investment", "lending", "fintech"],
        "healthcare": ["pharmaceutical", "biotech", "medical", "healthcare", "drug"],
        "energy": ["oil", "gas", "renewable", "solar", "wind", "energy"],
        "consumer": ["retail", "e-commerce", "consumer", "brand", "food"],
    }

    KNOWN_TICKERS = {
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD",
        "DIS", "BAC", "ADBE", "CRM", "NFLX", "INTC", "AMD",
    }

    def extract(self, document: Document) -> Document:
        """Enrich document metadata with extracted financial fields."""
        text = document.content
        metadata = dict(document.metadata)

        # Extract filing type
        filing_match = self.FILING_TYPE_PATTERN.search(text)
        if filing_match:
            metadata["filing_type"] = filing_match.group(1).upper()

        # Extract date
        date_match = self.DATE_PATTERN.search(text)
        if date_match:
            metadata["filing_date"] = date_match.group(1)

        # Extract ticker (only known tickers to reduce noise)
        words = set(self.TICKER_PATTERN.findall(text))
        tickers = words & self.KNOWN_TICKERS
        if tickers:
            metadata["ticker"] = sorted(tickers)[0]  # Primary ticker
            if len(tickers) > 1:
                metadata["related_tickers"] = ",".join(sorted(tickers))

        # Detect sector
        text_lower = text.lower()
        for sector, keywords in self.SECTOR_KEYWORDS.items():
            if any(kw.lower() in text_lower for kw in keywords):
                metadata["sector"] = sector
                break

        document.metadata = metadata
        return document
