"""PII detection and redaction for input and output text.

Uses regex-based pattern matching for common PII types (SSN, credit cards,
email, phone numbers). For production use, integrate Microsoft Presidio.
"""

import re
from dataclasses import dataclass

import structlog

logger = structlog.get_logger()

# PII detection patterns
PII_PATTERNS = {
    "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "PHONE_US": re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "US_BANK_ACCOUNT": re.compile(r"\b\d{8,17}\b"),  # Broad — only used with context
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

# Redaction placeholders
REDACTION_MAP = {
    "US_SSN": "[SSN REDACTED]",
    "CREDIT_CARD": "[CREDIT CARD REDACTED]",
    "EMAIL": "[EMAIL REDACTED]",
    "PHONE_US": "[PHONE REDACTED]",
    "US_BANK_ACCOUNT": "[ACCOUNT NUMBER REDACTED]",
    "IP_ADDRESS": "[IP REDACTED]",
}


@dataclass
class PIIResult:
    contains_pii: bool
    redacted_text: str
    detected_types: list[str]
    detection_count: int


class PIIDetector:
    """Detects and redacts PII from text using pattern matching.

    Scans both input queries and output responses to prevent PII
    from being processed or leaked through the RAG pipeline.
    """

    def __init__(self, entities: list[str] | None = None):
        """Initialize with specific PII entity types to detect.

        Args:
            entities: List of entity types to scan for. If None, scans for
                      SSN, credit cards, email, and phone numbers (not bank
                      accounts or IPs which have high false positive rates).
        """
        if entities is None:
            # Default: high-confidence patterns only
            self._active_patterns = {
                k: v
                for k, v in PII_PATTERNS.items()
                if k in ("US_SSN", "CREDIT_CARD", "EMAIL", "PHONE_US")
            }
        else:
            self._active_patterns = {k: v for k, v in PII_PATTERNS.items() if k in entities}

    def scan(self, text: str) -> PIIResult:
        """Scan text for PII and return redacted version."""
        detected_types: list[str] = []
        redacted = text
        total_count = 0

        for pii_type, pattern in self._active_patterns.items():
            matches = pattern.findall(redacted)
            if matches:
                detected_types.append(pii_type)
                total_count += len(matches)
                placeholder = REDACTION_MAP.get(pii_type, "[REDACTED]")
                redacted = pattern.sub(placeholder, redacted)

        if detected_types:
            logger.warning(
                "pii_detected",
                types=detected_types,
                count=total_count,
            )

        return PIIResult(
            contains_pii=len(detected_types) > 0,
            redacted_text=redacted,
            detected_types=detected_types,
            detection_count=total_count,
        )

    def scan_and_block(self, text: str) -> PIIResult:
        """Scan text and flag if PII is found (without redacting)."""
        result = self.scan(text)
        return result
