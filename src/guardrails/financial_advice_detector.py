"""Detects financial advice in LLM outputs.

Uses a hybrid regex + LLM approach: fast regex patterns catch obvious
advice language, with an optional LLM classifier for nuanced cases.
Critical for regulatory compliance in financial RAG systems.
"""

import re
from dataclasses import dataclass

import structlog

from src.generation.llm_service import LLMService

logger = structlog.get_logger()

# Regex patterns that indicate investment advice
ADVICE_PATTERNS = [
    # Direct recommendations
    (
        r"\b(you should|i recommend|i suggest|i advise|we recommend)\b"
        r".*\b(buy|sell|hold|invest|trade)\b"
    ),
    r"\b(buy|sell|short|go long)\b.*\b(stock|share|bond|option|future|etf|crypto)\b",
    # Price predictions
    r"\bprice target\b.*\$[\d,]+",
    (
        r"\b(will|going to|expected to|predicted to)\b"
        r".*\b(rise|fall|crash|surge|plummet|skyrocket|moon)\b"
    ),
    # Guarantees
    r"\b(guaranteed|certain|definitely|surely|absolutely)\b.*\b(return|profit|gain|growth)\b",
    # Tax/legal advice
    r"\b(tax deduction|tax advice|tax strategy|write off|tax shelter)\b",
    r"\b(you can deduct|claim as deduction|reduce your tax)\b",
    # Urgency-based advice
    r"\b(act now|don't miss|limited time|before it's too late|hurry)\b",
    # Self-aware disclaimer (if it disclaims, it was likely advising)
    r"\bthis is not financial advice\b",
]

ADVICE_CHECK_PROMPT = """You are a financial compliance checker.

Determine if the response contains investment advice, price predictions,
buy/sell recommendations, or tax/legal advice.

RESEARCH (allowed): Presenting facts from filings, explaining metrics,
comparing historical data, quoting what a company reported, describing
market conditions factually.

ADVICE (not allowed): Recommending actions to take, predicting future
prices, suggesting specific trades, providing tax strategies, making
guarantees about returns, urging urgency in financial decisions.

Response to check:
{response}

Answer with ONLY one word: "research" or "advice"."""


@dataclass
class AdviceDetectionResult:
    contains_advice: bool
    confidence: str  # "high" (regex match) or "moderate" (LLM detected)
    matched_pattern: str | None = None


class FinancialAdviceDetector:
    """Detects financial advice in LLM outputs to ensure regulatory compliance."""

    def __init__(self, llm: LLMService | None = None):
        self._llm = llm
        self._patterns = [re.compile(p, re.IGNORECASE) for p in ADVICE_PATTERNS]

    def check(self, response_text: str) -> AdviceDetectionResult:
        """Check if a response contains financial advice."""
        # Tier 1: Fast regex scan
        for pattern in self._patterns:
            match = pattern.search(response_text)
            if match:
                logger.warning(
                    "financial_advice_detected",
                    matched=match.group()[:100],
                    confidence="high",
                )
                return AdviceDetectionResult(
                    contains_advice=True,
                    confidence="high",
                    matched_pattern=match.group()[:100],
                )

        # Tier 2: LLM classification (if available)
        if self._llm:
            return self._check_llm(response_text)

        return AdviceDetectionResult(contains_advice=False, confidence="high")

    def _check_llm(self, response_text: str) -> AdviceDetectionResult:
        """Use LLM to detect subtle financial advice."""
        try:
            response = self._llm.generate(
                messages=[
                    {
                        "role": "user",
                        "content": ADVICE_CHECK_PROMPT.format(response=response_text[:2000]),
                    }
                ],
                temperature=0.0,
                max_tokens=10,
            )
            result = response.content.strip().lower()

            if "advice" in result:
                logger.warning("financial_advice_llm_detected")
                return AdviceDetectionResult(
                    contains_advice=True,
                    confidence="moderate",
                )

            return AdviceDetectionResult(contains_advice=False, confidence="moderate")
        except Exception:
            logger.warning("advice_detection_llm_failed", exc_info=True)
            return AdviceDetectionResult(contains_advice=False, confidence="low")
