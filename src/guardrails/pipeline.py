"""Guardrails pipeline orchestrator.

Composes input and output guardrails into a sequential pipeline with
fast-fail ordering: cheapest/fastest checks run first. If a fast regex
catches the problem, the expensive LLM classifier is never invoked.
"""

from dataclasses import dataclass, field
from enum import StrEnum

import structlog

from src.guardrails.content_safety import ContentSafetyGuardrail
from src.guardrails.financial_advice_detector import FinancialAdviceDetector
from src.guardrails.pii_detector import PIIDetector
from src.guardrails.prompt_injection import PromptInjectionDetector
from src.guardrails.topic_filter import TopicClass, TopicFilter

logger = structlog.get_logger()

FINANCIAL_DISCLAIMER = (
    "\n\n---\n*This information is sourced from SEC filings and financial documents "
    "for research purposes only. It does not constitute investment advice. "
    "Always consult a qualified financial advisor before making investment decisions.*"
)

FINANCIAL_KEYWORDS = [
    "revenue",
    "earnings",
    "stock",
    "share",
    "profit",
    "loss",
    "dividend",
    "valuation",
    "margin",
    "eps",
    "filing",
    "10-k",
    "10-q",
]

BLOCKED_RESPONSE = (
    "I'm unable to process this request. TradeIQ is a financial research assistant "
    "focused on SEC filings, earnings reports, and market analysis. "
    "Please ask a question related to financial research."
)

ADVICE_BLOCKED_RESPONSE = (
    "I've detected that my response may contain investment advice, which I'm not "
    "authorized to provide. Let me rephrase with only factual information from "
    "the source documents. Please ask your question again for a research-focused answer."
)


class GuardrailAction(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"
    MODIFY = "modify"


@dataclass
class GuardrailResult:
    action: GuardrailAction
    guardrail_name: str
    reason: str | None = None
    modified_text: str | None = None
    details: dict[str, str | float | bool] = field(default_factory=dict)


class GuardrailsPipeline:
    """Orchestrates input and output guardrails in sequence.

    Input guards run before the RAG chain (fast-fail ordering):
      1. Prompt injection detection (~50ms, local model)
      2. Topic filter regex tier (~0ms)
      3. PII scan + redact (~1ms)
      4. Topic filter LLM tier (~200ms, conditional)

    Output guards run after generation:
      1. Financial advice regex check (~0ms)
      2. PII scan on output (~1ms)
      3. Content safety / toxicity (~100ms)
      4. Disclaimer injection (~0ms)
    """

    def __init__(
        self,
        injection_detector: PromptInjectionDetector | None = None,
        topic_filter: TopicFilter | None = None,
        pii_detector: PIIDetector | None = None,
        advice_detector: FinancialAdviceDetector | None = None,
        content_safety: ContentSafetyGuardrail | None = None,
    ):
        self._injection_detector = injection_detector
        self._topic_filter = topic_filter
        self._pii_detector = pii_detector
        self._advice_detector = advice_detector
        self._content_safety = content_safety

    def run_input_guards(self, query: str) -> GuardrailResult:
        """Run all input guardrails in fast-fail order."""
        # 1. Prompt injection detection
        if self._injection_detector:
            result = self._injection_detector.check(query)
            if result.is_injection:
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    guardrail_name="prompt_injection",
                    reason="Prompt injection attempt detected",
                    details={
                        "confidence": result.confidence,
                        "label": result.label,
                    },
                )

        # 2. Topic filter (regex tier — fast)
        if self._topic_filter:
            topic_result = self._topic_filter.check(query)
            if topic_result.classification == TopicClass.OFF_TOPIC:
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    guardrail_name="topic_filter",
                    reason=topic_result.reason,
                    details={"tier": topic_result.tier},
                )

        # 3. PII scan + redact
        if self._pii_detector:
            pii_result = self._pii_detector.scan(query)
            if pii_result.contains_pii:
                return GuardrailResult(
                    action=GuardrailAction.MODIFY,
                    guardrail_name="pii_detector",
                    reason="PII detected and redacted from query",
                    modified_text=pii_result.redacted_text,
                    details={
                        "detected_types": ", ".join(pii_result.detected_types),
                        "count": float(pii_result.detection_count),
                    },
                )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            guardrail_name="input_pipeline",
            reason="All input guards passed",
        )

    def run_output_guards(self, response: str) -> GuardrailResult:
        """Run all output guardrails in fast-fail order."""
        # 1. Financial advice detection (regex + optional LLM)
        if self._advice_detector:
            advice_result = self._advice_detector.check(response)
            if advice_result.contains_advice:
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    guardrail_name="financial_advice",
                    reason="Response contains financial advice",
                    details={
                        "confidence": advice_result.confidence,
                        "matched": advice_result.matched_pattern or "",
                    },
                )

        # 2. PII scan on output
        if self._pii_detector:
            pii_result = self._pii_detector.scan(response)
            if pii_result.contains_pii:
                return GuardrailResult(
                    action=GuardrailAction.MODIFY,
                    guardrail_name="pii_detector",
                    reason="PII detected and redacted from output",
                    modified_text=pii_result.redacted_text,
                    details={
                        "detected_types": ", ".join(pii_result.detected_types),
                    },
                )

        # 3. Content safety (OpenAI Moderation API)
        if self._content_safety:
            safety_result = self._content_safety.check(response)
            if safety_result.flagged:
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    guardrail_name="content_safety",
                    reason="Response flagged for harmful content",
                    details={
                        "categories": ", ".join(safety_result.flagged_categories),
                        "highest_score": safety_result.highest_score,
                    },
                )

        # 4. Add financial disclaimer if response contains financial content
        if self._should_add_disclaimer(response):
            return GuardrailResult(
                action=GuardrailAction.MODIFY,
                guardrail_name="disclaimer",
                reason="Financial disclaimer appended",
                modified_text=response + FINANCIAL_DISCLAIMER,
            )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            guardrail_name="output_pipeline",
            reason="All output guards passed",
        )

    @staticmethod
    def _should_add_disclaimer(text: str) -> bool:
        """Check if the response discusses financial topics that need a disclaimer."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in FINANCIAL_KEYWORDS)
