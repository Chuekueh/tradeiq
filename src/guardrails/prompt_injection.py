"""Prompt injection detection using a fine-tuned DeBERTa classifier.

Uses ProtectAI's deberta-v3-base-prompt-injection-v2 model to detect
prompt injection attempts before they reach the RAG pipeline. Runs
locally on CPU with ~50ms latency per classification.
"""

from dataclasses import dataclass

import structlog

logger = structlog.get_logger()


@dataclass
class InjectionResult:
    is_injection: bool
    confidence: float
    label: str


class PromptInjectionDetector:
    """Detects prompt injection attempts using a fine-tuned DeBERTa classifier.

    This guard runs BEFORE the query enters the RAG pipeline, blocking
    attempts to override system instructions or extract sensitive data
    through crafted queries.
    """

    def __init__(self, threshold: float = 0.85):
        self._threshold = threshold
        self._classifier = None

    def _load_model(self) -> None:
        """Lazy-load the model on first use to avoid slow startup."""
        if self._classifier is not None:
            return

        try:
            from transformers import pipeline

            self._classifier = pipeline(
                "text-classification",
                model="ProtectAI/deberta-v3-base-prompt-injection-v2",
                truncation=True,
                max_length=512,
            )
            logger.info("prompt_injection_model_loaded")
        except Exception:
            logger.warning("prompt_injection_model_unavailable", exc_info=True)
            self._classifier = None

    def check(self, text: str) -> InjectionResult:
        """Check if the input text contains a prompt injection attempt."""
        self._load_model()

        if self._classifier is None:
            # Model not available — fail-open
            return InjectionResult(is_injection=False, confidence=0.0, label="SAFE")

        try:
            result = self._classifier(text)[0]
            is_injection = result["label"] == "INJECTION" and result["score"] >= self._threshold

            if is_injection:
                logger.warning(
                    "prompt_injection_detected",
                    confidence=round(result["score"], 3),
                    text_preview=text[:100],
                )

            return InjectionResult(
                is_injection=is_injection,
                confidence=round(result["score"], 4),
                label=result["label"],
            )
        except Exception:
            logger.warning("prompt_injection_check_failed", exc_info=True)
            return InjectionResult(is_injection=False, confidence=0.0, label="ERROR")
