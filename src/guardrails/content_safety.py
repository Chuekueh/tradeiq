"""Content safety guardrail using OpenAI's Moderation API.

Detects toxic, harmful, or inappropriate content in both inputs
and outputs. Free to use with any OpenAI API key.
"""

from dataclasses import dataclass

import structlog
from openai import OpenAI

logger = structlog.get_logger()


@dataclass
class ModerationResult:
    flagged: bool
    flagged_categories: list[str]
    highest_score: float
    highest_category: str


class ContentSafetyGuardrail:
    """Uses OpenAI Moderation API for toxicity and harmful content detection.

    The Moderation API is free and fast (~100ms). It checks for hate speech,
    harassment, self-harm, sexual content, and violence.
    """

    def __init__(self, api_key: str):
        self._client = OpenAI(api_key=api_key)

    def check(self, text: str) -> ModerationResult:
        """Check text for harmful content."""
        try:
            response = self._client.moderations.create(
                model="omni-moderation-latest",
                input=text,
            )
            result = response.results[0]

            # Collect flagged categories
            categories = result.categories
            flagged_cats: list[str] = []
            scores: dict[str, float] = {}

            for category_name in [
                "harassment",
                "harassment_threatening",
                "hate",
                "hate_threatening",
                "self_harm",
                "self_harm_instructions",
                "self_harm_intent",
                "sexual",
                "sexual_minors",
                "violence",
                "violence_graphic",
            ]:
                is_flagged = getattr(categories, category_name, False)
                score = getattr(result.category_scores, category_name, 0.0)
                scores[category_name] = score
                if is_flagged:
                    flagged_cats.append(category_name)

            # Find highest scoring category
            highest_cat = max(scores, key=lambda k: scores[k]) if scores else ""
            highest_score = scores.get(highest_cat, 0.0)

            if result.flagged:
                logger.warning(
                    "content_safety_flagged",
                    categories=flagged_cats,
                    highest_score=round(highest_score, 4),
                )

            return ModerationResult(
                flagged=result.flagged,
                flagged_categories=flagged_cats,
                highest_score=round(highest_score, 4),
                highest_category=highest_cat,
            )
        except Exception:
            logger.warning("content_safety_check_failed", exc_info=True)
            # Fail-open on API error
            return ModerationResult(
                flagged=False,
                flagged_categories=[],
                highest_score=0.0,
                highest_category="",
            )
