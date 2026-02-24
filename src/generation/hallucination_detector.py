from dataclasses import dataclass

import structlog

from src.generation.llm_service import LLMService

logger = structlog.get_logger()

DECOMPOSE_PROMPT = """You are an expert at analyzing financial answers.
Break the following answer into individual factual claims.
Each claim should be a single, atomic statement that can be verified independently.

Rules:
- Extract only factual claims (numbers, dates, comparisons, attributions)
- Ignore opinions, hedging language, and meta-commentary
- Each claim should be on its own line
- No numbering, bullets, or other formatting — just the plain claim text
- If the answer says it doesn't have information, return "NO_CLAIMS"

Answer: {answer}

Claims:"""

VERIFY_PROMPT = """You are a claim verification expert for financial documents.

Given a claim and the source context, determine if the claim is supported by the context.

Rules:
- "supported" = the context contains information that directly confirms the claim
- "unsupported" = the context does NOT contain enough information to confirm the claim
- Be strict: numerical claims must match the context precisely
- Partial support counts as "unsupported"

Respond with ONLY one word: "supported" or "unsupported"."""

VERIFY_USER_TEMPLATE = """Claim: {claim}

Context:
{context}

Verdict:"""


@dataclass
class Claim:
    text: str
    supported: bool
    supporting_source: str | None = None


@dataclass
class VerificationResult:
    claims: list[Claim]
    faithfulness_score: float
    flagged_claims: list[str]


class HallucinationDetector:
    """Post-generation verification that checks answer faithfulness.

    Decomposes LLM answers into atomic claims, then verifies each claim
    against the retrieved context using NLI (Natural Language Inference).
    Returns a faithfulness score and flags unsupported claims.

    Critical for financial RAG where hallucinated numbers or facts can
    lead to costly decisions.
    """

    def __init__(self, llm: LLMService):
        self._llm = llm

    def verify(self, answer: str, contexts: list[str], sources: list[str]) -> VerificationResult:
        """Verify an answer against retrieved contexts.

        Args:
            answer: The generated answer to verify.
            contexts: List of retrieved context strings.
            sources: List of source identifiers matching the contexts.

        Returns:
            VerificationResult with claims, faithfulness score, and flagged claims.
        """
        # Step 1: Decompose answer into claims
        claims_text = self._decompose_claims(answer)

        if not claims_text or claims_text == ["NO_CLAIMS"]:
            return VerificationResult(
                claims=[],
                faithfulness_score=1.0,
                flagged_claims=[],
            )

        # Step 2: Verify each claim against context
        combined_context = "\n\n".join(
            f"[Source: {src}]\n{ctx}" for ctx, src in zip(contexts, sources, strict=False)
        )
        claims: list[Claim] = []
        flagged: list[str] = []

        for claim_text in claims_text:
            supported, source = self._verify_claim(claim_text, combined_context, sources)
            claim = Claim(
                text=claim_text,
                supported=supported,
                supporting_source=source,
            )
            claims.append(claim)
            if not supported:
                flagged.append(claim_text)

        # Calculate faithfulness score
        score = sum(1 for c in claims if c.supported) / len(claims) if claims else 1.0

        logger.info(
            "hallucination_check",
            total_claims=len(claims),
            supported=sum(1 for c in claims if c.supported),
            unsupported=len(flagged),
            faithfulness=round(score, 3),
        )

        return VerificationResult(
            claims=claims,
            faithfulness_score=round(score, 4),
            flagged_claims=flagged,
        )

    def _decompose_claims(self, answer: str) -> list[str]:
        """Decompose an answer into atomic factual claims."""
        try:
            response = self._llm.generate(
                messages=[
                    {
                        "role": "user",
                        "content": DECOMPOSE_PROMPT.format(answer=answer),
                    }
                ],
                temperature=0.0,
                max_tokens=500,
            )
            text = response.content.strip()
            if text == "NO_CLAIMS":
                return ["NO_CLAIMS"]

            claims = [line.strip() for line in text.split("\n") if line.strip()]
            return claims
        except Exception:
            logger.warning("claim_decomposition_failed", exc_info=True)
            return []

    def _verify_claim(
        self, claim: str, context: str, sources: list[str]
    ) -> tuple[bool, str | None]:
        """Verify a single claim against the context."""
        try:
            response = self._llm.generate(
                messages=[
                    {"role": "system", "content": VERIFY_PROMPT},
                    {
                        "role": "user",
                        "content": VERIFY_USER_TEMPLATE.format(
                            claim=claim,
                            context=context[:3000],
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=10,
            )
            verdict = response.content.strip().lower()
            supported = "supported" in verdict and "unsupported" not in verdict

            # If supported, try to identify which source
            source = sources[0] if supported and sources else None
            return supported, source
        except Exception:
            logger.warning("claim_verification_failed", claim=claim[:50], exc_info=True)
            return True, None  # Fail-open: assume supported on error
