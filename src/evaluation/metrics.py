import math

from src.generation.llm_service import LLMService


class RAGMetrics:
    """Retrieval and generation quality metrics for RAG evaluation."""

    @staticmethod
    def context_precision(retrieved_sources: list[str], relevant_sources: list[str]) -> float:
        """Fraction of retrieved documents that are relevant."""
        if not retrieved_sources:
            return 0.0
        relevant_set = set(relevant_sources)
        hits = sum(1 for s in retrieved_sources if s in relevant_set)
        return hits / len(retrieved_sources)

    @staticmethod
    def context_recall(retrieved_sources: list[str], relevant_sources: list[str]) -> float:
        """Fraction of relevant documents that were retrieved."""
        if not relevant_sources:
            return 1.0
        retrieved_set = set(retrieved_sources)
        hits = sum(1 for s in relevant_sources if s in retrieved_set)
        return hits / len(relevant_sources)

    @staticmethod
    def mrr(retrieved_sources: list[str], relevant_sources: list[str]) -> float:
        """Mean Reciprocal Rank — how high is the first relevant result."""
        relevant_set = set(relevant_sources)
        for i, source in enumerate(retrieved_sources):
            if source in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_sources: list[str], relevant_sources: list[str], k: int = 5) -> float:
        """Normalized Discounted Cumulative Gain at k."""
        relevant_set = set(relevant_sources)

        # DCG
        dcg = 0.0
        for i, source in enumerate(retrieved_sources[:k]):
            rel = 1.0 if source in relevant_set else 0.0
            dcg += rel / math.log2(i + 2)

        # Ideal DCG
        ideal_rels = sorted(
            [1.0 if s in relevant_set else 0.0 for s in retrieved_sources[:k]],
            reverse=True,
        )
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def faithfulness(answer: str, contexts: list[str], llm: LLMService) -> float:
        """LLM-as-judge: Is the answer faithful to the provided context?"""
        context_text = "\n\n".join(contexts)
        response = llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an evaluation judge. Score how faithful the answer is to "
                        "the provided context. A faithful answer only contains information "
                        "supported by the context. Score from 0.0 to 1.0. "
                        "Respond with ONLY a number."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context_text}\n\nAnswer:\n{answer}\n\nFaithfulness score:"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=10,
        )
        try:
            return min(max(float(response.content.strip()), 0.0), 1.0)
        except ValueError:
            return 0.0

    @staticmethod
    def answer_relevancy(question: str, answer: str, llm: LLMService) -> float:
        """LLM-as-judge: Is the answer relevant to the question?"""
        response = llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an evaluation judge. Score how relevant the answer is to "
                        "the question. A relevant answer directly addresses what was asked. "
                        "Score from 0.0 to 1.0. Respond with ONLY a number."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question:\n{question}\n\nAnswer:\n{answer}\n\nRelevancy score:",
                },
            ],
            temperature=0.0,
            max_tokens=10,
        )
        try:
            return min(max(float(response.content.strip()), 0.0), 1.0)
        except ValueError:
            return 0.0
