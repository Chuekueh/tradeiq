from openai import OpenAI

from src.config import Settings


class QueryTransformer:
    """Transforms user queries for better retrieval."""

    def __init__(self, settings: Settings):
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model

    def expand_query(self, query: str) -> list[str]:
        """Generate multiple query variations for broader retrieval."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate 3 alternative phrasings of the given financial query. "
                        "Each should capture the same intent but use different terminology. "
                        "Return only the queries, one per line, no numbering."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.7,
            max_tokens=200,
        )
        text = response.choices[0].message.content or ""
        variants = [line.strip() for line in text.strip().split("\n") if line.strip()]
        return [query] + variants

    def generate_hypothetical_document(self, query: str) -> str:
        """HyDE: Generate a hypothetical answer to use as the query embedding."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial analyst. Given a question, write a short paragraph "
                        "that would answer it, as if it were extracted from an SEC filing or "
                        "financial report. Be specific and use financial terminology."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        return response.choices[0].message.content or query
