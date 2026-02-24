import re
from dataclasses import dataclass


@dataclass
class ParsedResponse:
    answer: str
    cited_sources: list[str]


class ResponseParser:
    """Parses LLM responses to extract citations and structure."""

    SOURCE_PATTERN = re.compile(r"\[Source:\s*([^\]]+)\]")

    def parse(self, raw_response: str) -> ParsedResponse:
        cited_sources = list(set(self.SOURCE_PATTERN.findall(raw_response)))
        return ParsedResponse(answer=raw_response, cited_sources=cited_sources)
