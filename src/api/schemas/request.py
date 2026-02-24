from pydantic import BaseModel, Field


class QueryFilters(BaseModel):
    ticker: str | None = Field(None, description="Filter by stock ticker (e.g., AAPL)")
    filing_type: str | None = Field(None, description="Filter by filing type (e.g., 10-K)")
    sector: str | None = Field(None, description="Filter by sector (e.g., technology)")

    def to_chroma_where(self) -> dict | None:
        conditions = []
        if self.ticker:
            conditions.append({"ticker": self.ticker.upper()})
        if self.filing_type:
            conditions.append({"filing_type": self.filing_type.upper()})
        if self.sector:
            conditions.append({"sector": self.sector.lower()})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    session_id: str | None = Field(None, description="Conversation session ID")
    filters: QueryFilters | None = None
    top_k: int = Field(5, ge=1, le=20)
    stream: bool = Field(False)


class IngestRequest(BaseModel):
    source_path: str = Field(..., description="Path to file or directory to ingest")
