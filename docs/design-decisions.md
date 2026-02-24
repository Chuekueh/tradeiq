# Architecture Decision Records

## ADR-001: ChromaDB as Vector Store

**Context**: Need a vector database for storing and searching document embeddings.

**Decision**: ChromaDB with persistent storage.

**Rationale**:
- Clean Python-native API with metadata filtering
- Built-in persistence (no external server needed)
- Supports cosine similarity and metadata WHERE clauses
- Easy local development; no Docker dependency for the DB itself
- In production, would consider Qdrant or Weaviate for horizontal scaling

## ADR-002: Hybrid Search over Pure Vector Search

**Context**: Financial documents contain both technical terms (ticker symbols, "10-K", "EPS") and conceptual queries ("risk factors", "growth outlook").

**Decision**: BM25 + vector search with Reciprocal Rank Fusion.

**Rationale**:
- BM25 excels at exact keyword matching (ticker symbols, financial terms)
- Vector search captures semantic similarity (conceptual queries)
- RRF provides a simple, parameter-free way to merge ranked lists
- Alpha parameter (default 0.7) allows tuning the balance

## ADR-003: Cross-Encoder Reranking

**Context**: Bi-encoder (embedding) similarity is fast but imprecise. For top-k results, we need higher precision.

**Decision**: Retrieve broadly (top 15) with hybrid search, then rerank to top 3-5 with a cross-encoder.

**Rationale**:
- Cross-encoders score (query, document) pairs jointly, yielding much better relevance
- `ms-marco-MiniLM-L-6-v2` is small and fast enough for real-time use
- The retrieve-then-rerank pattern is industry standard in production RAG

## ADR-004: OpenAI gpt-4o-mini as LLM

**Context**: Need a capable LLM for answer generation.

**Decision**: OpenAI gpt-4o-mini as the primary (and only) LLM provider.

**Rationale**:
- Extremely cost-effective (~$0.15/1M input tokens)
- Strong instruction following and financial knowledge
- Consistent API and streaming support
- Demonstrates API integration skills for portfolio

## ADR-005: SQLite for Conversation Memory

**Context**: Need to persist conversation history across sessions.

**Decision**: SQLite via Python's built-in sqlite3 module.

**Rationale**:
- Zero-dependency (built into Python)
- Persists across restarts (unlike in-memory stores)
- Fast enough for conversation-scale data
- No external database server needed
- In production, would use PostgreSQL or Redis for multi-instance deployments

## ADR-006: Recursive Character Splitting for Chunking

**Context**: Need to split financial documents into chunks for embedding.

**Decision**: RecursiveCharacterTextSplitter with Markdown-aware separators.

**Rationale**:
- Respects document structure (splits on headers first, then paragraphs)
- 512-token chunks with 50-token overlap balances context and specificity
- Markdown separators align well with SEC filing section structure
- Chunk overlap prevents losing context at boundaries
