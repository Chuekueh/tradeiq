# TradeIQ - Financial Research Knowledge Base

> Production-grade RAG system for SEC filings, earnings reports, and trading strategies — featuring 7 advanced retrieval techniques and a 5-layer guardrails pipeline.

[![CI](https://github.com/Chuekueh/tradeiq/actions/workflows/ci.yml/badge.svg)](https://github.com/Chuekueh/tradeiq/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Architecture

```
User Query
    │
    ▼
┌─────────────────────── GUARDRAILS (Input) ──────────────────────┐
│  Prompt Injection Detection (DeBERTa) → Topic Filter (regex+LLM)│
│  → PII Redaction (regex) → Content Safety (OpenAI Moderation)   │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
                    ┌─── Query Router ───┐
                    │ Simple │ Moderate  │ Complex
                    │ top_k=3│ top_k=5  │ top_k=8
                    └────────┬──────────┘
                             ▼
                  ┌─── Hybrid Search ───┐
                  │  BM25     Vector    │
                  │  Sparse   Dense     │
                  └────────┬────────────┘
                           ▼
                  Reciprocal Rank Fusion
                           ▼
                  Cross-Encoder Reranking
                           ▼
                  Corrective RAG (relevance gate)
                           ▼
                  Lost-in-Middle Reordering
                           ▼
                  Context Builder + Memory
                           ▼
                    OpenAI gpt-4o-mini
                           ▼
┌─────────────────── GUARDRAILS (Output) ─────────────────────────┐
│  Financial Advice Detection → PII Scan → Content Safety         │
│  → Hallucination Detection → Financial Disclaimer               │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
                    Response with Citations
```

## Features

### Core RAG Pipeline
- **Hybrid Search** — BM25 + vector similarity with Reciprocal Rank Fusion
- **Cross-Encoder Reranking** — Retrieve broadly, rerank precisely with `ms-marco-MiniLM-L-6-v2`
- **SEC EDGAR Integration** — Real 10-K/10-Q filings from public EDGAR API
- **Metadata Filtering** — Filter by ticker, filing type, sector, date
- **Source Citations** — Every answer includes exact filing sections with relevance scores
- **Conversation Memory** — SQLite-backed session persistence across restarts
- **Streaming Responses** — Real-time token-by-token generation
- **Evaluation Framework** — Context precision/recall, faithfulness, MRR, nDCG metrics

### Advanced RAG Techniques
- **Semantic Chunking** — Embedding-based boundary detection using sentence similarity instead of fixed-size splits. Computes cosine similarity between consecutive sentences and splits at statistical breakpoints (percentile-based).
- **Contextual Retrieval** — LLM-generated context prepended to each chunk before embedding (Anthropic's technique). Reduces retrieval failures by providing document-level context to isolated chunks.
- **Parent-Child Retrieval** — Index small chunks (256 tokens) for precise matching, return larger parent chunks (1024 tokens) for richer LLM context. Small-to-big retrieval with deduplication.
- **Corrective RAG (CRAG)** — Post-retrieval quality gate that grades each document's relevance. Filters irrelevant documents; returns honest "insufficient information" when all retrieved docs are off-target.
- **Lost-in-the-Middle Reordering** — Reorders documents so the most relevant appear at the beginning and end of context, avoiding the "middle" positions where LLMs perform worst (Liu et al. 2023).
- **Adaptive Query Routing** — Classifies queries by complexity (simple/moderate/complex) and dynamically adjusts retrieval strategy, top_k, and whether to use multi-query expansion.
- **Hallucination Detection** — Decomposes LLM answers into atomic claims, then verifies each claim against retrieved context using NLI. Returns a faithfulness score and flags unsupported claims.

### Guardrails & Safety
- **Prompt Injection Detection** — Fine-tuned DeBERTa classifier (`ProtectAI/deberta-v3-base-prompt-injection-v2`) blocks injection attempts before they reach the pipeline. ~50ms local inference.
- **Topic Filter** — Two-tier system: fast regex patterns (~0ms) catch obvious off-topic queries, with an LLM classifier (~200ms) for ambiguous cases. Keeps the system focused on financial research.
- **PII Detection & Redaction** — Regex-based scanning for SSN, credit cards, email, phone numbers. Detects and redacts PII in both inputs and outputs to prevent data leakage.
- **Financial Advice Detection** — Regex + LLM hybrid detector blocks investment recommendations, price predictions, and guaranteed-return claims. Critical for regulatory compliance.
- **Content Safety** — OpenAI Moderation API checks for toxic, harmful, or inappropriate content in LLM outputs.
- **Financial Disclaimer** — Automatically appends research-only disclaimers to responses containing financial content.

### Production Infrastructure
- **Docker & Docker Compose** — Containerized deployment
- **CI/CD** — GitHub Actions with lint, test, and evaluation pipelines
- **Structured Logging** — Production-ready observability with structlog
- **Health Checks** — API health and readiness endpoints

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key

### Local Development

```bash
# Clone and install
git clone https://github.com/Chuekueh/tradeiq.git
cd tradeiq
pip install -e ".[dev,frontend]"

# Configure
cp .env.example .env
# Edit .env and add your OpenAI API key

# Generate sample data
python scripts/generate_synthetic_data.py
python scripts/fetch_sec_filings.py   # Optional: fetch real SEC filings

# Ingest into vector store
python scripts/ingest_data.py --source data/raw/

# Start the API
make run

# In another terminal, start the frontend
make run-frontend
```

Visit `http://localhost:8501` for the chat UI and `http://localhost:8000/docs` for the API docs.

### Docker

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key

docker compose up --build
```

## Example Queries

- "What were Apple's main risk factors in their latest 10-K filing?"
- "Compare revenue growth between MSFT and GOOGL"
- "What trading strategies work best in high-volatility markets?"
- "Summarize the key takeaways from Tesla's Q3 earnings call"
- "What is pairs trading and how does it achieve market neutrality?"

## Project Structure

```
src/
  ingestion/       # Document loaders, chunkers (recursive + semantic), metadata extraction
    chunkers/      # Recursive, semantic chunking, parent-child splitting
    processors/    # Context enrichment (contextual retrieval)
  embeddings/      # Sentence-transformers embedding service
  vectorstore/     # ChromaDB with persistence and metadata filtering
  retrieval/       # Hybrid search, reranking, CRAG, query routing, parent-child retrieval
  generation/      # LLM service, prompts, RAG chain, hallucination detection
  guardrails/      # Input/output safety: injection, topic, PII, advice, content safety
  memory/          # SQLite conversation memory
  evaluation/      # RAG metrics and evaluation pipeline
  api/             # FastAPI application with versioned routes
  frontend/        # Streamlit chat interface
tests/
  unit/            # 106 unit tests across all modules
```

## Configuration

All features are configurable via environment variables or `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| `RAG_CORRECTIVE_RAG_ENABLED` | `true` | Enable CRAG relevance grading |
| `RAG_ADAPTIVE_ROUTING_ENABLED` | `true` | Enable query complexity routing |
| `RAG_HALLUCINATION_DETECTION_ENABLED` | `true` | Enable post-generation claim verification |
| `RAG_GUARDRAILS_ENABLED` | `true` | Enable the full guardrails pipeline |
| `RAG_PROMPT_INJECTION_DETECTION_ENABLED` | `true` | Enable DeBERTa injection detection |
| `RAG_TOPIC_FILTER_ENABLED` | `true` | Enable on-topic enforcement |
| `RAG_PII_DETECTION_ENABLED` | `true` | Enable PII redaction |
| `RAG_FINANCIAL_ADVICE_DETECTION_ENABLED` | `true` | Enable advice blocking |
| `RAG_CONTENT_SAFETY_ENABLED` | `true` | Enable OpenAI Moderation |
| `RAG_CHUNKING_STRATEGY` | `recursive` | `recursive` or `semantic` |
| `RAG_CONTEXTUAL_RETRIEVAL_ENABLED` | `false` | Enable LLM context prepending |
| `RAG_PARENT_CHILD_ENABLED` | `true` | Enable parent-child retrieval |

## Evaluation

Run the evaluation pipeline:

```bash
make evaluate
```

This computes: context precision, context recall, MRR, nDCG@5, faithfulness (LLM-as-judge), and answer relevancy across a curated financial Q&A dataset.

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | OpenAI gpt-4o-mini |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | ChromaDB |
| Search | BM25 + Dense + Reciprocal Rank Fusion |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Chunking | Recursive + Semantic (embedding-based) |
| Guardrails | DeBERTa (injection), OpenAI Moderation, regex + LLM |
| Backend | FastAPI |
| Frontend | Streamlit |
| Memory | SQLite |
| Containers | Docker + Docker Compose |
| CI/CD | GitHub Actions |

## Development

```bash
make lint          # Lint with ruff
make format        # Auto-format
make test          # Run all tests
make test-unit     # Run unit tests only
make type-check    # Type check with mypy
```

## License

MIT
