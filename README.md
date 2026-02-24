# TradeIQ - Financial Research Knowledge Base

> RAG-powered Q&A system for SEC filings, earnings reports, and trading strategies.

[![CI](https://github.com/yourusername/tradeiq/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/tradeiq/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Architecture

```
User Query
    |
    v
[Streamlit Frontend] --HTTP--> [FastAPI Backend]
                                      |
                                      v
                              [Query Transformer]
                                      |
                                      v
                              [Hybrid Search]
                              /              \
                    [BM25 Sparse]    [ChromaDB Dense]
                              \              /
                              [Reciprocal Rank Fusion]
                                      |
                                      v
                              [Cross-Encoder Reranker]
                                      |
                                      v
                              [Context Builder + Memory]
                                      |
                                      v
                              [OpenAI gpt-4o-mini]
                                      |
                                      v
                              [Response with Citations]
```

## Features

- **Hybrid Search** - BM25 + vector similarity with Reciprocal Rank Fusion
- **Cross-Encoder Reranking** - Retrieve broadly, rerank precisely
- **SEC EDGAR Integration** - Real 10-K/10-Q filings from public EDGAR API
- **Metadata Filtering** - Filter by ticker, filing type, sector, date
- **Source Citations** - Every answer includes exact filing sections with relevance scores
- **Conversation Memory** - SQLite-backed session persistence across restarts
- **Streaming Responses** - Real-time token-by-token generation
- **Evaluation Framework** - Context precision/recall, faithfulness, MRR, nDCG metrics
- **Production-Ready** - Docker, CI/CD, structured logging, health checks

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key

### Local Development

```bash
# Clone and install
git clone https://github.com/yourusername/tradeiq.git
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
  ingestion/     # Document loaders, chunkers, metadata extraction
  embeddings/    # Sentence-transformers embedding service
  vectorstore/   # ChromaDB with persistence and metadata filtering
  retrieval/     # Hybrid search, cross-encoder reranking, query transformation
  generation/    # LLM service, prompts, RAG chain orchestration
  memory/        # SQLite conversation memory
  evaluation/    # RAG metrics and evaluation pipeline
  api/           # FastAPI application with versioned routes
  frontend/      # Streamlit chat interface
```

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
