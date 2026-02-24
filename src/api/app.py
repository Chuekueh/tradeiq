from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import app_state
from src.api.middleware import RequestLoggingMiddleware
from src.api.routes import conversations, health, ingest, query

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("starting_tradeiq")
    app_state.initialize()
    doc_count = app_state.vector_store.count() if app_state.vector_store else 0
    logger.info("tradeiq_ready", documents=doc_count)
    yield
    logger.info("shutting_down_tradeiq")
    app_state.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(
        title="TradeIQ - Financial Research Knowledge Base",
        description="RAG-powered Q&A for SEC filings, earnings, and trading strategies",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)

    # Routes
    app.include_router(query.router, prefix="/api/v1", tags=["Query"])
    app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])
    app.include_router(health.router, prefix="/api/v1", tags=["Health"])
    app.include_router(conversations.router, prefix="/api/v1", tags=["Conversations"])

    return app
