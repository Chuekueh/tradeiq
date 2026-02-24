from fastapi import APIRouter, HTTPException

from src.api.dependencies import app_state
from src.api.schemas.request import IngestRequest
from src.api.schemas.response import IngestResponse
from src.ingestion.chunkers.recursive_chunker import RecursiveChunker
from src.ingestion.loaders.html_loader import HTMLLoader
from src.ingestion.loaders.json_loader import JSONLoader
from src.ingestion.loaders.markdown_loader import MarkdownLoader
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.processors.metadata_extractor import MetadataExtractor
from src.ingestion.processors.text_cleaner import TextCleaner

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    if not app_state.settings or not app_state.vector_store or not app_state.embedding_service:
        raise HTTPException(status_code=503, detail="Services not initialized")

    pipeline = IngestionPipeline(
        loaders=[MarkdownLoader(), HTMLLoader(), JSONLoader()],
        chunker=RecursiveChunker(
            chunk_size=app_state.settings.chunk_size,
            chunk_overlap=app_state.settings.chunk_overlap,
        ),
        metadata_extractor=MetadataExtractor(),
        text_cleaner=TextCleaner(),
    )

    chunks, result = pipeline.process_directory(request.source_path)

    if not chunks:
        return IngestResponse(
            documents_loaded=0, chunks_created=0, chunks_stored=0, source=request.source_path
        )

    texts = [chunk.content for chunk in chunks]
    embeddings = app_state.embedding_service.embed_documents(texts)

    app_state.vector_store.add(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[chunk.metadata for chunk in chunks],
    )

    return IngestResponse(
        documents_loaded=result.documents_loaded,
        chunks_created=result.chunks_created,
        chunks_stored=len(chunks),
        source=request.source_path,
    )
