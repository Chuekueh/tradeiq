"""Ingest documents into the TradeIQ vector store."""

import argparse
import time

from src.config import get_settings
from src.embeddings.embedding_service import EmbeddingService
from src.ingestion.chunkers.recursive_chunker import RecursiveChunker
from src.ingestion.loaders.html_loader import HTMLLoader
from src.ingestion.loaders.json_loader import JSONLoader
from src.ingestion.loaders.markdown_loader import MarkdownLoader
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.processors.metadata_extractor import MetadataExtractor
from src.ingestion.processors.text_cleaner import TextCleaner
from src.vectorstore.chroma_store import ChromaVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into TradeIQ")
    parser.add_argument("--source", type=str, default="data/raw/", help="Source directory")
    args = parser.parse_args()

    settings = get_settings()
    print(f"Ingesting documents from: {args.source}")
    start = time.perf_counter()

    # Initialize components
    pipeline = IngestionPipeline(
        loaders=[MarkdownLoader(), HTMLLoader(), JSONLoader()],
        chunker=RecursiveChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
        metadata_extractor=MetadataExtractor(),
        text_cleaner=TextCleaner(),
    )

    embedding_service = EmbeddingService(model_name=settings.embedding_model.value)
    vector_store = ChromaVectorStore(settings)

    # Process all files
    chunks, result = pipeline.process_directory(args.source)

    if not chunks:
        print("No documents found to ingest.")
        return

    print(f"Processed {result.documents_loaded} documents into {result.chunks_created} chunks")

    # Generate embeddings
    print("Generating embeddings...")
    texts = [chunk.content for chunk in chunks]
    embeddings = embedding_service.embed_documents(texts)

    # Store in vector database
    print("Storing in ChromaDB...")
    vector_store.add(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[chunk.metadata for chunk in chunks],
    )

    elapsed = time.perf_counter() - start
    print(f"\nIngestion complete!")
    print(f"  Documents: {result.documents_loaded}")
    print(f"  Chunks: {result.chunks_created}")
    print(f"  Total in store: {vector_store.count()}")
    print(f"  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
