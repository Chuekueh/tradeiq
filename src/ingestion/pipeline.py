from dataclasses import dataclass
from pathlib import Path

import structlog

from src.ingestion.chunkers.base import BaseChunker, Chunk
from src.ingestion.loaders.base import BaseLoader
from src.ingestion.processors.metadata_extractor import MetadataExtractor
from src.ingestion.processors.text_cleaner import TextCleaner

logger = structlog.get_logger()


@dataclass
class IngestionResult:
    source: str
    documents_loaded: int
    chunks_created: int
    chunks_stored: int


class IngestionPipeline:
    def __init__(
        self,
        loaders: list[BaseLoader],
        chunker: BaseChunker,
        metadata_extractor: MetadataExtractor,
        text_cleaner: TextCleaner,
    ):
        self._loaders = loaders
        self._chunker = chunker
        self._metadata_extractor = metadata_extractor
        self._text_cleaner = text_cleaner

    def _get_loader(self, path: str) -> BaseLoader | None:
        for loader in self._loaders:
            if loader.supports(path):
                return loader
        return None

    def process_file(self, file_path: str) -> list[Chunk]:
        """Load, clean, extract metadata, and chunk a single file."""
        loader = self._get_loader(file_path)
        if loader is None:
            logger.warning("no_loader_found", path=file_path)
            return []

        documents = loader.load(file_path)
        all_chunks: list[Chunk] = []

        for doc in documents:
            doc.content = self._text_cleaner.clean(doc.content)
            doc = self._metadata_extractor.extract(doc)
            chunks = self._chunker.chunk(doc)
            all_chunks.extend(chunks)

        logger.info(
            "file_processed",
            path=file_path,
            documents=len(documents),
            chunks=len(all_chunks),
        )
        return all_chunks

    def process_directory(self, dir_path: str) -> tuple[list[Chunk], IngestionResult]:
        """Process all supported files in a directory."""
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        all_chunks: list[Chunk] = []
        doc_count = 0

        for file_path in sorted(path.rglob("*")):
            if file_path.is_file() and self._get_loader(str(file_path)):
                chunks = self.process_file(str(file_path))
                all_chunks.extend(chunks)
                doc_count += 1

        result = IngestionResult(
            source=dir_path,
            documents_loaded=doc_count,
            chunks_created=len(all_chunks),
            chunks_stored=0,  # Updated after vector store insertion
        )
        logger.info("directory_processed", result=result)
        return all_chunks, result
