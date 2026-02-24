import hashlib

import nltk
import numpy as np
import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

from src.ingestion.chunkers.base import BaseChunker, Chunk
from src.ingestion.loaders.base import Document

nltk.download("punkt_tab", quiet=True)

logger = structlog.get_logger()


class SemanticChunker(BaseChunker):
    """Splits documents at semantic boundaries using embedding similarity."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        breakpoint_percentile_threshold: int = 25,
        max_chunk_size: int = 1500,
        min_sentences_for_semantic: int = 3,
    ):
        self._model = SentenceTransformer(embedding_model)
        self._breakpoint_percentile = breakpoint_percentile_threshold
        self._max_chunk_size = max_chunk_size
        self._min_sentences = min_sentences_for_semantic
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
            keep_separator=True,
        )
        logger.info(
            "semantic_chunker_initialized",
            model=embedding_model,
            breakpoint_percentile=breakpoint_percentile_threshold,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = sent_tokenize(document.content)

        # Fall back for short documents
        if len(sentences) < self._min_sentences:
            return self._fallback_chunk(document)

        # Compute sentence embeddings
        embeddings = self._model.encode(sentences, show_progress_bar=False)

        # Compute cosine similarities between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        if not similarities:
            return self._fallback_chunk(document)

        # Find breakpoints where similarity drops below threshold
        threshold = np.percentile(similarities, self._breakpoint_percentile)
        breakpoints = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]

        # Group sentences into chunks
        groups = self._group_sentences(sentences, breakpoints)

        # Build Chunk objects
        chunks: list[Chunk] = []
        chunk_index = 0
        for group in groups:
            text = " ".join(group)

            # If chunk is too large, split with fallback
            if len(text) > self._max_chunk_size:
                sub_texts = self._fallback_splitter.split_text(text)
                for sub_text in sub_texts:
                    chunk_id = hashlib.sha256(
                        f"{document.doc_id}:{chunk_index}:{sub_text[:100]}".encode()
                    ).hexdigest()[:16]
                    chunks.append(
                        Chunk(
                            content=sub_text,
                            metadata={
                                **document.metadata,
                                "chunk_index": chunk_index,
                                "chunking_strategy": "semantic",
                            },
                            source=document.source,
                            chunk_id=chunk_id,
                            chunk_index=chunk_index,
                            doc_id=document.doc_id,
                        )
                    )
                    chunk_index += 1
            else:
                chunk_id = hashlib.sha256(
                    f"{document.doc_id}:{chunk_index}:{text[:100]}".encode()
                ).hexdigest()[:16]
                chunks.append(
                    Chunk(
                        content=text,
                        metadata={
                            **document.metadata,
                            "chunk_index": chunk_index,
                            "chunking_strategy": "semantic",
                        },
                        source=document.source,
                        chunk_id=chunk_id,
                        chunk_index=chunk_index,
                        doc_id=document.doc_id,
                    )
                )
                chunk_index += 1

        # Add total_chunks to metadata
        for c in chunks:
            c.metadata["total_chunks"] = len(chunks)

        logger.info(
            "semantic_chunking_complete",
            doc_id=document.doc_id[:16],
            sentences=len(sentences),
            breakpoints=len(breakpoints),
            chunks=len(chunks),
        )
        return chunks

    def _fallback_chunk(self, document: Document) -> list[Chunk]:
        """Fall back to recursive character splitting."""
        texts = self._fallback_splitter.split_text(document.content)
        chunks: list[Chunk] = []
        for i, text in enumerate(texts):
            chunk_id = hashlib.sha256(f"{document.doc_id}:{i}:{text[:100]}".encode()).hexdigest()[
                :16
            ]
            chunks.append(
                Chunk(
                    content=text,
                    metadata={
                        **document.metadata,
                        "chunk_index": i,
                        "total_chunks": len(texts),
                        "chunking_strategy": "recursive_fallback",
                    },
                    source=document.source,
                    chunk_id=chunk_id,
                    chunk_index=i,
                    doc_id=document.doc_id,
                )
            )
        return chunks

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def _group_sentences(sentences: list[str], breakpoints: list[int]) -> list[list[str]]:
        """Group sentences into chunks based on breakpoint indices."""
        groups: list[list[str]] = []
        start = 0
        for bp in breakpoints:
            if start < bp:
                groups.append(sentences[start:bp])
            start = bp
        # Add remaining sentences
        if start < len(sentences):
            groups.append(sentences[start:])
        return groups
