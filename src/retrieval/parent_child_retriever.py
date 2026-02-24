import hashlib

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.chunkers.base import BaseChunker, Chunk
from src.ingestion.loaders.base import Document
from src.retrieval.hybrid_search import HybridSearcher
from src.vectorstore.base import SearchResult
from src.vectorstore.chroma_store import ChromaVectorStore

logger = structlog.get_logger()


class ParentChildChunker(BaseChunker):
    """Creates two levels of chunks: parent (large) and child (small).

    Child chunks are indexed for precise embedding search.
    Parent chunks are returned at retrieval time for richer LLM context.
    """

    def __init__(
        self,
        parent_chunk_size: int = 1024,
        parent_chunk_overlap: int = 100,
        child_chunk_size: int = 256,
        child_chunk_overlap: int = 25,
    ):
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
            keep_separator=True,
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
            keep_separator=True,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        """Create parent and child chunks.

        Returns both parent and child chunks. Child chunks reference their
        parent via parent_chunk_id. The caller should embed only child chunks
        and store parent chunks separately for later retrieval.
        """
        parent_texts = self._parent_splitter.split_text(document.content)
        all_chunks: list[Chunk] = []
        child_index = 0

        for p_idx, parent_text in enumerate(parent_texts):
            parent_id = hashlib.sha256(
                f"{document.doc_id}:parent:{p_idx}:{parent_text[:100]}".encode()
            ).hexdigest()[:16]

            # Create parent chunk
            parent_chunk = Chunk(
                content=parent_text,
                metadata={
                    **document.metadata,
                    "chunk_index": p_idx,
                    "total_chunks": len(parent_texts),
                    "chunk_type": "parent",
                },
                source=document.source,
                chunk_id=parent_id,
                chunk_index=p_idx,
                doc_id=document.doc_id,
                parent_chunk_id=None,
            )
            all_chunks.append(parent_chunk)

            # Create child chunks from this parent
            child_texts = self._child_splitter.split_text(parent_text)
            for c_idx, child_text in enumerate(child_texts):
                child_id = hashlib.sha256(
                    f"{document.doc_id}:child:{child_index}:{child_text[:100]}".encode()
                ).hexdigest()[:16]

                child_chunk = Chunk(
                    content=child_text,
                    metadata={
                        **document.metadata,
                        "chunk_index": child_index,
                        "chunk_type": "child",
                        "parent_chunk_id": parent_id,
                        "child_position": c_idx,
                    },
                    source=document.source,
                    chunk_id=child_id,
                    chunk_index=child_index,
                    doc_id=document.doc_id,
                    parent_chunk_id=parent_id,
                )
                all_chunks.append(child_chunk)
                child_index += 1

        logger.info(
            "parent_child_chunking",
            doc_id=document.doc_id[:16],
            parents=len(parent_texts),
            children=child_index,
        )
        return all_chunks


class ParentChildRetriever:
    """Retrieves child chunks then expands to parent chunks for richer context.

    Search is performed on child chunks (small, precise embeddings).
    Results are expanded to their parent chunks (large, context-rich).
    Duplicate parents are deduplicated.
    """

    def __init__(
        self,
        searcher: HybridSearcher,
        vector_store: ChromaVectorStore,
    ):
        self._searcher = searcher
        self._vector_store = vector_store

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """Search child chunks and expand to parent chunks."""
        # Search with more candidates to account for deduplication
        child_results = self._searcher.search(query, top_k=top_k * 2, where=where)

        if not child_results:
            return []

        # Expand to parent chunks
        seen_parent_ids: set[str] = set()
        parent_results: list[SearchResult] = []

        for child in child_results:
            parent_id = child.metadata.get("parent_chunk_id")

            # If no parent reference, use the child itself
            if not parent_id or parent_id in seen_parent_ids:
                if child.chunk_id not in seen_parent_ids:
                    seen_parent_ids.add(child.chunk_id)
                    if not parent_id:
                        parent_results.append(child)
                continue

            seen_parent_ids.add(str(parent_id))

            # Look up parent chunk from vector store
            parent_docs = self._vector_store.get_by_ids([str(parent_id)])
            if parent_docs:
                parent = parent_docs[0]
                parent.score = child.score  # Inherit child's relevance score
                parent_results.append(parent)
            else:
                # Fallback: use child if parent not found
                parent_results.append(child)

            if len(parent_results) >= top_k:
                break

        logger.info(
            "parent_child_retrieval",
            child_candidates=len(child_results),
            parent_results=len(parent_results),
        )
        return parent_results[:top_k]
