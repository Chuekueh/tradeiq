import chromadb
import structlog

from src.config import Settings
from src.vectorstore.base import BaseVectorStore, SearchResult

logger = structlog.get_logger()


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, settings: Settings):
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "chroma_initialized",
            collection=settings.chroma_collection_name,
            count=self._collection.count(),
        )

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str | int | float | bool]],
    ) -> None:
        # ChromaDB has a batch limit; process in chunks of 5000
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            end = i + batch_size
            self._collection.upsert(
                ids=ids[i:end],
                documents=documents[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],  # type: ignore[arg-type]
            )
        logger.info("documents_added", count=len(ids))

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[SearchResult]:
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        search_results: list[SearchResult] = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist, id_ in zip(
                results["documents"][0],
                results["metadatas"][0],  # type: ignore[index]
                results["distances"][0],  # type: ignore[index]
                results["ids"][0],
                strict=True,
            ):
                # ChromaDB returns cosine distance; convert to similarity
                score = 1.0 - dist
                search_results.append(
                    SearchResult(
                        content=doc,
                        metadata=meta,  # type: ignore[arg-type]
                        source=str(meta.get("file_name", "")),
                        chunk_id=id_,
                        score=score,
                    )
                )

        return search_results

    def count(self) -> int:
        return self._collection.count()

    def delete(self, ids: list[str]) -> None:
        self._collection.delete(ids=ids)
        logger.info("documents_deleted", count=len(ids))

    def get_all_documents(self) -> list[str]:
        """Retrieve all document texts for BM25 index building."""
        result = self._collection.get(include=["documents"])
        return result["documents"] or []

    def get_all_ids(self) -> list[str]:
        """Retrieve all document IDs."""
        result = self._collection.get()
        return result["ids"]

    def get_by_ids(self, ids: list[str]) -> list[SearchResult]:
        """Retrieve specific documents by their IDs."""
        if not ids:
            return []

        result = self._collection.get(ids=ids, include=["documents", "metadatas"])
        search_results: list[SearchResult] = []

        if result["documents"]:
            for doc, meta, id_ in zip(
                result["documents"],
                result["metadatas"],  # type: ignore[arg-type]
                result["ids"],
                strict=True,
            ):
                search_results.append(
                    SearchResult(
                        content=doc,
                        metadata=meta,  # type: ignore[arg-type]
                        source=str(meta.get("file_name", "")),
                        chunk_id=id_,
                        score=0.0,
                    )
                )

        return search_results
