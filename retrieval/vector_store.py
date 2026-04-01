"""
vector_store.py

ChromaDB interface — two collections: one for cases, one for statutes.
Stores chunk embeddings with metadata for doc-level aggregation.
"""

import chromadb
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

CASE_COLLECTION    = "aila_cases"
STATUTE_COLLECTION = "aila_statutes"


class LegalVectorStore:
    def __init__(self):
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self._client   = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._cases    = self._client.get_or_create_collection(
            name=CASE_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._statutes = self._client.get_or_create_collection(
            name=STATUTE_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def _collection(self, doc_type: str):
        return self._cases if doc_type == "case" else self._statutes

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]], doc_type: str):
        """
        Add a batch of chunks + their embeddings to the appropriate collection.
        chunks: list of chunk dicts from chunked_corpus.jsonl
        """
        col = self._collection(doc_type)
        col.add(
            ids        = [c["chunk_id"] for c in chunks],
            embeddings = embeddings,
            documents  = [c["text"] for c in chunks],
            metadatas  = [
                {
                    "doc_id":  c["doc_id"],
                    "role":    c.get("role", "GENERAL"),
                    "weight":  float(c.get("weight", 1.0)),
                    "title":   c.get("title", ""),
                }
                for c in chunks
            ],
        )

    def query(self, embedding: list[float], doc_type: str, top_k: int = 200) -> dict:
        """
        Query the collection and return raw ChromaDB results.
        top_k is set high so doc-level aggregation has enough candidates.
        """
        col = self._collection(doc_type)
        return col.query(
            query_embeddings=[embedding],
            n_results=min(top_k, col.count()),
            include=["metadatas", "distances"],
        )

    def count(self, doc_type: str) -> int:
        return self._collection(doc_type).count()

    def is_indexed(self, doc_type: str) -> bool:
        return self.count(doc_type) > 0
