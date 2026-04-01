"""
dense_retriever.py

Dense retrieval using InLegalBERT embeddings + ChromaDB.
Aggregates chunk-level cosine similarities → document-level score
using max-pooling with role weights (same strategy as BM25 retriever).
"""

import sys
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from retrieval.embedder import LegalEmbedder
from retrieval.vector_store import LegalVectorStore


class DenseRetriever:
    def __init__(self):
        self._store    = LegalVectorStore()
        self._embedder = LegalEmbedder()

    def retrieve(self, query: str, doc_type: str, top_k: int = 100) -> list[tuple[str, float]]:
        """
        Embed query → search ChromaDB → aggregate chunk scores → ranked doc list.
        Returns list of (doc_id, score) sorted descending.
        """
        query_embedding = self._embedder.encode_single(query)

        # Fetch more candidates than needed so aggregation has room
        fetch_k = min(top_k * 10, self._store.count(doc_type))
        results = self._store.query(query_embedding, doc_type=doc_type, top_k=fetch_k)

        metadatas = results["metadatas"][0]
        distances = results["distances"][0]   # cosine distance (lower = more similar)

        # Convert distance → similarity, apply role weight, max-pool per doc
        doc_scores: dict[str, float] = defaultdict(float)
        for meta, dist in zip(metadatas, distances):
            similarity = 1.0 - dist                  # cosine similarity
            weighted   = similarity * meta["weight"] # role-aware boost
            doc_id     = meta["doc_id"]
            doc_scores[doc_id] = max(doc_scores[doc_id], weighted)

        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
