"""
dense_retriever.py

Dense retrieval using InLegalBERT embeddings + ChromaDB.

Query encoding strategy:
  AILA queries are full case documents (thousands of words). InLegalBERT
  truncates at 512 tokens. Rather than encoding the full document, we
  extract the first 300 words — Indian legal cases always front-load
  the key facts, issue, and legal context in the opening paragraph,
  which is the most signal-dense part for retrieval.

  Note: InLegalBERT is an MLM model, not fine-tuned for semantic similarity.
  Performance improves significantly after fine-tuning on IndicLegalQA
  (see training/finetune_qa.py). Current results reflect zero-shot retrieval.

Also applies task-specific role weights (Kalamkar et al., LREC 2022).
"""

import sys
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from retrieval.embedder import LegalEmbedder
from retrieval.vector_store import LegalVectorStore
from preprocessing.chunker import TASK_WEIGHTS

QUERY_WORDS = 300   # first ~300 words of query — most informationally dense part


def truncate_query(text: str) -> str:
    """Use only the first QUERY_WORDS words of the query document."""
    words = text.split()
    return " ".join(words[:QUERY_WORDS])


class DenseRetriever:
    def __init__(self):
        self._store    = LegalVectorStore()
        self._embedder = LegalEmbedder()

    def retrieve(self, query: str, doc_type: str, top_k: int = 100) -> list[tuple[str, float]]:
        """
        Encode query (first 300 words) → search ChromaDB → aggregate → ranked list.
        Returns list of (doc_id, score) sorted descending.
        """
        task_weights    = TASK_WEIGHTS[doc_type]
        query_truncated = truncate_query(query)
        query_embedding = self._embedder.encode_single(query_truncated)

        fetch_k = min(top_k * 10, self._store.count(doc_type))
        results = self._store.query(query_embedding, doc_type=doc_type, top_k=fetch_k)

        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        doc_scores: dict[str, float] = defaultdict(float)
        for meta, dist in zip(metadatas, distances):
            similarity = 1.0 - dist
            role       = meta.get("role", "GENERAL")
            weight     = task_weights.get(role, 1.0)
            weighted   = similarity * weight
            doc_id     = meta["doc_id"]
            doc_scores[doc_id] = max(doc_scores[doc_id], weighted)

        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
