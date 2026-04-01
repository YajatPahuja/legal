"""
bm25_retriever.py

BM25 baseline retriever over the chunked AILA 2019 corpus.

Since documents are split into chunks, we:
  1. Score every chunk against the query
  2. Aggregate chunk scores → document score (max pooling)
  3. Return ranked list of doc_ids separated by type (case / statute)

Usage:
    from retrieval.bm25_retriever import BM25Retriever
    r = BM25Retriever()
    results = r.retrieve(query_text, doc_type="case", top_k=100)
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from rank_bm25 import BM25Okapi

BASE_DIR      = Path(__file__).resolve().parent.parent
CHUNKS_FILE   = BASE_DIR / "data" / "processed" / "chunked_corpus.jsonl"


def tokenize(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric. Simple but effective for BM25."""
    return re.findall(r'[a-z0-9]+', text.lower())


class BM25Retriever:
    def __init__(self):
        print("Loading chunked corpus...")
        self._chunks = self._load_chunks()

        # Separate indexes for cases and statutes
        self._case_chunks    = [c for c in self._chunks if c["doc_type"] == "case"]
        self._statute_chunks = [c for c in self._chunks if c["doc_type"] == "statute"]

        print(f"  {len(self._case_chunks)} case chunks | {len(self._statute_chunks)} statute chunks")
        print("Building BM25 indexes...")
        self._bm25_cases    = BM25Okapi([tokenize(c["text"]) for c in self._case_chunks])
        self._bm25_statutes = BM25Okapi([tokenize(c["text"]) for c in self._statute_chunks])
        print("BM25 ready.")

    def _load_chunks(self) -> list[dict]:
        chunks = []
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        return chunks

    def retrieve(self, query: str, doc_type: str, top_k: int = 100) -> list[tuple[str, float]]:
        """
        Retrieve top_k documents of the given type for a query.

        Returns list of (doc_id, score) sorted by score descending.
        Uses max-pooling over chunk scores to get document-level score.
        Role weights from chunking are applied as a multiplier.
        """
        tokens = tokenize(query)

        if doc_type == "case":
            bm25   = self._bm25_cases
            chunks = self._case_chunks
        else:
            bm25   = self._bm25_statutes
            chunks = self._statute_chunks

        scores = bm25.get_scores(tokens)

        # Aggregate: doc_id → max(chunk_score * role_weight)
        doc_scores: dict[str, float] = defaultdict(float)
        for chunk, score in zip(chunks, scores):
            weighted = score * chunk.get("weight", 1.0)
            doc_scores[chunk["doc_id"]] = max(doc_scores[chunk["doc_id"]], weighted)

        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
