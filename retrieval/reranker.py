"""
reranker.py

Cross-encoder re-ranker over the top-K candidates from the hybrid retriever.

Why a cross-encoder:
  Bi-encoders (BM25, mpnet) score query and doc independently — fast but lossy.
  A cross-encoder jointly encodes (query, chunk) and attends across both,
  catching fine-grained relevance signals a bi-encoder misses.
  Standard second-stage in modern retrieval pipelines.

Model: cross-encoder/ms-marco-MiniLM-L-12-v2
  Trained on MS MARCO passage reranking. Strong general-purpose cross-encoder.
  Small enough to rerank 50 candidates × ~15 chunks each in reasonable time on MPS.
  A domain-tuned swap (e.g. InLegalBERT fine-tuned on IndicLegalQA) plugs in
  behind the same interface later.

Ranking strategy:
  1. For each candidate doc, score every chunk against the query.
  2. Aggregate chunk scores back to doc level by taking the max (the single
     most relevant chunk dominates, mirroring how a human skims a case).
  3. Re-sort candidates by aggregated score.

Query handling:
  AILA queries are full case documents (thousands of words). The cross-encoder
  has a 512-token budget for (query, passage) together. We reuse the dense
  retriever's 300-word truncation so query context remains consistent across
  retrieval and re-ranking stages.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

import torch
from sentence_transformers import CrossEncoder

BASE_DIR    = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunked_corpus.jsonl"

from retrieval.dense_retriever import truncate_query

MODEL_NAME         = "cross-encoder/ms-marco-MiniLM-L-12-v2"
MAX_LENGTH         = 512
BATCH_SIZE         = 128   # larger batches → better MPS/GPU utilisation
MAX_CHUNKS_PER_DOC = 3     # only score the first N chunks per doc; max-pool picks the best


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Reranker:
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        self.device = device or _detect_device()
        print(f"Loading cross-encoder {model_name} on {self.device}...")
        self.model = CrossEncoder(model_name, max_length=MAX_LENGTH, device=self.device)
        self._chunks_by_doc = self._load_chunks()
        total_docs = len(self._chunks_by_doc)
        total_chunks = sum(len(v) for v in self._chunks_by_doc.values())
        print(f"Reranker ready. {total_chunks} chunks across {total_docs} docs loaded.")

    def _load_chunks(self) -> dict[str, list[dict]]:
        by_doc: dict[str, list[dict]] = defaultdict(list)
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                by_doc[c["doc_id"]].append(c)
        return by_doc

    def _score_chunks(self, query: str, candidates: list[tuple[str, float]]
                      ) -> list[tuple[dict, float]]:
        """
        Score every chunk of every candidate doc against the query.
        Returns list of (chunk_dict, score) in candidate/chunk order.
        """
        q_trunc = truncate_query(query)

        pairs  = []
        chunks = []
        for doc_id, _ in candidates:
            for c in self._chunks_by_doc.get(doc_id, [])[:MAX_CHUNKS_PER_DOC]:
                pairs.append((q_trunc, c["text"]))
                chunks.append(c)

        if not pairs:
            return []

        scores = self.model.predict(
            pairs,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return list(zip(chunks, (float(s) for s in scores)))

    def rerank(self, query: str, candidates: list[tuple[str, float]],
               top_k: int = 100) -> list[tuple[str, float]]:
        """
        Doc-level rerank: max-pool chunk scores per doc.
        Returns (doc_id, score) sorted descending.
        """
        if not candidates:
            return []

        scored = self._score_chunks(query, candidates)

        doc_scores: dict[str, float] = {}
        for chunk, score in scored:
            doc_id = chunk["doc_id"]
            prev = doc_scores.get(doc_id)
            if prev is None or score > prev:
                doc_scores[doc_id] = score

        for doc_id, _ in candidates:
            doc_scores.setdefault(doc_id, float("-inf"))

        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def rerank_chunks(self, query: str, candidates: list[tuple[str, float]],
                      top_n: int = 5) -> list[tuple[dict, float]]:
        """
        Chunk-level rerank for RAG context assembly.
        Returns the top-N (chunk_dict, score) globally, not aggregated to doc.
        """
        if not candidates:
            return []
        scored = self._score_chunks(query, candidates)
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]
