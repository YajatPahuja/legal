"""
hybrid_retriever.py

Fuses BM25 (lexical, NER-removed) and dense (mpnet, role-weighted) rankings
using Reciprocal Rank Fusion (RRF).

RRF is rank-based, not score-based — scale differences between BM25's
unbounded scores and dense's cosine similarities don't matter.

    rrf_score(doc) = Σ  w_i / (k + rank_i(doc))
                   i=1..N

k=60 is the standard value from Cormack et al. (2009).

Per-task weight overrides let us lean on BM25 where it dominates top-ranks
(statutes: section numbers are exact-match signals) and dense where
paraphrase matters (cases).
"""

import sys
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever

K_RRF = 60

TASK_FUSION_WEIGHTS = {
    "case":    {"bm25": 1.0, "dense": 1.0},
    "statute": {"bm25": 1.5, "dense": 1.0},
}


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever = None, dense: DenseRetriever = None):
        self._bm25  = bm25  or BM25Retriever()
        self._dense = dense or DenseRetriever()

    def retrieve(self, query: str, doc_type: str, top_k: int = 100,
                 k_rrf: int = K_RRF,
                 fetch_k: int = 200) -> list[tuple[str, float]]:
        weights = TASK_FUSION_WEIGHTS[doc_type]

        bm25_ranked  = self._bm25.retrieve(query,  doc_type=doc_type, top_k=fetch_k)
        dense_ranked = self._dense.retrieve(query, doc_type=doc_type, top_k=fetch_k)

        fused: dict[str, float] = defaultdict(float)
        for rank, (doc_id, _) in enumerate(bm25_ranked, start=1):
            fused[doc_id] += weights["bm25"] / (k_rrf + rank)
        for rank, (doc_id, _) in enumerate(dense_ranked, start=1):
            fused[doc_id] += weights["dense"] / (k_rrf + rank)

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
