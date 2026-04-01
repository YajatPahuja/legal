"""
metrics.py

Standard IR evaluation metrics used in AILA 2019:
  - MAP  (Mean Average Precision)
  - NDCG@k (Normalized Discounted Cumulative Gain)
  - MRR  (Mean Reciprocal Rank)
  - P@k  (Precision at k)
"""

import math


def average_precision(ranked_doc_ids: list[str], relevant_ids: set[str]) -> float:
    """AP for a single query."""
    if not relevant_ids:
        return 0.0
    hits, total = 0, 0.0
    for i, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_ids:
            hits += 1
            total += hits / i
    return total / len(relevant_ids)


def ndcg_at_k(ranked_doc_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """NDCG@k for a single query (binary relevance)."""
    dcg = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)

    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant_ids), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def reciprocal_rank(ranked_doc_ids: list[str], relevant_ids: set[str]) -> float:
    """RR for a single query."""
    for i, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def precision_at_k(ranked_doc_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """P@k for a single query."""
    hits = sum(1 for doc_id in ranked_doc_ids[:k] if doc_id in relevant_ids)
    return hits / k


def evaluate_all(results: dict[str, list[str]], relevance: dict[str, list[str]]) -> dict:
    """
    Compute MAP, NDCG@10, MRR, P@5, P@10 across all queries.

    Args:
        results:   { query_id: [ranked doc_id list] }
        relevance: { query_id: [relevant doc_id list] }

    Returns dict of metric → score.
    """
    ap_list, ndcg10_list, rr_list, p5_list, p10_list = [], [], [], [], []

    for qid, ranked in results.items():
        rel = set(relevance.get(qid, []))
        if not rel:
            continue
        ap_list.append(average_precision(ranked, rel))
        ndcg10_list.append(ndcg_at_k(ranked, rel, k=10))
        rr_list.append(reciprocal_rank(ranked, rel))
        p5_list.append(precision_at_k(ranked, rel, k=5))
        p10_list.append(precision_at_k(ranked, rel, k=10))

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "MAP":     mean(ap_list),
        "NDCG@10": mean(ndcg10_list),
        "MRR":     mean(rr_list),
        "P@5":     mean(p5_list),
        "P@10":    mean(p10_list),
        "num_queries_evaluated": len(ap_list),
    }
