"""
evaluate_reranked.py

Two-stage retrieval evaluation:
  Stage 1: hybrid BM25+dense (RRF) → top RERANK_K candidates
  Stage 2: cross-encoder rerank  → final ranking

Prints side-by-side with BM25 / Dense / Hybrid / +Reranker and saves results.

Run from project root:
    python evaluation/evaluate_reranked.py
"""

import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from evaluation.metrics import evaluate_all

PROC_DIR       = BASE_DIR / "data" / "processed"
QUERIES_FILE   = PROC_DIR / "queries.json"
RELEVANCE_FILE = PROC_DIR / "relevance.json"
BM25_FILE      = PROC_DIR / "bm25_results.json"
DENSE_FILE     = PROC_DIR / "dense_results.json"
HYBRID_FILE    = PROC_DIR / "hybrid_results.json"
RESULTS_FILE   = PROC_DIR / "reranked_results.json"

# Reranking is expensive — trim the first-stage list to something manageable.
# Every relevant doc the reranker can surface must be in here.
RERANK_K_CASES    = 50
RERANK_K_STATUTES = 30

FINAL_K_CASES    = 100
FINAL_K_STATUTES = 50


def run():
    queries   = json.loads(QUERIES_FILE.read_text())
    relevance = json.loads(RELEVANCE_FILE.read_text())
    rel_cases = relevance["prior_cases"]
    rel_stats = relevance["statutes"]

    hybrid   = HybridRetriever()
    reranker = Reranker()

    case_results, stat_results = {}, {}

    print(f"\nRunning hybrid → rerank for {len(queries)} queries...")
    for qid, query_text in queries.items():
        case_cand = hybrid.retrieve(query_text, doc_type="case",    top_k=RERANK_K_CASES)
        stat_cand = hybrid.retrieve(query_text, doc_type="statute", top_k=RERANK_K_STATUTES)

        case_ranked = reranker.rerank(query_text, case_cand, top_k=FINAL_K_CASES)
        stat_ranked = reranker.rerank(query_text, stat_cand, top_k=FINAL_K_STATUTES)

        case_results[qid] = [doc_id for doc_id, _ in case_ranked]
        stat_results[qid] = [doc_id for doc_id, _ in stat_ranked]
        print(f"  {qid} done")

    rr_case = evaluate_all(case_results, rel_cases)
    rr_stat = evaluate_all(stat_results, rel_stats)

    # Load previous stages for comparison
    def _load(path):
        if path.exists():
            d = json.loads(path.read_text())
            return d["task1_cases"]["metrics"], d["task2_statutes"]["metrics"]
        return {}, {}

    bm25_c,  bm25_s  = _load(BM25_FILE)
    dense_c, dense_s = _load(DENSE_FILE)
    hy_c,    hy_s    = _load(HYBRID_FILE)

    metrics = ["MAP", "NDCG@10", "MRR", "P@5", "P@10"]
    header_cases    = f"{'BM25':>10} {'Dense':>10} {'Hybrid':>10} {'+Rerank':>10}"
    header_statutes = f"{'BM25':>10} {'Dense':>10} {'Hybrid':>10} {'+Rerank':>10}"
    print("\n" + "=" * 110)
    print(f"{'':10} {'──── Task 1: Cases ────':^42}  {'──── Task 2: Statutes ────':^42}")
    print(f"{'Metric':<10} {header_cases}  {header_statutes}")
    print("=" * 110)
    for m in metrics:
        print(
            f"{m:<10} "
            f"{bm25_c.get(m,0):>10.4f} {dense_c.get(m,0):>10.4f} {hy_c.get(m,0):>10.4f} {rr_case[m]:>10.4f}  "
            f"{bm25_s.get(m,0):>10.4f} {dense_s.get(m,0):>10.4f} {hy_s.get(m,0):>10.4f} {rr_stat[m]:>10.4f}"
        )
    print("=" * 110)

    output = {
        "model": "Hybrid + CrossEncoder(ms-marco-MiniLM-L-12-v2)",
        "rerank_k_cases":    RERANK_K_CASES,
        "rerank_k_statutes": RERANK_K_STATUTES,
        "task1_cases":    {"metrics": rr_case, "ranked_results": case_results},
        "task2_statutes": {"metrics": rr_stat, "ranked_results": stat_results},
    }
    RESULTS_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved → {RESULTS_FILE}")


if __name__ == "__main__":
    run()
