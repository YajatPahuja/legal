"""
evaluate_hybrid.py

Evaluates hybrid BM25 + dense retrieval (RRF fusion) and prints a side-by-side
comparison with BM25 and dense baselines.

Run from project root:
    python evaluation/evaluate_hybrid.py
"""

import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from retrieval.hybrid_retriever import HybridRetriever
from evaluation.metrics import evaluate_all

PROC_DIR       = BASE_DIR / "data" / "processed"
QUERIES_FILE   = PROC_DIR / "queries.json"
RELEVANCE_FILE = PROC_DIR / "relevance.json"
BM25_FILE      = PROC_DIR / "bm25_results.json"
DENSE_FILE     = PROC_DIR / "dense_results.json"
RESULTS_FILE   = PROC_DIR / "hybrid_results.json"


def run():
    queries   = json.loads(QUERIES_FILE.read_text())
    relevance = json.loads(RELEVANCE_FILE.read_text())
    rel_cases = relevance["prior_cases"]
    rel_stats = relevance["statutes"]

    retriever = HybridRetriever()

    case_results, stat_results = {}, {}

    print(f"\nRetrieving for {len(queries)} queries (hybrid)...")
    for qid, query_text in queries.items():
        case_ranked = retriever.retrieve(query_text, doc_type="case",    top_k=100)
        stat_ranked = retriever.retrieve(query_text, doc_type="statute", top_k=50)
        case_results[qid] = [doc_id for doc_id, _ in case_ranked]
        stat_results[qid] = [doc_id for doc_id, _ in stat_ranked]
        print(f"  {qid} done")

    hy_case = evaluate_all(case_results, rel_cases)
    hy_stat = evaluate_all(stat_results, rel_stats)

    # Load prior results for comparison
    bm25_case = bm25_stat = dense_case = dense_stat = {}
    if BM25_FILE.exists():
        bm25 = json.loads(BM25_FILE.read_text())
        bm25_case = bm25["task1_cases"]["metrics"]
        bm25_stat = bm25["task2_statutes"]["metrics"]
    if DENSE_FILE.exists():
        dn = json.loads(DENSE_FILE.read_text())
        dense_case = dn["task1_cases"]["metrics"]
        dense_stat = dn["task2_statutes"]["metrics"]

    metrics = ["MAP", "NDCG@10", "MRR", "P@5", "P@10"]
    print("\n" + "=" * 100)
    print(f"{'':12} {'──────── Task 1: Cases ────────':^42}  {'──────── Task 2: Statutes ────────':^42}")
    print(f"{'Metric':<12} {'BM25':>12} {'Dense':>12} {'Hybrid':>12}   {'BM25':>12} {'Dense':>12} {'Hybrid':>12}")
    print("=" * 100)
    for m in metrics:
        print(f"{m:<12} "
              f"{bm25_case.get(m, 0):>12.4f} {dense_case.get(m, 0):>12.4f} {hy_case[m]:>12.4f}   "
              f"{bm25_stat.get(m, 0):>12.4f} {dense_stat.get(m, 0):>12.4f} {hy_stat[m]:>12.4f}")
    print("=" * 100)

    output = {
        "model": "Hybrid-RRF (BM25 + mpnet)",
        "task1_cases":    {"metrics": hy_case, "ranked_results": case_results},
        "task2_statutes": {"metrics": hy_stat, "ranked_results": stat_results},
    }
    RESULTS_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved → {RESULTS_FILE}")


if __name__ == "__main__":
    run()
