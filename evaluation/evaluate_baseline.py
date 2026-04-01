"""
evaluate_baseline.py

Runs BM25 retrieval on all 50 AILA queries and reports metrics
for both Task 1 (prior case retrieval) and Task 2 (statute retrieval).

Run from project root:
    python evaluation/evaluate_baseline.py
"""

import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from retrieval.bm25_retriever import BM25Retriever
from evaluation.metrics import evaluate_all

PROC_DIR      = BASE_DIR / "data" / "processed"
QUERIES_FILE  = PROC_DIR / "queries.json"
RELEVANCE_FILE = PROC_DIR / "relevance.json"
RESULTS_FILE  = PROC_DIR / "bm25_results.json"


def run():
    # Load data
    queries   = json.loads(QUERIES_FILE.read_text())
    relevance = json.loads(RELEVANCE_FILE.read_text())
    rel_cases = relevance["prior_cases"]
    rel_stats = relevance["statutes"]

    retriever = BM25Retriever()

    case_results   = {}
    statue_results = {}

    print(f"\nRetrieving for {len(queries)} queries...")
    for qid, query_text in queries.items():
        # Task 1: prior case retrieval (top 100)
        case_ranked = retriever.retrieve(query_text, doc_type="case", top_k=100)
        case_results[qid] = [doc_id for doc_id, _ in case_ranked]

        # Task 2: statute retrieval (top 50, only 197 statutes total)
        stat_ranked = retriever.retrieve(query_text, doc_type="statute", top_k=50)
        statue_results[qid] = [doc_id for doc_id, _ in stat_ranked]

    # Evaluate
    case_metrics = evaluate_all(case_results, rel_cases)
    stat_metrics = evaluate_all(statue_results, rel_stats)

    # Print results table
    print("\n" + "=" * 55)
    print(f"{'Metric':<15} {'Task1: Cases':>18} {'Task2: Statutes':>18}")
    print("=" * 55)
    for metric in ["MAP", "NDCG@10", "MRR", "P@5", "P@10"]:
        print(f"{metric:<15} {case_metrics[metric]:>18.4f} {stat_metrics[metric]:>18.4f}")
    print("=" * 55)
    print(f"{'Queries eval':<15} {case_metrics['num_queries_evaluated']:>18} {stat_metrics['num_queries_evaluated']:>18}")

    # Save results for later comparison
    output = {
        "model": "BM25-baseline",
        "task1_cases": {"metrics": case_metrics, "ranked_results": case_results},
        "task2_statutes": {"metrics": stat_metrics, "ranked_results": statue_results},
    }
    RESULTS_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved → {RESULTS_FILE}")


if __name__ == "__main__":
    run()
