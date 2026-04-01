"""
evaluate_dense.py

Evaluates dense (InLegalBERT) retrieval on all 50 AILA queries.
Compares results against the BM25 baseline.

Run from project root (after index_corpus.py has completed):
    python evaluation/evaluate_dense.py
"""

import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from retrieval.dense_retriever import DenseRetriever
from evaluation.metrics import evaluate_all

PROC_DIR       = BASE_DIR / "data" / "processed"
QUERIES_FILE   = PROC_DIR / "queries.json"
RELEVANCE_FILE = PROC_DIR / "relevance.json"
BM25_FILE      = PROC_DIR / "bm25_results.json"
RESULTS_FILE   = PROC_DIR / "dense_results.json"


def run():
    queries   = json.loads(QUERIES_FILE.read_text())
    relevance = json.loads(RELEVANCE_FILE.read_text())
    rel_cases = relevance["prior_cases"]
    rel_stats = relevance["statutes"]

    retriever = DenseRetriever()

    case_results   = {}
    statue_results = {}

    print(f"\nRetrieving for {len(queries)} queries (dense)...")
    for qid, query_text in queries.items():
        case_ranked = retriever.retrieve(query_text, doc_type="case",    top_k=100)
        stat_ranked = retriever.retrieve(query_text, doc_type="statute", top_k=50)
        case_results[qid]   = [doc_id for doc_id, _ in case_ranked]
        statue_results[qid] = [doc_id for doc_id, _ in stat_ranked]
        print(f"  {qid} done")

    dense_case = evaluate_all(case_results, rel_cases)
    dense_stat = evaluate_all(statue_results, rel_stats)

    # Load BM25 baseline for comparison
    bm25_case, bm25_stat = {}, {}
    if BM25_FILE.exists():
        bm25 = json.loads(BM25_FILE.read_text())
        bm25_case = bm25["task1_cases"]["metrics"]
        bm25_stat = bm25["task2_statutes"]["metrics"]

    # Print comparison table
    metrics = ["MAP", "NDCG@10", "MRR", "P@5", "P@10"]
    print("\n" + "=" * 75)
    print(f"{'':15} {'── Task 1: Cases ──':^28} {'── Task 2: Statutes ──':^28}")
    print(f"{'Metric':<15} {'BM25':>12} {'Dense':>12} {'BM25':>12} {'Dense':>12}")
    print("=" * 75)
    for m in metrics:
        bc = bm25_case.get(m, 0)
        dc = dense_case.get(m, 0)
        bs = bm25_stat.get(m, 0)
        ds = dense_stat.get(m, 0)
        print(f"{m:<15} {bc:>12.4f} {dc:>12.4f} {bs:>12.4f} {ds:>12.4f}")
    print("=" * 75)

    output = {
        "model": "Dense-InLegalBERT",
        "task1_cases":     {"metrics": dense_case,  "ranked_results": case_results},
        "task2_statutes":  {"metrics": dense_stat, "ranked_results": statue_results},
    }
    RESULTS_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved → {RESULTS_FILE}")


if __name__ == "__main__":
    run()
