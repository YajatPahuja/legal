"""
run_demo.py

CLI entrypoint for the RAG pipeline. Loads an AILA query (by id or via
--query-text), runs hybrid → rerank → prompt → generate, and prints the
assembled prompt + model answer.

Run from project root:
    # Dry-run (no LLM call, prints prompt only)
    python rag/run_demo.py --qid AILA_Q1

    # Anthropic
    RAG_BACKEND=anthropic ANTHROPIC_API_KEY=sk-... \
        python rag/run_demo.py --qid AILA_Q1

    # Ollama (requires `ollama serve` + `ollama pull llama3.1`)
    RAG_BACKEND=ollama python rag/run_demo.py --qid AILA_Q1

    # Free-form query
    python rag/run_demo.py --query-text "A bank employee was dismissed after..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag.pipeline import RAGPipeline

QUERIES_FILE = BASE_DIR / "data" / "processed" / "queries.json"


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--qid", help="AILA query id, e.g. AILA_Q1")
    src.add_argument("--query-text", help="Free-form query text")
    ap.add_argument("--top-n-cases", type=int, default=3)
    ap.add_argument("--top-n-statutes", type=int, default=3)
    ap.add_argument("--save", help="Optional path to dump full result JSON")
    args = ap.parse_args()

    if args.qid:
        queries = json.loads(QUERIES_FILE.read_text())
        if args.qid not in queries:
            sys.exit(f"Unknown qid {args.qid!r}. Available: {list(queries)[:5]}...")
        query_text = queries[args.qid]
        print(f"\nQuery: {args.qid}  ({len(query_text.split())} words)")
    else:
        query_text = args.query_text
        print(f"\nQuery (free-form, {len(query_text.split())} words)")

    pipe = RAGPipeline()
    print(f"Backend: {pipe.generator.name}\n")

    result = pipe.answer(
        query_text,
        top_n_cases=args.top_n_cases,
        top_n_statutes=args.top_n_statutes,
    )

    print("=" * 80)
    print("PROMPT")
    print("=" * 80)
    print(result["prompt"])
    print()
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(result["answer"])
    print()
    print("=" * 80)
    print("CITATIONS")
    print("=" * 80)
    for marker, doc_id in result["citations"].items():
        print(f"  {marker} → {doc_id}")

    if args.save:
        out = {
            "query":     result["query"],
            "backend":   result["backend"],
            "citations": result["citations"],
            "prompt":    result["prompt"],
            "answer":    result["answer"],
            "case_chunks": [
                {"doc_id": c["doc_id"], "chunk_id": c["chunk_id"],
                 "role": c["role"], "score": s}
                for c, s in result["case_chunks"]
            ],
            "statute_chunks": [
                {"doc_id": c["doc_id"], "chunk_id": c["chunk_id"],
                 "role": c["role"], "score": s}
                for c, s in result["statute_chunks"]
            ],
        }
        Path(args.save).write_text(json.dumps(out, indent=2))
        print(f"\nSaved → {args.save}")


if __name__ == "__main__":
    main()
