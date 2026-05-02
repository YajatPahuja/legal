"""
run_demo.py

Edit the CONFIG block below, then run:
    python rag/run_demo.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# ── CONFIG — edit these and run ──────────────────────────────────────────────

# Option A: pick an AILA query by id  (AILA_Q1 … AILA_Q50)
#           set to None to use QUERY_TEXT instead
QID = "AILA_Q2"

# Option B: paste your own query text here (used only when QID is None)
QUERY_TEXT = """
Paste your facts here.
"""

# How many top chunks to pull into the context
TOP_N_CASES    = 3
TOP_N_STATUTES = 3

# LLM backend: "extractive" | "dryrun" | "anthropic" | "ollama"
#   extractive — no LLM, shows retrieved passages verbatim (default, safe)
#   dryrun     — prints the assembled prompt, no generation
#   anthropic  — needs ANTHROPIC_API_KEY set below (or in your environment)
#   ollama     — needs `ollama serve` running + model pulled
BACKEND = "extractive"

# Only used when BACKEND = "anthropic"
ANTHROPIC_API_KEY = ""   # or leave blank to read from env

# Only used when BACKEND = "ollama"
OLLAMA_MODEL = "llama3.1"
OLLAMA_HOST  = "http://localhost:11434"

# Optional: set to a file path to save the full result as JSON, e.g. "out.json"
SAVE_TO = ""

# ── END CONFIG ────────────────────────────────────────────────────────────────


def main():
    # Resolve query text
    if QID is not None:
        queries_file = BASE_DIR / "data" / "processed" / "queries.json"
        queries = json.loads(queries_file.read_text())
        if QID not in queries:
            available = list(queries.keys())
            sys.exit(f"Unknown QID {QID!r}. Available ids: {available[:5]}...")
        query_text = queries[QID]
        print(f"\nQuery: {QID}  ({len(query_text.split())} words)")
    else:
        query_text = QUERY_TEXT.strip()
        if not query_text:
            sys.exit("Set either QID or QUERY_TEXT in the CONFIG block.")
        print(f"\nQuery: (custom, {len(query_text.split())} words)")

    # Wire backend via env var so make_generator() picks it up
    os.environ["RAG_BACKEND"] = BACKEND
    if BACKEND == "anthropic" and ANTHROPIC_API_KEY:
        os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
    if BACKEND == "ollama":
        os.environ["RAG_MODEL"]    = OLLAMA_MODEL
        os.environ["OLLAMA_HOST"]  = OLLAMA_HOST

    from rag.pipeline import RAGPipeline
    pipe = RAGPipeline()
    print(f"Backend: {pipe.generator.name}\n")

    result = pipe.answer(
        query_text,
        top_n_cases=TOP_N_CASES,
        top_n_statutes=TOP_N_STATUTES,
    )

    sep = "=" * 80
    print(sep, "PROMPT", sep, result["prompt"], sep="\n")
    print()
    print(sep, "ANSWER", sep, result["answer"], sep="\n")
    print()
    print(sep)
    print("CITATIONS")
    print(sep)
    for marker, doc_id in result["citations"].items():
        print(f"  [{marker}] → {doc_id}")

    if SAVE_TO:
        out = {
            "qid":       QID,
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
        Path(SAVE_TO).write_text(json.dumps(out, indent=2))
        print(f"\nSaved → {SAVE_TO}")


if __name__ == "__main__":
    main()
