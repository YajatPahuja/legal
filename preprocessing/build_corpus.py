"""
build_corpus.py

Merges Object_casedocs + Object_statutes into a single unified JSONL corpus.
Also parses queries and relevance judgments.

Run:
    python preprocessing/build_corpus.py
"""

import os
import re
import json
from pathlib import Path
from tqdm import tqdm
from text_cleaner import parse_case_doc, parse_statute_doc

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
ARCHIVE_DIR    = BASE_DIR / "archive"
CASEDOCS_DIR   = ARCHIVE_DIR / "Object_casedocs"
STATUTES_DIR   = ARCHIVE_DIR / "Object_statutes"
QUERY_FILE     = ARCHIVE_DIR / "Query_doc.txt"
REL_CASES_FILE = ARCHIVE_DIR / "relevance_judgments_priorcases.txt"
REL_STAT_FILE  = ARCHIVE_DIR / "relevance_judgments_statutes.txt"
OUT_DIR        = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Build unified corpus ────────────────────────────────────────────────────
def build_corpus():
    corpus = []

    # Case documents
    case_files = sorted(CASEDOCS_DIR.glob("C*.txt"),
                        key=lambda p: int(re.search(r'\d+', p.stem).group()))
    print(f"Processing {len(case_files)} case documents...")
    for fp in tqdm(case_files):
        doc_id = fp.stem  # e.g. "C1"
        parsed = parse_case_doc(str(fp))
        if not parsed:
            continue
        corpus.append({
            "doc_id":    doc_id,
            "doc_type":  "case",
            "case_name": parsed["case_name"],
            "court":     parsed["court"],
            "date":      parsed["date"],
            "title":     parsed["case_name"],
            "text":      parsed["full_text"],
        })

    # Statute documents
    stat_files = sorted(STATUTES_DIR.glob("S*.txt"),
                        key=lambda p: int(re.search(r'\d+', p.stem).group()))
    print(f"Processing {len(stat_files)} statute documents...")
    for fp in tqdm(stat_files):
        doc_id = fp.stem  # e.g. "S1"
        parsed = parse_statute_doc(str(fp))
        corpus.append({
            "doc_id":      doc_id,
            "doc_type":    "statute",
            "title":       parsed["title"],
            "description": parsed["description"],
            "text":        parsed["full_text"],
        })

    out_path = OUT_DIR / "unified_corpus.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\nCorpus saved → {out_path}")
    print(f"  Cases:    {sum(1 for d in corpus if d['doc_type']=='case')}")
    print(f"  Statutes: {sum(1 for d in corpus if d['doc_type']=='statute')}")
    return corpus


# ── 2. Parse queries ───────────────────────────────────────────────────────────
def parse_queries():
    """
    Query_doc.txt format (pipe-delimited):
        AILA_Q1||<full case text>
        AILA_Q2||<full case text>
        ...
    Each query is a complete current case document.
    The task: find which corpus cases/statutes are relevant to it.
    """
    queries = {}
    with open(QUERY_FILE, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "||" not in line:
                continue
            qid, text = line.split("||", 1)
            qid  = qid.strip()
            text = re.sub(r'\s+', ' ', text.strip())
            queries[qid] = text

    out_path = OUT_DIR / "queries.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)

    print(f"Queries saved → {out_path}  ({len(queries)} queries)")
    return queries


# ── 3. Parse relevance judgments ───────────────────────────────────────────────
def parse_relevance(filepath: Path) -> dict:
    """
    TREC-style format: <query_id> Q0 <doc_id> <relevance>
    relevance = 1 → relevant, 0 → not relevant.
    Returns: { query_id: [doc_id, ...] }  (only relevant docs)
    """
    relevance = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            # Format: AILA_Q1  Q0  C168  0
            qid    = parts[0]
            doc_id = parts[2]
            score  = parts[3]
            if score == "1":
                relevance.setdefault(qid, []).append(doc_id)
    return relevance


def build_relevance():
    rel_cases = parse_relevance(REL_CASES_FILE)
    rel_stats  = parse_relevance(REL_STAT_FILE)

    out = {
        "prior_cases": rel_cases,
        "statutes":    rel_stats,
    }
    out_path = OUT_DIR / "relevance.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Relevance saved → {out_path}")
    total_case_rels = sum(len(v) for v in rel_cases.values())
    total_stat_rels  = sum(len(v) for v in rel_stats.values())
    print(f"  Prior case relevance pairs: {total_case_rels}")
    print(f"  Statute relevance pairs:    {total_stat_rels}")
    return out


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Step 1/3: Building unified corpus")
    print("=" * 50)
    build_corpus()

    print("\n" + "=" * 50)
    print("Step 2/3: Parsing queries")
    print("=" * 50)
    parse_queries()

    print("\n" + "=" * 50)
    print("Step 3/3: Parsing relevance judgments")
    print("=" * 50)
    build_relevance()

    print("\nDone. All files saved to data/processed/")
