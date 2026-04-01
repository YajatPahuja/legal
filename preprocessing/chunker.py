"""
chunker.py

Splits documents from unified_corpus.jsonl into retrieval-ready chunks.

Novel strategy: Rhetorical role-aware chunking for case documents.
  - Detect role boundaries using regex patterns
  - Each chunk is tagged with its rhetorical role
  - Role weights are applied during retrieval scoring

For statutes (short): no chunking needed, entire doc = one chunk.

Run:
    python preprocessing/chunker.py
"""

import re
import json
from pathlib import Path
from tqdm import tqdm

BASE_DIR  = Path(__file__).resolve().parent.parent
PROC_DIR  = BASE_DIR / "data" / "processed"
IN_FILE   = PROC_DIR / "unified_corpus.jsonl"
OUT_FILE  = PROC_DIR / "chunked_corpus.jsonl"

# ── Rhetorical role patterns ───────────────────────────────────────────────────
# These are keyword-based heuristics to detect role section boundaries.
ROLE_PATTERNS = [
    ("FACTS",       re.compile(
        r'\b(facts?|background|brief facts?|factual background|case background|'
        r'brief background|facts? of the case)\b', re.IGNORECASE)),
    ("ARGUMENTS",   re.compile(
        r'\b(argued?|submitted?|contended?|counsel.*?submits?|learned counsel|'
        r'argument[s]? on behalf|it is argued|petitioner.*?submits?|respondent.*?submits?)\b',
        re.IGNORECASE)),
    ("ANALYSIS",    re.compile(
        r'\b(we have considered|having considered|on consideration|'
        r'in our (view|opinion)|the court (holds?|finds?|observes?)|'
        r'we (hold|find|observe|are of the view))\b', re.IGNORECASE)),
    ("RATIO",       re.compile(
        r'\b(ratio decidendi|the law (is|has been)|it is (settled|well.settled|established)|'
        r'it is a settled (law|proposition|principle)|legal proposition|'
        r'principle of law|this court has (held|laid down))\b', re.IGNORECASE)),
    ("STATUTE_REF", re.compile(
        r'\b(section \d+|article \d+|IPC|CrPC|CPC|Constitution of India|'
        r'under (the )?act|under section|by virtue of)\b', re.IGNORECASE)),
    ("RULING",      re.compile(
        r'\b(accordingly|in the result|for (the )?foregoing reasons|'
        r'the appeal (is|stands?)|the petition (is|stands?)|we (allow|dismiss|'
        r'partly allow)|order accordingly|disposed? of|set aside)\b', re.IGNORECASE)),
]

# Higher weight = more important for retrieval
ROLE_WEIGHTS = {
    "RATIO":       1.5,
    "ANALYSIS":    1.3,
    "RULING":      1.2,
    "STATUTE_REF": 1.2,
    "ARGUMENTS":   1.0,
    "FACTS":       0.9,
    "GENERAL":     0.8,
}

MAX_CHUNK_CHARS  = 1500   # ~300-400 tokens for BERT
OVERLAP_CHARS    = 200    # overlap between consecutive chunks


def detect_role(text: str) -> str:
    """Return the dominant rhetorical role for a text segment."""
    scores = {role: 0 for role, _ in ROLE_PATTERNS}
    for role, pattern in ROLE_PATTERNS:
        scores[role] = len(pattern.findall(text))
    best_role = max(scores, key=scores.get)
    return best_role if scores[best_role] > 0 else "GENERAL"


def split_into_sentences(text: str) -> list[str]:
    """Sentence-level split for legal text."""
    # Split on ". " followed by capital letter, or newlines
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_case(doc: dict) -> list[dict]:
    """
    Role-aware chunking for case documents.
    Groups sentences into chunks of ~MAX_CHUNK_CHARS with overlap,
    detecting and tagging the dominant rhetorical role per chunk.
    """
    text      = doc["text"]
    sentences = split_into_sentences(text)
    chunks    = []
    current   = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) > MAX_CHUNK_CHARS and current:
            chunk_text = " ".join(current)
            role       = detect_role(chunk_text)
            chunks.append({
                "doc_id":    doc["doc_id"],
                "doc_type":  "case",
                "chunk_id":  f"{doc['doc_id']}_c{len(chunks)}",
                "role":      role,
                "weight":    ROLE_WEIGHTS[role],
                "title":     doc.get("title", ""),
                "court":     doc.get("court", ""),
                "date":      doc.get("date", ""),
                "text":      chunk_text,
            })
            # Overlap: keep last N chars worth of sentences
            overlap_text = ""
            for s in reversed(current):
                if len(overlap_text) + len(s) < OVERLAP_CHARS:
                    overlap_text = s + " " + overlap_text
                else:
                    break
            current     = overlap_text.split(". ")
            current_len = len(overlap_text)

        current.append(sent)
        current_len += len(sent) + 1

    # Last chunk
    if current:
        chunk_text = " ".join(current)
        role       = detect_role(chunk_text)
        chunks.append({
            "doc_id":   doc["doc_id"],
            "doc_type": "case",
            "chunk_id": f"{doc['doc_id']}_c{len(chunks)}",
            "role":     role,
            "weight":   ROLE_WEIGHTS[role],
            "title":    doc.get("title", ""),
            "court":    doc.get("court", ""),
            "date":     doc.get("date", ""),
            "text":     chunk_text,
        })

    return chunks


def chunk_statute(doc: dict) -> list[dict]:
    """
    Statutes are short (~200-300 words) — keep as single chunk.
    Title is prepended to improve semantic matching.
    """
    text = f"{doc.get('title', '')}. {doc.get('text', '')}".strip()
    return [{
        "doc_id":      doc["doc_id"],
        "doc_type":    "statute",
        "chunk_id":    f"{doc['doc_id']}_c0",
        "role":        "STATUTE_REF",
        "weight":      ROLE_WEIGHTS["STATUTE_REF"],
        "title":       doc.get("title", ""),
        "description": doc.get("description", ""),
        "text":        text,
    }]


def build_chunked_corpus():
    all_chunks   = []
    case_count   = 0
    stat_count   = 0
    chunk_count  = 0

    with open(IN_FILE, "r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]

    print(f"Chunking {len(docs)} documents...")
    for doc in tqdm(docs):
        if doc["doc_type"] == "case":
            chunks = chunk_case(doc)
            case_count += 1
        else:
            chunks = chunk_statute(doc)
            stat_count += 1

        all_chunks.extend(chunks)
        chunk_count += len(chunks)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nChunked corpus saved → {OUT_FILE}")
    print(f"  Case docs:      {case_count}")
    print(f"  Statute docs:   {stat_count}")
    print(f"  Total chunks:   {chunk_count}")
    print(f"  Avg chunks/case: {chunk_count / max(case_count, 1):.1f}")

    # Role distribution
    from collections import Counter
    role_counts = Counter(c["role"] for c in all_chunks if c["doc_type"] == "case")
    print("\n  Role distribution (cases):")
    for role, cnt in role_counts.most_common():
        print(f"    {role:<15} {cnt}")


if __name__ == "__main__":
    build_chunked_corpus()
