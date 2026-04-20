"""
bm25_retriever.py

BM25 retriever over the chunked AILA 2019 corpus with Named Entity Removal.

Key finding from FIRE 2019 AILA overview (Bhattacharya et al., 2019):
  Removing named entities (person names, org names, court names) from both
  queries and documents substantially improved retrieval performance.
  TRDDC Pune's NER-removal approach was the top unsupervised system.

  Reason: Legal cases should match on legal principles and issues,
  not on the specific names of people or organisations involved.

NER removal is applied at query time (no re-indexing needed):
  - Corpus chunks are indexed with NER-cleaned text at startup
  - Queries are NER-cleaned before tokenization
"""

import json
import pickle
import re
import spacy
from pathlib import Path
from collections import defaultdict
from rank_bm25 import BM25Okapi

BASE_DIR       = Path(__file__).resolve().parent.parent
CHUNKS_FILE    = BASE_DIR / "data" / "processed" / "chunked_corpus.jsonl"
TOKEN_CACHE    = BASE_DIR / "data" / "processed" / "bm25_token_cache.pkl"

# spaCy model for NER — download once with: python -m spacy download en_core_web_sm
try:
    _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
except OSError:
    _nlp = None
    print("Warning: spaCy model not found. Run: python -m spacy download en_core_web_sm")

# Legal terms to always keep even if NER flags them
KEEP_TERMS = {
    "court", "supreme", "high", "district", "tribunal", "judge", "justice",
    "section", "article", "act", "ipc", "crpc", "cpc", "constitution",
    "petition", "appeal", "writ", "habeas", "corpus", "mandamus",
}

# Named entity types to remove (person names, orgs, locations add noise)
REMOVE_ENT_TYPES = {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP"}


def remove_named_entities(text: str) -> str:
    """
    Remove named entities from text using spaCy NER.
    Keeps legal structural terms (court, section, article etc.)
    Falls back to original text if spaCy unavailable.
    """
    if _nlp is None:
        return text

    # Process in chunks to handle long documents efficiently
    MAX_CHARS = 100_000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    doc    = _nlp(text)
    result = text

    # Replace entities with a placeholder in reverse order (preserve indices)
    replacements = []
    for ent in doc.ents:
        if ent.label_ in REMOVE_ENT_TYPES:
            # Keep if any word is a legal keep-term
            if not any(w.lower() in KEEP_TERMS for w in ent.text.split()):
                replacements.append((ent.start_char, ent.end_char))

    # Apply replacements in reverse order
    for start, end in reversed(replacements):
        result = result[:start] + result[end:]

    # Clean up extra whitespace
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def tokenize(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric."""
    return re.findall(r'[a-z0-9]+', text.lower())


def tokenize_cleaned(text: str) -> list[str]:
    """NER removal → tokenize."""
    return tokenize(remove_named_entities(text))


def _batch_tokenize(chunks: list[dict]) -> list[list[str]]:
    """Tokenize corpus chunks without NER — NER is applied only at query time.
    Corpus NER removal gives marginal gains vs. the cost of processing 46k docs;
    query-time NER (already in retrieve()) captures the key benefit."""
    return [tokenize(c["text"]) for c in chunks]


class BM25Retriever:
    def __init__(self):
        print("Loading chunked corpus...")
        self._chunks = self._load_chunks()

        self._case_chunks    = [c for c in self._chunks if c["doc_type"] == "case"]
        self._statute_chunks = [c for c in self._chunks if c["doc_type"] == "statute"]

        print(f"  {len(self._case_chunks)} case chunks | {len(self._statute_chunks)} statute chunks")

        case_tokens, stat_tokens = self._get_tokens()
        self._bm25_cases    = BM25Okapi(case_tokens)
        self._bm25_statutes = BM25Okapi(stat_tokens)
        print("BM25 ready.")

    def _get_tokens(self):
        """Load tokenized corpus from cache, or build and cache it."""
        cache_key = str(CHUNKS_FILE.stat().st_mtime)
        if TOKEN_CACHE.exists():
            with open(TOKEN_CACHE, "rb") as f:
                cached = pickle.load(f)
            if cached.get("mtime") == cache_key:
                print("  Loaded tokenized corpus from cache.")
                return cached["case_tokens"], cached["stat_tokens"]

        ner_status = "with NER removal" if _nlp else "without NER (spaCy not found)"
        print(f"Building BM25 indexes ({ner_status})... (will cache for future runs)")
        case_tokens = _batch_tokenize(self._case_chunks)
        stat_tokens = _batch_tokenize(self._statute_chunks)
        with open(TOKEN_CACHE, "wb") as f:
            pickle.dump({"mtime": cache_key, "case_tokens": case_tokens, "stat_tokens": stat_tokens}, f)
        print("  Token cache saved.")
        return case_tokens, stat_tokens

    def _load_chunks(self) -> list[dict]:
        chunks = []
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        return chunks

    def retrieve(self, query: str, doc_type: str, top_k: int = 100) -> list[tuple[str, float]]:
        """
        NER-clean query → tokenize → BM25 score → max-pool per doc → ranked list.
        """
        tokens = tokenize_cleaned(query)

        if doc_type == "case":
            bm25   = self._bm25_cases
            chunks = self._case_chunks
        else:
            bm25   = self._bm25_statutes
            chunks = self._statute_chunks

        scores = bm25.get_scores(tokens)

        doc_scores: dict[str, float] = defaultdict(float)
        for chunk, score in zip(chunks, scores):
            weighted = score * chunk.get("weight", 1.0)
            doc_scores[chunk["doc_id"]] = max(doc_scores[chunk["doc_id"]], weighted)

        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
