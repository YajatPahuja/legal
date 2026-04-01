"""
index_corpus.py

One-time script: embeds all chunks with InLegalBERT and stores in ChromaDB.

Runtime estimate (MacBook Air M-series, CPU):
  ~46k case chunks × 512 tokens, batch_size=32 → ~30–45 min
  197 statute chunks → ~1 min

Run once from project root:
    python retrieval/index_corpus.py

Re-running is safe — skips if already indexed.
"""

import sys
import json
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from retrieval.embedder import LegalEmbedder
from retrieval.vector_store import LegalVectorStore

CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunked_corpus.jsonl"
BATCH_SIZE  = 32   # lower to 16 if you get memory errors


def load_chunks_by_type() -> tuple[list[dict], list[dict]]:
    case_chunks, stat_chunks = [], []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            if chunk["doc_type"] == "case":
                case_chunks.append(chunk)
            else:
                stat_chunks.append(chunk)
    return case_chunks, stat_chunks


def index_chunks(chunks: list[dict], doc_type: str, embedder: LegalEmbedder, store: LegalVectorStore):
    print(f"\nIndexing {len(chunks)} {doc_type} chunks...")
    for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
        batch      = chunks[i : i + BATCH_SIZE]
        texts      = [c["text"] for c in batch]
        embeddings = embedder.encode(texts, batch_size=BATCH_SIZE)
        store.add_chunks(batch, embeddings, doc_type=doc_type)


def main():
    store    = LegalVectorStore()
    embedder = LegalEmbedder()

    case_chunks, stat_chunks = load_chunks_by_type()
    print(f"Loaded {len(case_chunks)} case chunks, {len(stat_chunks)} statute chunks.")

    # Index cases
    if store.is_indexed("case"):
        print(f"\nCase index already exists ({store.count('case')} chunks). Skipping.")
    else:
        index_chunks(case_chunks, "case", embedder, store)
        print(f"Case index complete. Total: {store.count('case')} chunks.")

    # Index statutes
    if store.is_indexed("statute"):
        print(f"Statute index already exists ({store.count('statute')} chunks). Skipping.")
    else:
        index_chunks(stat_chunks, "statute", embedder, store)
        print(f"Statute index complete. Total: {store.count('statute')} chunks.")

    print("\nIndexing complete. ChromaDB saved to data/chroma_db/")


if __name__ == "__main__":
    main()
