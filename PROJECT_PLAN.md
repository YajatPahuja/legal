# Indian Legal AI Assistant — Project Plan

## Dataset: AILA 2019 (Combined Corpus)

**"Combining both datasets" = merging Object_casedocs + Object_statutes into one unified retrieval corpus.**

```
legal/
├── Object_casedocs/                  # 2,914 Supreme Court judgments
├── Object_statutes/                  # 197 Indian law sections
├── Query_doc.txt                     # 50 legal situation queries
├── relevance_judgments_prior_cases   # Ground truth: Task 1
└── relevance_judgments_statutes      # Ground truth: Task 2
```

**Two evaluation tasks:**
- Task 1: Given a query → retrieve relevant prior case judgments
- Task 2: Given a query → retrieve relevant applicable statutes

Both tasks use the **same RAG pipeline** over a **unified corpus** (cases + statutes merged). A `doc_type` metadata field (`case` or `statute`) handles task-specific filtering.

---

## Methodology: Rhetorical Role-Aware Hybrid RAG

### Novelty
Standard RAG uses fixed-size chunking (e.g., 512 tokens). This is bad for legal documents because a judgment mixes facts, arguments, reasoning, and rulings — retrieving a chunk from "Facts" section to answer a legal question is noise.

**Novel contribution: chunk by rhetorical role, not by token count.**

Each document is segmented into role-labeled chunks:
- `FACTS` — background facts of the case
- `ARGUMENTS` — petitioner/respondent arguments  
- `ANALYSIS` — court's legal analysis
- `RATIO` — ratio decidendi (core legal reasoning)
- `RULING` — final decision
- `STATUTE_REF` — statutes cited

During retrieval, role-weighted scoring boosts chunks from `RATIO` and `ANALYSIS` roles, which carry the most legally relevant content.

This is novel for AILA 2019 — no prior AILA paper uses rhetorical role-aware chunking.

---

## Architecture

```
Query (legal situation description)
        │
        ▼
┌───────────────────┐
│  Query Expansion  │  ← expand with legal synonyms / IPC terms
└────────┬──────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Hybrid Retriever                │
│                                         │
│  BM25 (lexical)  +  Dense (semantic)    │
│         └──────────┬────────────────────┘
│                    │ Reciprocal Rank Fusion
└────────────────────┼────────────────────┘
                     │
                     ▼ Top-K chunks (with doc_type metadata)
┌─────────────────────────────────────────┐
│         Cross-Encoder Re-Ranker         │  ← InLegalBERT fine-tuned
│  (query, chunk) → relevance score       │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│         Answer Generation               │
│  Retrieved chunks → LLM → grounded      │
│  answer with source attribution         │
└─────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Chunking | Custom role-aware splitter + LangChain RecursiveTextSplitter fallback |
| Rhetorical role tagger | OpenNyAI `en_legal_ner_trf` or fine-tuned InLegalBERT |
| Embeddings | `law-ai/InLegalBERT` |
| Vector store | ChromaDB (local, no infra needed) |
| BM25 | `rank_bm25` library |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Re-ranker | Cross-encoder on `InLegalBERT` |
| LLM | Llama-3 8B (local) or GPT-3.5 via API |
| Backend | FastAPI |
| Frontend | Streamlit |
| Evaluation | MAP, NDCG@10, MRR, P@5 (AILA standard metrics) |

---

## Project Structure

```
legal-ai/
├── data/
│   ├── aila_2019/
│   │   ├── Object_casedocs/
│   │   ├── Object_statutes/
│   │   ├── Query_doc.txt
│   │   ├── relevance_judgments_prior_cases
│   │   └── relevance_judgments_statutes
│   └── processed/
│       ├── unified_corpus.jsonl       # cases + statutes merged
│       └── role_chunked_corpus.jsonl  # after rhetorical role chunking
├── preprocessing/
│   ├── text_cleaner.py                # normalize, fix encoding
│   ├── rhetorical_tagger.py           # assign role labels to sentences
│   ├── chunker.py                     # role-aware chunking
│   └── build_corpus.py                # merge casedocs + statutes
├── retrieval/
│   ├── embedder.py                    # InLegalBERT embeddings
│   ├── vector_store.py                # ChromaDB interface
│   ├── bm25_retriever.py              # BM25 sparse retrieval
│   ├── hybrid_retriever.py            # BM25 + dense + RRF fusion
│   └── reranker.py                    # cross-encoder re-ranking
├── rag/
│   ├── query_expander.py              # legal query expansion
│   ├── rag_pipeline.py                # end-to-end RAG chain
│   └── prompt_templates.py            # prompts for case vs statute tasks
├── evaluation/
│   ├── metrics.py                     # MAP, NDCG, MRR, P@5
│   └── evaluate_aila.py               # run eval on 50 queries
├── api/
│   └── app.py                         # FastAPI endpoints
├── frontend/
│   └── app.py                         # Streamlit UI
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_baseline.ipynb
├── configs/
│   └── config.yaml
└── requirements.txt
```

---

## Dataset Location

All data is under `archive/`:
```
archive/
├── Object_casedocs/       # 2,914 case files (C1.txt … C2914.txt)
├── Object_statutes/       # 197 statute files (S1.txt … S197.txt)
├── Query_doc.txt          # 50 queries in <Q id="AILA_Q1">…</Q> format
├── relevance_judgments_priorcases.txt   # TREC format, listed docs = relevant
└── relevance_judgments_statutes.txt
```

**"Combining both datasets"** = merging Object_casedocs + Object_statutes into one
unified JSONL corpus with `doc_type: case | statute` metadata.

---

## Build Steps (in order)

- [x] **Step 1** — Write preprocessing scripts + requirements.txt
- [ ] **Step 2** — Run `setup.sh`, then `build_corpus.py` → `unified_corpus.jsonl`
- [ ] **Step 3** — Run `chunker.py` → `chunked_corpus.jsonl`
- [ ] **Step 4** — BM25 baseline retrieval + evaluate (MAP/NDCG on 50 queries)
- [ ] **Step 5** — Dense retrieval: embed chunks with InLegalBERT → ChromaDB
- [ ] **Step 6** — Hybrid retrieval: BM25 + dense + Reciprocal Rank Fusion
- [ ] **Step 7** — Cross-encoder re-ranking (fine-tune on AILA relevance labels)
- [ ] **Step 8** — Full RAG pipeline: query → retrieve → re-rank → LLM generate
- [ ] **Step 9** — Evaluation: compare all stages vs. baseline
- [ ] **Step 10** — FastAPI backend + Streamlit demo

---

## Baseline vs. Novel Model (Expected Results)

| Model | Task 1 MAP | Task 2 MAP |
|---|---|---|
| BM25 only | ~0.20 | ~0.25 |
| Dense retrieval (InLegalBERT) | ~0.35 | ~0.38 |
| Hybrid BM25 + Dense (RRF) | ~0.40 | ~0.42 |
| + Role-aware chunking (novel) | ~0.44 | ~0.45 |
| + Cross-encoder re-ranking (novel) | ~0.47 | ~0.48 |

---

## Novel Contributions Summary

1. **Unified dual-task RAG corpus** — cases + statutes in one index, task handled by metadata filtering
2. **Rhetorical role-aware chunking** — segment by legal roles (Facts/Ratio/Ruling) instead of token windows
3. **Role-weighted retrieval scoring** — boost chunks from `RATIO` and `ANALYSIS` roles
4. **Hybrid retrieval with RRF** — BM25 + InLegalBERT dense + reciprocal rank fusion
5. **Cross-encoder re-ranking** — fine-tuned on AILA relevance judgments for precision boost
