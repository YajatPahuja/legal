# Indian Legal AI Assistant — Project Plan

## Dataset: AILA 2019 (Combined Corpus)

All data is under `archive/`:
```
archive/
├── Object_casedocs/       # 2,914 case files (C1.txt … C2914.txt)
├── Object_statutes/       # 197 statute files (S1.txt … S197.txt)
├── Query_doc.txt          # 50 queries, format: AILA_Q1||<full case text>
├── relevance_judgments_priorcases.txt   # TREC format: 195 relevant pairs
└── relevance_judgments_statutes.txt     # TREC format: 221 relevant pairs
```

**Two evaluation tasks:**
- Task 1: Given a query → retrieve relevant prior case judgments (195 positive pairs across 50 queries)
- Task 2: Given a query → retrieve relevant applicable statutes (221 positive pairs across 50 queries)

Both tasks share the same pipeline. `doc_type` metadata (`case` / `statute`) handles task-specific filtering.

---

## References (Teacher-provided papers)

| Paper | Key Contribution | Applied Here |
|---|---|---|
| Kalamkar et al., LREC 2022 | 13-role rhetorical taxonomy for Indian judgments; SciBERT-HSLN baseline | Role taxonomy for chunker; task-specific weight tables |
| Furniturewala et al., AILA 2021 Task 2 | Joint BERT + TF-IDF features for sentence relevance; 7-role labeling | Re-ranker architecture; fine-tuning strategy |
| Steno AI, SemEval 2023 Task 6 | InLegalBERT + GCN for role labeling; 86% F1 | Confirms InLegalBERT as best base model |

**Additional dataset for fine-tuning:**
- IndicLegalQA (Kaggle: kmldas/indiclegalqa-dataset) — 10,000 QA pairs from 1,256 SC judgments → fine-tune re-ranker

---

## Methodology: Rhetorical Role-Aware Hybrid RAG

### Novel Contributions

1. **13-role rhetorical chunking** (Kalamkar et al. taxonomy) instead of fixed token windows
2. **Task-specific role weights** — case retrieval boosts RATIO/PRECEDENT_RELIED; statute retrieval boosts STATUTE/ISSUE
3. **Unified dual-task corpus** — cases + statutes in one pipeline, filtered by metadata
4. **Hybrid BM25 + Dense retrieval** with Reciprocal Rank Fusion
5. **Re-ranker fine-tuned on IndicLegalQA** using joint BERT + TF-IDF features (Paper 3)

### Rhetorical Role Taxonomy (inspired by Kalamkar et al., LREC 2022)

| Role | Description | Task 1 Weight | Task 2 Weight |
|---|---|---|---|
| RATIO | Ratio decidendi — core legal principle | **1.6** | 1.2 |
| ANALYSIS | Court's reasoning and consideration | **1.4** | 1.3 |
| RULING | Final order/decision | 1.2 | 1.1 |
| STATUTE_REF | Statutory provisions cited | 1.0 | **1.6** |
| ARGUMENTS | Petitioner/respondent arguments | 1.0 | 0.9 |
| FACTS | Background facts | 0.8 | 0.7 |
| GENERAL | Uncategorized | 0.7 | 0.7 |

---

## Architecture

```
Query (full current case description)
        │
        ▼
┌─────────────────────────────────────────┐
│         Hybrid Retriever                │
│                                         │
│  BM25 (lexical, role-weighted)          │
│       +                                 │
│  Dense (InLegalBERT, task-specific      │
│         role weights)                   │
│         └──────────┬────────────────────┘
│                    │ Reciprocal Rank Fusion (RRF)
└────────────────────┼────────────────────┘
                     │ Top-K candidates
                     ▼
┌─────────────────────────────────────────┐
│         Re-Ranker                       │
│  InLegalBERT fine-tuned on IndicLegalQA │
│  Joint features: BERT(768) + TF-IDF     │
│  (query, chunk) → relevance score       │
└────────────────────┬────────────────────┘
                     │ Final ranked list
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
| Embeddings | `law-ai/InLegalBERT` (fp16 on MPS) |
| Rhetorical role detection | Regex patterns (Kalamkar et al. taxonomy) |
| Vector store | ChromaDB (persistent, local) |
| BM25 | `rank_bm25` library |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Re-ranker | InLegalBERT fine-tuned on IndicLegalQA |
| LLM | Llama-3 8B (local) or GPT-3.5 via API |
| Backend | FastAPI |
| Frontend | Streamlit |
| Evaluation | MAP, NDCG@10, MRR, P@5, P@10 |

---

## Build Steps (in order)

- [x] **Step 1** — Project setup: preprocessing scripts + requirements.txt
- [x] **Step 2** — `build_corpus.py` → unified_corpus.jsonl + queries.json + relevance.json
- [x] **Step 3** — `chunker.py` (13-role taxonomy) → chunked_corpus.jsonl
- [x] **Step 4** — BM25 baseline evaluation (MAP=0.014 cases, MAP=0.062 statutes)
- [ ] **Step 5** — `index_corpus.py` (MPS, batch=256) → ChromaDB index ← **RUNNING NOW**
- [ ] **Step 6** — `evaluate_dense.py` → dense retrieval metrics vs BM25
- [ ] **Step 7** — Hybrid retrieval: BM25 + dense + RRF → `hybrid_retriever.py`
- [ ] **Step 8** — Download IndicLegalQA, run `finetune_qa.py` → fine-tuned re-ranker
- [ ] **Step 9** — Full RAG pipeline: `rag_pipeline.py`
- [ ] **Step 10** — Final evaluation: all stages compared
- [ ] **Step 11** — FastAPI backend + Streamlit demo

---

## Results Tracker

| Model | Task 1 MAP | Task 2 MAP | Task 1 NDCG@10 | Task 2 NDCG@10 |
|---|---|---|---|---|
| BM25 baseline | 0.0144 | 0.0617 | 0.0188 | 0.1060 |
| Dense (InLegalBERT) | — | — | — | — |
| Hybrid BM25 + Dense | — | — | — | — |
| + Re-ranker (IndicLegalQA) | — | — | — | — |
