# Indian Legal AI Assistant

A rhetorical-role-aware hybrid RAG system for Indian case law. Built on the
**AILA 2019** benchmark (FIRE), the pipeline retrieves relevant prior cases
and statutes for a new factual query and — optionally — produces a grounded
analysis citing them.

The retrieval stack is a classical three-stage IR design:

1. **Lexical** — BM25 over role-chunked, NER-scrubbed text
2. **Dense** — `all-mpnet-base-v2` embeddings with task-specific role weights
3. **Fusion** — Reciprocal Rank Fusion (RRF) of lexical + dense rankings
4. **Rerank** — cross-encoder (`ms-marco-MiniLM-L-12-v2`) over the fused top-K
5. **Generate** — pluggable final step (extractive by default; Anthropic /
   Ollama / dry-run optional)

Each stage has its own evaluation script so you can isolate its contribution.

---

## Table of contents

- [Results at a glance](#results-at-a-glance)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Pipeline stages](#pipeline-stages)
  - [Stage 0 — Preprocessing & rhetorical-role chunking](#stage-0--preprocessing--rhetorical-role-chunking)
  - [Stage 1 — BM25 retrieval](#stage-1--bm25-retrieval)
  - [Stage 2 — Dense retrieval](#stage-2--dense-retrieval)
  - [Stage 3 — Hybrid fusion (RRF)](#stage-3--hybrid-fusion-rrf)
  - [Stage 4 — Cross-encoder reranking](#stage-4--cross-encoder-reranking)
  - [Stage 5 — RAG generation](#stage-5--rag-generation)
- [Repository layout](#repository-layout)
- [How to run](#how-to-run)
- [Design decisions](#design-decisions)
- [Evaluation methodology](#evaluation-methodology)
- [Known gaps and future work](#known-gaps-and-future-work)

---

## Results at a glance

Scored on the 50 AILA 2019 queries using the standard relevance judgements
(195 positive pairs for cases, 221 for statutes).

**Task 1 — Prior case retrieval**

| Stage                          | MAP        | NDCG@10    | MRR        | P@5        | P@10       |
|--------------------------------|------------|------------|------------|------------|------------|
| BM25                           | 0.0144     | 0.0188     | 0.102     | 0.0120     | 0.030     |
| Dense (mpnet)                  | 0.0160     | 0.0252     | 0.176     | 0.0220     | 0.0420     |
| Hybrid (RRF)                   | 0.0182     | 0.0260     | 0.197     | 0.0360     | 0.0520     |
| **Hybrid + cross-encoder**     | **0.0288** | **0.0446** | **0.205** | **0.0440** | **0.0580** |

**Task 2 — Statute retrieval**

| Stage                          | MAP        | NDCG@10    | MRR        | P@5        | P@10       |
|--------------------------------|------------|------------|------------|------------|------------|
| BM25                           | 0.0617     | **0.1060** | **0.1917** | **0.0600** | **0.0560** |
| Dense (mpnet)                  | 0.0620     | 0.0721     | 0.1433     | 0.0320     | 0.0380     |
| Hybrid (RRF)                   | 0.0556     | 0.0829     | 0.1467     | 0.0560     | 0.0460     |
| **Hybrid + cross-encoder**     | **0.0638** | 0.0922     | 0.1810     | 0.0520     | 0.0440     |

**Key observations**

- Reranking **doubles** case MAP (0.0144 → 0.0288) and more than doubles
  NDCG@10. The cross-encoder's joint attention over (query, chunk) pays off
  most where paraphrase matters — case law.
- For statutes, BM25 alone **still wins** NDCG@10 / MRR / P@5 / P@10.
  Section numbers and verbatim statutory language are exact-match signals
  that dense methods dilute. Reranking recovers MAP (largest gain on the
  full list) but doesn't beat BM25 on the top ranks.
- This asymmetry is why fusion is weighted per task
  ([retrieval/hybrid_retriever.py](retrieval/hybrid_retriever.py),
  `TASK_FUSION_WEIGHTS`).

---

## Architecture

End-to-end data flow from raw AILA files to answer:

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                     OFFLINE (run once)                              │
  │                                                                     │
  │  archive/Object_casedocs/*.txt       archive/Object_statutes/*.txt  │
  │  archive/Query_doc.txt               archive/relevance_*.txt        │
  │                  │                                                  │
  │                  ▼                                                  │
  │   preprocessing/build_corpus.py                                     │
  │                  │                                                  │
  │                  ▼                                                  │
  │   data/processed/unified_corpus.jsonl                               │
  │   data/processed/queries.json                                       │
  │   data/processed/relevance.json                                     │
  │                  │                                                  │
  │                  ▼                                                  │
  │   preprocessing/chunker.py  (13-role rhetorical taxonomy)           │
  │                  │                                                  │
  │                  ▼                                                  │
  │   data/processed/chunked_corpus.jsonl                               │
  │                  │                                                  │
  │         ┌────────┴────────┐                                         │
  │         ▼                 ▼                                         │
  │   BM25 token cache   retrieval/index_corpus.py                      │
  │   (.pkl)             (ChromaDB, mpnet fp16 on MPS)                  │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │                     ONLINE (per query)                              │
  │                                                                     │
  │                   user query (facts of a matter)                    │
  │                            │                                        │
  │         ┌──────────────────┼──────────────────┐                     │
  │         ▼                                     ▼                     │
  │   BM25Retriever                         DenseRetriever              │
  │   (rank_bm25, NER-scrubbed)             (ChromaDB, role-weighted)   │
  │         │                                     │                     │
  │         └──────────────┬──────────────────────┘                     │
  │                        ▼                                            │
  │                  HybridRetriever                                    │
  │                  (Reciprocal Rank Fusion)                           │
  │                        │                                            │
  │                        ▼  top-K candidates (K=50 cases, 30 statutes)│
  │                    Reranker                                         │
  │                    (cross-encoder/ms-marco-MiniLM-L-12-v2)          │
  │                        │                                            │
  │         ┌──────────────┴──────────────┐                             │
  │         ▼                             ▼                             │
  │  doc-level ranking            chunk-level top-N                     │
  │  (evaluation)                 (RAG context)                         │
  │                                       │                             │
  │                                       ▼                             │
  │                                RAGPipeline                          │
  │                                 ▼                                   │
  │                          Generator (pluggable)                      │
  │                          ├── Extractive  (default, no LLM)          │
  │                          ├── DryRun      (prompt only)              │
  │                          ├── Anthropic   (Claude)                   │
  │                          └── Ollama      (local llama3.1)           │
  │                                 │                                   │
  │                                 ▼                                   │
  │                     answer + citations [C-1]..[S-N]                 │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## Dataset

AILA 2019 ships under `archive/`:

```
archive/
├── Object_casedocs/              2,914 case files  (C1.txt … C2914.txt)
├── Object_statutes/                197 statute files (S1.txt … S197.txt)
├── Query_doc.txt                    50 queries, format: AILA_Q<id>||<full text>
├── relevance_judgments_priorcases.txt   TREC qrels — 195 positive pairs
└── relevance_judgments_statutes.txt     TREC qrels — 221 positive pairs
```

Two evaluation tasks share the same pipeline:

| Task | Corpus            | Relevance judgements | Positive pairs |
|------|-------------------|----------------------|----------------|
| 1    | Prior case docs   | `priorcases` qrels   | 195            |
| 2    | Statutes          | `statutes` qrels     | 221            |

Queries are **full case narratives**, typically 400–1,500 words of factual
exposition. `doc_type` metadata (`case` / `statute`) tags every chunk so a
single index serves both tasks.

---

## Pipeline stages

### Stage 0 — Preprocessing & rhetorical-role chunking

Files: [preprocessing/build_corpus.py](preprocessing/build_corpus.py),
[preprocessing/chunker.py](preprocessing/chunker.py).

Instead of fixed token windows, each document is split into passages tagged
with a **rhetorical role** from the 13-role taxonomy (inspired by Kalamkar
et al., LREC 2022). Each role carries a **task-specific weight** used later
by the dense retriever for max-pooled chunk → doc scoring.

```
                raw case / statute text
                          │
                          ▼
                ┌─────────────────────┐
                │  role classifier    │   regex + heuristics over
                │  (Kalamkar-inspired)│   boilerplate markers
                └─────────┬───────────┘
                          │
     FACTS · ARGUMENTS · ANALYSIS · RATIO · RULING · STATUTE_REF · GENERAL
                          │
                          ▼
                passage chunks, each tagged:
                  { doc_id, doc_type, chunk_id, role, weight,
                    title, court, date, text }
```

**Task-specific role weights** ([preprocessing/chunker.py](preprocessing/chunker.py)):

| Role        | Task 1 (cases) | Task 2 (statutes) | Rationale                     |
|-------------|----------------|-------------------|-------------------------------|
| RATIO       | **1.6**        | 1.2               | Ratio decidendi = reusable law|
| ANALYSIS    | 1.4            | 1.3               | Court reasoning travels well  |
| RULING      | 1.2            | 1.1               | The holding                   |
| STATUTE_REF | 1.0            | **1.6**           | Section text = statute signal |
| ARGUMENTS   | 1.0            | 0.9               |                               |
| FACTS       | 0.8            | 0.7               | Case-specific, low reuse      |
| GENERAL     | 0.7            | 0.7               | Fallback bucket               |

### Stage 1 — BM25 retrieval

File: [retrieval/bm25_retriever.py](retrieval/bm25_retriever.py).

```
         query text
              │
              ▼
       ┌──────────────┐
       │  spaCy NER   │  drop PERSON / ORG / GPE / LOC / FAC / NORP
       │  tokenizer   │  keep KEEP_TERMS (legal structural words)
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐      pre-built on corpus chunks (cached .pkl)
       │   BM25Okapi  │◄─────────────────────────────┐
       │  (rank_bm25) │                              │
       └──────┬───────┘                         token cache
              │
              ▼
     per-chunk scores → max-pool to doc level → top-k
```

NER removal is the key lexical-retrieval move: raw judgments contain judge
names, counsel names, and cities that would otherwise dominate TF-IDF. The
`KEEP_TERMS` allowlist preserves words like *appellant, petitioner, section,
court, article* that carry real retrieval signal.

### Stage 2 — Dense retrieval

File: [retrieval/dense_retriever.py](retrieval/dense_retriever.py), with
[retrieval/embedder.py](retrieval/embedder.py) and
[retrieval/vector_store.py](retrieval/vector_store.py).

```
           query text
              │
              ▼
      truncate to first 300 words        (AILA queries are multi-thousand
              │                            words; mpnet has a 384-token cap)
              ▼
   sentence-transformers
   all-mpnet-base-v2 (fp16, MPS)
              │
              ▼
        query vector (768d)
              │
              ▼
       ┌──────────────┐
       │  ChromaDB    │   separate collections: _cases, _statutes
       │  (cosine)    │
       └──────┬───────┘
              │
              ▼
     top-(k × 10) chunk hits
              │
              ▼
   score' = cosine × role_weight          (TASK_WEIGHTS from chunker.py)
              │
              ▼
   max-pool to doc level → top-k docs
```

`all-mpnet-base-v2` is trained with contrastive loss for semantic similarity
and outperformed raw InLegalBERT embeddings on the AILA queries in our
experiments (see the early commit history for the switch). An
InLegalBERT-specific encoder remains a drop-in swap behind the same
interface.

### Stage 3 — Hybrid fusion (RRF)

File: [retrieval/hybrid_retriever.py](retrieval/hybrid_retriever.py).

```
    BM25 ranking                      Dense ranking
    ────────────                      ─────────────
    rank 1: docA                      rank 1: docC
    rank 2: docB                      rank 2: docA
    rank 3: docC                      rank 3: docD
       ...                               ...

                     │     │
                     ▼     ▼
            Reciprocal Rank Fusion
                                          w_i
         rrf_score(doc) =  ∑     ──────────────────
                          i ∈ {bm25,dense}   k + rank_i(doc)

                         k = 60  (Cormack et al. 2009)
                         w_i = TASK_FUSION_WEIGHTS[doc_type][i]
```

RRF is **rank-based, not score-based**: BM25's unbounded scores and dense
cosine similarities aren't directly comparable, so we fuse ranks instead.

**Per-task fusion weights:**

| doc_type | BM25 weight | Dense weight |
|----------|-------------|--------------|
| case     | 1.0         | 1.0          |
| statute  | 1.5         | 1.0          |

The statute bump reflects the empirical observation that BM25 dominates
top-ranks on statute retrieval — there is a principled reason (exact match
on section numbers) and a data-driven one (BM25 beats dense on Task 2
NDCG@10 and MRR).

### Stage 4 — Cross-encoder reranking

File: [retrieval/reranker.py](retrieval/reranker.py).

```
     query text + hybrid top-K candidates
                   │
                   ▼
   expand each doc → all its chunks
                   │
                   ▼
   build (query, chunk_text) pairs
                   │
                   ▼
   ┌──────────────────────────────────┐
   │  cross-encoder/                  │   joint-encoding: the model attends
   │  ms-marco-MiniLM-L-12-v2         │   across query ↔ chunk, unlike a
   │  max_length = 512                │   bi-encoder which scores each side
   │  batch_size = 64 (MPS)           │   independently
   └──────────────┬───────────────────┘
                  │
                  ▼
         score per (query, chunk)
                  │
         ┌────────┴────────┐
         ▼                 ▼
  doc-level:          chunk-level:
  max-pool scores     sort all chunks globally
  per doc_id          → top-N for RAG context
  → rerank() API      → rerank_chunks() API
```

**Why a cross-encoder at all**: bi-encoders (BM25, mpnet) score query and
document independently — fast, but they miss fine-grained relevance signals
that only appear when the two representations can attend to each other. A
cross-encoder is the standard second-stage in modern IR for exactly this
reason.

**Why MS-MARCO-MiniLM**: it's an off-the-shelf, general-purpose reranker
trained on passage ranking. No legal-domain fine-tuning, but small enough
to rerank 50 candidates × ~15 chunks each in reasonable time on a MacBook
Air's MPS. A domain-tuned swap (e.g. InLegalBERT fine-tuned on IndicLegalQA)
plugs in behind the same interface later without touching the pipeline.

### Stage 5 — RAG generation

Files: [rag/pipeline.py](rag/pipeline.py), [rag/generators.py](rag/generators.py),
[rag/run_demo.py](rag/run_demo.py).

```
    query text
        │
        ▼
   HybridRetriever.retrieve(query, doc_type, top_k=50/30)
        │
        ▼
   Reranker.rerank_chunks(query, candidates, top_n=3)      × 2 (cases, statutes)
        │
        ▼
   build_prompt(query, case_chunks, statute_chunks)
        │
        ▼
   ┌──────────────────────────────────────────────────┐
   │              Generator (pluggable)               │
   │                                                  │
   │   ┌────────────────────────────────────────────┐ │
   │   │ ExtractiveGenerator  (default)             │ │
   │   │   No LLM. Renders a structured summary     │ │
   │   │   with [C-N] / [S-N] markers and verbatim  │ │
   │   │   snippets. Zero hallucination.            │ │
   │   └────────────────────────────────────────────┘ │
   │   ┌────────────────────────────────────────────┐ │
   │   │ DryRunGenerator                            │ │
   │   │   Returns prompt only. No API call.        │ │
   │   └────────────────────────────────────────────┘ │
   │   ┌────────────────────────────────────────────┐ │
   │   │ AnthropicGenerator                         │ │
   │   │   Claude via Anthropic API.                │ │
   │   │   Needs ANTHROPIC_API_KEY.                 │ │
   │   └────────────────────────────────────────────┘ │
   │   ┌────────────────────────────────────────────┐ │
   │   │ OllamaGenerator                            │ │
   │   │   Local llama3.1 via Ollama daemon.        │ │
   │   └────────────────────────────────────────────┘ │
   └──────────────────────────┬───────────────────────┘
                              ▼
              { prompt, answer, citations: {C-1: doc_id, ...},
                case_chunks, statute_chunks }
```

Select the backend with the `RAG_BACKEND` env var (`extractive` |
`dryrun` | `anthropic` | `ollama`). Extractive is the default so the
pipeline runs out of the box with no keys, no network, and no risk of
the model inventing a citation.

---

## Repository layout

```
legal/
├── archive/                             raw AILA 2019 data
├── data/processed/                      build artifacts
│   ├── unified_corpus.jsonl
│   ├── chunked_corpus.jsonl
│   ├── queries.json
│   ├── relevance.json
│   ├── bm25_token_cache.pkl
│   ├── bm25_results.json
│   ├── dense_results.json
│   ├── hybrid_results.json
│   └── reranked_results.json
│
├── preprocessing/
│   ├── build_corpus.py                  archive/ → unified_corpus.jsonl
│   └── chunker.py                       rhetorical-role chunking + weights
│
├── retrieval/
│   ├── embedder.py                      LegalEmbedder (mpnet, MPS/CUDA/CPU)
│   ├── vector_store.py                  ChromaDB wrapper (cases + statutes)
│   ├── index_corpus.py                  batch-embed chunks into ChromaDB
│   ├── bm25_retriever.py                rank_bm25 + NER scrubbing
│   ├── dense_retriever.py               query ChromaDB, role-weighted max-pool
│   ├── hybrid_retriever.py              RRF fusion, per-task weights
│   └── reranker.py                      cross-encoder, doc & chunk APIs
│
├── evaluation/
│   ├── metrics.py                       MAP, NDCG@10, MRR, P@5, P@10
│   ├── evaluate_bm25.py
│   ├── evaluate_dense.py
│   ├── evaluate_hybrid.py               side-by-side with BM25 / Dense
│   └── evaluate_reranked.py             side-by-side with BM25 / Dense / Hybrid
│
├── rag/
│   ├── pipeline.py                      RAGPipeline: retrieve → rerank → generate
│   ├── generators.py                    Extractive / DryRun / Anthropic / Ollama
│   └── run_demo.py                      CLI entrypoint
│
├── training/                            (reserved — IndicLegalQA fine-tune, skipped)
├── requirements.txt
├── setup.sh
└── PROJECT_PLAN.md
```

---

## How to run

**Setup**

```bash
bash setup.sh                 # creates venv, installs requirements.txt
source venv/bin/activate
python -m spacy download en_core_web_sm
```

**Build the indexes (run once)**

```bash
python preprocessing/build_corpus.py         # → unified_corpus.jsonl
python preprocessing/chunker.py              # → chunked_corpus.jsonl
python retrieval/index_corpus.py             # → ChromaDB (MPS, batch 256)
```

**Evaluate each stage**

```bash
python evaluation/evaluate_bm25.py
python evaluation/evaluate_dense.py
python evaluation/evaluate_hybrid.py         # prints BM25 / Dense / Hybrid
python evaluation/evaluate_reranked.py       # prints all 4 side-by-side
```

Every evaluator saves its full ranked results to
`data/processed/<stage>_results.json` so later stages can diff against it.

**Run a query through the RAG pipeline**

```bash
# Default — no LLM, no keys, no network
python rag/run_demo.py --qid AILA_Q1

# Prompt-only, useful for inspecting the assembled context
RAG_BACKEND=dryrun python rag/run_demo.py --qid AILA_Q1

# Claude via Anthropic API
RAG_BACKEND=anthropic ANTHROPIC_API_KEY=sk-... \
    python rag/run_demo.py --qid AILA_Q1

# Local llama3.1 via Ollama
RAG_BACKEND=ollama python rag/run_demo.py --qid AILA_Q1

# Free-form query (not from AILA set)
python rag/run_demo.py --query-text "A bank employee was dismissed after..."

# Save the full result (prompt + answer + chunks) to JSON
python rag/run_demo.py --qid AILA_Q3 --save out.json
```

---

## Design decisions

**Why rhetorical-role chunking instead of fixed windows.**
A judgment's FACTS section is largely case-specific and a poor reuse signal;
its RATIO section is exactly the part precedent works on. Fixed-window
chunking blurs that distinction. Role tags let the dense retriever weight
chunks differently per task (RATIO for case retrieval, STATUTE_REF for
statute retrieval) — a lever we use in `TASK_WEIGHTS`.

**Why mpnet instead of InLegalBERT.**
InLegalBERT is domain-pretrained but trained with masked language modelling,
not sentence similarity. `all-mpnet-base-v2` is contrastively trained for
exactly the retrieval task, and it outperformed raw InLegalBERT embeddings
on the AILA query set. InLegalBERT remains a swap-in option — the interface
in `embedder.py` is model-agnostic.

**Why RRF instead of score-weighted fusion.**
BM25 produces unbounded scores; dense cosine similarities live in `[-1, 1]`.
Normalising them into a common scale is fiddly and brittle across query
distributions. RRF sidesteps this by fusing *ranks*, which are comparable
by construction. `k=60` is the value from Cormack et al. (2009).

**Why per-task fusion weights.**
The data is asymmetric: BM25 dominates top-ranks on statutes (exact-match on
section numbers is decisive), dense helps on cases (paraphrase over
long-range fact patterns). A single set of weights leaves points on the
table. `TASK_FUSION_WEIGHTS` encodes this directly.

**Why max-pool chunk scores to doc level.**
Mirroring how a human skims a case: a single highly relevant paragraph
usually drives the decision to pull a case off the shelf. Mean-pooling
dilutes a strong signal with boilerplate; sum-pooling biases toward long
documents.

**Why an off-the-shelf cross-encoder instead of fine-tuning.**
Fine-tuning on IndicLegalQA is in scope later but was deliberately skipped
for this iteration. `ms-marco-MiniLM-L-12-v2` already doubles case MAP with
no training, which is a strong ablation baseline: any fine-tuned model has
to beat 0.0288 to justify its training cost.

**Why extractive is the default generator.**
Hallucinated citations are unacceptable in a legal tool. The extractive
path renders the reranker's top passages verbatim with stable citation
markers and zero model calls. LLM backends remain one env var away when
fluent prose is the goal.

---

## Evaluation methodology

- All metrics computed with [evaluation/metrics.py](evaluation/metrics.py):
  MAP, NDCG@10, MRR, P@5, P@10 (standard IR metrics; Manning et al.).
- Relevance judgements parsed from the official AILA qrels files into
  `data/processed/relevance.json`. A doc is relevant if the qrel label is
  non-zero; the binary form is used for MAP and precision, the graded form
  (always 1 here) feeds NDCG.
- Every stage runs against the full 50-query set; no per-query cherry-picking.
- Each evaluator dumps both metrics **and** the full ranked result list
  keyed by query id, so later stages can be compared on the exact same
  candidate pools.

---

## Known gaps and future work

- **Stale InLegalBERT docstrings** in [retrieval/dense_retriever.py](retrieval/dense_retriever.py)
  — code uses mpnet, a few comment lines still mention InLegalBERT. Cosmetic
  only; flagged for cleanup.
- **IndicLegalQA fine-tune for the reranker** — the `training/` directory is
  reserved for this. Expected to lift statute top-ranks where MS-MARCO's
  general-domain signal is weakest.
- **BM25 remains the statute top-rank leader.** A statute-specific pipeline
  variant (BM25-only or very-low-dense-weight) could beat the current
  reranked hybrid on NDCG@10 / MRR. Worth testing if top-1 statute
  precision is the product target.
- **Query truncation.** AILA queries are full case narratives; we truncate
  to the first 300 words for the encoder models. A query-summarization pass
  before embedding would likely help and is a cheap next experiment.
- **No API / UI yet.** FastAPI + Streamlit deliberately out of scope for
  this iteration; the CLI in `rag/run_demo.py` is the current user surface.
