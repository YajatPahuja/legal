"""
pipeline.py

End-to-end RAG pipeline for the Indian legal assistant.

Flow:
  query (case facts)
    → hybrid retrieve (BM25 + dense, RRF)           # first-stage
    → cross-encoder rerank                          # second-stage
    → pull top-N chunks per task (cases, statutes)
    → assemble prompt with numbered citations
    → LLM generator (dry-run / Anthropic / Ollama)

Design notes:
  - We rerank at the CHUNK level (not doc level) here. For evaluation we
    max-pool to docs because the AILA relevance judgements are doc-level;
    for generation the LLM wants the specific passage, not the whole case.
  - Citations are [C-1]..[C-N] for cases and [S-1]..[S-M] for statutes.
    The prompt tells the model to use those markers, and the pipeline
    returns a citation map so the caller can resolve [C-1] → doc_id.
  - Prompt length is bounded by CHUNK_CHAR_CAP per chunk. Raw legal chunks
    can be ~1.5k chars each; capping keeps a 3-case + 3-statute prompt
    comfortably under most model limits without truncating mid-sentence
    logic heuristically (we cut on a sentence boundary when possible).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from rag.generators import Generator, make_generator


RERANK_K_CASES    = 50
RERANK_K_STATUTES = 30

TOP_N_CASES    = 3
TOP_N_STATUTES = 3

CHUNK_CHAR_CAP = 1200


SYSTEM_PREAMBLE = (
    "You are a legal research assistant for Indian law. You will be given "
    "the facts of a new matter, then a set of retrieved passages from prior "
    "cases and statutes. Your job is to analyze which of the retrieved "
    "items are actually applicable to the facts.\n\n"
    "Rules:\n"
    "  - Cite every claim using the bracketed markers provided: [C-1], "
    "[C-2], ... for cases and [S-1], [S-2], ... for statutes.\n"
    "  - If a retrieved passage is NOT relevant, say so explicitly — do "
    "not force a fit.\n"
    "  - Do not invent citations, case names, or section numbers that are "
    "not in the retrieved passages.\n"
    "  - Keep the analysis grounded in the retrieved text; flag gaps "
    "where a confident answer would need material not shown."
)


def _trim_chunk(text: str, cap: int = CHUNK_CHAR_CAP) -> str:
    if len(text) <= cap:
        return text
    cut = text[:cap]
    # Prefer ending on sentence boundary within the last 200 chars.
    for sep in (". ", "? ", "! "):
        idx = cut.rfind(sep, cap - 200)
        if idx != -1:
            return cut[:idx + 1].rstrip() + " […]"
    return cut.rstrip() + " […]"


def _format_chunk_block(marker: str, chunk: dict, score: float) -> str:
    title = chunk.get("title") or chunk.get("doc_id", "")
    court = chunk.get("court") or ""
    date  = chunk.get("date") or ""
    role  = chunk.get("role") or ""
    header_bits = [f"{marker} {chunk['doc_id']}"]
    if title:  header_bits.append(title)
    if court:  header_bits.append(court)
    if date:   header_bits.append(date)
    header = " | ".join(header_bits)
    meta = f"(role: {role}, rerank score: {score:.3f})"
    body = _trim_chunk(chunk["text"])
    return f"{header}\n{meta}\n{body}"


class RAGPipeline:
    def __init__(self,
                 hybrid: HybridRetriever | None = None,
                 reranker: Reranker | None = None,
                 generator: Generator | None = None):
        self.hybrid    = hybrid    or HybridRetriever()
        self.reranker  = reranker  or Reranker()
        self.generator = generator or make_generator()

    def _retrieve_reranked_chunks(self, query: str, doc_type: str,
                                  rerank_k: int, top_n: int
                                  ) -> list[tuple[dict, float]]:
        candidates = self.hybrid.retrieve(query, doc_type=doc_type, top_k=rerank_k)
        return self.reranker.rerank_chunks(query, candidates, top_n=top_n)

    def build_prompt(self, query: str,
                     case_chunks: list[tuple[dict, float]],
                     statute_chunks: list[tuple[dict, float]]) -> str:
        lines: list[str] = [SYSTEM_PREAMBLE, "", "## Facts of the matter", query.strip(), ""]

        lines.append("## Retrieved prior cases")
        if case_chunks:
            for i, (chunk, score) in enumerate(case_chunks, start=1):
                lines.append(_format_chunk_block(f"[C-{i}]", chunk, score))
                lines.append("")
        else:
            lines.append("(none retrieved)")
            lines.append("")

        lines.append("## Retrieved statutes")
        if statute_chunks:
            for i, (chunk, score) in enumerate(statute_chunks, start=1):
                lines.append(_format_chunk_block(f"[S-{i}]", chunk, score))
                lines.append("")
        else:
            lines.append("(none retrieved)")
            lines.append("")

        lines.extend([
            "## Task",
            "1. Identify which of the cases [C-*] and statutes [S-*] above are "
            "genuinely applicable to the facts. Explain the applicability in "
            "2–4 sentences each, citing the marker.",
            "2. List any items you judge NOT applicable and say briefly why.",
            "3. End with a short 'Bottom line' paragraph summarising the "
            "likely legal position, citing markers.",
        ])
        return "\n".join(lines)

    def answer(self, query: str,
               rerank_k_cases: int = RERANK_K_CASES,
               rerank_k_statutes: int = RERANK_K_STATUTES,
               top_n_cases: int = TOP_N_CASES,
               top_n_statutes: int = TOP_N_STATUTES,
               ) -> dict[str, Any]:
        case_chunks    = self._retrieve_reranked_chunks(
            query, "case",    rerank_k_cases,    top_n_cases)
        statute_chunks = self._retrieve_reranked_chunks(
            query, "statute", rerank_k_statutes, top_n_statutes)

        prompt = self.build_prompt(query, case_chunks, statute_chunks)
        answer = self.generator.generate(
            prompt,
            query=query,
            case_chunks=case_chunks,
            statute_chunks=statute_chunks,
        )

        citations = {
            f"C-{i}": chunk["doc_id"]
            for i, (chunk, _) in enumerate(case_chunks, start=1)
        }
        citations.update({
            f"S-{i}": chunk["doc_id"]
            for i, (chunk, _) in enumerate(statute_chunks, start=1)
        })

        return {
            "query":          query,
            "case_chunks":    case_chunks,
            "statute_chunks": statute_chunks,
            "citations":      citations,
            "prompt":         prompt,
            "answer":         answer,
            "backend":        getattr(self.generator, "name", "unknown"),
        }
