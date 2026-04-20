"""
generators.py

Pluggable backends for the final "generate" step of the RAG pipeline.

All generators share the same signature:
    .generate(prompt: str, **context) -> str

`context` is optional structured data (query, case_chunks, statute_chunks)
that extractive backends use directly. LLM backends ignore it and work off
the assembled prompt string.

Four backends ship:
  - ExtractiveGenerator  NO LLM. Composes a structured response from the
                         reranked chunks. Zero cost, zero hallucination.
                         Default — safest for a legal assistant where
                         invented citations are unacceptable.
  - DryRunGenerator      Returns a marker string so you can inspect the
                         assembled prompt without calling anything.
  - AnthropicGenerator   Claude via the Anthropic API. Needs
                         `pip install anthropic` and ANTHROPIC_API_KEY.
  - OllamaGenerator      Local model via Ollama server (default
                         http://localhost:11434). Needs `ollama` running
                         and the model pulled (e.g. `ollama pull llama3.1`).

Imports are lazy so the base module works without optional deps installed.
Swap in a new backend by implementing .generate() — nothing else in the
pipeline needs to know which model is behind it.
"""

from __future__ import annotations

import os
from typing import Protocol


class Generator(Protocol):
    def generate(self, prompt: str, **context) -> str: ...


class DryRunGenerator:
    """Returns a marker string; the pipeline prints the prompt separately."""
    name = "dryrun"

    def generate(self, prompt: str, **context) -> str:
        return "[DRY RUN — no LLM called. Inspect the assembled prompt above.]"


class ExtractiveGenerator:
    """
    No LLM. Builds a deterministic, citation-safe response from the reranked
    chunks. The returned string preserves the [C-*] / [S-*] markers from the
    pipeline and cannot invent text outside the retrieved passages.
    """
    name = "extractive"

    SNIPPET_CHARS = 400  # per-chunk quote length in the summary

    def _snippet(self, text: str) -> str:
        text = text.strip().replace("\n", " ")
        if len(text) <= self.SNIPPET_CHARS:
            return text
        cut = text[:self.SNIPPET_CHARS]
        idx = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
        if idx > self.SNIPPET_CHARS - 200:
            return cut[:idx + 1].rstrip() + " […]"
        return cut.rstrip() + " […]"

    def _format_items(self, chunks, prefix):
        if not chunks:
            return f"  (no {prefix.lower()} items retrieved)"
        lines = []
        for i, (c, score) in enumerate(chunks, start=1):
            marker = f"[{prefix}-{i}]"
            title  = c.get("title") or c.get("doc_id", "")
            court  = c.get("court") or ""
            date   = c.get("date") or ""
            role   = c.get("role") or ""
            head_bits = [b for b in (title, court, date) if b]
            head = f"{marker} {c['doc_id']}"
            if head_bits:
                head += " — " + " | ".join(head_bits)
            head += f"  (role: {role}, rerank: {score:.3f})"
            lines.append(head)
            lines.append(f"  > {self._snippet(c['text'])}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def generate(self, prompt: str, *,
                 query: str | None = None,
                 case_chunks=None,
                 statute_chunks=None,
                 **_) -> str:
        case_chunks    = case_chunks    or []
        statute_chunks = statute_chunks or []
        parts = [
            "EXTRACTIVE SUMMARY (no LLM — content is drawn verbatim from "
            "retrieved passages; relevance judgements and reasoning are left "
            "to the reader).",
            "",
            "### Most relevant prior cases",
            self._format_items(case_chunks, "C"),
            "",
            "### Most relevant statutes",
            self._format_items(statute_chunks, "S"),
            "",
            "### How to use",
            "  - Open each cited passage in context before relying on it.",
            "  - The reranker score indicates the cross-encoder's confidence; "
            "higher is better but not a substitute for reading the case.",
            "  - For a prose analysis, switch RAG_BACKEND to `anthropic` or "
            "`ollama`.",
        ]
        return "\n".join(parts)


class AnthropicGenerator:
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-6",
                 api_key: str | None = None,
                 max_tokens: int = 1024,
                 temperature: float = 0.2):
        import anthropic  # lazy
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompt: str, **_) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


class OllamaGenerator:
    name = "ollama"

    def __init__(self, model: str = "llama3.1",
                 host: str = "http://localhost:11434",
                 temperature: float = 0.2):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature

    def generate(self, prompt: str, **_) -> str:
        import urllib.request
        import json
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body.get("response", "")


def make_generator(backend: str | None = None) -> Generator:
    """
    Build a generator from env/config. backend overrides RAG_BACKEND env var.
    Defaults to extractive so the pipeline works out of the box without any
    API keys, network calls, or hallucination risk.
    """
    choice = (backend or os.environ.get("RAG_BACKEND") or "extractive").lower()
    if choice == "extractive":
        return ExtractiveGenerator()
    if choice == "dryrun":
        return DryRunGenerator()
    if choice == "anthropic":
        return AnthropicGenerator(
            model=os.environ.get("RAG_MODEL", "claude-sonnet-4-6"),
        )
    if choice == "ollama":
        return OllamaGenerator(
            model=os.environ.get("RAG_MODEL", "llama3.1"),
            host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        )
    raise ValueError(f"Unknown RAG_BACKEND: {choice!r}")
