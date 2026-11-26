"""LLM backends.

Every backend receives the question plus the retrieved chunks and must answer using
citations of the form ``[source#n]``. The extractive backend needs no API key, which
keeps tests and the CI eval deterministic and free.
"""

from __future__ import annotations

import os
import re
from typing import Protocol

from .bm25 import stems
from .chunking import Chunk

SYSTEM_PROMPT = """You answer questions about company policies using ONLY the provided context.

Rules:
- Every factual sentence must end with a citation in square brackets, e.g. [travel-policy.md#2].
- Only cite chunk ids that appear in the context.
- If the context does not contain the answer, reply exactly: I don't know based on the available documents.
- Be concise: at most 3 sentences."""

REFUSAL = "I don't know based on the available documents."


class LLM(Protocol):
    name: str

    def answer(self, question: str, context: list[Chunk]) -> str: ...


def format_context(context: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.id}] ({c.title})\n{c.text}" for c in context)


class ExtractiveLLM:
    """Offline baseline: returns the context sentence with the most query-term overlap."""

    name = "extractive"

    def answer(self, question: str, context: list[Chunk]) -> str:
        terms = set(stems(question))
        best: tuple[int, str, str] | None = None
        for chunk in context:
            for sentence in re.split(r"(?<=[.!?])\s+", chunk.text):
                overlap = len(terms & set(stems(sentence)))
                if overlap and (best is None or overlap > best[0]):
                    best = (overlap, sentence.strip(), chunk.id)
        if best is None:
            return REFUSAL
        return f"{best[1]} [{best[2]}]"


class ClaudeLLM:
    """Claude backend. Requires ANTHROPIC_API_KEY."""

    name = "claude"

    def __init__(self, model: str | None = None, max_tokens: int = 300) -> None:
        import anthropic  # imported lazily so the offline path has no hard dependency

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("RAG_MODEL", "claude-sonnet-4-5")
        self.max_tokens = max_tokens

    def answer(self, question: str, context: list[Chunk]) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{format_context(context)}\n\nQuestion: {question}",
                }
            ],
        )
        return "".join(b.text for b in message.content if b.type == "text").strip()


def get_llm(backend: str | None = None) -> LLM:
    backend = backend or os.environ.get("RAG_BACKEND", "extractive")
    if backend == "extractive":
        return ExtractiveLLM()
    if backend == "claude":
        return ClaudeLLM()
    raise ValueError(f"unknown backend: {backend}")
