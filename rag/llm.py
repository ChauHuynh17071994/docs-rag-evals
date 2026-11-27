"""LLM backends.

Every backend receives the question plus the retrieved chunks and must answer using
citations of the form ``[source#n]``. The extractive backend needs no API key, which
keeps tests and the CI eval deterministic and free.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
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

# List prices in USD per million tokens (input, output). Override with RAG_PRICE_IN / RAG_PRICE_OUT
# when a model is missing here or prices change.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-opus-4-5": (5.0, 25.0),
}


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES_PER_MTOK.get(model, (0.0, 0.0))
    price_in = float(os.environ.get("RAG_PRICE_IN", price_in))
    price_out = float(os.environ.get("RAG_PRICE_OUT", price_out))
    return round((input_tokens * price_in + output_tokens * price_out) / 1_000_000, 6)


class LLM(Protocol):
    name: str

    def answer(self, question: str, context: list[Chunk]) -> str: ...


def generate(llm: LLM, question: str, context: list[Chunk]) -> tuple[str, Usage]:
    """Answer and report token usage. Backends without ``generate`` report zero usage."""
    if hasattr(llm, "generate"):
        return llm.generate(question, context)
    return llm.answer(question, context), Usage()


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
        return self.generate(question, context)[0]

    def generate(self, question: str, context: list[Chunk]) -> tuple[str, Usage]:
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
        text = "".join(b.text for b in message.content if b.type == "text").strip()
        tokens_in, tokens_out = message.usage.input_tokens, message.usage.output_tokens
        return text, Usage(tokens_in, tokens_out, estimate_cost(self.model, tokens_in, tokens_out))


def get_llm(backend: str | None = None) -> LLM:
    backend = backend or os.environ.get("RAG_BACKEND", "extractive")
    if backend == "extractive":
        return ExtractiveLLM()
    if backend == "claude":
        return ClaudeLLM()
    raise ValueError(f"unknown backend: {backend}")
