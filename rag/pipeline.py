"""Answer pipeline: retrieve -> refuse if weak evidence -> generate -> validate citations."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .index import Hit, Index
from .llm import LLM, REFUSAL

CITATION = re.compile(r"\[([^\[\]]+?\.md#\d+)\]")


@dataclass
class Answer:
    question: str
    text: str
    refused: bool
    citations: list[str]
    invalid_citations: list[str]
    retrieved: list[str]
    top_score: float
    latency_ms: float
    backend: str
    meta: dict = field(default_factory=dict)

    @property
    def grounded(self) -> bool:
        """Answered, cited at least one chunk, and cited nothing outside the retrieved context."""
        return not self.refused and bool(self.citations) and not self.invalid_citations


class RagPipeline:
    def __init__(
        self,
        index: Index,
        llm: LLM,
        k: int = 4,
        min_score: float = 1.5,
        min_similarity: float = 0.22,
    ) -> None:
        self.index = index
        self.llm = llm
        self.k = k
        self.min_score = min_score
        self.min_similarity = min_similarity

    def retrieve(self, question: str) -> list[Hit]:
        return self.index.search(question, self.k)

    def ask(self, question: str) -> Answer:
        start = time.perf_counter()
        hits = self.retrieve(question)
        top = max((h.score for h in hits), default=0.0)
        top_dense = max((h.dense for h in hits), default=0.0)
        retrieved = [h.chunk.id for h in hits]
        meta = {"retriever": self.index.retriever, "top_dense": round(top_dense, 3)}

        # Weak evidence from every retriever: refuse without spending an LLM call.
        if top < self.min_score and top_dense < self.min_similarity:
            return self._result(question, REFUSAL, [], retrieved, top, start, meta)

        text = self.llm.answer(question, [h.chunk for h in hits])
        return self._result(question, text, retrieved, retrieved, top, start, meta)

    def _result(self, question, text, allowed, retrieved, top, start, meta) -> Answer:
        cited = list(dict.fromkeys(CITATION.findall(text)))
        allowed_set = set(allowed)
        return Answer(
            question=question,
            text=text,
            refused=text.strip() == REFUSAL,
            citations=[c for c in cited if c in allowed_set],
            invalid_citations=[c for c in cited if c not in allowed_set],
            retrieved=retrieved,
            top_score=round(top, 3),
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            backend=self.llm.name,
            meta=meta,
        )
