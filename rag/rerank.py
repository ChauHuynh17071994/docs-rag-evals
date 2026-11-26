"""Second-stage reranking of retrieved candidates.

``LexicalReranker`` is an offline, deterministic reranker that looks at signals a
first-stage retriever ignores:

- coverage:  share of distinct query terms that appear in the chunk
- proximity: how tightly the matched terms cluster (smallest window containing them)
- title:     share of query terms that appear in the section heading

The first-stage rank is kept as a small prior so the reranker only reorders when it
has a reason to. A cross-encoder or an LLM judge can replace it behind the same
``rerank(question, hits)`` interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .bm25 import stems
from .index import Hit


class Reranker(Protocol):
    name: str

    def rerank(self, question: str, hits: list[Hit]) -> list[Hit]: ...


def smallest_window(positions: dict[str, list[int]]) -> int | None:
    """Length of the shortest token span that contains at least one position of every term."""
    events = sorted((pos, term) for term, plist in positions.items() for pos in plist)
    need, counts, have = len(positions), {}, 0
    best, left = None, 0
    for pos, term in events:
        counts[term] = counts.get(term, 0) + 1
        if counts[term] == 1:
            have += 1
        while have == need:
            span = pos - events[left][0] + 1
            best = span if best is None else min(best, span)
            left_term = events[left][1]
            counts[left_term] -= 1
            if counts[left_term] == 0:
                have -= 1
            left += 1
    return best


@dataclass
class LexicalReranker:
    w_coverage: float = 1.0
    w_proximity: float = 0.5
    w_title: float = 0.3
    w_prior: float = 0.2
    name: str = "lexical"

    def score(self, terms: set[str], hit: Hit, rank: int) -> float:
        body = stems(hit.chunk.text)
        positions: dict[str, list[int]] = {}
        for i, token in enumerate(body):
            if token in terms:
                positions.setdefault(token, []).append(i)
        coverage = len(positions) / len(terms)
        window = smallest_window(positions) if len(positions) > 1 else None
        proximity = len(positions) / window if window else 0.0
        title = len(terms & set(stems(hit.chunk.title))) / len(terms)
        prior = 1.0 / rank
        return (
            self.w_coverage * coverage
            + self.w_proximity * proximity
            + self.w_title * title
            + self.w_prior * prior
        )

    def rerank(self, question: str, hits: list[Hit]) -> list[Hit]:
        terms = set(stems(question))
        if not terms:
            return list(hits)
        scored = [(self.score(terms, h, r), -r, h) for r, h in enumerate(hits, start=1)]
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [h for _, _, h in scored]
