"""Search index over chunks, with JSON persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .bm25 import BM25
from .chunking import Chunk, load_corpus
from .embeddings import DenseIndex, Embedder, HashingEmbedder

RETRIEVERS = ("bm25", "dense", "hybrid")
RRF_K = 60  # standard constant from Cormack et al. (2009); dampens the weight of top ranks


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # BM25 score (0 when the chunk was found only by dense search)
    dense: float = 0.0  # cosine similarity (0 when dense search is off)


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: sum 1 / (k + rank) over every ranking a doc appears in."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda s: (-s[1], s[0]))


class Index:
    def __init__(
        self, chunks: list[Chunk], retriever: str = "bm25", embedder: Embedder | None = None
    ) -> None:
        if not chunks:
            raise ValueError("cannot build an index from zero chunks")
        if retriever not in RETRIEVERS:
            raise ValueError(f"unknown retriever {retriever!r}; choose from {RETRIEVERS}")
        self.chunks = chunks
        self.retriever = retriever
        # The section title is indexed too: headings carry a lot of signal in policy docs.
        texts = [f"{c.title} {c.text}" for c in chunks]
        self._bm25 = BM25(texts)
        self._dense = DenseIndex(texts, embedder or HashingEmbedder()) if retriever != "bm25" else None

    @classmethod
    def from_folder(cls, folder: str | Path, max_chars: int = 800, **kwargs) -> Index:
        return cls(load_corpus(folder, max_chars), **kwargs)

    def search(self, query: str, k: int = 4, candidates: int = 20) -> list[Hit]:
        lexical = dict(self._bm25.search(query, max(k, candidates)))
        if self._dense is None:
            order = list(lexical)
            dense: dict[int, float] = {}
        else:
            dense = dict(self._dense.search(query, max(k, candidates)))
            if self.retriever == "dense":
                order = list(dense)
            else:
                order = [i for i, _ in rrf([list(lexical), list(dense)])]
        return [Hit(self.chunks[i], lexical.get(i, 0.0), dense.get(i, 0.0)) for i in order[:k]]

    def get(self, chunk_id: str) -> Chunk | None:
        return next((c for c in self.chunks if c.id == chunk_id), None)

    def save(self, path: str | Path) -> None:
        payload = {"version": 1, "chunks": [asdict(c) for c in self.chunks]}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> Index:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError(f"unsupported index version: {payload.get('version')}")
        return cls([Chunk(**c) for c in payload["chunks"]], **kwargs)
