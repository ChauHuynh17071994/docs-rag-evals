"""Search index over chunks, with JSON persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .bm25 import BM25
from .chunking import Chunk, load_corpus


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("cannot build an index from zero chunks")
        self.chunks = chunks
        # The section title is indexed too: headings carry a lot of signal in policy docs.
        self._bm25 = BM25([f"{c.title} {c.text}" for c in chunks])

    @classmethod
    def from_folder(cls, folder: str | Path, max_chars: int = 800) -> Index:
        return cls(load_corpus(folder, max_chars))

    def search(self, query: str, k: int = 4) -> list[Hit]:
        return [Hit(self.chunks[i], s) for i, s in self._bm25.search(query, k)]

    def get(self, chunk_id: str) -> Chunk | None:
        return next((c for c in self.chunks if c.id == chunk_id), None)

    def save(self, path: str | Path) -> None:
        payload = {"version": 1, "chunks": [asdict(c) for c in self.chunks]}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Index:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError(f"unsupported index version: {payload.get('version')}")
        return cls([Chunk(**c) for c in payload["chunks"]])
