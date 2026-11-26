"""Dense retrieval.

``HashingEmbedder`` maps text to a fixed-size vector of hashed character n-grams
(the "hashing trick"), L2-normalised. It needs no model download or API key, is
deterministic, and catches what exact-token BM25 misses: morphological variants
("password" vs "passwords") and partial overlaps. It is a stand-in for a neural
embedding model; any object with an ``embed(texts) -> list[list[float]]`` method
can replace it.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol

from .bm25 import tokenize

Vector = list[float]


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> list[Vector]: ...


def _bucket(gram: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    # Signed hashing keeps collisions from only ever adding similarity.
    return value % dim, 1.0 if (value >> 63) & 1 else -1.0


class HashingEmbedder:
    """Character n-gram hashing embedder (offline, deterministic)."""

    name = "hashing"

    def __init__(self, dim: int = 2048, ngram_range: tuple[int, int] = (3, 5)) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.ngram_range = ngram_range

    def _grams(self, text: str) -> list[str]:
        grams: list[str] = []
        lo, hi = self.ngram_range
        for token in tokenize(text):
            padded = f"<{token}>"
            for n in range(lo, hi + 1):
                grams.extend(padded[i : i + n] for i in range(max(1, len(padded) - n + 1)))
        return grams

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            for gram in self._grams(text):
                slot, sign = _bucket(gram, self.dim)
                vec[slot] += sign
            vectors.append(normalize(vec))
        return vectors


class SentenceTransformerEmbedder:
    """Optional neural embedder (``pip install sentence-transformers``). Not used in CI."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # lazy, optional dependency

        self.model = SentenceTransformer(model)
        self.name = f"st:{model.rsplit('/', 1)[-1]}"

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [list(map(float, v)) for v in self.model.encode(list(texts), normalize_embeddings=True)]


def normalize(vec: Vector) -> Vector:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity of two already-normalised vectors."""
    return sum(x * y for x, y in zip(a, b, strict=True))


class DenseIndex:
    def __init__(self, documents: Sequence[str], embedder: Embedder) -> None:
        self.embedder = embedder
        self.vectors = embedder.embed(documents)

    def search(self, query: str, k: int = 5, min_similarity: float = 0.0) -> list[tuple[int, float]]:
        (q,) = self.embedder.embed([query])
        scored = [(i, cosine(q, v)) for i, v in enumerate(self.vectors)]
        scored = [s for s in scored if s[1] > min_similarity]
        scored.sort(key=lambda s: (-s[1], s[0]))
        return scored[:k]


def get_embedder(name: str = "hashing") -> Embedder:
    if name == "hashing":
        return HashingEmbedder()
    if name.startswith("st:"):
        return SentenceTransformerEmbedder(name[3:])
    raise ValueError(f"unknown embedder: {name}")


__all__ = [
    "DenseIndex",
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "cosine",
    "get_embedder",
    "normalize",
]
