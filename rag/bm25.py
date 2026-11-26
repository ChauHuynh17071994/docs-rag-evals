"""Okapi BM25, implemented without external dependencies."""

from __future__ import annotations

import math
import re
from collections import Counter

TOKEN = re.compile(r"[a-z0-9]+")

# Small English stopword list: enough to stop "the" and "is" from dominating scores.
STOPWORDS = frozenset(
    """a an and are as at be by can do does for from how i if in is it its many may much
    must of on or should than that the their them there this to up us was we what when where
    which who will with within you your""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if len(t) > 1 and t not in STOPWORDS]


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(d) for d in documents]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / len(self.docs) if self.docs else 0.0
        self.tf = [Counter(d) for d in self.docs]
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(set(doc))
        n = len(self.docs)
        # BM25+ style idf floor at 0 so very common terms never subtract score.
        self.idf = {t: max(0.0, math.log((n - f + 0.5) / (f + 0.5) + 1)) for t, f in df.items()}

    def score(self, query: str, index: int) -> float:
        tf = self.tf[index]
        norm = self.k1 * (1 - self.b + self.b * self.doc_len[index] / (self.avgdl or 1))
        total = 0.0
        for term in tokenize(query):
            freq = tf.get(term, 0)
            if freq:
                total += self.idf.get(term, 0.0) * freq * (self.k1 + 1) / (freq + norm)
        return total

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        scored = [(i, self.score(query, i)) for i in range(len(self.docs))]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda s: (-s[1], s[0]))
        return scored[:k]
