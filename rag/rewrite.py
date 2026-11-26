"""Query rewriting for short, keyword-style questions.

Users type "wfh days" or "pw length", not full questions. Short queries carry few
terms, use abbreviations, and often miss the vocabulary of the documents. A rewriter
expands them before retrieval; the generator still sees the user's original question.

- ``GlossaryRewriter`` (offline): expands abbreviations and synonyms from a domain
  glossary, only for queries with at most ``max_terms`` content terms.
- ``ClaudeRewriter`` (optional): asks Claude to rewrite the query in the documents'
  vocabulary. Requires ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Protocol

from .bm25 import tokenize

# A small HR / IT / finance glossary of the kind a company keeps for its intranet search.
DEFAULT_GLOSSARY: dict[str, str] = {
    "wfh": "work remotely remote work",
    "remote": "work remotely",
    "pw": "password",
    "pwd": "password",
    "mfa": "multi-factor authentication",
    "2fa": "multi-factor authentication",
    "per diem": "daily allowance",
    "vacation": "annual leave",
    "holiday": "annual leave",
    "pto": "annual leave",
    "cap": "maximum",
    "limit": "maximum",
    "sick note": "doctor's certificate sick leave",
    "po": "purchase order",
    "pr": "purchase requisition",
    "payment terms": "invoices paid net",
    "expenses": "expense claims",
    "laptop": "laptop device",
}


class Rewriter(Protocol):
    name: str

    def rewrite(self, question: str) -> str: ...


class GlossaryRewriter:
    name = "glossary"

    def __init__(self, glossary: dict[str, str] | None = None, max_terms: int = 3) -> None:
        self.glossary = {k.lower(): v for k, v in (glossary or DEFAULT_GLOSSARY).items()}
        self.max_terms = max_terms
        # Longest phrases first so "sick note" wins over a hypothetical "note".
        keys = sorted(self.glossary, key=len, reverse=True)
        self._pattern = re.compile(r"\b(" + "|".join(map(re.escape, keys)) + r")\b") if keys else None

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> GlossaryRewriter:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), **kwargs)

    def rewrite(self, question: str) -> str:
        if len(tokenize(question)) > self.max_terms or self._pattern is None:
            return question
        expansions = [self.glossary[m] for m in self._pattern.findall(question.lower())]
        if not expansions:
            return question
        return f"{question} {' '.join(dict.fromkeys(expansions))}"


REWRITE_PROMPT = """Rewrite the search query so it matches the wording of company policy documents.
Expand abbreviations, add likely synonyms, keep it under 20 words, and output only the query."""


class ClaudeRewriter:
    name = "claude"

    def __init__(self, model: str | None = None, max_terms: int = 3) -> None:
        import anthropic  # lazy: optional dependency path

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("RAG_MODEL", "claude-sonnet-4-5")
        self.max_terms = max_terms

    def rewrite(self, question: str) -> str:
        if len(tokenize(question)) > self.max_terms:
            return question
        message = self.client.messages.create(
            model=self.model,
            max_tokens=60,
            temperature=0,
            system=REWRITE_PROMPT,
            messages=[{"role": "user", "content": question}],
        )
        text = "".join(b.text for b in message.content if b.type == "text").strip()
        return f"{question} {text}" if text else question


def get_rewriter(name: str | None) -> Rewriter | None:
    if not name or name == "none":
        return None
    if name == "glossary":
        return GlossaryRewriter()
    if name == "claude":
        return ClaudeRewriter()
    raise ValueError(f"unknown rewriter: {name}")
