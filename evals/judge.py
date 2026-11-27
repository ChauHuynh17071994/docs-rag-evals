"""Faithfulness judges: is every claim in an answer supported by the chunks it cites?

- ``LexicalJudge`` (offline, deterministic): a sentence counts as supported when most of its
  content terms appear in one of the cited chunks. Cheap and reproducible, but it can only
  catch claims that use words absent from the source; it cannot catch a wrong number phrased
  with the source's own words.
- ``ClaudeJudge`` (needs ANTHROPIC_API_KEY): asks Claude to check each sentence against the
  cited chunks and return a JSON verdict.

Refusals are not judged: there is no claim to check.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Protocol

from rag.bm25 import stems
from rag.chunking import Chunk
from rag.llm import format_context

CITATION = re.compile(r"\[[^\[\]]+?\.md#\d+\]")
SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Verdict:
    score: float  # fraction of answer sentences supported by the cited chunks, 0..1
    faithful: bool  # every sentence supported
    reason: str = ""


class Judge(Protocol):
    name: str

    def judge(self, question: str, answer: str, cited: list[Chunk]) -> Verdict: ...


def claims(answer: str) -> list[str]:
    """Split an answer into sentences with citation markers removed."""
    parts = (CITATION.sub("", s).strip() for s in SENTENCE.split(answer))
    return [p for p in parts if stems(p)]


class LexicalJudge:
    name = "lexical"

    def __init__(self, min_overlap: float = 0.6) -> None:
        self.min_overlap = min_overlap

    def judge(self, question: str, answer: str, cited: list[Chunk]) -> Verdict:
        sentences = claims(answer)
        if not sentences:
            return Verdict(0.0, False, "no claims")
        if not cited:
            return Verdict(0.0, False, "no cited chunks")
        sources = [set(stems(f"{c.title} {c.text}")) for c in cited]
        unsupported = []
        for sentence in sentences:
            terms = set(stems(sentence))
            best = max(len(terms & src) / len(terms) for src in sources)
            if best < self.min_overlap:
                unsupported.append(sentence)
        score = round(1 - len(unsupported) / len(sentences), 3)
        reason = "unsupported: " + " | ".join(unsupported) if unsupported else ""
        return Verdict(score, not unsupported, reason)


JUDGE_PROMPT = """You check whether an answer is faithful to its sources.

For each sentence of the answer, decide whether the cited context fully supports it.
A sentence is unsupported if it adds facts, numbers or conditions that the context does not state.

Reply with JSON only: {"supported": <int>, "total": <int>, "reason": "<one short sentence>"}"""


class ClaudeJudge:
    name = "claude"

    def __init__(self, model: str | None = None) -> None:
        import anthropic  # lazy: the offline judge has no dependency on the SDK

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("RAG_JUDGE_MODEL", "claude-sonnet-4-5")

    def judge(self, question: str, answer: str, cited: list[Chunk]) -> Verdict:
        prompt = f"Context:\n{format_context(cited)}\n\nQuestion: {question}\n\nAnswer: {answer}"
        message = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            temperature=0,
            system=JUDGE_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in message.content if b.type == "text")
        return parse_verdict(text)


def parse_verdict(text: str) -> Verdict:
    """Parse the judge's JSON reply; tolerate surrounding prose or code fences."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return Verdict(0.0, False, f"unparseable judge reply: {text[:80]}")
    data = json.loads(match.group(0))
    total = max(int(data.get("total", 0)), 1)
    supported = min(max(int(data.get("supported", 0)), 0), total)
    return Verdict(round(supported / total, 3), supported == total, str(data.get("reason", "")))


def get_judge(name: str | None) -> Judge | None:
    if not name or name == "none":
        return None
    if name == "lexical":
        return LexicalJudge()
    if name == "claude":
        return ClaudeJudge()
    raise ValueError(f"unknown judge: {name}")
