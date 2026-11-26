"""Markdown-aware chunking.

Documents are split on headings so that every chunk keeps the section it belongs to.
Each chunk gets a stable id ``<file>#<n>`` that the model must use in citations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    title: str
    text: str


def chunk_markdown(source: str, markdown: str, max_chars: int = 800) -> list[Chunk]:
    """Split a Markdown document into heading-scoped chunks."""
    sections: list[tuple[str, list[str]]] = []
    doc_title = source
    current_title = ""
    current_lines: list[str] = []

    for line in markdown.splitlines():
        match = HEADING.match(line)
        if match:
            if current_lines:
                sections.append((current_title, current_lines))
            level, heading = len(match.group(1)), match.group(2).strip()
            if level == 1:
                doc_title = heading
                current_title = heading
            else:
                current_title = f"{doc_title} > {heading}"
            current_lines = []
        elif line.strip():
            current_lines.append(line.strip())
    if current_lines:
        sections.append((current_title, current_lines))

    chunks: list[Chunk] = []
    for title, lines in sections:
        for piece in _split_long(" ".join(lines), max_chars):
            chunks.append(Chunk(id=f"{source}#{len(chunks)}", source=source, title=title, text=piece))
    return chunks


def _split_long(text: str, max_chars: int) -> list[str]:
    """Split on sentence boundaries so no piece exceeds max_chars (unless one sentence does)."""
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces, buf = [], ""
    for sentence in sentences:
        if buf and len(buf) + len(sentence) + 1 > max_chars:
            pieces.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if buf:
        pieces.append(buf)
    return pieces


def load_corpus(folder: str | Path, max_chars: int = 800) -> list[Chunk]:
    """Chunk every Markdown file in a folder, in a deterministic order."""
    chunks: list[Chunk] = []
    for path in sorted(Path(folder).glob("*.md")):
        chunks.extend(chunk_markdown(path.name, path.read_text(encoding="utf-8"), max_chars))
    return chunks
