"""Compare retrieval configurations on the eval set and print a Markdown table.

Usage::

    python -m evals.compare                    # all retrievers, offline backend
    python -m evals.compare --out results.md   # also write the table to a file
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.index import RETRIEVERS, Index
from rag.llm import get_llm
from rag.pipeline import RagPipeline
from rag.rerank import LexicalReranker
from rag.rewrite import GlossaryRewriter

from .run_eval import evaluate, load_dataset

COLUMNS = ["recall@k", "mrr", "answer_match", "cites_gold", "refusal_acc", "avg_latency_ms"]


def configs() -> list[dict]:
    return [
        {"retriever": r, "rerank": rerank, "rewrite": rewrite}
        for r in RETRIEVERS
        for rewrite in (False, True)
        for rerank in (False, True)
    ]


def label(config: dict) -> str:
    return " + ".join(str(v) if not isinstance(v, bool) else k for k, v in config.items() if v)


def compare(docs: str, dataset: str, backend: str | None = None, k: int = 4) -> list[tuple[str, dict]]:
    data = load_dataset(dataset)
    llm = get_llm(backend)
    results = []
    for config in configs():
        index_opts = {key: config[key] for key in ("retriever",) if key in config}
        reranker = LexicalReranker() if config.get("rerank") else None
        rewriter = GlossaryRewriter() if config.get("rewrite") else None
        pipeline = RagPipeline(
            Index.from_folder(docs, **index_opts), llm, k=k, reranker=reranker, rewriter=rewriter
        )
        results.append((label(config), evaluate(pipeline, data)["metrics"]))
    return results


def to_markdown(results: list[tuple[str, dict]]) -> str:
    lines = ["| config | " + " | ".join(COLUMNS) + " |", "|---" * (len(COLUMNS) + 1) + "|"]
    for name, metrics in results:
        cells = [f"{metrics[c]:.3f}" if c != "avg_latency_ms" else f"{metrics[c]:.2f}" for c in COLUMNS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--docs", default="data/docs")
    parser.add_argument("--dataset", default="evals/dataset.jsonl")
    parser.add_argument("--backend", choices=["extractive", "claude"])
    parser.add_argument("-k", type=int, default=4)
    parser.add_argument("--out", help="write the Markdown table to this file")
    args = parser.parse_args(argv)

    table = to_markdown(compare(args.docs, args.dataset, args.backend, args.k))
    print(table, end="")
    if args.out:
        Path(args.out).write_text(table, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
