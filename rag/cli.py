"""Command line interface: ``rag ingest``, ``rag ask``, ``rag eval``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

from .index import RETRIEVERS, Index
from .llm import get_llm
from .pipeline import RagPipeline
from .rerank import LexicalReranker
from .rewrite import get_rewriter


def _pipeline(args: argparse.Namespace) -> RagPipeline:
    opts = {"retriever": args.retriever}
    index = Index.load(args.index, **opts) if args.index else Index.from_folder(args.docs, **opts)
    return RagPipeline(
        index,
        get_llm(args.backend),
        k=args.k,
        min_score=args.min_score,
        min_similarity=args.min_similarity,
        reranker=LexicalReranker() if args.rerank else None,
        rewriter=get_rewriter(args.rewrite),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag", description=__doc__)
    parser.add_argument("--docs", default="data/docs", help="folder of Markdown documents")
    parser.add_argument("--index", help="prebuilt index.json (skips chunking)")
    parser.add_argument("--backend", choices=["extractive", "claude"], help="LLM backend")
    parser.add_argument("-k", type=int, default=4, help="chunks to retrieve")
    parser.add_argument(
        "--retriever",
        choices=RETRIEVERS,
        default=os.environ.get("RAG_RETRIEVER", "bm25"),
        help="bm25 (lexical), dense (hashing embeddings) or hybrid (reciprocal rank fusion)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        default=os.environ.get("RAG_RERANK", "") == "1",
        help="rerank a 20-chunk candidate pool before keeping the top k",
    )
    parser.add_argument(
        "--rewrite",
        choices=["none", "glossary", "claude"],
        default=os.environ.get("RAG_REWRITE", "none"),
        help="rewrite short queries (<= 3 terms) before retrieval",
    )
    parser.add_argument("--min-score", type=float, default=1.5, help="BM25 refusal threshold")
    parser.add_argument(
        "--min-similarity", type=float, default=0.22, help="dense cosine refusal threshold"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="chunk documents and write index.json")
    ingest.add_argument("--out", default="index.json")

    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", help="print the full answer object")

    ev = sub.add_parser("eval", help="run the evaluation suite")
    ev.add_argument("--dataset", default="evals/dataset.jsonl")
    ev.add_argument("--report", default="eval_report.json")
    ev.add_argument("--gate", action="store_true", help="exit 1 if metrics fall below thresholds")
    ev.add_argument("--markdown", help="also write a per-question Markdown report to this path")

    args = parser.parse_args(argv)

    if args.command == "ingest":
        index = Index.from_folder(args.docs)
        index.save(args.out)
        print(f"Indexed {len(index.chunks)} chunks -> {args.out}")
        return 0

    if args.command == "ask":
        answer = _pipeline(args).ask(args.question)
        print(json.dumps(asdict(answer), indent=2) if args.json else answer.text)
        return 0

    from evals.run_eval import run  # local import: evals is a dev-time package

    return run(_pipeline(args), args.dataset, args.report, gate=args.gate, markdown=args.markdown)


if __name__ == "__main__":
    sys.exit(main())
