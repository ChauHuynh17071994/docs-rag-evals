"""HTTP API. Run with: ``uvicorn rag.api:app --reload``."""

from __future__ import annotations

import os
from dataclasses import asdict
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import __version__
from .index import Index
from .llm import get_llm
from .pipeline import RagPipeline
from .rerank import LexicalReranker
from .rewrite import get_rewriter

app = FastAPI(title="docs-rag-evals", version=__version__)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class AskResponse(BaseModel):
    answer: str
    refused: bool
    grounded: bool
    citations: list[str]
    invalid_citations: list[str]
    retrieved: list[str]
    top_score: float
    latency_ms: float
    backend: str


@lru_cache(maxsize=1)
def get_pipeline() -> RagPipeline:
    """Build the pipeline from environment variables (same names as the CLI flags).

    ``RAG_INDEX`` points at a prebuilt ``index.json`` (as written by ``rag ingest``); when it
    is unset the documents in ``RAG_DOCS`` are chunked at startup.
    """
    retriever = os.environ.get("RAG_RETRIEVER", "bm25")
    index_path = os.environ.get("RAG_INDEX")
    if index_path:
        index = Index.load(index_path, retriever=retriever)
    else:
        index = Index.from_folder(os.environ.get("RAG_DOCS", "data/docs"), retriever=retriever)
    reranker = LexicalReranker() if os.environ.get("RAG_RERANK") == "1" else None
    rewriter = get_rewriter(os.environ.get("RAG_REWRITE", "none"))
    return RagPipeline(index, get_llm(), reranker=reranker, rewriter=rewriter)


@app.get("/health")
def health() -> dict:
    pipeline = get_pipeline()
    return {
        "status": "ok",
        "chunks": len(pipeline.index.chunks),
        "backend": pipeline.llm.name,
        "retriever": pipeline.index.retriever,
        "reranker": pipeline.reranker.name if pipeline.reranker else None,
        "rewriter": pipeline.rewriter.name if pipeline.rewriter else None,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = get_pipeline().ask(request.question)
    data = asdict(result)
    data["answer"] = data.pop("text")
    return AskResponse(grounded=result.grounded, **data)
