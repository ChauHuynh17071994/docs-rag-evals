"""HTTP API. Run with: ``uvicorn rag.api:app --reload``."""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from functools import lru_cache

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from . import __version__
from .index import Index
from .llm import get_llm
from .observability import configure_logging, request_context
from .pipeline import RagPipeline
from .rerank import LexicalReranker
from .rewrite import get_rewriter

log = configure_logging().getChild("api")
app = FastAPI(title="docs-rag-evals", version=__version__)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Assign a request id (or reuse X-Request-ID), log one JSON line per request."""
    with request_context(request.headers.get("x-request-id")) as rid:
        start = time.perf_counter()
        request.state.request_id = rid
        try:
            response = await call_next(request)
        except Exception:
            log.exception("request failed", extra={"method": request.method, "path": request.url.path})
            raise
        response.headers["X-Request-ID"] = rid
        log.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            },
        )
        return response


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
    request_id: str | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    usage: dict[str, float] = Field(default_factory=dict)


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
def ask(body: AskRequest, request: Request) -> AskResponse:
    result = get_pipeline().ask(body.question)
    data = asdict(result)
    data["answer"] = data.pop("text")
    meta = data.pop("meta")
    data.pop("question")
    response = AskResponse(
        grounded=result.grounded,
        request_id=getattr(request.state, "request_id", None),
        timings=meta.get("timings", {}),
        usage=meta.get("usage", {}),
        **data,
    )
    log.info(
        "answer",
        extra={
            "backend": result.backend,
            "refused": result.refused,
            "grounded": result.grounded,
            "citations": len(result.citations),
            "invalid_citations": len(result.invalid_citations),
            "top_score": result.top_score,
            "latency_ms": result.latency_ms,
            **response.timings,
            **response.usage,
        },
    )
    return response
