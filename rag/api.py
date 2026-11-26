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
    docs = os.environ.get("RAG_DOCS", "data/docs")
    retriever = os.environ.get("RAG_RETRIEVER", "bm25")
    reranker = LexicalReranker() if os.environ.get("RAG_RERANK") == "1" else None
    return RagPipeline(Index.from_folder(docs, retriever=retriever), get_llm(), reranker=reranker)


@app.get("/health")
def health() -> dict:
    pipeline = get_pipeline()
    return {
        "status": "ok",
        "chunks": len(pipeline.index.chunks),
        "backend": pipeline.llm.name,
        "retriever": pipeline.index.retriever,
        "reranker": pipeline.reranker.name if pipeline.reranker else None,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = get_pipeline().ask(request.question)
    data = asdict(result)
    data["answer"] = data.pop("text")
    return AskResponse(grounded=result.grounded, **data)
