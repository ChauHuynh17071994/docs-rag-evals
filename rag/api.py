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
    return RagPipeline(Index.from_folder(docs), get_llm())


@app.get("/health")
def health() -> dict:
    pipeline = get_pipeline()
    return {"status": "ok", "chunks": len(pipeline.index.chunks), "backend": pipeline.llm.name}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = get_pipeline().ask(request.question)
    data = asdict(result)
    data["answer"] = data.pop("text")
    return AskResponse(grounded=result.grounded, **data)
