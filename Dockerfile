# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so code changes do not invalidate this layer.
COPY pyproject.toml README.md LICENSE ./
COPY rag/__init__.py rag/__init__.py
RUN pip install .  && pip uninstall -y docs-rag-evals

COPY rag/ rag/
COPY data/ data/
RUN pip install --no-deps . \
    && python -m rag.cli --docs data/docs ingest --out /app/index.json \
    && useradd --create-home --uid 1000 rag

USER rag

ENV RAG_INDEX=/app/index.json \
    RAG_BACKEND=extractive \
    RAG_RETRIEVER=bm25

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health').status != 200)"

CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
