# Changelog

## 0.3.0 — production touches

- Dockerfile with a prebuilt index, non-root user and health check; `docker-compose.yml`; CI builds the image and smoke-tests `/ask` in the container.
- JSON-lines logging with request ids (`X-Request-ID`), one log line per request and per answer.
- Per-stage timings (`rewrite_ms`, `retrieve_ms`, `generate_ms`) and token/cost accounting for the Claude backend, returned by `/ask`.
- Faithfulness judges (`--judge lexical|claude`) with `faithfulness` and `fully_faithful` metrics in the eval report; CI runs the lexical judge.
- Single-page web UI at `/` and a `/chunks/{id}` endpoint to inspect citations.
- API reads a prebuilt index from `RAG_INDEX` and honors `RAG_REWRITE`.

## 0.2.0 — retrieval quality

- Offline hashing embedder, dense and hybrid (reciprocal rank fusion) retrieval behind `--retriever`.
- Lexical reranker behind `--rerank`, with before/after results.
- Glossary and Claude query rewriting for short queries (`--rewrite`), plus 8 short keyword questions in the eval set.
- Configuration comparison script and per-question Markdown report published to the CI job summary.

## 0.1.0 — core pipeline

- Markdown chunking, BM25 retrieval, threshold refusals, citation validation.
- Offline extractive and Claude backends, CLI, FastAPI service.
- Eval suite (recall@k, MRR, answer match, citation validity, refusal accuracy) with a CI quality gate.
