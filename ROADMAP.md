# Roadmap

Progress tracker for this project. Each item is one small, reviewable commit.

## v0.1 — core pipeline
- [x] Project scaffold, license, sample corpus
- [x] Markdown-aware chunking with stable chunk ids
- [x] BM25 retriever (no external dependencies)
- [x] Index build + JSON persistence
- [x] LLM interface with offline extractive backend
- [x] Claude backend with citation-only prompt
- [x] Answer pipeline: retrieve → threshold refusal → generate → validate citations
- [x] CLI: ingest / ask / eval
- [x] FastAPI service (/health, /ask)
- [x] Unit tests for chunking, BM25, pipeline, API
- [x] Eval dataset incl. unanswerable questions
- [x] Eval runner: recall@k, MRR, citation validity, refusal accuracy, keyword match
- [x] CI: lint + tests + eval quality gate
- [x] README with architecture, metrics, how to run

## v0.2 — retrieval quality
- [x] Hybrid retrieval (BM25 + embeddings) behind a flag
  - [x] Offline hashing embedder + dense vector search
  - [x] Reciprocal rank fusion and `--retriever bm25|dense|hybrid` flag
  - [x] Config comparison script and measured results in README
- [x] Reranking step and before/after eval comparison
  - [x] Lexical coverage/proximity reranker behind `--rerank`
  - [x] Before/after eval table in README
- [x] Query rewriting for short queries
  - [x] Short keyword-style questions in the eval set
  - [x] Rule-based rewriter (offline) + optional Claude rewriter
- [x] Per-question eval report as Markdown artifact in CI
  - [x] Markdown report writer
  - [x] Upload report and post it to the CI job summary

## v0.3 — production touches
- [x] Dockerfile + docker-compose
  - [x] Dockerfile with a prebuilt index and non-root user
  - [x] docker-compose service with health check and .env support
- [x] Structured logging and latency/cost tracking per request
  - [x] JSON logger with request ids
  - [x] Per-stage timings and token/cost accounting in the pipeline
  - [x] Request logging middleware and usage fields in the API
- [ ] LLM-as-judge faithfulness metric (optional, Claude backend)
  - [x] Judge interface with an offline lexical judge and a Claude judge
  - [ ] `--judge` flag and faithfulness metric in the eval report
- [ ] Simple web UI
  - [ ] Single-page UI served by the API
  - [ ] README section for UI, Docker and observability
