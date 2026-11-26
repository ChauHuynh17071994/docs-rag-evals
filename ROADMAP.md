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
- [ ] FastAPI service (/health, /ask)
- [ ] Unit tests for chunking, BM25, pipeline, API
- [ ] Eval dataset incl. unanswerable questions
- [ ] Eval runner: recall@k, MRR, citation validity, refusal accuracy, keyword match
- [ ] CI: lint + tests + eval quality gate
- [ ] README with architecture, metrics, how to run

## v0.2 — retrieval quality
- [ ] Hybrid retrieval (BM25 + embeddings) behind a flag
- [ ] Reranking step and before/after eval comparison
- [ ] Query rewriting for short queries
- [ ] Per-question eval report as Markdown artifact in CI

## v0.3 — production touches
- [ ] Dockerfile + docker-compose
- [ ] Structured logging and latency/cost tracking per request
- [ ] LLM-as-judge faithfulness metric (optional, Claude backend)
- [ ] Simple web UI
