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
- [ ] Hybrid retrieval (BM25 + embeddings) behind a flag
  - [x] Offline hashing embedder + dense vector search
  - [x] Reciprocal rank fusion and `--retriever bm25|dense|hybrid` flag
  - [ ] Config comparison script and measured results in README
- [ ] Reranking step and before/after eval comparison
  - [ ] Lexical coverage/proximity reranker behind `--rerank`
  - [ ] Before/after eval table in README
- [ ] Query rewriting for short queries
  - [ ] Short keyword-style questions in the eval set
  - [ ] Rule-based rewriter (offline) + optional Claude rewriter
- [ ] Per-question eval report as Markdown artifact in CI
  - [ ] Markdown report writer
  - [ ] Upload report and post it to the CI job summary

## v0.3 — production touches
- [ ] Dockerfile + docker-compose
- [ ] Structured logging and latency/cost tracking per request
- [ ] LLM-as-judge faithfulness metric (optional, Claude backend)
- [ ] Simple web UI
