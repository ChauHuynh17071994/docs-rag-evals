from fastapi.testclient import TestClient

from rag.api import app

client = TestClient(app)


def test_health():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] > 0


def test_ask_returns_grounded_answer():
    body = client.post("/ask", json={"question": "How fast must a stolen laptop be reported?"}).json()
    assert "2 hours" in body["answer"]
    assert body["grounded"] is True
    assert body["citations"] == ["it-security.md#1"]


def test_ask_validates_input():
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_health_reports_retriever():
    assert client.get("/health").json()["retriever"] == "bm25"


def test_pipeline_loads_prebuilt_index_and_rewriter(tmp_path, monkeypatch):
    from rag import api
    from rag.index import Index

    path = tmp_path / "index.json"
    Index.from_folder("data/docs").save(path)
    monkeypatch.setenv("RAG_INDEX", str(path))
    monkeypatch.setenv("RAG_REWRITE", "glossary")
    api.get_pipeline.cache_clear()
    try:
        pipeline = api.get_pipeline()
        assert len(pipeline.index.chunks) == len(Index.from_folder("data/docs").chunks)
        assert pipeline.rewriter is not None and pipeline.rewriter.name == "glossary"
    finally:
        api.get_pipeline.cache_clear()


def test_ask_returns_request_id_timings_and_usage():
    response = client.post(
        "/ask",
        json={"question": "How fast must a stolen laptop be reported?"},
        headers={"X-Request-ID": "test-req-1"},
    )
    body = response.json()
    assert response.headers["X-Request-ID"] == "test-req-1"
    assert body["request_id"] == "test-req-1"
    assert {"rewrite_ms", "retrieve_ms", "generate_ms"} <= set(body["timings"])
    assert body["usage"]["cost_usd"] == 0.0


def test_requests_are_logged_as_json():
    import io
    import json
    import logging

    from rag.observability import JsonFormatter

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("rag")
    logger.addHandler(handler)
    try:
        client.get("/health", headers={"X-Request-ID": "log-check"})
    finally:
        logger.removeHandler(handler)
    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    request_log = next(r for r in lines if r["msg"] == "request")
    assert request_log["request_id"] == "log-check"
    assert request_log["path"] == "/health"
    assert request_log["status"] == 200
    assert request_log["duration_ms"] >= 0
