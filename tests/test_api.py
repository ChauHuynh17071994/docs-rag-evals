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
