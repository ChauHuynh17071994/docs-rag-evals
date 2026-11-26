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
