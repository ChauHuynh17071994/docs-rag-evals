import json

from rag.cli import main


def test_ask_with_hybrid_retriever_prints_answer_object(capsys):
    code = main(["--retriever", "hybrid", "ask", "--json", "What is the minimum password length?"])
    body = json.loads(capsys.readouterr().out)
    assert code == 0
    assert body["meta"]["retriever"] == "hybrid"
    assert body["citations"] == ["it-security.md#0"]


def test_retriever_defaults_to_env(monkeypatch, capsys):
    monkeypatch.setenv("RAG_RETRIEVER", "dense")
    main(["ask", "--json", "How quickly must a stolen laptop be reported?"])
    assert json.loads(capsys.readouterr().out)["meta"]["retriever"] == "dense"
