import json

import pytest

from rag.index import Index
from rag.llm import ExtractiveLLM
from rag.pipeline import RagPipeline
from rag.rewrite import GlossaryRewriter, get_rewriter


def test_short_queries_are_expanded_with_glossary_terms():
    out = GlossaryRewriter().rewrite("wfh days")
    assert out.startswith("wfh days ")
    assert "work remotely" in out


def test_multiword_entries_win_and_long_questions_are_untouched():
    rewriter = GlossaryRewriter()
    assert "certificate" in rewriter.rewrite("sick note")
    long_q = "How many vacation days do full-time employees get per year?"
    assert rewriter.rewrite(long_q) == long_q


def test_queries_without_glossary_terms_are_unchanged():
    assert GlossaryRewriter().rewrite("parking rules") == "parking rules"


def test_glossary_can_be_loaded_from_file(tmp_path):
    path = tmp_path / "glossary.json"
    path.write_text(json.dumps({"OOO": "out of office leave"}))
    assert GlossaryRewriter.from_file(path).rewrite("ooo policy").endswith("out of office leave")


def test_pipeline_retrieves_with_rewritten_query_and_records_it():
    pipeline = RagPipeline(Index.from_folder("data/docs"), ExtractiveLLM(), rewriter=GlossaryRewriter())
    answer = pipeline.ask("per diem abroad")
    assert answer.retrieved[0] == "travel-policy.md#2"
    assert "daily allowance" in answer.meta["rewritten_query"]
    assert answer.question == "per diem abroad"


def test_get_rewriter():
    assert get_rewriter(None) is None and get_rewriter("none") is None
    assert get_rewriter("glossary").name == "glossary"
    with pytest.raises(ValueError):
        get_rewriter("magic")
