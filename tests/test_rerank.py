from rag.chunking import Chunk
from rag.index import Hit, Index
from rag.llm import ExtractiveLLM
from rag.pipeline import RagPipeline
from rag.rerank import LexicalReranker, smallest_window


def hit(cid, title, text):
    return Hit(Chunk(cid, "d.md", title, text), score=1.0)


def test_smallest_window():
    assert smallest_window({"a": [0, 10], "b": [3, 11]}) == 2
    assert smallest_window({"a": [5]}) == 1


def test_reranker_prefers_full_coverage_over_first_stage_rank():
    hits = [
        hit("d.md#0", "Travel", "Hotel bookings go through the travel portal."),
        hit("d.md#1", "Travel > Hotels", "The maximum hotel rate in a capital city is 160 euros."),
    ]
    out = LexicalReranker().rerank("maximum hotel rate capital", hits)
    assert [h.chunk.id for h in out] == ["d.md#1", "d.md#0"]


def test_reranker_keeps_order_when_question_has_no_terms():
    hits = [hit("d.md#0", "A", "x"), hit("d.md#1", "B", "y")]
    assert LexicalReranker().rerank("what is the", hits) == hits


def test_pipeline_reranks_a_candidate_pool_and_records_it():
    pipeline = RagPipeline(Index.from_folder("data/docs"), ExtractiveLLM(), k=1, reranker=LexicalReranker())
    answer = pipeline.ask("What is the maximum hotel rate in a capital city?")
    assert answer.retrieved == ["travel-policy.md#4"]
    assert answer.meta["reranker"] == "lexical"
