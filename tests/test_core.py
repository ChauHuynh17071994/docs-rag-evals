from rag.bm25 import BM25, tokenize
from rag.chunking import chunk_markdown, load_corpus
from rag.index import Index
from rag.llm import REFUSAL, ExtractiveLLM
from rag.pipeline import RagPipeline

DOC = """# Policy

## Hotels
The maximum hotel rate is 160 euros per night.

## Meals
The daily allowance is 52 euros.
"""


def test_chunks_are_scoped_to_headings_with_stable_ids():
    chunks = chunk_markdown("policy.md", DOC)
    assert [c.id for c in chunks] == ["policy.md#0", "policy.md#1"]
    assert chunks[0].title == "Policy > Hotels"
    assert "160 euros" in chunks[0].text


def test_long_sections_are_split_on_sentences():
    text = "# T\n\n" + " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = chunk_markdown("t.md", text, max_chars=200)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What is the Hotel rate?") == ["hotel", "rate"]


def test_bm25_ranks_relevant_document_first():
    bm25 = BM25(["hotel rate per night", "daily meal allowance", "password length rules"])
    assert bm25.search("meal allowance")[0][0] == 1
    assert bm25.search("unrelated query terms") == []


def test_index_roundtrip(tmp_path):
    index = Index.from_folder("data/docs")
    path = tmp_path / "index.json"
    index.save(path)
    loaded = Index.load(path)
    assert [c.id for c in loaded.chunks] == [c.id for c in index.chunks]


def test_pipeline_answers_with_valid_citation():
    pipeline = RagPipeline(Index.from_folder("data/docs"), ExtractiveLLM())
    answer = pipeline.ask("What is the daily allowance abroad?")
    assert "52 euros" in answer.text
    assert answer.grounded
    assert answer.citations == ["travel-policy.md#2"]


def test_pipeline_refuses_when_evidence_is_weak():
    pipeline = RagPipeline(Index.from_folder("data/docs"), ExtractiveLLM())
    answer = pipeline.ask("What is the CEO's salary?")
    assert answer.refused
    assert answer.text == REFUSAL


class FakeLLM:
    name = "fake"

    def __init__(self, text):
        self.text = text

    def answer(self, question, context):
        return self.text


def test_citations_outside_retrieved_context_are_flagged():
    pipeline = RagPipeline(Index.from_folder("data/docs"), FakeLLM("Made up. [secret.md#9]"))
    answer = pipeline.ask("What is the daily allowance abroad?")
    assert answer.invalid_citations == ["secret.md#9"]
    assert not answer.grounded
