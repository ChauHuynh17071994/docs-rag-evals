from evals.compare import compare, to_markdown


def test_compare_covers_every_retriever_and_renders_markdown():
    results = compare("data/docs", "evals/dataset.jsonl")
    names = [name for name, _ in results]
    assert names == ["bm25", "bm25 + rerank", "dense", "dense + rerank", "hybrid", "hybrid + rerank"]
    table = to_markdown(results)
    assert table.splitlines()[0].startswith("| config | recall@k")
    assert len(table.splitlines()) == 2 + len(results)
