from evals.compare import compare, to_markdown


def test_compare_covers_every_retriever_and_renders_markdown():
    results = compare("data/docs", "evals/dataset.jsonl")
    names = [name for name, _ in results]
    assert len(names) == 12
    assert names[:4] == ["bm25", "bm25 + rerank", "bm25 + rewrite", "bm25 + rerank + rewrite"]
    table = to_markdown(results)
    assert table.splitlines()[0].startswith("| config | recall@k")
    assert len(table.splitlines()) == 2 + len(results)
