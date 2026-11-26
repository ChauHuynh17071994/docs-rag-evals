from evals.report import to_markdown
from evals.run_eval import evaluate, load_dataset
from rag.cli import main
from rag.index import Index
from rag.llm import ExtractiveLLM
from rag.pipeline import RagPipeline


def test_report_has_summary_and_one_row_per_question():
    data = load_dataset("evals/dataset.jsonl")
    result = evaluate(RagPipeline(Index.from_folder("data/docs"), ExtractiveLLM()), data)
    md = to_markdown(result)
    assert "| recall@k |" in md
    assert md.count("\n| q") + md.count("\n| s") + md.count("\n| u") == len(data)
    assert "retriever=bm25" in md
    assert "**Needs attention:**" in md


def test_cell_escapes_pipes_in_answers():
    from evals.report import _cell

    assert _cell("a | b") == "a \\| b"
    assert _cell("x" * 200).endswith("…")


def test_cli_writes_markdown_report(tmp_path):
    md = tmp_path / "report.md"
    code = main(["eval", "--report", str(tmp_path / "r.json"), "--markdown", str(md)])
    assert code == 0
    assert md.read_text().startswith("## Eval report")
