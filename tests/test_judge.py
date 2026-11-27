from evals.judge import LexicalJudge, claims, get_judge, parse_verdict
from rag.chunking import Chunk

LAPTOP = Chunk(
    id="it-security.md#1",
    source="it-security.md",
    title="Lost devices",
    text="A lost or stolen laptop or phone must be reported to the IT service desk within 2 hours.",
)


def test_claims_strip_citations_and_split_sentences():
    answer = "Report it within 2 hours. [it-security.md#1] Call the desk. [it-security.md#1]"
    assert claims(answer) == ["Report it within 2 hours.", "Call the desk."]


def test_lexical_judge_accepts_extracted_sentence():
    verdict = LexicalJudge().judge("q", f"{LAPTOP.text} [{LAPTOP.id}]", [LAPTOP])
    assert verdict.faithful and verdict.score == 1.0


def test_lexical_judge_flags_unsupported_sentence():
    answer = f"{LAPTOP.text} [{LAPTOP.id}] Managers receive a bonus for quick reporting. [{LAPTOP.id}]"
    verdict = LexicalJudge().judge("q", answer, [LAPTOP])
    assert not verdict.faithful
    assert verdict.score == 0.5
    assert "bonus" in verdict.reason


def test_lexical_judge_needs_cited_chunks():
    assert LexicalJudge().judge("q", "Something true.", []).score == 0.0


def test_parse_verdict_handles_fenced_json_and_clamps():
    v = parse_verdict('```json\n{"supported": 2, "total": 3, "reason": "one extra fact"}\n```')
    assert (v.score, v.faithful, v.reason) == (0.667, False, "one extra fact")
    assert parse_verdict('{"supported": 9, "total": 2}').faithful
    assert parse_verdict("no json here").score == 0.0


def test_get_judge():
    assert get_judge("none") is None
    assert get_judge("lexical").name == "lexical"


def test_claude_judge_without_network():
    from types import SimpleNamespace

    from evals.judge import ClaudeJudge

    reply = SimpleNamespace(content=[SimpleNamespace(type="text", text='{"supported": 1, "total": 1}')])
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return reply

    judge = ClaudeJudge.__new__(ClaudeJudge)  # skip __init__: no client or API key needed
    judge.client = SimpleNamespace(messages=SimpleNamespace(create=create))
    judge.model = "claude-sonnet-4-5"
    assert judge.judge("q", "a", [LAPTOP]).faithful
    assert LAPTOP.id in seen["messages"][0]["content"]


class EmbellishingLLM:
    """Copies the first chunk's first sentence, then adds an unsupported claim."""

    name = "embellishing"

    def answer(self, question, context):
        first = context[0]
        return f"{first.text.split('. ')[0]}. [{first.id}] Exceptions need CEO sign-off. [{first.id}]"


def test_eval_with_judge_reports_faithfulness_and_lists_unfaithful_answers():
    from evals.judge import LexicalJudge
    from evals.report import to_markdown
    from evals.run_eval import evaluate, load_dataset
    from rag.index import Index
    from rag.pipeline import RagPipeline

    data = load_dataset("evals/dataset.jsonl")[:5]
    result = evaluate(RagPipeline(Index.from_folder("data/docs"), EmbellishingLLM()), data, LexicalJudge())
    assert result["judge"] == "lexical"
    assert result["metrics"]["fully_faithful"] == 0.0
    assert 0.0 < result["metrics"]["faithfulness"] < 1.0
    md = to_markdown(result)
    assert "### Unfaithful answers" in md and "CEO sign-off" in md


def test_eval_without_judge_has_no_faithfulness_metric():
    from evals.run_eval import evaluate, load_dataset
    from rag.index import Index
    from rag.llm import ExtractiveLLM
    from rag.pipeline import RagPipeline

    result = evaluate(
        RagPipeline(Index.from_folder("data/docs"), ExtractiveLLM()), load_dataset("evals/dataset.jsonl")[:3]
    )
    assert "faithfulness" not in result["metrics"]
