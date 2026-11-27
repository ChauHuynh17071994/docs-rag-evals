"""Evaluation suite.

Metrics
- recall@k        gold chunk appears in the retrieved set (answerable questions)
- mrr             mean reciprocal rank of the first gold chunk
- answer_match    answer contains every expected key phrase (case-insensitive)
- citation_valid  answered questions whose citations all point at retrieved chunks
- cites_gold      answered questions that cite at least one gold chunk
- refusal_acc     correct behaviour on unanswerable questions (refuses) and
                  answerable ones (does not refuse)
- faithfulness    (with --judge) mean share of answer sentences supported by the
                  chunks the answer cites; refusals are not judged
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from rag.pipeline import RagPipeline

# Quality gate for CI with the offline extractive backend. Raise these as the system improves.
THRESHOLDS = {
    "recall@k": 0.85,
    "mrr": 0.70,
    "citation_valid": 1.0,
    "refusal_acc": 0.80,
}


def load_dataset(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def evaluate(pipeline: RagPipeline, dataset: list[dict], judge=None) -> dict:
    rows = []
    for item in dataset:
        answer = pipeline.ask(item["question"])
        verdict = None
        if judge is not None and not answer.refused:
            cited = [c for c in (pipeline.index.get(cid) for cid in answer.citations) if c]
            verdict = asdict(judge.judge(item["question"], answer.text, cited))
        gold = set(item["gold"])
        answerable = bool(gold)
        rank = next((i + 1 for i, cid in enumerate(answer.retrieved) if cid in gold), None)
        text = answer.text.lower()
        rows.append(
            {
                "id": item["id"],
                "answerable": answerable,
                "hit": rank is not None,
                "rr": 1 / rank if rank else 0.0,
                "match": answerable and all(p.lower() in text for p in item["expect"]),
                "answered": not answer.refused,
                "citation_valid": not answer.invalid_citations and bool(answer.citations),
                "cites_gold": bool(gold & set(answer.citations)),
                "refusal_ok": answer.refused != answerable,
                "judge": verdict,
                "answer": asdict(answer),
            }
        )

    answerable = [r for r in rows if r["answerable"]]
    answered = [r for r in rows if r["answered"]]

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 3) if values else 0.0

    metrics = {
        "n": len(rows),
        "n_answerable": len(answerable),
        "recall@k": mean([r["hit"] for r in answerable]),
        "mrr": mean([r["rr"] for r in answerable]),
        "answer_match": mean([r["match"] for r in answerable]),
        "citation_valid": mean([r["citation_valid"] for r in answered]),
        "cites_gold": mean([r["cites_gold"] for r in answered if r["answerable"]]),
        "refusal_acc": mean([r["refusal_ok"] for r in rows]),
        "avg_latency_ms": mean([r["answer"]["latency_ms"] for r in rows]),
    }
    judged = [r["judge"] for r in rows if r["judge"] is not None]
    if judge is not None:
        metrics["faithfulness"] = mean([v["score"] for v in judged])
        metrics["fully_faithful"] = mean([v["faithful"] for v in judged])
    result = {"backend": pipeline.llm.name, "k": pipeline.k, "metrics": metrics, "rows": rows}
    if judge is not None:
        result["judge"] = judge.name
    return result


def run(
    pipeline: RagPipeline,
    dataset: str,
    report: str,
    gate: bool = False,
    markdown: str | None = None,
    judge: str | None = None,
) -> int:
    from .judge import get_judge

    result = evaluate(pipeline, load_dataset(dataset), judge=get_judge(judge))
    Path(report).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if markdown:
        from .report import to_markdown  # local import: report imports THRESHOLDS from here

        Path(markdown).write_text(to_markdown(result), encoding="utf-8")

    metrics = result["metrics"]
    judged_by = f" judge={result['judge']}" if "judge" in result else ""
    print(f"backend={result['backend']} k={result['k']} n={metrics['n']}{judged_by}")
    for name, value in metrics.items():
        if name.startswith("n"):
            continue
        floor = THRESHOLDS.get(name)
        status = "" if floor is None else ("  PASS" if value >= floor else f"  FAIL (< {floor})")
        print(f"  {name:<15}{value:>8}{status}")

    failed = [r["id"] for r in result["rows"] if not (r["hit"] or not r["answerable"])]
    if failed:
        print(f"  retrieval misses: {', '.join(failed)}")

    if gate and any(metrics[k] < v for k, v in THRESHOLDS.items()):
        return 1
    return 0


if __name__ == "__main__":
    from rag.cli import main

    sys.exit(main(["eval", *sys.argv[1:]]))
