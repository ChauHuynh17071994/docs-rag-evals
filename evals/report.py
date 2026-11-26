"""Render an eval result as a Markdown report: summary metrics plus one row per question."""

from __future__ import annotations

from .run_eval import THRESHOLDS


def _mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def _cell(text: str, limit: int = 90) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def to_markdown(result: dict, title: str = "Eval report") -> str:
    metrics = result["metrics"]
    first = result["rows"][0]["answer"]["meta"] if result["rows"] else {}
    config = ", ".join(
        f"{key}={first.get(key) or 'none'}" for key in ("retriever", "reranker") if key in first
    )
    lines = [
        f"## {title}",
        "",
        f"backend=`{result['backend']}` k={result['k']} n={metrics['n']}"
        + (f" ({config})" if config else ""),
        "",
        "| metric | value | gate |",
        "|---|---|---|",
    ]
    for name, value in metrics.items():
        if name.startswith("n"):
            continue
        floor = THRESHOLDS.get(name)
        gate = "" if floor is None else f"{_mark(value >= floor)} ≥ {floor}"
        lines.append(f"| {name} | {value} | {gate} |")

    lines += [
        "",
        "### Per question",
        "",
        "| id | question | retrieved@1 | hit | answer ok | refusal ok | answer |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in result["rows"]:
        answer = row["answer"]
        top = answer["retrieved"][0] if answer["retrieved"] else "—"
        match = _mark(row["match"]) if row["answerable"] else "n/a"
        hit = _mark(row["hit"]) if row["answerable"] else "n/a"
        lines.append(
            f"| {row['id']} | {_cell(answer['question'], 60)} | `{top}` | {hit} | {match} "
            f"| {_mark(row['refusal_ok'])} | {_cell(answer['text'])} |"
        )

    failures = [r["id"] for r in result["rows"] if not r["refusal_ok"] or (r["answerable"] and not r["hit"])]
    lines += ["", f"**Needs attention:** {', '.join(failures) if failures else 'none'}", ""]
    return "\n".join(lines)
