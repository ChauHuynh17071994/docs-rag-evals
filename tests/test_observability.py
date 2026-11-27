import json
import logging

from rag.observability import JsonFormatter, configure_logging, get_request_id, request_context


def _record(msg="hello", **extra):
    record = logging.makeLogRecord({"name": "rag.test", "levelname": "INFO", "msg": msg})
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_includes_extra_fields():
    line = JsonFormatter().format(_record(latency_ms=1.5, backend="extractive"))
    payload = json.loads(line)
    assert payload["msg"] == "hello"
    assert payload["level"] == "info"
    assert payload["latency_ms"] == 1.5
    assert payload["backend"] == "extractive"
    assert "request_id" not in payload


def test_request_context_binds_and_resets_id():
    assert get_request_id() is None
    with request_context("abc123") as rid:
        assert rid == "abc123"
        assert json.loads(JsonFormatter().format(_record()))["request_id"] == "abc123"
    assert get_request_id() is None


def test_request_context_generates_id():
    with request_context() as rid:
        assert len(rid) == 16


def test_configure_logging_is_idempotent():
    logger = configure_logging("debug")
    configure_logging("debug")
    assert sum(isinstance(h.formatter, JsonFormatter) for h in logger.handlers) == 1
    assert logger.level == logging.DEBUG
