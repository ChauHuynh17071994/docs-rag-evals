"""Structured (JSON lines) logging with a per-request id.

``configure_logging()`` installs a JSON formatter on the ``rag`` logger. Every record carries
the current request id (set with ``request_context``) plus any ``extra={...}`` fields, so logs
can be filtered and aggregated without parsing free text.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import uuid
from collections.abc import Iterator
from contextvars import ContextVar
from datetime import datetime, timezone

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

# Attributes every LogRecord has; anything else was passed via ``extra=`` and is logged as a field.
_STANDARD = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName"}


def get_request_id() -> str | None:
    return _request_id.get()


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


@contextlib.contextmanager
def request_context(request_id: str | None = None) -> Iterator[str]:
    """Bind a request id to every log record emitted inside the block."""
    rid = request_id or new_request_id()
    token = _request_id.set(rid)
    try:
        yield rid
    finally:
        _request_id.reset(token)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = get_request_id()
        if rid:
            payload["request_id"] = rid
        payload.update({k: v for k, v in vars(record).items() if k not in _STANDARD})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str | None = None) -> logging.Logger:
    """Attach a single JSON stderr handler to the ``rag`` logger (idempotent)."""
    logger = logging.getLogger("rag")
    logger.setLevel((level or os.environ.get("RAG_LOG_LEVEL", "INFO")).upper())
    if not any(getattr(h, "_rag_json", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handler._rag_json = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.propagate = False
    return logger
