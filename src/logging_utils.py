"""Structured JSON request logging for the API layer.

Writes one JSON line per request to logs/requests.jsonl, appended and
flushed immediately -- the same incremental-JSONL convention used
throughout this project's eval/ harnesses. Exists to answer, after the
fact, "when an answer fails, was it retrieval or generation": every row
carries per-stage latency and status, plus which clauses were retrieved,
without ever storing the chunk text itself.

Never logs: API keys, the X-API-Key header, or full retrieved chunk text
(only clause labels) -- enforced by this module only accepting the
specific fields below, not an arbitrary dict a caller could accidentally
widen to include a header or chunk body.
"""

import json
import threading
import time
from pathlib import Path

LOG_PATH = Path("logs/requests.jsonl")
_lock = threading.Lock()


def log_request(
    *,
    request_id: str,
    arm: str,
    question: str,
    retrieved_clauses: list[str],
    abstained: bool | None,
    latency_ms: dict[str, float],
    tokens: dict[str, int] | None,
    status: str,
    error: str | None = None,
) -> None:
    """Append one structured JSON line describing a single /query request."""
    row = {
        "request_id": request_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "arm": arm,
        "question": question,
        "retrieved_clauses": retrieved_clauses,
        "abstained": abstained,
        "latency_ms": latency_ms,
        "tokens": tokens,
        "status": status,
        "error": error,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with LOG_PATH.open("a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
