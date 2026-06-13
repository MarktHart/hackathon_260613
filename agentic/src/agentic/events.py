"""Append-only event log for pipeline state transitions.

Every state transition emits one JSON line to `<state_dir>/events.jsonl`. A
future dashboard tails that file (and/or receives `AGENTIC_EVENT_WEBHOOK`
POSTs) without caring how the pipeline is implemented.

Event vocabulary (extend cautiously — schema is a contract):
    task_claimed       slug
    benchmark_written  slug
    benchmark_reviewed slug
    attempt_started    slug
    attempt_done       slug
    graded             slug, attempt, verdict_path
    pipeline_idle      reason
    pipeline_failed    slug, reason
    pipeline_paused    slug, stage
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic.config import settings

EVENT_FILE = "events.jsonl"


def _state_dir() -> Path:
    p = Path(settings.state_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def emit(event_type: str, **fields: Any) -> dict[str, Any]:
    """Append one event record. Returns the record for inspection.

    POSTs to `AGENTIC_EVENT_WEBHOOK` on a best-effort basis (failures don't
    propagate — the local log is the source of truth).
    """
    record: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "type": event_type,
        **fields,
    }
    line = json.dumps(record, default=str)

    events_path = _state_dir() / EVENT_FILE
    with events_path.open("a") as f:
        f.write(line + "\n")

    if settings.event_webhook:
        try:
            req = urllib.request.Request(
                settings.event_webhook,
                data=line.encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass  # best-effort

    return record


def read_events() -> list[dict[str, Any]]:
    """Read all events. Useful for tests and the future dashboard backend."""
    path = _state_dir() / EVENT_FILE
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events
