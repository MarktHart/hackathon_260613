"""Fold the event log + block state into a dashboard view model.

Read-only: consumes `events.read_events()` and `blocks.load_state()`, never
writes. The output is plain dicts (JSON-ready) so `server` can ship them over
HTTP/SSE without a serialization layer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from agentic.blocks import BlockState, load_state, parse_blocks
from agentic.config import settings
from agentic.dashboard.pricing import cost_for
from agentic.events import read_events

# Pipeline stage order (mirrors pipeline.py's usage.stage() blocks).
STAGE_ORDER = ["picker", "reviewer", "solver", "jury"]

# Block statuses that mean an agent is actively working the problem.
ACTIVE_STATUSES = {"claimed", "pending_solver", "solving", "awaiting_jury"}

# Map raw event types to the stage they belong to, for "current stage" display.
_EVENT_STAGE = {
    "task_claimed": "claimed",
    "picker_attempt": "picker",
    "benchmark_written": "picker",
    "benchmark_smoke": "picker",
    "benchmark_reviewed": "reviewer",
    "attempt_started": "solver",
    "solver_main_run": "solver",
    "solver_app_boot": "solver",
    "attempt_done": "solver",
    "graded": "jury",
}

# An active slug with no event newer than this is shown as "stalled", not
# "running" (its process likely died). Generous vs. the longest wall_clock.
STALL_AFTER_S = 1200


def _epoch(ts: str | None) -> float:
    if not ts:
        return 0.0
    try:
        return datetime.fromisoformat(ts).timestamp()
    except ValueError:
        return 0.0


def _empty_tokens() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }


def _add_tokens(dst: dict[str, int], ev: dict[str, Any]) -> None:
    for k in _empty_tokens():
        dst[k] += int(ev.get(k) or 0)


def build_view(
    events: list[dict[str, Any]] | None = None,
    states: dict[str, BlockState] | None = None,
    now_iso: str | None = None,
) -> dict[str, Any]:
    """Build the full dashboard snapshot from events + block state."""
    events = read_events() if events is None else events
    states = load_state() if states is None else states
    titles = {b.slug: b.title for b in parse_blocks()}

    now = _epoch(now_iso) if now_iso else datetime.now(UTC).timestamp()

    # Per-slug accumulators.
    problems: dict[str, dict[str, Any]] = {}
    cost_series: list[dict[str, Any]] = []
    cumulative_cost = 0.0

    def problem(slug: str) -> dict[str, Any]:
        if slug not in problems:
            problems[slug] = {
                "slug": slug,
                "title": titles.get(slug, slug),
                "status": "pending",
                "current_stage": None,
                "last_event_ts": None,
                "claimed_at": None,
                "attempt": None,
                "verdict_path": None,
                "per_model": {},  # model -> tokens + cost
                "stages": {},  # stage -> interval + tokens + cost
                "totals": {**_empty_tokens(), "cost_usd": 0.0, "n_calls": 0},
            }
        return problems[slug]

    for ev in events:
        slug = ev.get("slug")
        etype = ev.get("type")
        ts = ev.get("ts")
        if not isinstance(slug, str):
            continue
        p = problem(slug)

        # Track recency + current stage from any slug-bearing event.
        if ts and (p["last_event_ts"] is None or ts > p["last_event_ts"]):
            p["last_event_ts"] = ts
            stage = _EVENT_STAGE.get(etype or "")
            if stage:
                p["current_stage"] = stage
        if etype == "task_claimed":
            p["claimed_at"] = ts

        if etype != "model_usage":
            continue

        # --- token + cost accounting ---
        model = ev.get("model") or "unknown"
        stage = ev.get("stage") or "unknown"
        call_cost = cost_for(
            model=model,
            input_tokens=int(ev.get("input_tokens") or 0),
            output_tokens=int(ev.get("output_tokens") or 0),
            cache_read_input_tokens=int(ev.get("cache_read_input_tokens") or 0),
            cache_creation_input_tokens=int(ev.get("cache_creation_input_tokens") or 0),
            reported_cost_usd=float(ev.get("cost_usd") or 0.0),
        )

        pm = p["per_model"].setdefault(model, {**_empty_tokens(), "cost_usd": 0.0, "n_calls": 0})
        _add_tokens(pm, ev)
        pm["cost_usd"] += call_cost
        pm["n_calls"] += 1

        _add_tokens(p["totals"], ev)
        p["totals"]["cost_usd"] += call_cost
        p["totals"]["n_calls"] += 1

        st = p["stages"].setdefault(
            stage,
            {"stage": stage, "started_ts": ts, "ended_ts": ts, **_empty_tokens(), "cost_usd": 0.0},
        )
        if ts and (st["started_ts"] is None or ts < st["started_ts"]):
            st["started_ts"] = ts
        if ts and (st["ended_ts"] is None or ts > st["ended_ts"]):
            st["ended_ts"] = ts
        _add_tokens(st, ev)
        st["cost_usd"] += call_cost

        cumulative_cost += call_cost
        cost_series.append({"ts": ts, "cumulative_cost_usd": round(cumulative_cost, 6)})

    # Merge block state (authoritative status) + derive liveness.
    for slug, state in states.items():
        p = problem(slug)
        p["status"] = state.status
        p["attempt"] = state.attempt
        p["verdict_path"] = state.verdict_path
        if state.claimed_at:
            p["claimed_at"] = state.claimed_at

    # Finalize each problem: ordered timeline + running/stalled flags.
    out_problems: list[dict[str, Any]] = []
    for p in problems.values():
        p["stage_timeline"] = [
            p["stages"][s] for s in STAGE_ORDER if s in p["stages"]
        ] + [v for k, v in p["stages"].items() if k not in STAGE_ORDER]
        del p["stages"]

        active = p["status"] in ACTIVE_STATUSES
        age = now - _epoch(p["last_event_ts"]) if p["last_event_ts"] else None
        p["is_stalled"] = bool(active and age is not None and age > STALL_AFTER_S)
        p["is_running"] = bool(active and not p["is_stalled"])
        p["last_event_age_s"] = round(age, 1) if age is not None else None
        out_problems.append(p)

    # Sort: running first, then most-recent activity.
    out_problems.sort(
        key=lambda p: (not p["is_running"], -_epoch(p["last_event_ts"])),
    )

    grand = {**_empty_tokens(), "cost_usd": 0.0, "n_calls": 0}
    for p in out_problems:
        for k in _empty_tokens():
            grand[k] += p["totals"][k]
        grand["cost_usd"] += p["totals"]["cost_usd"]
        grand["n_calls"] += p["totals"]["n_calls"]

    return {
        "generated_at": now_iso or datetime.now(UTC).isoformat(),
        "running_count": sum(1 for p in out_problems if p["is_running"]),
        "stalled_count": sum(1 for p in out_problems if p["is_stalled"]),
        "problems": out_problems,
        "totals": grand,
        "cost_series": cost_series,
        "state_dir": str(settings.state_dir),
    }
