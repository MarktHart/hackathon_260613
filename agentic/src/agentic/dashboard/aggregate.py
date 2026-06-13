"""Fold the event log + block state into a dashboard view model.

Read-only: consumes `events.read_events()` and `blocks.load_state()`, never
writes. The output is plain dicts (JSON-ready) so `server` can ship them over
HTTP/SSE without a serialization layer.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic.blocks import BlockState, load_state, parse_blocks
from agentic.config import settings
from agentic.dashboard.pricing import cost_for
from agentic.events import read_events

# Human-judged rubric criteria in the order the jury writes them (see verdict.py).
# Each is a {"score": int, "note": str} object in verdict.json.
VERDICT_CRITERIA = [
    "architecture_fit",
    "baseline_comparison",
    "faithfulness",
    "operating_range",
    "hardcoded_weights_bonus",
    "visual_judgement",
    "visualisation_rationale",
]

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

# Rank verdicts so we can pick the single "best" attempt per problem. Higher is
# better; ties break on the mean rubric score. An ungraded attempt ranks below
# every graded one.
_VERDICT_RANK = {"pass": 3, "borderline": 2, "fail": 1, "unscored": 0}


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


def _load_verdict(path: str | None) -> dict[str, Any] | None:
    """Load + summarize the jury's verdict.json for the dashboard.

    Returns None if there's no verdict yet or the file is missing/unreadable
    (e.g. the attempt dir was cleaned up). The shape is UI-ready: a flat list of
    scored criteria plus the headline `overall`/`notes` and a mean of the scored
    criteria (0 = N/A entries are excluded from the mean).
    """
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return None

    criteria: list[dict[str, Any]] = []
    scores: list[int] = []
    for key in VERDICT_CRITERIA:
        c = data.get(key)
        if not isinstance(c, dict):
            continue
        score = int(c.get("score") or 0)
        criteria.append({"key": key, "score": score, "note": c.get("note") or ""})
        if score > 0:  # 0 == not applicable; keep out of the mean.
            scores.append(score)

    return {
        "overall": data.get("overall") or "unscored",
        "notes": data.get("notes") or "",
        "criteria": criteria,
        "score_avg": round(sum(scores) / len(scores), 2) if scores else None,
    }


def _attempt_sort_key(name: str) -> tuple[int, int]:
    """Order attempt folders chronologically: first_pass, pass_2, pass_3, …"""
    if name == "first_pass":
        return (0, 0)
    if name.startswith("pass_"):
        try:
            return (1, int(name.split("_", 1)[1]))
        except ValueError:
            return (2, 0)
    return (3, 0)


def _verdict_quality(verdict: dict[str, Any] | None) -> tuple[int, float]:
    """Sort key for picking the best attempt: (overall rank, mean score)."""
    if not verdict:
        return (-1, 0.0)
    return (_VERDICT_RANK.get(verdict.get("overall"), 0), verdict.get("score_avg") or 0.0)


def _load_attempts(slug: str) -> list[dict[str, Any]]:
    """Enumerate every attempt folder under `experiments/<slug>/`.

    Each attempt is one solver pass (first_pass, pass_2, …). We surface its
    verdict (if graded) and whether it ships a Gradio `app.py` the dashboard
    can launch. Read-only; missing/unreadable files degrade to None/False.
    """
    goal_dir = Path("experiments") / slug
    if not goal_dir.is_dir():
        return []
    attempts: list[dict[str, Any]] = []
    for d in goal_dir.iterdir():
        if not d.is_dir():
            continue
        # Attempt folders follow the picker's naming (see pipeline._suggest_attempt_name);
        # everything else here is goal scaffolding (__pycache__, results, symlinks).
        if d.name != "first_pass" and not d.name.startswith("pass_"):
            continue
        attempts.append(
            {
                "name": d.name,
                "verdict": _load_verdict(str(d / "verdict.json")),
                "has_app": (d / "app.py").is_file(),
            }
        )
    attempts.sort(key=lambda a: _attempt_sort_key(a["name"]))
    return attempts


def build_view(
    events: list[dict[str, Any]] | None = None,
    states: dict[str, BlockState] | None = None,
    now_iso: str | None = None,
) -> dict[str, Any]:
    """Build the full dashboard snapshot from events + block state."""
    events = read_events() if events is None else events
    states = load_state() if states is None else states
    blocks = parse_blocks()
    titles = {b.slug: b.title for b in blocks}

    now = _epoch(now_iso) if now_iso else datetime.now(UTC).timestamp()

    # Per-slug accumulators.
    problems: dict[str, dict[str, Any]] = {}
    cost_series: list[dict[str, Any]] = []
    cumulative_cost = 0.0
    cumulative_by_model: dict[str, float] = {}
    cumulative_tokens = 0
    cumulative_tokens_by_model: dict[str, int] = {}

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
                "verdict": None,
                "per_model": {},  # model -> tokens + cost
                "stages": {},  # stage -> interval + tokens + cost
                "totals": {**_empty_tokens(), "cost_usd": 0.0, "n_calls": 0},
            }
        return problems[slug]

    # Seed every BLOCKS.md task so pending (never-claimed) ones are visible
    # and startable from the dashboard, even with no events/state yet.
    for b in blocks:
        problem(b.slug)

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

        call_tokens = int(ev.get("input_tokens") or 0) + int(ev.get("output_tokens") or 0)
        cumulative_cost += call_cost
        cumulative_by_model[model] = cumulative_by_model.get(model, 0.0) + call_cost
        cumulative_tokens += call_tokens
        cumulative_tokens_by_model[model] = cumulative_tokens_by_model.get(model, 0) + call_tokens
        # Each point carries the running totals (cost and tokens) plus a snapshot
        # of every model's cumulative cost/tokens so far, so the UI can plot
        # either metric, as a total or per-model, on a shared x-axis (the global
        # call index). Models unseen at this point are simply absent (treated as
        # 0 by the chart).
        cost_series.append(
            {
                "ts": ts,
                "cumulative_cost_usd": round(cumulative_cost, 6),
                "by_model": {m: round(c, 6) for m, c in cumulative_by_model.items()},
                "cumulative_tokens": cumulative_tokens,
                "tokens_by_model": dict(cumulative_tokens_by_model),
            }
        )

    # Merge block state (authoritative status) + derive liveness.
    for slug, state in states.items():
        p = problem(slug)
        p["status"] = state.status
        p["attempt"] = state.attempt
        p["verdict_path"] = state.verdict_path
        p["verdict"] = _load_verdict(state.verdict_path)
        if state.claimed_at:
            p["claimed_at"] = state.claimed_at

    # Finalize each problem: ordered timeline + running/stalled flags.
    out_problems: list[dict[str, Any]] = []
    for p in problems.values():
        p["stage_timeline"] = [p["stages"][s] for s in STAGE_ORDER if s in p["stages"]] + [
            v for k, v in p["stages"].items() if k not in STAGE_ORDER
        ]
        del p["stages"]

        # Enumerate every attempt on disk and let the row's headline verdict be
        # the best of them (block state only tracks the latest attempt).
        attempts = _load_attempts(p["slug"])
        p["attempts"] = attempts
        best = max(attempts, key=lambda a: _verdict_quality(a["verdict"]), default=None)
        p["best_attempt"] = None
        if best and best["verdict"]:
            p["verdict"] = best["verdict"]
            p["best_attempt"] = best["name"]

        active = p["status"] in ACTIVE_STATUSES
        age = now - _epoch(p["last_event_ts"]) if p["last_event_ts"] else None
        p["is_stalled"] = bool(active and age is not None and age > STALL_AFTER_S)
        p["is_running"] = bool(active and not p["is_stalled"])
        # Startable = not actively running. Pending/graded/failed/stalled tasks
        # can be (re)launched; a graded/failed re-run needs --force (server adds it).
        p["is_startable"] = not p["is_running"]
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
        "cost_models": sorted(cumulative_by_model),
        "state_dir": str(settings.state_dir),
    }
