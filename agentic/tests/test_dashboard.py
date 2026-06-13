"""Dashboard aggregation tests — pure functions, no server, no filesystem.

`build_view` takes events + states directly, so we exercise token/cost
rollup, stage timeline, and liveness without booting FastAPI.
"""

from __future__ import annotations

import pytest

from agentic.blocks import Block, BlockState
from agentic.dashboard import aggregate, launcher
from agentic.dashboard.aggregate import STALL_AFTER_S, build_view
from agentic.dashboard.launcher import UnknownSlug, launch_slug
from agentic.dashboard.pricing import cost_for


def _usage(slug: str, stage: str, model: str, inp: int, out: int, ts: str) -> dict:
    return {
        "ts": ts,
        "type": "model_usage",
        "slug": slug,
        "stage": stage,
        "model": model,
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cost_usd": 0.0,
    }


def test_pricing_fills_token_only_tiers() -> None:
    # Nemotron: $1/M in, $3/M out.
    c = cost_for(
        model="nvidia/Nemotron-3-Ultra-550b-a55b", input_tokens=1_000_000, output_tokens=1_000_000
    )
    assert round(c, 6) == 4.0
    # Provider-reported cost wins over the table.
    assert (
        cost_for(model="anything", input_tokens=999, output_tokens=999, reported_cost_usd=0.5)
        == 0.5
    )
    # Unknown model with no reported cost → 0.
    assert cost_for(model="mystery", input_tokens=1000, output_tokens=1000) == 0.0


def test_tokens_and_cost_roll_up_per_problem() -> None:
    events = [
        {"ts": "2026-06-13T10:00:00+00:00", "type": "task_claimed", "slug": "a", "title": "A"},
        _usage(
            "a",
            "picker",
            "nvidia/Cosmos3-Super-Reasoner",
            1_000_000,
            1_000_000,
            "2026-06-13T10:00:01+00:00",
        ),
        _usage(
            "a",
            "solver",
            "nvidia/Cosmos3-Super-Reasoner",
            1_000_000,
            0,
            "2026-06-13T10:00:02+00:00",
        ),
    ]
    states = {"a": BlockState(slug="a", status="solving")}
    v = build_view(events=events, states=states, now_iso="2026-06-13T10:00:03+00:00")

    p = next(p for p in v["problems"] if p["slug"] == "a")
    assert p["totals"]["input_tokens"] == 2_000_000
    assert p["totals"]["output_tokens"] == 1_000_000
    assert p["totals"]["n_calls"] == 2
    # Cosmos: $0.10/M in, $0.30/M out → in 2M*0.1 + out 1M*0.3 = 0.2 + 0.3 = 0.5
    assert round(p["totals"]["cost_usd"], 6) == 0.5
    # Two stages in canonical order.
    assert [s["stage"] for s in p["stage_timeline"]] == ["picker", "solver"]
    # Grand total mirrors the single problem.
    assert round(v["totals"]["cost_usd"], 6) == 0.5
    assert len(v["cost_series"]) == 2


def test_running_vs_stalled_liveness() -> None:
    events = [
        _usage("a", "picker", "nvidia/Cosmos3-Super-Reasoner", 1, 1, "2026-06-13T10:00:00+00:00")
    ]
    states = {"a": BlockState(slug="a", status="solving")}

    fresh = build_view(events=events, states=states, now_iso="2026-06-13T10:00:30+00:00")
    pa = fresh["problems"][0]
    assert pa["is_running"] and not pa["is_stalled"]
    assert fresh["running_count"] == 1

    # Same state, but the last event is older than the stall window.
    late = "2026-06-13T10:00:00+00:00"
    far = f"2026-06-13T10:{STALL_AFTER_S // 60 + 5:02d}:00+00:00"
    stalled = build_view(
        events=[_usage("a", "picker", "m", 1, 1, late)], states=states, now_iso=far
    )
    ps = stalled["problems"][0]
    assert ps["is_stalled"] and not ps["is_running"]
    assert stalled["stalled_count"] == 1


def test_pending_blocks_are_visible_and_startable(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two BLOCKS.md tasks, neither with events or state.
    monkeypatch.setattr(
        aggregate,
        "parse_blocks",
        lambda: [Block(slug="x", title="X", spec=""), Block(slug="y", title="Y", spec="")],
    )
    v = build_view(events=[], states={}, now_iso="2026-06-13T10:00:00+00:00")
    by = {p["slug"]: p for p in v["problems"]}
    assert set(by) == {"x", "y"}
    assert by["x"]["status"] == "pending"
    assert by["x"]["is_startable"] and not by["x"]["is_running"]


def test_running_block_is_not_startable() -> None:
    events = [
        _usage("a", "solver", "nvidia/Cosmos3-Super-Reasoner", 1, 1, "2026-06-13T10:00:00+00:00")
    ]
    states = {"a": BlockState(slug="a", status="solving")}
    v = build_view(events=events, states=states, now_iso="2026-06-13T10:00:05+00:00")
    p = v["problems"][0]
    assert p["is_running"] and not p["is_startable"]


def _att(name: str, overall: str | None, avg: float | None) -> dict:
    verdict = None if overall is None else {"overall": overall, "score_avg": avg, "criteria": [], "notes": ""}
    return {"name": name, "verdict": verdict, "has_app": True}


def test_row_verdict_is_best_over_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    # first_pass failed; pass_2 was good — the row should headline pass_2.
    monkeypatch.setattr(
        aggregate,
        "_load_attempts",
        lambda slug: [_att("first_pass", "fail", 2.0), _att("pass_2", "good", 4.5)],
    )
    states = {"a": BlockState(slug="a", status="graded")}
    v = build_view(events=[], states=states, now_iso="2026-06-13T10:00:00+00:00")
    p = next(p for p in v["problems"] if p["slug"] == "a")
    assert p["best_attempt"] == "pass_2"
    assert p["verdict"]["overall"] == "good"
    assert [a["name"] for a in p["attempts"]] == ["first_pass", "pass_2"]


def test_best_breaks_ties_on_mean_score(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both borderline; the higher mean score wins.
    monkeypatch.setattr(
        aggregate,
        "_load_attempts",
        lambda slug: [_att("first_pass", "borderline", 2.5), _att("pass_2", "borderline", 3.8)],
    )
    v = build_view(events=[], states={"a": BlockState(slug="a", status="graded")}, now_iso="2026-06-13T10:00:00+00:00")
    p = next(p for p in v["problems"] if p["slug"] == "a")
    assert p["best_attempt"] == "pass_2"


def test_launch_rejects_unknown_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    # Validation happens before any subprocess spawn, so this never launches.
    monkeypatch.setattr(launcher, "known_slugs", lambda: {"real_task"})
    with pytest.raises(UnknownSlug):
        launch_slug("not_a_task")


def test_graded_problem_is_not_running() -> None:
    events = [
        _usage("a", "jury", "nvidia/Cosmos3-Super-Reasoner", 10, 10, "2026-06-13T10:00:00+00:00")
    ]
    states = {"a": BlockState(slug="a", status="graded")}
    v = build_view(events=events, states=states, now_iso="2026-06-13T10:00:01+00:00")
    p = v["problems"][0]
    assert not p["is_running"] and not p["is_stalled"]
    assert v["running_count"] == 0
