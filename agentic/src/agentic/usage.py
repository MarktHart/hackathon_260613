"""Per-slug token / cost accounting across the two model call sites.

The pipeline binds a `UsageTracker` for the duration of `run_pipeline(slug)`
via the `track(slug)` context manager. The two call sites
(`runner.run_at_tier_agentic`, `runner.run_at_tier_completion`) feed every
API response through `record(...)`. Each call emits a `model_usage` event;
on exit the tracker emits a `usage_summary` event aggregated per model.

Cost is captured when the provider returns it directly (Claude Agent SDK's
`ResultMessage.total_cost_usd`). OpenAI-compatible providers (Nebius)
report tokens only — cost is left at 0.0 and can be computed downstream
from the per-model token totals.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from agentic.events import emit


@dataclass
class ModelUsage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0
    n_calls: int = 0


@dataclass
class UsageTracker:
    slug: str
    per_model: dict[str, ModelUsage] = field(default_factory=dict)

    def add(
        self,
        *,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        bucket = self.per_model.setdefault(model, ModelUsage(model=model))
        bucket.input_tokens += input_tokens
        bucket.output_tokens += output_tokens
        bucket.cache_creation_input_tokens += cache_creation_input_tokens
        bucket.cache_read_input_tokens += cache_read_input_tokens
        bucket.cost_usd += cost_usd
        bucket.n_calls += 1

    def summary(self) -> dict[str, Any]:
        per_model = {m: vars(u) for m, u in self.per_model.items()}
        return {
            "slug": self.slug,
            "total_input_tokens": sum(u.input_tokens for u in self.per_model.values()),
            "total_output_tokens": sum(u.output_tokens for u in self.per_model.values()),
            "total_cache_creation_input_tokens": sum(
                u.cache_creation_input_tokens for u in self.per_model.values()
            ),
            "total_cache_read_input_tokens": sum(
                u.cache_read_input_tokens for u in self.per_model.values()
            ),
            "total_cost_usd": sum(u.cost_usd for u in self.per_model.values()),
            "per_model": per_model,
        }


_TRACKER: ContextVar[UsageTracker | None] = ContextVar("agentic_usage_tracker", default=None)
_STAGE: ContextVar[str | None] = ContextVar("agentic_usage_stage", default=None)


@contextmanager
def track(slug: str) -> Iterator[UsageTracker]:
    """Bind a tracker for the duration of one pipeline run.

    Emits `usage_summary` on exit even if the body raises.
    """
    tracker = UsageTracker(slug=slug)
    token = _TRACKER.set(tracker)
    try:
        yield tracker
    finally:
        _TRACKER.reset(token)
        emit("usage_summary", **tracker.summary())


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Tag every `record` inside this block with `stage=name` on its event."""
    token = _STAGE.set(name)
    try:
        yield
    finally:
        _STAGE.reset(token)


def record(
    *,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cost_usd: float = 0.0,
) -> None:
    """Record one model call. Aggregates into the active tracker (if any) and
    emits a `model_usage` event."""
    tracker = _TRACKER.get()
    slug = tracker.slug if tracker else None
    if tracker is not None:
        tracker.add(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cost_usd=cost_usd,
        )
    emit(
        "model_usage",
        slug=slug,
        stage=_STAGE.get(),
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cost_usd=cost_usd,
    )


def record_sdk_result(message: Any, *, default_model: str) -> None:
    """Pull usage out of a Claude Agent SDK `ResultMessage` and record it.

    Prefers `model_usage` (per-model breakdown emitted when multiple models
    are used in one session) over the flat `usage` + `total_cost_usd`. Safe
    to call on non-result messages — they're ignored.
    """
    if type(message).__name__ != "ResultMessage":
        return

    model_usage = getattr(message, "model_usage", None) or {}
    if model_usage:
        for model, entry in model_usage.items():
            entry = entry or {}
            record(
                model=model,
                input_tokens=int(entry.get("inputTokens") or entry.get("input_tokens") or 0),
                output_tokens=int(entry.get("outputTokens") or entry.get("output_tokens") or 0),
                cache_creation_input_tokens=int(
                    entry.get("cacheCreationInputTokens")
                    or entry.get("cache_creation_input_tokens")
                    or 0
                ),
                cache_read_input_tokens=int(
                    entry.get("cacheReadInputTokens") or entry.get("cache_read_input_tokens") or 0
                ),
                cost_usd=float(entry.get("costUSD") or entry.get("cost_usd") or 0.0),
            )
        return

    usage = getattr(message, "usage", None) or {}
    record(
        model=default_model,
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens") or 0),
        cache_read_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
        cost_usd=float(getattr(message, "total_cost_usd", None) or 0.0),
    )
