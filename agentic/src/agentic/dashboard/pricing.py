"""Token → USD conversion for tiers that report tokens but not cost.

The Claude/Expert tier returns `cost_usd` directly from the Agent SDK, so we
never price it here. The Nebius OpenAI-compatible tiers (Nemotron, Cosmos)
report tokens with `cost_usd = 0.0`; we fill that gap from this table.

Rates are USD per 1M tokens. Keep model keys in sync with `config.TIER`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rate:
    input_per_mtok: float
    output_per_mtok: float


# Source: Nebius Token Factory pricing (confirmed 2026-06-13).
PRICING: dict[str, Rate] = {
    "nvidia/Nemotron-3-Ultra-550b-a55b": Rate(input_per_mtok=1.00, output_per_mtok=3.00),
    "nvidia/Cosmos3-Super-Reasoner": Rate(input_per_mtok=0.10, output_per_mtok=0.30),
}


def cost_for(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    reported_cost_usd: float = 0.0,
) -> float:
    """Best estimate of USD for one model call.

    Prefer the provider-reported cost when present (Claude). Otherwise price
    from the local table. Cache-read/creation tokens are billed as input here
    (the token-only tiers don't use caching, so this only matters if a future
    provider does); refine per-provider if that changes.
    """
    if reported_cost_usd:
        return reported_cost_usd
    rate = PRICING.get(model)
    if rate is None:
        return 0.0
    billed_input = input_tokens + cache_read_input_tokens + cache_creation_input_tokens
    return (
        billed_input / 1_000_000 * rate.input_per_mtok
        + output_tokens / 1_000_000 * rate.output_per_mtok
    )
