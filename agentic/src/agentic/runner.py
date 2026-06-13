"""Tier-routed model dispatch.

- `run_at_tier_agentic` — one Claude Agent SDK loop at an `agentic`-mode tier
  (the EXPERT tier — reviewer + jury). Streams SDK messages.
- `run_at_tier_completion` — one OpenAI-compatible chat completion at a
  `completion`-mode tier (STANDARD + QUICK — picker + solver) via the Nebius
  Token Factory. Returns the assistant's full text content.

Prompt caching: tier-1 stages share long, *stable* system prompts across runs.
The Anthropic API auto-caches identical prefixes within a 5-minute window,
so the discipline is to keep `system_prompt` byte-identical between calls
(one constant per stage, no f-string interpolation) and append run-specific
context in the user prompt only.

Wall-clock budgets (`TIER.wall_clock_s`) are enforced by the *caller* via
`asyncio.wait_for` around the consumer — see `pipeline._drain_with_timeout`.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, query

from agentic import usage
from agentic.config import TIER, Tier


def options_at_tier(tier: Tier, **overrides: Any) -> ClaudeAgentOptions:
    """Build `ClaudeAgentOptions` for an `agentic`-mode tier.

    Sets `effort` (extended-thinking budget) when the tier configures one —
    e.g. tier 1 defaults to `"high"`.
    """
    cfg = TIER[tier]
    if cfg.mode != "agentic":
        raise ValueError(f"Tier {tier.value} is mode={cfg.mode!r}; use `run_at_tier_completion`.")
    kwargs: dict[str, Any] = {"model": cfg.model, "max_turns": cfg.max_turns}
    if cfg.effort is not None:
        kwargs["effort"] = cfg.effort
    base = ClaudeAgentOptions(**kwargs)
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


async def run_at_tier_agentic(
    tier: Tier,
    prompt: str,
    *,
    system_prompt: str | None = None,
    allowed_tools: list[str] | None = None,
    cwd: str | None = None,
) -> AsyncIterator[Any]:
    """Run one Claude Agent SDK loop at an `agentic`-mode tier. Streams messages."""
    overrides: dict[str, Any] = {}
    if system_prompt is not None:
        overrides["system_prompt"] = system_prompt
    if allowed_tools is not None:
        overrides["allowed_tools"] = allowed_tools
    if cwd is not None:
        overrides["cwd"] = cwd
    options = options_at_tier(tier, **overrides)
    async for message in query(prompt=prompt, options=options):
        usage.record_sdk_result(message, default_model=TIER[tier].model)
        yield message


async def run_at_tier_completion(
    tier: Tier,
    prompt: str,
    *,
    system_prompt: str = "",
) -> str:
    """Run one OpenAI-compatible chat completion at a `completion`-mode tier.

    The Nebius Token Factory uses OpenAI's wire format, so the OpenAI SDK is
    the cleanest client. No tool use — completion-mode tiers emit file-block
    output that the pipeline parses and applies via `file_blocks.apply_blocks`.
    """
    from openai import AsyncOpenAI

    cfg = TIER[tier]
    if cfg.mode != "completion":
        raise ValueError(f"Tier {tier.value} is mode={cfg.mode!r}; use `run_at_tier_agentic`.")
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise OSError(f"Tier {tier.value} expects env var {cfg.api_key_env}; not set.")

    client = AsyncOpenAI(base_url=cfg.api_base, api_key=api_key)
    response = await client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    u = getattr(response, "usage", None)
    if u is not None:
        usage.record(
            model=cfg.model,
            input_tokens=int(getattr(u, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(u, "completion_tokens", 0) or 0),
        )
    return response.choices[0].message.content or ""
