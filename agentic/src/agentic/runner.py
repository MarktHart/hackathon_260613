"""Top-level entrypoints.

- `run_task` — free-form orchestrator (used by `agentic run` / `agentic run-goal`).
- `run_at_tier_agentic` — one Claude Agent SDK loop at an `agentic`-mode tier.
- `run_at_tier_completion` — one OpenAI-compatible chat completion at a
  `completion`-mode tier via the Nebius Token Factory.

Prompt caching: tier-1 stages (reviewer, jury) share a long, *stable* system
prompt across runs. Anthropic's API auto-caches identical prefixes within a
5-minute window, so the design discipline is to keep `system_prompt` byte-
identical between calls (one constant per stage, no string interpolation) and
appendrun-specific context in the user prompt only. The TIER.wall_clock_s
budget per call is enforced by the *caller* via `asyncio.wait_for` — pipelines
wrap each stage so a runaway model can be cut off without leaking the
subprocess.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, query

from agentic.agents import DEFAULT_AGENTS
from agentic.config import TIER, Tier, settings
from agentic.tools import external_llm_mcp_server

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the orchestrator of an agentic system. You can:

- Use built-in tools (Read, Write, Edit, Bash, Glob, Grep, WebSearch, WebFetch) directly.
- Delegate focused sub-tasks via the Agent tool. Available agents:
  * `researcher` — read-only codebase + web investigation.
  * `experimenter` — implements one attempt at a mech-interp goal. Brief it
    with the goal directory (`experiments/<goal>/`). It will read
    `README_EXPERIMENT.md` for shape and the goal's `README.md` for spec.
  * `external-dispatcher` — forwards a self-contained prompt to a non-Claude LLM.

Experiments live under `experiments/<goal>/<attempt_name>/`. Every attempt is a
uv workspace member that symlinks `pyproject.toml` to `experiments/pyproject.toml`,
so they share one venv with torch, transformers, datasets, gradio. Run them with
`uv run python experiments/<goal>/<attempt_name>/main.py` from the repo root.
Experiment scripts use `agentic.experiments.results_dir(__file__)` to write
artefacts under `experiments/<goal>/<attempt_name>/results/<run-id>/`. The full
attempt convention lives in `README_EXPERIMENT.md` — point sub-agents at it.

Prefer delegation over doing everything inline when a sub-task is well-scoped.
Reach for `external-dispatcher` when another model is a better fit (cost,
latency, or a capability Claude doesn't have in this context).
"""


def _default_options(**overrides: Any) -> ClaudeAgentOptions:
    base = ClaudeAgentOptions(
        model=settings.orchestrator_model,
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        allowed_tools=[
            "Read",
            "Write",
            "Edit",
            "Bash",
            "Glob",
            "Grep",
            "WebSearch",
            "WebFetch",
            "Agent",
            "mcp__external-llm__dispatch_to_external_llm",
        ],
        agents=DEFAULT_AGENTS,
        mcp_servers={"external-llm": external_llm_mcp_server},
        max_turns=settings.max_turns,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


async def run_task(
    prompt: str,
    *,
    options: ClaudeAgentOptions | None = None,
) -> AsyncIterator[Any]:
    """Run `prompt` against the orchestrator, yielding each message as it streams."""
    async for message in query(prompt=prompt, options=options or _default_options()):
        yield message


def options_at_tier(tier: Tier, **overrides: Any) -> ClaudeAgentOptions:
    """Build `ClaudeAgentOptions` for an `agentic`-mode tier."""
    cfg = TIER[tier]
    if cfg.mode != "agentic":
        raise ValueError(f"Tier {tier.value} is mode={cfg.mode!r}; use `run_at_tier_completion`.")
    base = ClaudeAgentOptions(
        model=cfg.model,
        max_turns=cfg.max_turns,
    )
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
    """Run one Claude Agent SDK loop at an `agentic`-mode tier. Streams messages.

    The caller is expected to enforce `TIER[tier].wall_clock_s` via
    `asyncio.wait_for` around the consumer — see `pipeline._drain_with_timeout`.
    """
    overrides: dict[str, Any] = {}
    if system_prompt is not None:
        overrides["system_prompt"] = system_prompt
    if allowed_tools is not None:
        overrides["allowed_tools"] = allowed_tools
    if cwd is not None:
        overrides["cwd"] = cwd
    options = options_at_tier(tier, **overrides)
    async for message in query(prompt=prompt, options=options):
        yield message


async def run_at_tier_completion(
    tier: Tier,
    prompt: str,
    *,
    system_prompt: str = "",
) -> str:
    """Run one OpenAI-compatible chat completion at a `completion`-mode tier.

    Returns the assistant's full text content. The Nebius Token Factory uses
    OpenAI's wire format, so the OpenAI SDK is the cleanest client. No tool
    use — completion-mode tiers emit file-block output that the pipeline
    parses and applies via `file_blocks.apply_blocks`.
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
    return response.choices[0].message.content or ""
