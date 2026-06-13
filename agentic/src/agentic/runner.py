"""Top-level entrypoint: hand a task to the orchestrator and stream messages back."""

from collections.abc import AsyncIterator
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, query

from agentic.agents import DEFAULT_AGENTS
from agentic.config import settings
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
