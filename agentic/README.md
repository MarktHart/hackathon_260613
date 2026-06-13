# agentic

A rough agentic framework built on top of the **Claude Agent SDK**, with a
first-class **external-LLM dispatch** path via LiteLLM.

## Why this shape

- **Orchestration is Claude.** The Claude Agent SDK gives us the proven
  think → act → observe loop, sub-agents, hooks, permissions, sessions, and
  MCP — the same pieces Claude Code uses. The semantics line up with Anthropic's
  Managed Agents so we can move there later without rewriting agent logic.
- **External-LLM dispatch is a tool.** A single in-process MCP tool
  (`dispatch_to_external_llm`) backed by LiteLLM lets the orchestrator (or a
  dedicated sub-agent) call any provider — OpenAI, Gemini, Bedrock, Vertex,
  Ollama, or any OpenAI-compatible endpoint.
- **Sub-agents are typed.** `AgentDefinition`s in `agents/definitions.py` are
  the unit of specialisation. Add an agent → add a tool → done.

## Layout

```
agentic/
├── pyproject.toml
└── src/agentic/
    ├── config.py            # env-driven Settings
    ├── runner.py            # run_task() — main entrypoint
    ├── cli.py               # `agentic run "<task>"`
    ├── tools/
    │   └── external_llm.py  # @tool + in-process MCP server
    └── agents/
        └── definitions.py   # AgentDefinitions (researcher, external-dispatcher)
```

## Setup

```bash
uv sync
cp .env.example .env   # then fill in keys
```

Required env:

- `ANTHROPIC_API_KEY` — for the orchestrator.
- `OPENAI_API_KEY` (or whichever provider you pick for `AGENTIC_EXTERNAL_MODEL`).

Optional env:

- `AGENTIC_ORCHESTRATOR_MODEL` (default: `claude-opus-4-7`)
- `AGENTIC_EXTERNAL_MODEL` (default: `openai/gpt-4.1`) — LiteLLM `provider/model` format.
- `AGENTIC_EXTERNAL_API_BASE` — point at a local Ollama / LM Studio / LiteLLM proxy.
- `AGENTIC_MAX_TURNS` (default: `30`)

## Run

```bash
uv run agentic run "Summarise README.md, then ask the external LLM to critique the summary"
# or
uv run python examples/quickstart.py
```

## Extending

- **New sub-agent:** add an `AgentDefinition` to `agents/definitions.py`. Restrict
  its `tools=` to what it actually needs.
- **New tool:** add a `@tool` in `tools/`, expose it through `create_sdk_mcp_server`,
  then list it in `runner.py:_default_options.allowed_tools`.
- **Swap external provider:** change `AGENTIC_EXTERNAL_MODEL`. No code changes.

## Next moves

- Hooks for audit logging (`PostToolUse`) and cost tracking.
- Persistent sessions (`resume=` in `ClaudeAgentOptions`).
- Lift to Managed Agents when we want hosted sandboxes.
