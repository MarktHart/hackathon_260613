"""Sub-agent definitions the orchestrator can delegate to.

Each entry becomes invokable via the built-in `Agent` tool. The
`external-dispatcher` agent is the one that crosses the boundary into a
non-Claude model — keep its toolset narrow so it can't do anything other than
prepare a prompt and forward it.
"""

from claude_agent_sdk import AgentDefinition

DEFAULT_AGENTS: dict[str, AgentDefinition] = {
    "external-dispatcher": AgentDefinition(
        description=(
            "Forwards a self-contained task to an external LLM (OpenAI, Gemini, "
            "local model, etc.) via the dispatch_to_external_llm tool. Use this "
            "when the task needs a non-Claude model."
        ),
        prompt=(
            "You wrap a single external LLM call. Restate the task as a complete, "
            "self-contained prompt — the external model has no access to this "
            "conversation. Call `dispatch_to_external_llm` exactly once, then "
            "return its response verbatim along with a one-line note on which "
            "model was used."
        ),
        tools=["mcp__external-llm__dispatch_to_external_llm"],
    ),
    "researcher": AgentDefinition(
        description="Read-only research over the codebase and the web.",
        prompt=(
            "Investigate the question using Read, Glob, Grep, and WebSearch. "
            "Return a focused brief: findings, file:line citations, open questions."
        ),
        tools=["Read", "Glob", "Grep", "WebSearch", "WebFetch"],
    ),
    "experimenter": AgentDefinition(
        description=(
            "Implements one attempt at a mechanistic-interpretability goal under "
            "experiments/<goal>/<attempt_name>/. Reads the assigned goal's "
            "README.md and follows README_EXPERIMENT.md."
        ),
        prompt=(
            "You have been assigned a mech-interp experiment. Your sequence is:\n"
            "1. Read README_EXPERIMENT.md at the repo root for the expected shape.\n"
            "2. Read experiments/<goal>/README.md for the specific goal.\n"
            "3. Scaffold experiments/<goal>/<attempt_name>/ with a symlinked "
            "   pyproject.toml, write main.py (compute), app.py (Gradio), and "
            "   README.md (what + why-this-viz).\n"
            "4. Run main.py to produce results.\n"
            "5. Boot-check app.py: launch in the background with Bash "
            "   (`uv run python app.py &`), use Monitor on its stdout, wait "
            "   until you see 'Running on local URL', then kill the process. "
            "   A green compute step with a broken app.py is not done.\n"
            "Use the shared workspace venv — do not create a new one."
        ),
        tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Monitor"],
    ),
}
